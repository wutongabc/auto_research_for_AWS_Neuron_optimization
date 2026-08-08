# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger

from vllm_neuron import envs
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)

import vllm_neuron.functional as NF
from vllm_neuron.parallel.neuron_parallel_state import (
    get_neuron_ep_group,
)

from ..functional.moe.permute_routed_tokens import _bitcast

logger = init_logger(__name__)

_SUPPORTED_DISPATCH_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16]


class NeuronAll2AllManager(All2AllManagerBase):
    """
    All2All communication based on NKI EP kernels, for 2D Torus and NeuronSwitch topologies.

    2D Torus: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html
    NeuronSwitch: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

        # Collective topology
        self.is_neuron_switch = envs.VLLM_NEURON_SWITCH_CC

        # Dispatch/combine implementation
        self._dispatch_func = (
            self._dispatch_switch if self.is_neuron_switch else self._dispatch_torus
        )
        self._combine_func = (
            self._combine_switch if self.is_neuron_switch else self._combine_torus
        )

        # Collective dims
        self.num_tokens = None
        self.num_local_experts = None
        self.num_experts_per_tok = None
        self.num_dispatch_elements_per_tok = None

        # Dispatch metadata
        self.dispatch_recv_tokens = None

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        """
        EP dispatch with All2All-v. Each token is routed to ranks that contain its top-K experts.

        Args:
            hidden_states (torch.Tensor): [T, H] bf16 or fp8 hidden states.
            topk_weights (torch.Tensor): [T, E] bf16 expert affinities, with zeros for non-routed pairs.
            topk_ids (torch.Tensor): [T, K] int32 expert indices.
            is_sequence_parallel (bool): Whether dispatch() is being invoked with sequence parallel sharding.
                Currently, only is_sequence_parallel=True is supported.

        Returns:
            torch.Tensor: [T * world_size, num_dispatch_elements_per_tok], where each row is [hidden | affinities | tok_idx].

        Example:
            >>> dispatched = mgr.dispatch(hidden, topk_weights, topk_ids, is_sequence_parallel=True)
            >>> # Output buffer is statically shaped [T * world_size, num_dispatch_elements_per_tok];
            >>> # rows from each src rank are scattered into the buffer at fixed offsets of T * src_rank,
            >>> # i.e. src_rank's contributed rows occupy [src_rank * T : src_rank * T + dispatch_recv_tokens[src_rank]).
            >>> # The remaining rows in each src's slot (up to T) are zero-padded. Per row layout (in dispatch_dtype):
            >>> #   [0 : H)                          -> hidden state
            >>> #   [H : H + n_local_experts)        -> per-local-expert affinities (bf16, bitcast)
            >>> #   [H + n_local_experts : ...)      -> source token index (int32, bitcast)
        """

        assert is_sequence_parallel, (
            f"NeuronAll2AllManager.dispatch() requires is_sequence_parallel=True but got {is_sequence_parallel=}"
        )

        # FIXME[cc bug]: convert all inputs to bf16, since native fp8 collective is inaccurate
        original_dtype = hidden_states.dtype
        hidden_states = _bitcast(hidden_states, torch.bfloat16)

        self._init_dispatch_combine_dims(hidden_states, topk_weights, topk_ids)
        dispatch_metadata = NF.build_all2all_dispatch_metadata(
            expert_index=topk_ids,
            num_experts=self.num_local_experts,
            num_elements_per_token=self.num_dispatch_elements_per_tok,
            group=get_neuron_ep_group(),
            recv_displs=None,
        )
        dispatch_data = NF.permute_routed_tokens(
            hidden_input=hidden_states,
            expert_index=topk_ids,
            expert_affinities_masked=topk_weights,
            group=get_neuron_ep_group(),
            is_sequence_parallel=is_sequence_parallel,
        )

        dispatch_recv, dispatch_metadata = self._dispatch_func(
            dispatch_data, dispatch_metadata
        )

        # Save dispatch metadata, which for reuse during combine()
        self.dispatch_recv_tokens = (
            dispatch_metadata[2, :] // self.num_dispatch_elements_per_tok
        )

        # FIXME[cc bug]: convert back to original dtype
        dispatch_recv = _bitcast(dispatch_recv, original_dtype)

        return dispatch_recv

    def _init_dispatch_combine_dims(self, hidden_states, topk_weights, topk_ids):
        """Compute T, E, K, dispatch elements per tok for use during dispatch/combine; validate sizes."""
        self.num_tokens, H = hidden_states.shape
        _, self.num_local_experts = topk_weights.shape
        _, self.num_experts_per_tok = topk_ids.shape

        assert self.num_local_experts % get_neuron_ep_group().world_size == 0, (
            f"Expected num_local_experts divisible by all2all group size, got E={self.num_local_experts}, all2all group size={get_neuron_ep_group().world_size}"
        )
        assert topk_weights.dtype == torch.bfloat16, (
            f"all2all only supports topk_weights.dtype == torch.bfloat16, got {topk_weights.dtype=}"
        )
        assert hidden_states.dtype in _SUPPORTED_DISPATCH_DTYPES, (
            f"all2all dispatch only supports hidden_states.dtype in {_SUPPORTED_DISPATCH_DTYPES}, got {hidden_states.dtype=}"
        )
        experts_per_dispatch_rank = (
            self.num_local_experts // get_neuron_ep_group().world_size
        )

        # H + (E/EP + 2) * adj_factor, where adj_factor = 1 when hidden is bf16 and 2 when hidden is fp8
        bf16_per_int32 = bf16_bytes = 2
        self.num_dispatch_elements_per_tok = int(
            H
            + (experts_per_dispatch_rank + bf16_per_int32)
            * (bf16_bytes / hidden_states.element_size())
        )

    def _dispatch_torus(self, data, metadata):
        raise NotImplementedError(
            "All2All communication not yet supported for 2D Torus topology."
        )

    def _dispatch_switch(self, data, metadata):
        """
        Input size: [T * K, num_dispatch_elements_per_tok]
        Output size: [T * all2all_group_size, num_dispatch_elements_per_tok]
        """

        # Output must be initialized with zeros
        output_recv = torch.zeros(
            (
                int(self.num_tokens * get_neuron_ep_group().world_size),
                int(self.num_dispatch_elements_per_tok),
            ),
            dtype=data.dtype,
            device=data.device,
        )

        # Execute all2all-v. Dispatch collective computes recv_counts for reuse during combine(), and does not use rdispls.
        output_recv, metadata = NF.all_to_all_v(
            input=data,
            output=output_recv,
            group=get_neuron_ep_group(),
            metadata=metadata,
            recv_counts_known=False,
            has_rdispls=False,
        )

        return output_recv, metadata

    def combine(self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False):
        """
        EP combine with All2All-v.

        Each token is returned to its dispatch source rank, and partial outputs are reduced. Requires dispatch() to have run first to populate collective metadata.

        Args:
            hidden_states (torch.Tensor): [T * all2all_group_size, H + 2] bf16 expert outputs,
                with the last 2 columns containing int32 token indices.
            is_sequence_parallel (bool): Whether dispatch() is being invoked with sequence parallel sharding.
                Currently, only is_sequence_parallel=True is supported.
        Returns:
            torch.Tensor: [T, H] bf16.

        Example:
            >>> reduced = mgr.combine(expert_out, is_sequence_parallel=True)
            >>> # Output buffer is statically shaped [T, H] bf16; rows correspond 1:1 to the
            >>> # source-rank tokens that were dispatched.
        """

        assert is_sequence_parallel, (
            f"NeuronAll2AllManager.combine() requires is_sequence_parallel=True but got {is_sequence_parallel=}"
        )

        assert all(
            v is not None
            for v in (
                self.num_tokens,
                self.num_local_experts,
                self.num_experts_per_tok,
                self.num_dispatch_elements_per_tok,
                self.dispatch_recv_tokens,
            )
        ), (
            "NeuronAll2AllManager.combine() requires dispatch() to run first to populate metadata."
        )

        # Compute combine send counts using recv counts saved during dispatch
        combine_send_counts = (
            self.dispatch_recv_tokens.to(torch.int32) * hidden_states.shape[1]
        )
        combine_metadata = NF.build_all2all_combine_metadata(
            send_counts=combine_send_counts,
            recv_displs=None,
        )
        combine_recv, combine_metadata = self._combine_func(
            hidden_states, combine_metadata
        )
        output_reduced = NF.topk_reduce(
            input=combine_recv,
            T=self.num_tokens,
            K=self.num_experts_per_tok,
            is_sequence_parallel=is_sequence_parallel,
        )
        return output_reduced

    def _combine_torus(self, data, metadata):
        raise NotImplementedError(
            "All2All communication not yet supported for 2D Torus topology."
        )

    def _combine_switch(self, data, metadata):
        """
        Input size: [T * all2all_group_size, H + 2]
        Output size: [T * all2all_group_size, H + 2]
        """

        # Output must be initialized with zeros
        output_recv = torch.zeros(
            (self.num_tokens * get_neuron_ep_group().world_size, data.shape[1]),
            dtype=data.dtype,
            device=data.device,
        )

        # Execute all2all-v. Combine collective does not compute recv_counts or use rdispls.
        output_recv, metadata = NF.all_to_all_v(
            input=data,
            output=output_recv,
            group=get_neuron_ep_group(),
            metadata=metadata,
            recv_counts_known=True,
            has_rdispls=False,
        )
        return output_recv, metadata
