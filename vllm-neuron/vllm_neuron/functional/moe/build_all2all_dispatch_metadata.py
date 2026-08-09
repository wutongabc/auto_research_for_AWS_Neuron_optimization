# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.distributed.parallel_state import GroupCoordinator

from vllm_neuron.nki.nki_hop import can_run_kernel
from ..cumsum import cumsum


def build_all2all_dispatch_metadata(
    expert_index: torch.Tensor,
    num_experts: int,
    num_elements_per_token: int,
    group: GroupCoordinator,
    recv_displs: torch.Tensor = None,
) -> torch.Tensor:
    """Build metadata for all2all dispatch. Computes send_counts and send_displs from expert_index
    and places recv_displs in row3 of the metadata buffer if provided.

    All metadata rows represent the number of elements being sent/recieved, not the number of tokens.

    Args:
        expert_index (torch.Tensor): [T, K] int32 tensor of expert indices per token.
        num_experts (int): Total number of experts.
        num_elements_per_token (int): Number of elements per token (e.g. hidden_size). Used to
            convert per-rank token counts into element counts for the all2all transfer.
        group (GroupCoordinator): The distributed group coordinator. Its world_size determines
            the number of destination ranks (replica_group_size).
        recv_displs (torch.Tensor, optional): Pre-computed recv displacements.

    Returns:
        torch.Tensor: [4, replica_group_size] uint32 tensor.
            Row 0: send counts (number of elements per rank), Row 1: send displacements,
            Row 2: recv counts (zeros), Row 3: recv displacements (zeros or provided).

    Example:
        >>> expert_index = torch.arange(8, dtype=torch.int32).unsqueeze(1)
        >>> meta = build_all2all_dispatch_metadata(expert_index, num_experts=8,
        ...     num_elements_per_token=1, group=group)
        >>> # meta[0] = [1,1,1,1,1,1,1,1], meta[1] = [0,1,2,3,4,5,6,7]
    """

    # Convert from GroupCoordinator -> size
    replica_group_size = group.world_size

    _validate_inputs(replica_group_size, num_experts, num_elements_per_token)

    if _can_use_kernel(expert_index):
        # TODO: NKI kernel dispatch
        pass

    return _torch_impl(
        expert_index,
        num_experts,
        num_elements_per_token,
        replica_group_size,
        recv_displs,
    )


def _validate_inputs(
    replica_group_size: int,
    num_experts: int,
    num_elements_per_token: int,
) -> None:
    """Validate inputs for build_all2all_dispatch_metadata."""
    assert num_experts % replica_group_size == 0, (
        f"num_experts must be divisible by replica_group_size, got {num_experts=}, {replica_group_size=}"
    )
    assert replica_group_size > 1, (
        f"expected replica_group_size > 1, got {replica_group_size=}"
    )
    assert num_elements_per_token > 0, (
        f"num_elements_per_token must be greater than 0, got {num_elements_per_token=}"
    )


def _can_use_kernel(
    expert_index: torch.Tensor,
) -> bool:
    """Check if the NKI kernel can be used."""
    if not can_run_kernel(expert_index):
        return False

    # TODO: add kernel integration
    return False


def _torch_impl(
    expert_index: torch.Tensor,
    num_experts: int,
    num_elements_per_token: int,
    replica_group_size: int,
    recv_displs: torch.Tensor = None,
) -> torch.Tensor:
    """PyTorch implementation of build_all2all_dispatch_metadata."""

    # Utilize int32 ops internally, which XLA natively supports
    zeros = torch.zeros(
        replica_group_size, dtype=torch.int32, device=expert_index.device
    )
    recv_counts = zeros
    recv_displs = recv_displs.to(torch.int32) if recv_displs is not None else zeros

    # Step 1: Map each (token, k) to its destination rank.
    # When a token is routed to multiple experts that are on the same destination rank,
    # the (token, rank) pair is de-duped: it counts as 1, not K.
    T, K = expert_index.shape
    n_local_experts = num_experts // replica_group_size
    dst_ranks = (expert_index // n_local_experts).to(torch.int32)  # [T, K]

    # Step 2: Build [T, replica_group_size] mask. scatter_ writes 1 at each (t, dst_rank)
    # pair; duplicate writes within the same row collapse to a single 1, which is what
    # de-dupes (token, rank) pairs that route to the same rank multiple times.
    rank_mask = torch.zeros(
        T, replica_group_size, dtype=torch.int32, device=expert_index.device
    )
    rank_mask.scatter_(1, dst_ranks, 1)

    # Step 3: Sum across tokens to get per-rank de-duped token counts, then scale by
    # num_elements_per_token to get per-rank send_counts in elements (matches the unit
    # of the all2all_v collective metadata).
    send_counts = rank_mask.sum(dim=0).to(torch.int32) * num_elements_per_token

    # Compute send_displs as cumsum(send_counts) with offset of 1
    send_displs = torch.zeros(
        replica_group_size, dtype=torch.int32, device=expert_index.device
    )
    send_displs[1:] = cumsum(send_counts[:-1].unsqueeze(0)).squeeze(0)

    # Stack metadata, with cast back to uint32
    return torch.stack([send_counts, send_displs, recv_counts, recv_displs], dim=0).to(
        torch.uint32
    )
