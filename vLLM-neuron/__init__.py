# SPDX-License-Identifier: Apache-2.0
"""Load upstream vLLM-Neuron and specialize Qwen3 MoE graph construction."""

from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import PathFinder
import importlib
import sys

_UPSTREAM_INIT = "/dev3/zigeng/bc/vllm-neuron/vllm_neuron/__init__.py"
with open(_UPSTREAM_INIT, "rb") as _source:
    exec(compile(_source.read(), _UPSTREAM_INIT, "exec"), globals(), globals())

_TARGET_MODEL = "vllm_neuron.model.qwen3.model_bf16"


def _patch_qwen3_moe(module):
    cls = module.Qwen3MoeExperts
    if getattr(cls, "_opt_block_size_patched", False):
        return
    original_init = cls.__init__

    def _init_with_smaller_blocks(self, config):
        original_init(self, config)
        self.block_size = 128

    cls.__init__ = _init_with_smaller_blocks
    cls._opt_block_size_patched = True

    original_prefill = cls.forward_prefill
    router_impl = importlib.import_module(
        "vllm_neuron.functional.moe.router"
    )
    router_topk_nki = router_impl.wrap_nki(router_impl.router_topk_jit)

    def _forward_prefill_with_nki_router(self, hidden_states, positions):
        if self.norm_topk_prob:
            return original_prefill(self, hidden_states, positions)

        torch = module.torch
        hidden_states = self.post_attention_layernorm(hidden_states)
        token_count = hidden_states.shape[0]
        expert_count = self.router_weight.shape[0]
        router_logits = torch.zeros(
            token_count,
            expert_count,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        expert_affinities = torch.zeros_like(router_logits)
        expert_indices = torch.zeros(
            token_count,
            self.top_k,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        _, _, expert_affinities = router_topk_nki[2](
            x=hidden_states,
            w=self.router_weight.T,
            w_bias=None,
            router_logits=router_logits,
            expert_affinities=expert_affinities,
            expert_index=expert_indices,
            act_fn=router_impl.RouterActFnType.SOFTMAX,
            k=self.top_k,
            x_hbm_layout=1,
            x_sb_layout=0,
            router_pre_norm=True,
            norm_topk_prob=False,
            use_indirect_dma_scatter=False,
            use_column_tiling=False,
            shard_on_tokens=True,
            skip_store_router_logits=True,
            skip_store_expert_index=True,
            use_PE_broadcast_w_bias=False,
        )
        if self.world_size > 1:
            expert_affinities = self.tp_group.all_gather(
                expert_affinities, dim=0
            )
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        last_real_idx = torch.argmax(positions)
        token_indices = torch.arange(
            positions.shape[0], device=positions.device
        )
        padding_mask = token_indices <= last_real_idx
        if self.ep_enabled:
            local_indices = torch.arange(
                self.ep_rank * self.num_experts,
                (self.ep_rank + 1) * self.num_experts,
                device=hidden_states.device,
                dtype=torch.int32,
            )
            expert_affinities = module.NF.get_local_expert_affinities(
                expert_affinities, local_indices
            )

        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = module.NF.build_blockwise_mapping(
            expert_affinities=expert_affinities,
            num_local_experts=self.num_experts,
            num_experts_per_token=self.top_k,
            block_size=self.block_size,
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
            padding_mask=padding_mask,
        )
        if self.fp8_weights_enabled and self.fp8_only:
            gate_up_weight = self._dequant_gate_up_for_prefill()
            down_weight = self._dequant_down_for_prefill()
        else:
            gate_up_weight = self.gate_up_proj_weight
            down_weight = self.down_proj_weight
        output = module.NF.moe_cte(
            implementation=module.MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_weight,
            down_proj_weight=down_weight,
            activation_function=module.ActFnType.SiLU,
            block_size=self.block_size,
            token_position_to_id=token_position_to_id.to(torch.int32),
            block_to_expert=block_to_expert.to(torch.int32),
            expert_affinities_scaling_mode=(
                module.ExpertAffinityScaleMode.POST_SCALE
            ),
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=module.nl.bfloat16,
        )
        if self.world_size > 1:
            output = self.tp_group.reduce_scatter(output, dim=0)
        return output

    cls.forward_prefill = _forward_prefill_with_nki_router

    if not getattr(module.NF.moe_cte, "_opt_skip_weight_patched", False):
        original_moe_cte = module.NF.moe_cte

        def _moe_cte_skip_padding_weights(*args, **kwargs):
            kwargs.setdefault("skip_weight", True)
            return original_moe_cte(*args, **kwargs)

        _moe_cte_skip_padding_weights._opt_skip_weight_patched = True
        module.NF.moe_cte = _moe_cte_skip_padding_weights


class _ModelPatchLoader(Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        _patch_qwen3_moe(module)


class _ModelPatchFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET_MODEL:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _ModelPatchLoader(spec.loader)
        return spec


_loaded_model = sys.modules.get(_TARGET_MODEL)
if _loaded_model is not None:
    _patch_qwen3_moe(_loaded_model)
elif not any(isinstance(finder, _ModelPatchFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _ModelPatchFinder())
