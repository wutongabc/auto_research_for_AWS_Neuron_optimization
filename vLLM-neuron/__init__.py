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

    segmented_impl = importlib.import_module(
        "vllm_neuron.functional.attention.attention_segmented_cte"
    )
    if segmented_impl._wrapped_attention_segmented_cte is None:
        from nkilib.core.attention.attention_segmented_cte import (
            attention_segmented_cte,
        )

        segmented_impl._SEGMENTED_KERNEL_HAS_FP8_PACKED = True
        segmented_kernel = segmented_impl.nki.jit()(attention_segmented_cte)
        segmented_impl._wrapped_attention_segmented_cte = segmented_impl.wrap_nki(
            segmented_kernel
        )

    if not getattr(module.NF.segmented_attention, "_opt_hybrid_patched", False):
        torch = segmented_impl.torch
        nki_segmented_attention = module.NF.segmented_attention
        fallback_attention = segmented_impl._torch_segmented_attention_impl

        def _hybrid_segmented_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            prior_tokens,
            block_size,
            kv_segment_size,
            scale=None,
            tp_q=True,
            tp_out=False,
            sliding_window=None,
            sink=None,
            fp8_packed=False,
            k_scale=None,
            v_scale=None,
        ):
            def _short_context(q_t, k_t, v_t, blocks_t, prior_t):
                return nki_segmented_attention(
                    q_t,
                    k_cache=k_t,
                    v_cache=v_t,
                    block_tables=blocks_t,
                    prior_tokens=prior_t,
                    block_size=block_size,
                    kv_segment_size=kv_segment_size,
                    scale=scale,
                    tp_q=tp_q,
                    tp_out=tp_out,
                    sliding_window=sliding_window,
                    sink=sink,
                    fp8_packed=fp8_packed,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

            def _long_context(q_t, k_t, v_t, blocks_t, prior_t):
                return fallback_attention(
                    q=q_t,
                    k_cache=k_t,
                    v_cache=v_t,
                    block_tables=blocks_t,
                    prior_tokens=prior_t,
                    block_size=block_size,
                    kv_segment_size=kv_segment_size,
                    scale=scale,
                    tp_q=tp_q,
                    tp_out=tp_out,
                    sliding_window=sliding_window,
                    sink=sink,
                )

            return torch.cond(
                prior_tokens.reshape(-1)[0] < 9216,
                _short_context,
                _long_context,
                (q, k_cache, v_cache, block_tables, prior_tokens),
            )

        _hybrid_segmented_attention._opt_hybrid_patched = True
        module.NF.segmented_attention = _hybrid_segmented_attention

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
