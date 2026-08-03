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

    if not getattr(module.NF.segmented_attention, "_opt_fused_scale_patched", False):
        segmented_impl = importlib.import_module(
            "vllm_neuron.functional.attention.attention_segmented_cte"
        )

        def _segmented_attention_with_kernel_scale(
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
            if scale is None:
                scale = 1.0 / (q.shape[2] ** 0.5)
            if not tp_q or fp8_packed:
                raise ValueError("optimized segmented attention expects TP Q and BF16 KV")
            result = segmented_impl._wrapped_attention_segmented_cte[2](
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                block_tables=block_tables,
                prior_tokens=prior_tokens,
                block_size=block_size,
                prior_seg_size=kv_segment_size,
                scale=scale,
                tp_q=True,
                tp_out=False,
                sliding_window=sliding_window if sliding_window else None,
                sink=sink,
                num_q_heads=q.shape[0],
                k_scale=k_scale,
                v_scale=v_scale,
            )
            return result.transpose(1, 2) if tp_out else result

        _segmented_attention_with_kernel_scale._opt_fused_scale_patched = True
        module.NF.segmented_attention = _segmented_attention_with_kernel_scale

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
