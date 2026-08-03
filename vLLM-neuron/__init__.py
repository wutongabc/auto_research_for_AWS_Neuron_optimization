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
    if not getattr(
        segmented_impl._torch_segmented_attention_impl,
        "_opt_bf16_bmm_patched",
        False,
    ):
        torch = segmented_impl.torch

        def _bf16_segmented_attention_impl(
            q,
            k_cache,
            v_cache,
            block_tables,
            prior_tokens,
            block_size,
            kv_segment_size,
            scale,
            tp_q=True,
            tp_out=False,
            sliding_window=None,
            sink=None,
        ):
            if not tp_q:
                q = q.transpose(1, 2)

            batch, query_len, head_dim = q.shape
            num_kv_heads = k_cache.shape[1]
            max_blocks = block_tables.shape[1]
            padded_kv_len = max_blocks * block_size
            prior_len = prior_tokens.reshape(-1)[0].to(torch.int64)

            block_ids = block_tables[0].clamp_min(0).to(torch.int64)
            k_blocks = k_cache[block_ids]
            v_blocks = v_cache[block_ids]
            k_seq = k_blocks.permute(1, 0, 2, 3).reshape(
                num_kv_heads, padded_kv_len, head_dim
            )
            v_seq = v_blocks.permute(1, 0, 2, 3).reshape(
                num_kv_heads, padded_kv_len, head_dim
            )
            heads_per_kv = batch // num_kv_heads
            if heads_per_kv > 1:
                k_seq = k_seq.repeat_interleave(heads_per_kv, dim=0)
                v_seq = v_seq.repeat_interleave(heads_per_kv, dim=0)

            q_pos = torch.arange(
                query_len, device=q.device, dtype=torch.int64
            ).unsqueeze(1)
            k_pos = torch.arange(
                padded_kv_len, device=q.device, dtype=torch.int64
            ).unsqueeze(0)
            allowed = (k_pos < prior_len + query_len) & (
                k_pos <= q_pos + prior_len
            )
            if sliding_window is not None and sliding_window > 0:
                allowed = allowed & (
                    k_pos > q_pos + prior_len - sliding_window
                )

            scores = torch.bmm(q, k_seq.transpose(1, 2)).float() * scale
            scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
            if sink is not None:
                sink_scores = sink.float().reshape(batch, 1, 1).expand(
                    batch, query_len, 1
                )
                scores = torch.cat([scores, sink_scores], dim=-1)

            weights = torch.nn.functional.softmax(scores, dim=-1)
            if sink is not None:
                weights = weights[:, :, :-1]
            weights = torch.nan_to_num(weights, nan=0.0).to(v_seq.dtype)
            output = torch.bmm(weights, v_seq).to(q.dtype)
            return output.transpose(1, 2) if tp_out else output

        _bf16_segmented_attention_impl._opt_bf16_bmm_patched = True
        segmented_impl._torch_segmented_attention_impl = (
            _bf16_segmented_attention_impl
        )

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
