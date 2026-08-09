# SPDX-License-Identifier: Apache-2.0
"""
Llama Static FP8 Implementation
===============================

Static FP8 variant of :mod:`vllm_neuron.model.llama3.model` for TRN2+.

Weights are stored in ``float8_e4m3fn`` with a single per-tensor dequant
scale. Activations are static-FP8 quantized for QKV/O proj and MLP. Both
weight and input scales are pre-broadcast to ``(128, 1)`` fp32 at load
time (or ``(128, 3)`` for fused QKV) to avoid runtime broadcast for
better performance.

Only :class:`LlamaAttention` and :class:`LlamaMLP` are provided here;
the decoder layer, model backbone, and causal-LM head still come from
:mod:`.model`.

This implementation has been tested against checkpoints quantized with
``quant_method='model_opt'`` and ``quant_algo='FP8'`` e.g.
nvidia/Llama-3.1-8B-Instruct-FP8.

Module responsibilities (differ from bf16 sibling)
--------------------------------------------------
:class:`LlamaAttention` is **bf16-in / bf16-out** on both prefill and
decode. The STATIC ``NF.qkv_proj`` and ``NF.o_proj`` kernels take a
bf16 activation and handle fp8 quant/dequant internally using
``qkv_in_scale`` / ``o_in_scale``. The decoder layer runs its plain
``input_layernorm`` in bf16 and passes the result straight to
``self_attn`` — there is no pre-attention ``rmsnorm_quant``.

:class:`LlamaMLP` owns **both the post-attention RMSNorm and the MLP**
on the prefill path: its ``forward`` internally calls
:func:`NF.rmsnorm_quant` (fused RMSNorm + static fp8 quant) to produce
fp8 activations for :func:`NF.mlp`, using a ``ln_w`` supplied by the
decoder layer. The decoder layer must therefore **skip** its plain
``post_attention_layernorm`` on prefill when dispatching to this MLP.
On decode this module is MLP-only — the attention megakernel already
emits bf16 and the MLP kernel handles the static re-quant internally.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Parallelism code; matches the bf16 sibling.
  # <-- STATIC-FP8: ...        Places that differ from bf16 (dtype, scales,
                                kernel kwargs).
"""

import logging

from torch import nn
from vllm.distributed.parallel_state import get_tp_group
from nkilib.core.utils.common_types import NormType, QuantizationType

import torch
import vllm_neuron.functional as NF
from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    with_rank_override,
)

from .config import LlamaConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Static-FP8 constants
# =============================================================================

# Static-FP8 weight dtype
_FP8_DTYPE = torch.float8_e4m3fn

# Partition dimension size the kernels expect for pre-broadcast scales.
_PMAX = 128

# Number of per-projection scales fused into the QKV weight_scale tensor
# (one each for Q, K, V). Matches the NF.qkv_proj STATIC contract of
# [1, 3] / [P_MAX, 3].
_QKV_FUSED = 3


# =============================================================================
# TRN2 / TRN3 fp8 range compensation
# =============================================================================
#
# Checkpoints typically calibrate against IEEE ``float8_e4m3fn`` (max
# 448). Neuron's fp8 implementation on Trn2 is ``float8_e4m3`` capped at 240.
# Therefore, we reinterpret ``float8_e4m3fn`` as ``float8_e4m3`` during compilation
# on Trn2 and out of range values will lead to incorrect results.
#
# We scale weights into Neuron range and compensate by scaling outputs back with
# inverse factor allowing out-of-range parameters while maintaining accuracy.
#
# The raw weight rescaling happens inside the weight loaders so it runs
# once per parameter at checkpoint load time and is transparent to
# downstream code. The scale rescaling is folded into the scale-loader
# transforms below.

# TODO: Support natively float8_e4m3fn on Trn3
_FP8_E4M3_MAX = 240.0
_FP8_E4M3FN_MAX = 448.0
_FP8_WEIGHT_DOWNSCALE = _FP8_E4M3_MAX / _FP8_E4M3FN_MAX
_FP8_SCALE_COMPENSATION = _FP8_E4M3FN_MAX / _FP8_E4M3_MAX


def _downscale_fp8_weight(w_fp8: torch.Tensor) -> torch.Tensor:
    """Bring a ModelOpt-calibrated fp8 weight tensor into Neuron range.

    ``w_fp8 * (240/448)``, clamp to ``[-240, 240]`` to absorb any bytes
    that land outside the range through rounding, and re-cast to
    ``float8_e4m3fn`` so downstream allocators and kernels see the
    expected dtype.
    """
    return (
        (w_fp8.float() * _FP8_WEIGHT_DOWNSCALE)
        .clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        .to(_FP8_DTYPE)
    )


def _wrap_with_fp8_downscale(
    loader: SafetensorsWeightLoader,
) -> SafetensorsWeightLoader:
    """Post-process an fp8 weight loader's output with the 240/448 downscale.

    Accepts any :class:`SafetensorsWeightLoader` (including the generic
    ``fused_qkv_weight_loader`` / ``sharding_weight_loader`` from
    :mod:`vllm_neuron.utils.weight_loader`) and returns a new loader
    whose ``transform`` appends :func:`_downscale_fp8_weight` to the
    original output. Non-fp8 loaders would be a bug to wrap here — we
    rely on the caller pairing this with a parameter that is
    ``float8_e4m3fn``.
    """
    base_transform = loader.transform or (lambda slices, rank: slices[0][:])

    def transform(slices, rank):
        return _downscale_fp8_weight(base_transform(slices, rank))

    return SafetensorsWeightLoader(transform=transform)


def _read_scalar_from_slice(slice_obj) -> torch.Tensor:
    """Read a scalar tensor from a ``PySafeSlice``, handling rank-0.

    ``PySafeSlice[:]`` raises ``"slice() cannot be applied to a 0-dim
    tensor"`` on rank-0 tensors (the shape ModelOpt uses for FP8
    scales). ``PySafeSlice[()]`` is the rank-0-safe alternative.
    Returns a rank-1 length-1 fp32 tensor so downstream code can
    treat both shapes uniformly.
    """
    shape = slice_obj.get_shape()
    raw = slice_obj[()] if len(shape) == 0 else slice_obj[:]
    flat = raw.to(torch.float32).reshape(-1)
    assert flat.numel() == 1, (
        f"_read_scalar_from_slice expects a scalar tensor, got shape "
        f"{tuple(shape)} with {flat.numel()} elements"
    )
    return flat


def _broadcast_scalar_scale_loader(
    is_weight_scale: bool,
) -> SafetensorsWeightLoader:
    """Load a scalar fp32 checkpoint tensor and broadcast it to ``[128, 1]``.

    Expects exactly one checkpoint slice containing a scalar (``()``) or
    rank-1 length-1 (``(1,)``) tensor. Returns a contiguous ``[128, 1]``
    fp32 tensor with every entry equal to the scalar.

    Args:
        is_weight_scale: When ``True``, the returned scale is
            pre-multiplied by :data:`_FP8_SCALE_COMPENSATION` to
            absorb the ``240/448`` weight downscale applied by
            :func:`_wrap_with_fp8_downscale`. When ``False`` (input
            scale), no compensation is applied — see the module-level
            comment above.
    """
    mult = _FP8_SCALE_COMPENSATION if is_weight_scale else 1.0

    def transform(slices, rank):
        assert len(slices) == 1, (
            "_broadcast_scalar_scale_loader expects exactly one slice, "
            f"got {len(slices)}"
        )
        scalar = _read_scalar_from_slice(slices[0])
        return (scalar * mult).expand(_PMAX, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _fused_qkv_weight_scale_loader() -> SafetensorsWeightLoader:
    """Stack three scalar Q/K/V weight scales into a ``[128, 3]`` fp32 tensor.

    Expects three checkpoint slices in order (Q, K, V), each a scalar
    fp32. Produces the ``[P_MAX, 3]`` pre-broadcast layout that the
    ``NF.qkv_proj`` STATIC kernel contract specifies, with each column
    pre-multiplied by :data:`_FP8_SCALE_COMPENSATION` (this is always
    a weight scale, never an input scale).
    """

    def transform(slices, rank):
        assert len(slices) == _QKV_FUSED, (
            f"_fused_qkv_weight_scale_loader expects {_QKV_FUSED} slices, "
            f"got {len(slices)}"
        )
        cols = [_read_scalar_from_slice(s) * _FP8_SCALE_COMPENSATION for s in slices]
        stacked = torch.stack(cols, dim=0).reshape(1, _QKV_FUSED)  # [1, 3]
        return stacked.expand(_PMAX, _QKV_FUSED).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _fused_qkv_input_scale_loader() -> SafetensorsWeightLoader:
    """Load a single per-tensor input scale for fused QKV, broadcast to ``[128, 1]``.

    ModelOpt emits one ``input_scale`` per projection (Q, K, V). The
    kernel's STATIC contract wants a single per-tensor input scale, and
    ModelOpt's static-FP8 calibration produces identical values across
    Q/K/V for fused QKV. We read all three and assert they agree; if a
    future checkpoint breaks this assumption the assertion catches it
    before the kernel silently miscomputes.

    No ``_FP8_SCALE_COMPENSATION`` multiplication: the runtime activation
    quantizers already produce Neuron-range values — see the
    module-level comment.
    """

    def transform(slices, rank):
        assert len(slices) == _QKV_FUSED, (
            f"_fused_qkv_input_scale_loader expects {_QKV_FUSED} slices, "
            f"got {len(slices)}"
        )
        scalars = [_read_scalar_from_slice(s) for s in slices]
        for idx, sc in enumerate(scalars[1:], start=1):
            assert torch.equal(scalars[0], sc), (
                "_fused_qkv_input_scale_loader expects identical "
                f"input_scale across Q/K/V for fused QKV; got "
                f"Q={scalars[0].item()} vs index {idx}={sc.item()}. "
                "Fused QKV with per-projection input scales is not "
                "supported by the STATIC kernel."
            )
        return scalars[0].expand(_PMAX, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


# =============================================================================
# Attention (static FP8)
# =============================================================================


class LlamaAttention(nn.Module):
    """GQA attention with TP head sharding, static FP8 weights and scales.

    Structurally identical to :class:`vllm_neuron.model.llama3.model.LlamaAttention`
    except:
      * QKV / O weights are allocated in ``float8_e4m3fn``.
      * Four scale buffers are registered (qkv weight/input, o weight/input).
      * Prefill QKV projection passes STATIC kwargs to :func:`NF.qkv_proj`.
      * Prefill output projection passes STATIC kwargs to :func:`NF.o_proj`.
      * Decode passes STATIC kwargs to :func:`NF.attention_decode`.

    Parallelism, KV cache handling, RoPE, and flash/segmented attention are
    unchanged from the bf16 sibling.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        # Compute dtype (bf16/fp32) used for activations outside the quantized
        # matmul. Weights are FP8 regardless of this.
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Dependent DP setup (decode-only Q/O sharding across DP) <<<
        self.attention_dp_size = (
            config.neuron_config.attention_dp_size if config.neuron_config else 1
        )
        self.attention_dp_group = None
        self.attention_dp_rank = 0
        if self.attention_dp_size > 1:
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_attention_dp_group,
                get_neuron_attention_dp_rank,
            )

            self.attention_dp_group = get_neuron_attention_dp_group()
            self.attention_dp_rank = get_neuron_attention_dp_rank()

        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_tp_group,
        )

        self.attn_tp_group = get_neuron_attention_tp_group()

        # Effective sharding degree for Q/O (TP for standard, TP*DDP for attention DP)
        effective_q_shards = self.world_size * self.attention_dp_size

        # >>> PARALLELISM: Head sharding calculation <<<
        self.num_attention_heads_per_rank = (
            self.num_attention_heads // effective_q_shards
        )

        self.kv_needs_a2a = (
            self.attention_dp_size > 1
            and self.num_key_value_heads > self.world_size
            and self.num_key_value_heads % effective_q_shards == 0
        )

        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1

        self.num_kv_heads_for_weight = (
            self.num_key_value_heads // effective_q_shards
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )
        num_kv_heads_for_weight = self.num_kv_heads_for_weight

        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // num_kv_heads_for_weight
        )

        self.num_q_heads_after_a2a = (
            self.num_attention_heads_per_rank * self.attention_dp_size
        )
        self.num_kv_heads_after_a2a = (
            num_kv_heads_for_weight * self.attention_dp_size
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )

        # >>> PARALLELISM: QKV weight shapes <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = num_kv_heads_for_weight * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in_features = (
            self.num_attention_heads * self.head_dim
        ) // effective_q_shards

        # <-- STATIC-FP8: weights live in fp8e4m3fn; scales are registered as
        # buffers so they move with the module but are not trainable.
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=_FP8_DTYPE)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=_FP8_DTYPE)
        )

        # <-- STATIC-FP8: dequant / input scales. Shapes match the kernel
        # contract ([_PMAX, 3] for fused QKV weight, [_PMAX, 1] elsewhere).
        self.register_buffer(
            "qkv_weight_scale",
            torch.empty(_PMAX, _QKV_FUSED, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "qkv_input_scale",
            torch.empty(_PMAX, 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "o_weight_scale",
            torch.empty(_PMAX, 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "o_input_scale",
            torch.empty(_PMAX, 1, dtype=torch.float32),
            persistent=False,
        )

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        # KV caches bound externally via bind_kv_cache()
        self.k_cache = None
        self.v_cache = None

        # KV cache quantization scales (populated at weight-load time).
        self.register_buffer("k_scale", None, persistent=False)
        self.register_buffer("v_scale", None, persistent=False)
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        self._setup_weight_loaders()

    # ------------------------------------------------------------------
    # Weight loaders
    # ------------------------------------------------------------------

    def _setup_weight_loaders(self):
        """Attach loaders for fp8 weights + static scales.

        Weight loaders mirror the bf16 sibling
        (``fused_qkv_weight_loader`` / ``sharding_weight_loader`` preserve
        dtype), wrapped in :func:`_wrap_with_fp8_downscale` so
        ModelOpt-calibrated values in ``(240, 448]`` are rescaled into
        Neuron's 240-capped fp8 range before they land on the parameter.
        Scale loaders are local to this module and apply the matching
        ``448/240`` compensation.
        """
        ddp = self.attention_dp_size
        effective_q_rank = self.attention_dp_rank + self.tp_group.rank_in_group * ddp
        o_shard_size = (self.num_attention_heads * self.head_dim) // (
            self.world_size * ddp
        )

        # QKV weight (fused Q/K/V, fp8e4m3fn). Downscale wrapper applies
        # the 240/448 factor once per-parameter at load time.
        set_weight_loader(
            self.qkv_proj_weight,
            _wrap_with_fp8_downscale(
                fused_qkv_weight_loader(
                    q_size=self.q_size,
                    kv_size=self.kv_size,
                    shard_dim=1,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                    num_kv_replicas=self.num_kv_replicas,
                    attention_dp_rank=self.attention_dp_rank,
                    attention_dp_size=ddp,
                    kv_sharded_across_attention_dp=self.kv_needs_a2a,
                )
            ),
        )

        # <-- STATIC-FP8: QKV weight scale stacks three scalars into [128, 3].
        set_weight_loader(self.qkv_weight_scale, _fused_qkv_weight_scale_loader())
        # <-- STATIC-FP8: QKV input scale collapses Q/K/V scalars to one [128, 1].
        set_weight_loader(self.qkv_input_scale, _fused_qkv_input_scale_loader())

        # O weight (fp8e4m3fn). Same downscale wrapper.
        set_weight_loader(
            self.o_proj_weight,
            _wrap_with_fp8_downscale(
                with_rank_override(
                    sharding_weight_loader(
                        shard_dim=0,
                        shard_size=o_shard_size,
                        num_shards=self.world_size * ddp,
                        is_storage_transposed=True,
                    ),
                    rank=effective_q_rank,
                )
            ),
        )

        # <-- STATIC-FP8: O weight + input scales (scalar → [128, 1]).
        # weight scale compensates for the 240/448 weight downscale;
        # input scale does not (activation quantizers are already
        # Neuron-range-native — see the scale-loader module comment).
        set_weight_loader(
            self.o_weight_scale,
            _broadcast_scalar_scale_loader(is_weight_scale=True),
        )
        set_weight_loader(
            self.o_input_scale,
            _broadcast_scalar_scale_loader(is_weight_scale=False),
        )

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        ln_w: torch.Tensor = None,
        eps: float = 1e-5,
    ):
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
                ln_w=ln_w,
                eps=eps,
            )

        # >>> PARALLELISM: All-gather from SP before attention <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Prefill path.

        Step 1 (QKV projection) and Step 5 (O projection) pass STATIC
        kernel kwargs. Steps 2–4 (RoPE, KV cache write, attention core)
        are unchanged from bf16.

        The activation dtype is bf16 on entry and stays bf16 throughout;
        the STATIC qkv_proj / o_proj kernels handle the internal fp8
        quant/dequant against the provided scales. We cast once at the
        top to match the reference FP8 Llama3 model's prefill contract.
        """
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        # <-- STATIC-FP8: explicit bf16 cast before the STATIC qkv_proj
        # kernel. Matches the reference model's contract; without it the
        # on-gen3+ nc_matmul transpose-mode path asserts
        # ``dst=bfloat16 but input=float8_e4m3`` because the accumulator
        # dtype is picked from ``hidden`` and the kernel sees a mismatch
        # with the fp8 weight tile.
        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # ── Step 1: QKV Projection (STATIC, fused RoPE) ─────────────────
        cos, sin = position_embeddings
        cos_cache = cos.unsqueeze(0)
        sin_cache = sin.unsqueeze(0)
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
            d_head=self.head_dim,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            quantization_type=QuantizationType.STATIC,
            qkv_w_scale=self.qkv_weight_scale,
            qkv_in_scale=self.qkv_input_scale,
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_kv_heads_for_weight,
        ).squeeze(0)

        qkv = qkv.to(self.dtype)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        k = k.view(tokens, self.num_kv_heads_for_weight, self.head_dim).transpose(0, 1)
        v = v.view(tokens, self.num_kv_heads_for_weight, self.head_dim).transpose(0, 1)

        # ── Step 3: KV cache update (unchanged) ────────────────────────
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size

        if self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            k_flat = (
                (k.reshape(-1, self.head_dim) * self.k_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.k_cache.dtype)
            )
            v_flat = (
                (v.reshape(-1, self.head_dim) * self.v_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.k_cache.dtype)
            )
        else:
            k_flat = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
            v_flat = v.reshape(-1, self.head_dim).to(self.k_cache.dtype)

        head_indices_for_put = torch.arange(
            self.num_key_value_heads_per_rank,
            dtype=torch.long,
            device=hidden_states.device,
        ).repeat_interleave(slot_mapping.shape[0])
        block_indices_for_put = block_indices.repeat(self.num_key_value_heads_per_rank)
        position_indices_for_put = position_indices.repeat(
            self.num_key_value_heads_per_rank
        )

        self.k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k_flat,
        )
        self.v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_flat,
        )

        # ── Step 4: Attention (unchanged) ───────────────────────────────
        if kv_segment_size:
            attn_output = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=block_table,
                prior_tokens=cached_seq_len,
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                tp_q=True,
                tp_out=True,
            )  # [Nh, Dh, T]
        else:
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)

            q_flash = q.transpose(1, 2)
            k_flash = k.transpose(1, 2)
            v_flash = v

            attn_output = NF.flash_attention(
                q_flash,
                k_flash,
                v_flash,
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )

        # ── Step 5: Output Projection (STATIC) ──────────────────────────
        # <-- STATIC-FP8: attn_output is bf16 from the attention core; it
        # needs to be quantized inside the kernel using o_input_scale.
        attn_output = attn_output.unsqueeze(0)  # [1, Nh, Dh, T]
        attn_output = NF.o_proj(
            attn_output,
            self.o_proj_weight,
            None,
            quantization_type=QuantizationType.STATIC,
            weight_scales=self.o_weight_scale,
            input_scales=self.o_input_scale,
        )
        attn_output = attn_output.squeeze(0)

        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
        ln_w: torch.Tensor = None,
        eps: float = 1e-5,
    ):
        """Decode path (fused megakernel) with STATIC QKV and STATIC output."""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B_local = block_table.shape[0]
        B = B_local * self.attention_dp_size
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank

        X = hidden_states.view(B, S_decode, hidden)

        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        cos_kernel = (
            cos[:, :half_d]
            .view(B_local, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        sin_kernel = (
            sin[:, :half_d]
            .view(B_local, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )

        pos_ids = positions.view(1, B_local * S_decode)
        attention_mask = NF.gen_attention_decode_mask(
            pos_ids=pos_ids.to(torch.float32),
            bs=B_local,
            q_head=self.num_q_heads_after_a2a,
            s_active=S_decode,
            s_prior=S_ctx,
            start_pos=None,
            block_len=block_size,
        )

        active_blocks_table = block_table

        output = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=ln_w is not None,
            rmsnorm_X_eps=eps if ln_w is not None else None,
            rmsnorm_X_gamma=ln_w.view(1, -1) if ln_w is not None else None,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,
            quantization_type_qkv=QuantizationType.STATIC,
            weight_dequant_scale_qkv=self.qkv_weight_scale,
            input_dequant_scale_qkv=self.qkv_input_scale,
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling / self.k_scale_float,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=slot_mapping.view(B_local, S_decode).to(torch.uint32),
            W_out=self.o_proj_weight,
            bias_out=None,
            quantization_type_out=QuantizationType.STATIC,
            weight_dequant_scale_out=self.o_weight_scale,
            input_dequant_scale_out=self.o_input_scale,
            transposed_out=False,
            out_in_sb=False,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            attention_dp=self.attention_dp_size,
            attention_dp_group=self.attention_dp_group.device_group
            if self.attention_dp_group
            else None,
            attention_dp_rank=self.attention_dp_rank,
            kv_needs_a2a=self.kv_needs_a2a,
        )

        self.attn_tp_group.all_reduce(output)
        return output


# =============================================================================
# MLP (static FP8)
# =============================================================================


class LlamaMLP(nn.Module):
    """Fused pre-MLP RMSNorm + SwiGLU MLP with TP intermediate sharding
    (static FP8).

    Unlike the bf16 sibling (which is MLP-only), this module owns **both
    the post-attention RMSNorm and the MLP** on the prefill path: the
    RMSNorm is fused with static fp8 quant via :func:`NF.rmsnorm_quant`
    inside ``forward`` to produce fp8 activations for :func:`NF.mlp`.
    The decoder layer must skip its plain ``post_attention_layernorm``
    on prefill and instead pass that norm's ``weight`` and
    ``config.rms_norm_eps`` into this module's ``forward``.

    Differs from :class:`vllm_neuron.model.llama3.model.LlamaMLP`:
      * gate/up/down weights are fp8e4m3fn.
      * Five scale buffers are registered (gate/up/down weight, gate_up/down input).
      * :func:`NF.mlp` is called with ``quantization_type=STATIC`` + scales.
      * :func:`NF.rmsnorm_quant` is fused into ``forward`` on prefill to
        produce fp8 activations; see the ``forward`` docstring.

    Pre-MLP RMSNorm fusion on prefill
    ---------------------------------
    The MLP kernel's STATIC path consumes fp8 activations (with
    ``gate_up_input_scale`` used internally to dequantize them back to
    the compute dtype for the gate/up matmul). To produce those fp8
    activations we fuse the pre-MLP RMSNorm with the static quant step
    via :func:`NF.rmsnorm_quant`. The norm weight (``ln_w``) and ``eps``
    are passed in from the decoder layer so the MLP does not have to
    own a duplicate RMSNorm parameter.

    Decoder-layer contract (prefill):
      * The decoder layer must **not** run the plain
        ``post_attention_layernorm`` before calling this MLP on prefill;
        instead it passes that norm's ``weight`` as ``ln_w`` and
        ``config.rms_norm_eps`` as ``eps``. Running the plain norm as
        well would double-apply RMSNorm.

    Decoder-layer contract (decode):
      * Unchanged. The preceding :class:`LlamaAttention.forward_decode`
        fused megakernel already emits bf16; the MLP kernel handles
        the static re-quantization of that bf16 input via
        ``gate_up_input_scale``. ``ln_w`` / ``eps`` are ignored on
        decode.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.mlp_dp_size = (
            config.neuron_config.mlp_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_mlp_tp_group,
            get_neuron_mlp_dp_group,
        )

        mlp_tp_group = get_neuron_mlp_tp_group()
        self.mlp_tp_group = mlp_tp_group
        self.mlp_tp_size = mlp_tp_group.world_size
        self.mlp_tp_rank = mlp_tp_group.rank_in_group
        self.mlp_dp_group = get_neuron_mlp_dp_group()

        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.intermediate_size // self.mlp_tp_size
        # <-- STATIC-FP8: record the compute dtype the MLP must emit so the
        # fp8 kernel output is cast back to bf16 (otherwise it would default
        # to ``hidden.dtype``, which on prefill is the fp8 output of
        # ``NF.rmsnorm_quant`` — the downstream residual add would crash).
        self.act_dtype = config.torch_dtype

        # <-- STATIC-FP8: fp8 weights, same 2D [H, I] / [I, H] layout as bf16.
        self.gate_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=_FP8_DTYPE,
            )
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=_FP8_DTYPE,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_rank,
                config.hidden_size,
                dtype=_FP8_DTYPE,
            )
        )

        # <-- STATIC-FP8: per-projection dequant scales + per-stage input
        # scales (gate/up share one input scale; down has its own because
        # it takes the activation function's output).
        for name in (
            "gate_weight_scale",
            "up_weight_scale",
            "down_weight_scale",
            "gate_up_input_scale",
            "down_input_scale",
        ):
            self.register_buffer(
                name,
                torch.empty(_PMAX, 1, dtype=torch.float32),
                persistent=False,
            )

        self._setup_weight_loaders(config)

    def _setup_weight_loaders(self, config):
        """Attach loaders for fp8 weights + scales.

        Weight loaders mirror bf16 (``sharding_weight_loader`` preserves
        dtype), wrapped in :func:`_wrap_with_fp8_downscale` so every fp8
        parameter lands in Neuron's 240-capped range. Scale loaders are
        the shared scalar → ``[128, 1]`` helper and apply the matching
        ``448/240`` compensation.
        """
        gate_up_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.mlp_tp_size,
            is_storage_transposed=True,
        )
        down_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.mlp_tp_size,
            is_storage_transposed=True,
        )

        gate_up_loader = with_rank_override(gate_up_loader, rank=self.mlp_tp_rank)
        down_loader = with_rank_override(down_loader, rank=self.mlp_tp_rank)

        # Downscale wrapper applies the 240/448 factor once per
        # parameter at load time — the scale loaders above already
        # carry the compensating 448/240 factor, so the dequantized
        # product ``weight * scale`` is unchanged.
        gate_up_loader = _wrap_with_fp8_downscale(gate_up_loader)
        down_loader = _wrap_with_fp8_downscale(down_loader)

        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

        # <-- STATIC-FP8: weight and input scales (scalar → [128, 1]).
        # Weight scales compensate for the 240/448 weight downscale;
        # input scales do not.
        set_weight_loader(
            self.gate_weight_scale,
            _broadcast_scalar_scale_loader(is_weight_scale=True),
        )
        set_weight_loader(
            self.up_weight_scale,
            _broadcast_scalar_scale_loader(is_weight_scale=True),
        )
        set_weight_loader(
            self.down_weight_scale,
            _broadcast_scalar_scale_loader(is_weight_scale=True),
        )
        set_weight_loader(
            self.gate_up_input_scale,
            _broadcast_scalar_scale_loader(is_weight_scale=False),
        )
        set_weight_loader(
            self.down_input_scale,
            _broadcast_scalar_scale_loader(is_weight_scale=False),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool,
        ln_w: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Run the pre-MLP RMSNorm + static-FP8 MLP.

        The norm is fused with the static fp8 quant. The fusion happens in
        different places on prefill vs decode (to match the reference):

        * **Prefill:** explicit :func:`NF.rmsnorm_quant` before
          :func:`NF.mlp`; kernel is called with ``norm_type=NO_NORM``
          since the norm already ran. Required because CTE mode
          (large-seqlen path) does not support fused norm+quant inside
          :func:`NF.mlp`.
        * **Decode:** :func:`NF.mlp` with ``norm_type=RMS_NORM`` and
          ``ln_w=ln_w.view(1, -1)``. TKG mode (small-B×S path) fuses
          the norm and the static quant in one kernel.

        Either way, the decoder layer must **skip** its plain
        ``post_attention_layernorm`` and forward the norm weight and eps
        via ``ln_w`` / ``eps`` here.

        Args:
            hidden_states: Residual-stream activation in the compute
                dtype (bf16). Shape ``[T_local, H]`` on prefill (SP
                slice) or ``[T, H]`` on decode.
            is_prefill: Prefill vs decode path selector (matches the
                bf16 sibling).
            ln_w: ``post_attention_layernorm.weight`` from the decoder
                layer. Required on both paths now.
            eps: RMSNorm epsilon. Required on both paths.
        """
        if ln_w is None:
            raise ValueError(
                "LlamaMLP (static FP8) requires ln_w on both prefill and "
                "decode; the decoder layer must pass "
                "post_attention_layernorm.weight."
            )

        if is_prefill:
            # <-- STATIC-FP8 (prefill): rmsnorm_quant in SP region, then gather.
            # Each rank normalizes+quantizes its local tokens before all-gather.
            hidden_states = NF.rmsnorm_quant(
                hidden_states,
                ln_w=ln_w,
                input_dequant_scale=self.gate_up_input_scale,
                eps=eps,
                quantization_type=QuantizationType.STATIC,
            )
            # >>> PARALLELISM: All-gather from SP after rmsnorm_quant <<<
            if self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            mlp_norm_type = NormType.NO_NORM
            mlp_ln_w = None
        else:
            # <-- STATIC-FP8 (decode): TKG mode fuses RMSNorm + static
            # quant inside NF.mlp; hand it the (1, H)-shaped gamma and
            # let the kernel do both.
            mlp_norm_type = NormType.RMS_NORM
            mlp_ln_w = ln_w.view(1, -1)

        # <-- STATIC-FP8: STATIC quant + scales on both paths.
        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
            eps=eps,
            ln_w=mlp_ln_w,
            norm_type=mlp_norm_type,
            quantization_type=QuantizationType.STATIC,
            gate_w_scale=self.gate_weight_scale,
            up_w_scale=self.up_weight_scale,
            down_w_scale=self.down_weight_scale,
            gate_up_in_scale=self.gate_up_input_scale,
            down_in_scale=self.down_input_scale,
            output_dtype="bfloat16",
        )

        # >>> PARALLELISM: Combine TP (+ MLP DP) shards <<<
        if is_prefill:
            if self.world_size > 1:
                output = self.tp_group.reduce_scatter(output, dim=0)
        else:
            self.mlp_tp_group.all_reduce(output)

        return output
