# SPDX-License-Identifier: Apache-2.0
"""
Llama Static MX FP8 Implementation
===========================

Static MX FP8 (STATIC_MX kernel) variant of
:mod:`vllm_neuron.model.llama3.model` for Trn3. Prefill always uses MX
(QKV CTE / O proj CTE / MLP CTE on the STATIC_MX kernel path).
``model_static_fp8`` is the Trn2 / non-MX sibling; the factory
(:func:`vllm_neuron.model.llama3.quantization.resolve_attention_mlp_classes`)
dispatches to the right module by platform.

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

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group
from nkilib.core.utils.common_types import (
    NormType,
    QKVWeightLayout,
    QuantizationType,
)

try:
    from nkilib.core.utils.common_types import MLPGateUpWeightLayout
except ImportError:
    from enum import IntEnum

    class MLPGateUpWeightLayout(IntEnum):
        CONTIGUOUS = 0
        H_X4_MIDDLE = 2


import vllm_neuron.functional as NF
from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX

from .config import LlamaConfig
from .model import apply_rotary_pos_emb  # shared with bf16
from . import weight_loaders_static_fp8 as _static_fp8_loaders
from . import weight_loaders_mx_fp8 as _mx_fp8_loaders

logger = logging.getLogger(__name__)


# =============================================================================
# Static-FP8 constants
# =============================================================================

# Static-FP8 weight dtype.
_FP8_DTYPE = torch.float8_e4m3fn

# Partition dimension size the kernels expect for pre-broadcast scales.
_PMAX = 128

# Number of per-projection scales fused into the QKV weight_scale tensor
# (one each for Q, K, V). Matches the NF.qkv_proj STATIC contract.
_QKV_FUSED = 3

# x4 packing factor for MX FP8: 4 ``float8_e4m3fn`` bytes per ``nl.float8_e4m3fn_x4``.
_Q_WIDTH = 4


# =============================================================================
# Loader selection: static FP8 vs MX FP8
# =============================================================================
#
# Both schemes consume the same ModelOpt static-FP8 checkpoint (plain
# fp8 weights, scalar fp32 weight + activation scales). They differ in
# the *layout* of the weights as they sit in HBM:
#
#   * static FP8: plain ``[H, I]`` / ``[H, fused]`` fp8.
#   * MX FP8 (Trn3 only): MLP gate/up/down weights are pre-swizzled
#     into the layout the trn3 nkilib MLP CTE STATIC_MX kernel expects
#     (other projections stay plain).
#
# The choice is made at module-construction time. Forward picks the
# matching ``QuantizationType`` (STATIC vs STATIC_MX) when calling NF
# kernel wrappers; the wrappers fall back from STATIC_MX to STATIC at
# runtime when per-call constraints fail (e.g. BxS%4 on the QKV TKG MX
# path).


def _pick_loader_module(use_mx: bool):
    """Return the loader module (``weight_loaders_*``) to use."""
    return _mx_fp8_loaders if use_mx else _static_fp8_loaders


# Names that the dual-buffer (monolithic MX prefill + STATIC decode)
# path duplicates. The MX layouts live under the canonical names; the
# STATIC layouts live under ``<name>_tkg``. Used by ``_TkgAlias`` to
# redirect the static loader-attach functions to write to the
# ``_tkg`` parameters without modifying the existing loader code.
#
# Scales are aliased too because the STATIC and MX scale loaders
# differ (STATIC applies a 240/448 compensation paired with the legacy
# weight downscale; MX does not). Sharing the buffer would let the
# STATIC attach overwrite the MX-loaded scale with a compensated
# value — wrong for the MX kernel.
_TKG_ALIASED_ATTRS = (
    "qkv_proj_weight",
    "qkv_weight_scale",
    "qkv_input_scale",
    "o_proj_weight",
    "o_weight_scale",
    "o_input_scale",
    "gate_proj_weight",
    "up_proj_weight",
    "down_proj_weight",
    "gate_weight_scale",
    "up_weight_scale",
    "down_weight_scale",
    "gate_up_input_scale",
    "down_input_scale",
)


class _TkgAlias:
    """Module proxy that aliases ``<name> -> module.<name>_tkg`` for the
    parameters the dual-buffer decode path duplicates. Other attribute
    reads pass through to the underlying module.

    The static FP8 loader-attach functions in ``weight_loaders_static_fp8``
    write to ``module.qkv_proj_weight`` etc. via ``set_weight_loader``.
    Wrapping the LlamaAttention / LlamaMLP module in this proxy and
    handing it to those attach functions causes the loaders to land on
    the ``_tkg`` parameters instead — no edits to the loader files.
    O proj scales are not aliased because they have the same shape on
    both schemes; the static loader can write to the existing buffer.
    """

    __slots__ = ("_module",)

    def __init__(self, module):
        object.__setattr__(self, "_module", module)

    def __getattr__(self, name: str):
        # __getattr__ only fires on misses; __slots__ + object.__setattr__
        # keep this loop-free.
        if name in _TKG_ALIASED_ATTRS:
            tkg_name = f"{name}_tkg"
            if hasattr(self._module, tkg_name):
                return getattr(self._module, tkg_name)
        return getattr(self._module, name)


# Per-call gating: not every kernel has a STATIC_MX implementation in
# nkilib today, so the ``self._use_mx_prefill`` / ``self._use_mx_decode``
# flags get split per call site (the prefill flag drives ``*_cte``
# helpers; the decode flag drives ``*_tkg`` helpers). The kernels that
# DO support STATIC_MX:
#   * MLP CTE / TKG  (prefill + decode)
#   * O proj CTE / TKG (prefill + decode)
#   * QKV TKG (decode) — but requires ``BxS % 4 == 0``, so only
#     spec-decode satisfies it. Plain decode (B=1, S=1) must stay on
#     STATIC. Today we keep this conservative and always emit STATIC
#     here; spec-decode wiring can flip it later.
#   * QKV CTE (prefill) — NOT yet implemented in nkilib; must stay on
#     STATIC.
# Using STATIC_MX for an unsupported kernel triggers a kernel-validator
# assertion at compile time, so each helper below picks the conservative
# value when the kernel isn't ready.


def _quant_type_mlp_cte(use_mx: bool) -> QuantizationType:
    """MLP CTE (prefill): STATIC_MX when MX is on, STATIC otherwise."""
    return QuantizationType.STATIC_MX if use_mx else QuantizationType.STATIC


def _quant_type_mlp_tkg(use_mx: bool) -> QuantizationType:
    """MLP TKG (decode): always STATIC. The dual-buffer path stores
    decode-side weights in flat 2-D fp8 (``*_tkg`` parameters) so the
    legacy STATIC TKG kernel can consume them; STATIC_MX MLP TKG is
    not currently exercised on this path."""
    del use_mx
    return QuantizationType.STATIC


def _quant_type_o_proj_cte(use_mx: bool) -> QuantizationType:
    """O proj CTE (prefill): STATIC_MX on trn3 when MX is on."""
    return QuantizationType.STATIC_MX if use_mx else QuantizationType.STATIC


def _quant_type_o_proj_tkg(use_mx: bool) -> QuantizationType:
    """O proj TKG (decode): always STATIC. The STATIC_MX TKG kernel
    asserts ``B*S % 4 == 0`` (``output_projection_tkg.py:617``); greedy
    decode (B=1, S=1) trips it at compile time. See
    ``_quant_type_qkv_tkg`` for the same reasoning."""
    del use_mx
    return QuantizationType.STATIC


def _quant_type_qkv_cte(use_mx: bool) -> QuantizationType:
    """QKV CTE (prefill): STATIC_MX-via-MX path on trn3.

    The patched QKV CTE kernel (CR-277644685) does NOT have a literal
    ``STATIC_MX`` enum branch — it implements per-tensor static dequant
    by setting ``QuantizationType.MX`` + ``MX_INTERLEAVED`` weight layout
    + supplying ``qkv_w_scale`` and ``qkv_in_scale`` (see
    ``qkv_cte_utils.py:761-781``). We rely on that contract here.
    """
    return QuantizationType.MX if use_mx else QuantizationType.STATIC


def _quant_type_qkv_tkg(use_mx: bool) -> QuantizationType:
    """QKV TKG (decode): always STATIC. The STATIC_MX TKG kernel
    asserts ``BxS % 4 == 0`` (``qkv_tkg_mx_utils.py:104``), which
    greedy decode (B=1, S=1) cannot satisfy — the assert fires at
    compile time even on prefill-only servers because vLLM still
    extracts the decode graph during warmup. Decode is where greedy
    happens, so STATIC is the correct dispatch regardless of
    ``use_mx``. (Prefill MX still works — see ``_quant_type_qkv_cte``.)
    """
    del use_mx
    return QuantizationType.STATIC


def _quant_type_rmsnorm(use_mx: bool) -> QuantizationType:
    """``rmsnorm_quant`` only understands STATIC; the OCP clamp range is
    chosen via the kernel's ``auto_resolve_fp8_dtype`` flag, not the
    ``QuantizationType`` enum."""
    del use_mx
    return QuantizationType.STATIC


def _qkv_weight_layout(use_mx: bool) -> QKVWeightLayout:
    """``MX_INTERLEAVED`` for the STATIC_MX-via-MX path,
    ``CONTIGUOUS`` otherwise (the legacy STATIC kernel only accepts
    contiguous fp8)."""
    return QKVWeightLayout.MX_INTERLEAVED if use_mx else QKVWeightLayout.CONTIGUOUS


def _mlp_gate_up_w_layout(use_mx: bool):
    """``H_X4_MIDDLE`` for STATIC_MX MLP CTE, ``CONTIGUOUS`` otherwise."""
    return (
        MLPGateUpWeightLayout.H_X4_MIDDLE
        if use_mx
        else MLPGateUpWeightLayout.CONTIGUOUS
    )


# Module-level toggles for the prefill / decode kernel paths. The factory
# only routes Trn3 STATIC_FP8 checkpoints here, so MX is the *intent*; the
# split lets a test author flip prefill or decode independently without
# re-introducing an env var. ``__init__`` snapshots these into instance
# attributes ``self._use_mx_prefill`` / ``self._use_mx_decode`` so a
# monkeypatch after import takes effect on subsequent constructions.
#
#   USE_MX_PREFILL: drives QKV CTE / O proj CTE / MLP CTE kernel choice
#                   (STATIC_MX vs STATIC) and the canonical weight-buffer
#                   dtypes / shapes (uint32 6-D vs fp8 2-D).
#   USE_MX_DECODE:  drives QKV TKG / O proj TKG / MLP TKG kernel choice.
#                   Today this MUST be False — every STATIC_MX TKG kernel
#                   asserts ``B*S % 4 == 0`` (qkv_tkg_mx_utils.py:104,
#                   output_projection_tkg.py:617) and greedy decode
#                   (B=1, S=1) trips it at compile time. ``__init__``
#                   asserts the constraint to keep the failure mode
#                   self-explanatory.
#
# When ``USE_MX_PREFILL != USE_MX_DECODE`` (i.e. today's "MX prefill +
# STATIC decode" dual-buffer config) the canonical weight buffers carry
# the prefill layout and ``*_tkg`` mirror buffers carry the decode
# layout. ``forward_decode`` reads through the mirrors. When both flags
# agree, mirrors are not registered.
USE_MX_PREFILL = True
USE_MX_DECODE = False


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

        num_kv_heads_for_weight = (
            self.num_key_value_heads // effective_q_shards
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )

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

        # Snapshot the module-level toggles so a later monkeypatch
        # doesn't desynchronise this layer's prefill and decode views.
        self._use_mx_prefill = USE_MX_PREFILL
        self._use_mx_decode = USE_MX_DECODE
        # ``*_tkg`` mirror buffers exist iff prefill and decode disagree
        # on layout (today's MX-prefill + STATIC-decode dual-buffer).
        self._dual_buffer = self._use_mx_prefill != self._use_mx_decode
        if self._use_mx_decode:
            raise NotImplementedError(
                "USE_MX_DECODE=True is not currently supported: every "
                "STATIC_MX TKG kernel asserts B*S % 4 == 0 (greedy "
                "decode trips it). Wire spec-decode and re-evaluate."
            )

        # QKV: STATIC fp8 takes ``[H, qkv_size] float8_e4m3fn``; MX takes
        # ``[H/4, qkv_size] uint32`` (x4-packed MX_INTERLEAVED, kernel
        # view-casts to ``nl.float8_e4m3fn_x4`` post-DMA — see
        # CR-277644685 / weight_loaders_mx_fp8.py).
        if self._use_mx_prefill:
            self.qkv_proj_weight = nn.Parameter(
                torch.empty(
                    self.hidden_size // _Q_WIDTH,
                    qkv_size,
                    dtype=torch.uint32,
                ),
                requires_grad=False,
            )
        else:
            self.qkv_proj_weight = nn.Parameter(
                torch.empty(self.hidden_size, qkv_size, dtype=_FP8_DTYPE)
            )

        # O proj: identical shape + dtype on both paths. STATIC_MX only
        # rearranges the byte content host-side (see
        # ``weight_pack_mx_fp8.mx_shuffle_o_proj``).
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=_FP8_DTYPE)
        )

        # ── Dual-buffer (prefill ≠ decode layout) ─────────────────────
        # When prefill and decode disagree on the QKV / O proj weight
        # layout the canonical buffers carry the prefill layout (MX
        # uint32 [H/4, fused] for QKV, byte-shuffled fp8 for O proj)
        # and ``*_tkg`` mirror buffers carry the decode (STATIC) layout.
        # Same source weights from the checkpoint, two physical HBM
        # copies. Memory cost: ~2× QKV + O proj weight size; trn3 has
        # the headroom.
        if self._dual_buffer:
            self.qkv_proj_weight_tkg = nn.Parameter(
                torch.empty(self.hidden_size, qkv_size, dtype=_FP8_DTYPE),
                requires_grad=False,
            )
            self.o_proj_weight_tkg = nn.Parameter(
                torch.empty(o_proj_in_features, self.hidden_size, dtype=_FP8_DTYPE),
                requires_grad=False,
            )

        # Dequant / input scale shapes:
        # * STATIC fp8 (trn2 / legacy)   : [_PMAX, 3] / [_PMAX, 1] (broadcast scalar)
        # * STATIC_MX QKV CTE-via-MX     : [1, 3]    / [1, 1]      (compact scalar)
        # * STATIC_MX O proj             : [_PMAX, 1] either column (broadcast scalar)
        if self._use_mx_prefill:
            qkv_w_scale_shape = (1, _QKV_FUSED)
            qkv_in_scale_shape = (1, 1)
        else:
            qkv_w_scale_shape = (_PMAX, _QKV_FUSED)
            qkv_in_scale_shape = (_PMAX, 1)
        self.register_buffer(
            "qkv_weight_scale",
            torch.empty(*qkv_w_scale_shape, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "qkv_input_scale",
            torch.empty(*qkv_in_scale_shape, dtype=torch.float32),
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

        # STATIC-shape scales for the dual-buffer decode path. The MX
        # and STATIC scale loaders differ (STATIC applies a 240/448
        # compensation, MX does not), so even where the buffer shape is
        # the same we keep separate buffers so the two loaders don't
        # stomp on each other's values.
        if self._dual_buffer:
            self.register_buffer(
                "qkv_weight_scale_tkg",
                torch.empty(_PMAX, _QKV_FUSED, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "qkv_input_scale_tkg",
                torch.empty(_PMAX, 1, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "o_weight_scale_tkg",
                torch.empty(_PMAX, 1, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "o_input_scale_tkg",
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
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        self._setup_weight_loaders(config)

    # ------------------------------------------------------------------
    # Weight loaders
    # ------------------------------------------------------------------

    def _setup_weight_loaders(self, config: LlamaConfig):
        """Attach weight loaders.

        On the MX path this attaches MX loaders to the canonical
        ``qkv_proj_weight`` / ``o_proj_weight`` parameters and, in
        addition, attaches STATIC loaders to ``qkv_proj_weight_tkg`` /
        ``o_proj_weight_tkg`` (and the corresponding ``_tkg`` scales)
        via :class:`_TkgAlias` so ``forward_decode`` has plain-fp8
        weights to read. On the STATIC path only the canonical buffers
        exist and a single attach call wires them.
        """
        common_kwargs = dict(
            q_size=self.q_size,
            kv_size=self.kv_size,
            world_size=self.world_size,
            num_kv_replicas=self.num_kv_replicas,
            attention_dp_size=self.attention_dp_size,
            attention_dp_rank=self.attention_dp_rank,
            kv_needs_a2a=self.kv_needs_a2a,
            num_attention_heads=self.num_attention_heads,
            head_dim=self.head_dim,
        )
        loaders = _pick_loader_module(self._use_mx_prefill)
        loaders.attach_attention_loaders(self, **common_kwargs)

        if self._dual_buffer:
            _static_fp8_loaders.attach_attention_loaders(
                _TkgAlias(self), **common_kwargs
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

        # ── Step 1: QKV Projection (STATIC or STATIC_MX-via-MX) ────────
        # ``_use_mx_prefill=False``: legacy STATIC kernel — CONTIGUOUS [H, fused] fp8.
        # ``_use_mx_prefill=True``: MX kernel + per-tensor scales + MX_INTERLEAVED
        # uint32 weight (CR-277644685). The kernel returns bf16 either way.
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
            quantization_type=_quant_type_qkv_cte(self._use_mx_prefill),
            qkv_w_scale=self.qkv_weight_scale,
            qkv_in_scale=self.qkv_input_scale,
            weight_layout=_qkv_weight_layout(self._use_mx_prefill),
            # The STATIC / MX kernels need the head split to shape the
            # matmul correctly. ``NF.qkv_proj`` infers these from the
            # tensor shapes in the bf16 path, but the quantized path
            # requires them explicitly. The MX-static-dequant validator
            # additionally requires ``d_head`` (qkv_cte_utils.py:781).
            d_head=self.head_dim,
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_key_value_heads_per_rank,
        ).squeeze(0)

        # <-- STATIC-FP8: QKV output is dequantized to bf16 by the kernel;
        # from here on the activation dtype matches the bf16 model.
        qkv = qkv.to(self.dtype)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )

        # ── Step 2: RoPE (unchanged) ────────────────────────────────────
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

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
            quantization_type=_quant_type_o_proj_cte(self._use_mx_prefill),
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
        """Decode path (fused megakernel) with STATIC QKV and STATIC output.

        Decode always uses the legacy STATIC TKG kernel. On a real DI
        prefill-only instance the QKV buffer is in MX_INTERLEAVED uint32
        layout (incompatible with this kernel), but vLLM never routes
        decode requests there, so the decode graph is only compiled —
        never executed. Tracing this function with fake tensors is
        therefore safe regardless of ``_use_mx_prefill`` / ``_use_mx_decode``.
        """
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

        k_cache = (
            self.k_cache.squeeze(1) if self.k_cache.dim() == 4 and nkh else self.k_cache
        )
        v_cache = (
            self.v_cache.squeeze(1) if self.v_cache.dim() == 4 and nkh else self.v_cache
        )

        active_blocks_table = block_table

        # <-- STATIC-FP8: pass STATIC quant enum + four scale tensors (qkv +
        # output). KV-scale fusion into softmax/W_out is unchanged from bf16.
        # Dual-buffer (prefill ≠ decode layout): read the ``_tkg``
        # decode-layout weights + scales; the canonical buffers carry
        # the prefill layout which the decode kernel can't consume.
        if self._dual_buffer:
            qkv_w = self.qkv_proj_weight_tkg
            qkv_w_scale = self.qkv_weight_scale_tkg
            qkv_in_scale = self.qkv_input_scale_tkg
            o_w = self.o_proj_weight_tkg
            o_w_scale = self.o_weight_scale_tkg
            o_in_scale = self.o_input_scale_tkg
        else:
            qkv_w = self.qkv_proj_weight
            qkv_w_scale = self.qkv_weight_scale
            qkv_in_scale = self.qkv_input_scale
            o_w = self.o_proj_weight
            o_w_scale = self.o_weight_scale
            o_in_scale = self.o_input_scale
        output, K_new, V_new = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=ln_w is not None,
            rmsnorm_X_eps=eps if ln_w is not None else None,
            rmsnorm_X_gamma=ln_w.view(1, -1) if ln_w is not None else None,
            W_qkv=qkv_w,
            bias_qkv=None,
            quantization_type_qkv=_quant_type_qkv_tkg(self._use_mx_decode),
            weight_dequant_scale_qkv=qkv_w_scale,
            input_dequant_scale_qkv=qkv_in_scale,
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=k_cache,
            V_cache=v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling / self.k_scale_float,
            sink=None,
            update_cache=False,
            W_out=o_w,
            bias_out=None,
            quantization_type_out=_quant_type_o_proj_tkg(self._use_mx_decode),
            weight_dequant_scale_out=o_w_scale,
            input_dequant_scale_out=o_in_scale,
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

        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        num_tokens = slot_mapping.shape[0]

        k_new = (
            K_new.permute(1, 2, 0)
            .reshape(B_local, nkh, S_decode, self.head_dim)
            .transpose(0, 1)
            .reshape(nkh, B_local * S_decode, self.head_dim)
        )
        k_new_flat = k_new.reshape(-1, self.head_dim)
        v_new_flat = V_new.transpose(0, 1).reshape(-1, self.head_dim)

        head_indices_for_put = torch.arange(
            nkh, dtype=torch.long, device=hidden_states.device
        ).repeat_interleave(num_tokens)
        block_indices_for_put = block_indices.repeat(nkh)
        position_indices_for_put = position_indices.repeat(nkh)

        self.k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k_new_flat.to(self.k_cache.dtype),
        )
        self.v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_new_flat.to(self.v_cache.dtype),
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

        # See ``LlamaAttention.__init__`` for the toggle rationale.
        self._use_mx_prefill = USE_MX_PREFILL
        self._use_mx_decode = USE_MX_DECODE
        self._dual_buffer = self._use_mx_prefill != self._use_mx_decode
        if self._use_mx_decode:
            raise NotImplementedError("USE_MX_DECODE=True is not currently supported.")

        # MX layouts (when ``_use_mx_prefill``): gate/up are 6-D ``H_X4_MIDDLE``
        # ``[_PMAX, H/512, I_padded/512, 4, _PMAX, 4]``, down is 4-D
        # ``[_PMAX, I_padded/512, H, 4]``. STATIC layouts are flat 2-D fp8.
        # The MX loaders pad the intermediate dim up to a multiple of 512
        # via ``ceil(I/_TILE_SIZE) * _TILE_SIZE``; the placeholder shape
        # must match what the loader emits or ``load_state_dict`` size-
        # checks fail (``strict=False`` only relaxes key presence).
        if self._use_mx_prefill:
            _TILE = _PMAX * _Q_WIDTH  # 512
            assert config.hidden_size % _TILE == 0
            n_h_tiles = config.hidden_size // _TILE
            n_i_tiles = (self.intermediate_size_per_rank + _TILE - 1) // _TILE
            self.gate_proj_weight = nn.Parameter(
                torch.zeros(
                    _PMAX,
                    n_h_tiles,
                    n_i_tiles,
                    _Q_WIDTH,
                    _PMAX,
                    _Q_WIDTH,
                    dtype=_FP8_DTYPE,
                )
            )
            self.up_proj_weight = nn.Parameter(
                torch.zeros(
                    _PMAX,
                    n_h_tiles,
                    n_i_tiles,
                    _Q_WIDTH,
                    _PMAX,
                    _Q_WIDTH,
                    dtype=_FP8_DTYPE,
                )
            )
            self.down_proj_weight = nn.Parameter(
                torch.zeros(
                    _PMAX,
                    n_i_tiles,
                    config.hidden_size,
                    _Q_WIDTH,
                    dtype=_FP8_DTYPE,
                )
            )
        else:
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

        # ── Dual-buffer (prefill ≠ decode layout) ─────────────────────
        # Allocate flat 2-D fp8 mirrors so ``forward_decode`` can read
        # the STATIC TKG layout when prefill is MX. The MLP scales are
        # the same shape on both schemes — reuse them.
        if self._dual_buffer:
            self.gate_proj_weight_tkg = nn.Parameter(
                torch.empty(
                    config.hidden_size,
                    self.intermediate_size_per_rank,
                    dtype=_FP8_DTYPE,
                ),
                requires_grad=False,
            )
            self.up_proj_weight_tkg = nn.Parameter(
                torch.empty(
                    config.hidden_size,
                    self.intermediate_size_per_rank,
                    dtype=_FP8_DTYPE,
                ),
                requires_grad=False,
            )
            self.down_proj_weight_tkg = nn.Parameter(
                torch.empty(
                    self.intermediate_size_per_rank,
                    config.hidden_size,
                    dtype=_FP8_DTYPE,
                ),
                requires_grad=False,
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

        # Dual-buffer (prefill ≠ decode): separate scale buffers for
        # the decode path because the STATIC scale loader applies a
        # 240/448 compensation that the MX kernel doesn't want. Buffer
        # shapes match the MX path (the kernels accept the same
        # ``[_PMAX, 1]`` broadcast).
        if self._dual_buffer:
            for name in (
                "gate_weight_scale_tkg",
                "up_weight_scale_tkg",
                "down_weight_scale_tkg",
                "gate_up_input_scale_tkg",
                "down_input_scale_tkg",
            ):
                self.register_buffer(
                    name,
                    torch.empty(_PMAX, 1, dtype=torch.float32),
                    persistent=False,
                )

        self._setup_weight_loaders(config)

    def _setup_weight_loaders(self, config):
        """Attach MLP loaders.

        With ``_use_mx_prefill`` on, the canonical gate/up/down weights live in
        the STATIC_MX 6-D / 4-D layouts (loaded via the MX loader) and
        ``forward_decode`` reads ``*_tkg`` flat-2-D fp8 buffers loaded
        via the STATIC loader through :class:`_TkgAlias`. Both copies
        share the MLP scales (same [_PMAX, 1] shape on both schemes).
        """
        common_kwargs = dict(
            intermediate_size_per_rank=self.intermediate_size_per_rank,
            mlp_tp_size=self.mlp_tp_size,
            mlp_tp_rank=self.mlp_tp_rank,
            hidden_size=config.hidden_size,
        )
        loaders = _pick_loader_module(self._use_mx_prefill)
        loaders.attach_mlp_loaders(self, **common_kwargs)

        if self._dual_buffer:
            _static_fp8_loaders.attach_mlp_loaders(_TkgAlias(self), **common_kwargs)

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

        # >>> PARALLELISM: All-gather from SP for full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        if is_prefill:
            # <-- STATIC-FP8 (prefill): explicit rmsnorm_quant → fp8
            # activation; the MLP kernel runs with NO_NORM because CTE
            # mode cannot fuse norm+quant internally.
            hidden_states = NF.rmsnorm_quant(
                hidden_states,
                ln_w=ln_w,
                input_dequant_scale=self.gate_up_input_scale,
                eps=eps,
                quantization_type=_quant_type_rmsnorm(self._use_mx_prefill),
            )
            mlp_norm_type = NormType.NO_NORM
            mlp_ln_w = None
        else:
            # <-- STATIC-FP8 (decode): TKG mode fuses RMSNorm + static
            # quant inside NF.mlp; hand it the (1, H)-shaped gamma and
            # let the kernel do both.
            mlp_norm_type = NormType.RMS_NORM
            mlp_ln_w = ln_w.view(1, -1)

        # Dual-buffer (prefill ≠ decode layout): decode reads ``*_tkg``
        # mirrors with the decode-layout weights + compensated scales.
        # When prefill and decode agree, both share the canonical buffers.
        if self._dual_buffer and not is_prefill:
            gate_w = self.gate_proj_weight_tkg
            up_w = self.up_proj_weight_tkg
            down_w = self.down_proj_weight_tkg
            gate_w_scale = self.gate_weight_scale_tkg
            up_w_scale = self.up_weight_scale_tkg
            down_w_scale = self.down_weight_scale_tkg
            gate_up_in_scale = self.gate_up_input_scale_tkg
            down_in_scale = self.down_input_scale_tkg
            mlp_quant_type = _quant_type_mlp_tkg(self._use_mx_decode)
            mlp_layout = MLPGateUpWeightLayout.CONTIGUOUS
        else:
            gate_w = self.gate_proj_weight
            up_w = self.up_proj_weight
            down_w = self.down_proj_weight
            gate_w_scale = self.gate_weight_scale
            up_w_scale = self.up_weight_scale
            down_w_scale = self.down_weight_scale
            gate_up_in_scale = self.gate_up_input_scale
            down_in_scale = self.down_input_scale
            mlp_quant_type = (
                _quant_type_mlp_cte(self._use_mx_prefill)
                if is_prefill
                else _quant_type_mlp_tkg(self._use_mx_decode)
            )
            mlp_layout = _mlp_gate_up_w_layout(
                self._use_mx_prefill if is_prefill else False
            )

        output = NF.mlp(
            hidden_states,
            gate_w,
            up_w,
            down_w,
            eps=eps,
            ln_w=mlp_ln_w,
            norm_type=mlp_norm_type,
            quantization_type=mlp_quant_type,
            gate_w_scale=gate_w_scale,
            up_w_scale=up_w_scale,
            down_w_scale=down_w_scale,
            gate_up_in_scale=gate_up_in_scale,
            down_in_scale=down_in_scale,
            gate_up_w_layout=mlp_layout,
            # <-- STATIC-FP8: explicitly emit the compute dtype so
            # downstream residual add / reduce-scatter see bf16.
            output_dtype=self.act_dtype,
        )

        # <-- STATIC-FP8: safety-net cast. The NIR declared output spec
        # doesn't always honour ``output_dtype`` on the STATIC path — it
        # can still emit fp8 — so match the reference and force the
        # compute dtype here. Without this, the residual add in the
        # decoder layer fails with a BF16 + FP8 promotion error during
        # graph capture.
        if output.dtype != self.act_dtype:
            output = output.to(self.act_dtype)

        # >>> PARALLELISM: Combine TP (+ MLP DP) shards <<<
        if is_prefill:
            if self.world_size > 1:
                output = self.tp_group.reduce_scatter(output, dim=0)
        else:
            self.mlp_tp_group.all_reduce(output)

        return output
