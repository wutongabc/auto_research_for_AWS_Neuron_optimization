# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Attention Block TKG Kernel

This kernel implements the attention block for token generation (TKG), fusing all stages in SBUF to avoid HBM round-trips.

It performs:
        +-------------------+   +-------------------+   +-------------------+   +-------------------+
        | Input X (HBM)     |-->|   RMSNorm X       |-->| QKV Projection    |-->|  Split QKV → Q,K  |-->
        | [B, S_tkg, H]     |   | (optional)        |   |                   |   |   (transpose)     |
        +-------------------+   +-------------------+   +-------------------+   +-------------------+

        +-------------------+   +-------------------+   +-------------------+   +-------------------+
    --> | RMSNorm Q/K       |-->|   RoPE Embedding  |-->| RMSNorm Q/K       |-->| Quantize K/V      |-->
        | (optional)        |   | (optional)        |   | (optional)        |   | to FP8 (optional) |
        +-------------------+   +-------------------+   +-------------------+   +-------------------+

        +-------------------+   +-----------------------+   +--------------------+   +-------------------+   +-------------------+
    --> | KVDP Input Gather |-->| Attention TKG         |-->| KVDP Output Gather |-->| KV-Cache Update   |-->| Output Projection |
        | (optional)        |   | (softmax(Q·Kᵀ/√d) @ V)|   | (optional)         |   | (optional)        |   | (optional)        |
        +-------------------+   +-----------------------+   +--------------------+   +-------------------+   +-------------------+

Features:
- Supports grouped-query attention (GQA) with a single key/value head
- LNC-2 sharding support
- KV data parallelism (KVDP) for multi-rank inference with sharded KV cache
- Operates with or without output projection
- Optimized for small batch_size * sequence length typical in decoding
- Optional FP8 KV cache quantization for memory-efficient inference
"""

from typing import Any, Dict, Optional, Tuple, Union

import nki
import nki.isa as nisa
import nki.language as nl

try:
    from nki.collectives import ReplicaGroup
except ImportError:
    ReplicaGroup = None  # not available in simulation runtime
from nki.isa.constants import oob_mode

from ...core.attention.attention_tkg import AttnTKGConfig, attention_tkg
from ...core.attention.attention_tkg_utils import is_fp8_e4m3
from ...core.embeddings.rope import RoPE_sbuf
from ...core.output_projection.output_projection_tkg import output_projection_tkg
from ...core.qkv.qkv import qkv
from ...core.utils.allocator import SbufManager, create_auto_alloc_manager
from ...core.utils.common_types import DtypeMode, NormType, QKVOutputLayout, QuantizationType
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import (
    div_ceil,
    get_max_positive_value_for_dtype,
    get_verified_program_sharding_info,
    is_hbm_buffer,
    resolve_fp8_e4m3_dtype,
)
from ...core.utils.logging import get_logger
from ...core.utils.tensor_view import TensorView
from .attention_block_tkg_sharding import (
    KVDPCollectiveMode,
    _KVDP_attention_input_collectives,
    _KVDP_attention_output_collectives,
)


# TODO(NKI-699): Refactor API to use configuration dataclasses for better clarity
# Note: Using keyword-only args (via *) to avoid breaking callers when adding/reordering
# parameters, and to improve readability given the large number of arguments.
@nki.jit
def attention_block_tkg(
    # -- input
    X: nl.ndarray,
    X_hidden_dim_actual: Optional[int],
    # -- rmsnorm X
    rmsnorm_X_enabled: bool,
    rmsnorm_X_eps: Optional[float],
    rmsnorm_X_gamma: Optional[nl.ndarray],
    # -- qkv projections
    W_qkv: nl.ndarray,
    bias_qkv: Optional[nl.ndarray],
    quantization_type_qkv: QuantizationType,
    weight_dequant_scale_qkv: Optional[nl.ndarray],
    input_dequant_scale_qkv: Optional[nl.ndarray],
    # -- Q/K processing: pre-RoPE RMSNorm
    rmsnorm_QK_pre_rope_enabled: bool,
    rmsnorm_QK_pre_rope_eps: float,
    rmsnorm_QK_pre_rope_W_Q: Optional[nl.ndarray],
    rmsnorm_QK_pre_rope_W_K: Optional[nl.ndarray],
    # -- Q/K processing: RoPE
    cos: Optional[nl.ndarray],
    sin: Optional[nl.ndarray],
    rope_contiguous_layout: bool,
    # -- Q/K processing: post-RoPE RMSNorm
    rmsnorm_QK_post_rope_enabled: bool,
    rmsnorm_QK_post_rope_eps: float,
    rmsnorm_QK_post_rope_W_Q: Optional[nl.ndarray],
    rmsnorm_QK_post_rope_W_K: Optional[nl.ndarray],
    # -- attention
    K_cache_transposed: bool,
    active_blocks_table: Optional[nl.ndarray],
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    attention_mask: nl.ndarray,
    sink: Optional[nl.ndarray],
    # -- KV cache update
    update_cache: bool,
    kv_cache_update_idx: Optional[nl.ndarray],
    # -- output projection
    W_out: Optional[nl.ndarray],
    bias_out: Optional[nl.ndarray],
    quantization_type_out: QuantizationType,
    weight_dequant_scale_out: Optional[nl.ndarray],
    input_dequant_scale_out: Optional[nl.ndarray],
    transposed_out: bool,
    # -- output
    out_in_sb: bool,
    # -- transposed input
    transposed_in: bool = False,
    # -- optional params with defaults
    softmax_scale: Optional[float] = None,
    enable_fa_s_prior_tiling: bool = True,
    fp8_packed: bool = False,
    k_scale: Optional[nl.ndarray] = None,
    v_scale: Optional[nl.ndarray] = None,
    sbm: Optional[SbufManager] = None,
    skip_attention: bool = False,
    is_h_transposed_by_4: bool = False,
    KVDP: int = 1,
    KVDP_replica_group: Optional[ReplicaGroup] = None,
    KVDP_collective_mode: Optional[KVDPCollectiveMode] = None,
    KVDP_rank: Optional[nl.ndarray] = None,
    pos_ids: Optional[nl.ndarray] = None,
    swa_start_pos_ids: Optional[nl.ndarray] = None,
    S_ctx: Optional[int] = None,
    max_context_len: Optional[nl.ndarray] = None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
):
    """
    Fused Attention Block for Token Generation (TKG).

    Performs end-to-end attention block computation optimized for autoregressive
    decoding with all stages fused in SBUF to avoid HBM round-trips. Intended for
    small batch sizes (B ≤ 16) and short sequence lengths (S_tkg ≤ 8) typical in
    token generation workloads.

    Dimensions:
        B: Batch size (≤ 16 recommended)
        B_attn: Batch size for attention = B/KVDP when KV data parallelism enabled, otherwise B
        S_tkg: Number of new tokens to generate (≤ 8 required)
        S_ctx: KV cache sequence length in current bucket
        S_max_ctx: Maximum KV cache capacity of current bucket
        H: Hidden dimension (must be multiple of 128)
        d_head: Head dimension (must be even)
        q_heads: Number of query heads
        kv_heads: 1 (GQA with single KV head)
        num_blocks: Number of blocks in block KV cache
        block_len: Block length for block KV cache

    Args:
        X (nl.ndarray): Input hidden states
            Shape:
                [B, S_tkg, H]                                       when in HBM (default)
                [H0=pmax, BxS, H1] where H1=lnc x (H//lnc//pmax)  when in SBUF
                [H0=pmax, n_prgs, H1_shard, BxS]                    when transposed_in=True (HBM)

            When in SBUF, the layout is obtained by rearranging HBM data:
                HBM: (BxS, lnc, H0, H1//lnc) -> SBUF: (H0, BxS, (lnc, H1//lnc))
            This interleaves H1//lnc values from each lnc chunk along the H dimension,
            matching qkv_tkg() kernel's expected SBUF input format.

            When transposed_in=True, X is in the transposed HBM layout produced by
            transposed_out from a preceding layer. The kernel loads only the per-NC
            shard (contiguous DMA), permutes to [H0, BxS, H1_shard] in SBUF, applies
            shard_on_h RMSNorm if enabled, and passes the result to QKV.
        X_hidden_dim_actual (Optional[int]): Actual hidden dim if X is padded

        rmsnorm_X_enabled (bool): Apply RMSNorm to X before QKV projection
        rmsnorm_X_eps (Optional[float]): RMSNorm epsilon (default 1e-3)
        rmsnorm_X_gamma (Optional[nl.ndarray]): [1, H] @ HBM, RMSNorm weights

        W_qkv (nl.ndarray): [H, d_head*(q_heads+2)] @ HBM, QKV projection weights
        bias_qkv (Optional[nl.ndarray]): [1, d_head*(q_heads+2)] @ HBM, QKV bias
        quantization_type_qkv (QuantizationType): Type of quantization for QKV projection (NONE, STATIC, ROW).
        weight_dequant_scale_qkv (Optional[nl.ndarray]): Weight dequantization scale for QKV projection.
            Shape: [PMAX, 3] @ HBM when quantization_type_qkv is STATIC.
            Shape: [PMAX, I] @ HBM when quantization_type_qkv is ROW.
        input_dequant_scale_qkv (Optional[nl.ndarray]): Input dequantization scale for QKV projection.
            Shape: [PMAX, 1] @ HBM when quantization_type_qkv is STATIC.

        rmsnorm_QK_pre_rope_enabled (bool): Apply RMSNorm to Q/K before RoPE
        rmsnorm_QK_pre_rope_eps (float): Pre-RoPE RMSNorm epsilon
        rmsnorm_QK_pre_rope_W_Q (Optional[nl.ndarray]): [1, d_head] @ HBM, Pre-RoPE Q gamma weights
        rmsnorm_QK_pre_rope_W_K (Optional[nl.ndarray]): [1, d_head] @ HBM, Pre-RoPE K gamma weights
        cos (Optional[nl.ndarray]): [d_head//2, B, S_tkg] @ HBM, RoPE cosine embeddings (None = skip RoPE)
        sin (Optional[nl.ndarray]): [d_head//2, B, S_tkg] @ HBM, RoPE sine embeddings (None = skip RoPE)
        rope_contiguous_layout (bool): True for contiguous halves, False for interleaved
        rmsnorm_QK_post_rope_enabled (bool): Apply RMSNorm to Q/K after RoPE
        rmsnorm_QK_post_rope_eps (float): Post-RoPE RMSNorm epsilon
        rmsnorm_QK_post_rope_W_Q (Optional[nl.ndarray]): [1, d_head] @ HBM, Post-RoPE Q weights
        rmsnorm_QK_post_rope_W_K (Optional[nl.ndarray]): [1, d_head] @ HBM, Post-RoPE K weights

        K_cache_transposed (bool): Whether K cache is stored transposed in HBM.
            If True: K cache is [B, d_head, S_ctx]. If False: K cache is [B, S_ctx, d_head].
            Must be False for block KV cache.
        active_blocks_table (Optional[nl.ndarray]): [B, num_blocks] @ HBM, Block indices for block KV cache
        K_cache (nl.ndarray): Key cache @ HBM.
            Flat KV: [B, d_head, S_max_ctx] if K_cache_transposed else [B, S_max_ctx, d_head].
            Block KV: [num_blocks, block_len, d_head] or [num_blocks, 1, block_len, d_head] with kv head dim.
            Block KV with fp8_packed: [num_blocks, block_len // 2, d_head, 2] fp8 or [num_blocks, 1, block_len // 2, d_head, 2] with kv head dim.
        V_cache (nl.ndarray): Value cache @ HBM.
            Flat KV: [B, S_max_ctx, d_head].
            Block KV: [num_blocks, block_len, d_head] or [num_blocks, 1, block_len, d_head] with kv head dim.
        attention_mask (nl.ndarray): Attention mask @ HBM.
            When pos_ids is None: [S_ctx, B, q_heads, S_tkg], full pre-generated attention mask.
            When pos_ids is provided: [S_tkg, B, q_heads, S_tkg], active-only portion of the mask.
        sink (Optional[nl.ndarray]): [H, 1] @ HBM, Attention sink tokens
        softmax_scale (Optional[float]): Scaling factor for attention scores. If None, defaults to (1/√D) / k_scale.
            When using FP8 KV cache (k_scale/v_scale provided) and softmax_scale is None, the kernel automatically
            divides by k_scale to dequantize the KV cache values in the QK matmul, effectively setting
            ``softmax_scale = (1/√D) / k_scale``. When softmax_scale is explicitly provided, the caller is
            responsible for incorporating k_scale (e.g., ``softmax_scale = scaling / k_scale``).
        enable_fa_s_prior_tiling: bool: Whether to enable flash attention (FA) for attention computation.
            When enabled, the attention computation is tiled along the context (s_prior) to reduce peak memory usage.
        fp8_packed (bool): When True with block KV, K_cache is fp8 [num_blocks, block_len // 2, d_head, 2]
            where the last dimension packs two consecutive sequence positions. Enables DMA transpose
            for FP8 block KV, avoiding the slower PE transpose fallback.

        k_scale (Optional[nl.ndarray]): Scale for K quantization to FP8. Shape (PMAX, 1) or (1, 1) @ HBM.
            Must contain a single scalar value (replicated or scalar). When provided with v_scale,
            enables FP8 KV cache quantization. Supported dtypes: float32, float16, bfloat16.
        v_scale (Optional[nl.ndarray]): Scale for V quantization to FP8. Shape (PMAX, 1) or (1, 1) @ HBM.
            Must contain a single scalar value (replicated or scalar). When provided with k_scale,
            enables FP8 KV cache quantization. Supported dtypes: float32, float16, bfloat16.

        update_cache (bool): Update KV cache with new tokens
        kv_cache_update_idx (Optional[nl.ndarray]): [B, S_tkg] for block KV, [B, 1] for flat KV.
            Per-token cache write positions (uint32_max = skip).
            For flat KV, only the start position is needed; consecutive tokens are assumed.
            IMPORTANT: When update_cache=True, kv_cache_update_idx values must NOT overlap with
            the prior cache read range (as determined by mask) used by attention. Overlapping
            read/write positions can cause a data race between cores in the current implementation.

        W_out (Optional[nl.ndarray]): [q_heads*d_head, H] @ HBM, Output projection weights.
            When using FP8 KV cache (k_scale/v_scale provided), the attention output has
            magnitude scaled by v_scale. The caller must absorb v_scale into W_out or its
            dequant scale to compensate: ``W_out = W_out_original / v_scale`` (for NONE
            quantization) or ``weight_dequant_scale_out = scale / v_scale`` (for ROW/STATIC).
        bias_out (Optional[nl.ndarray]): [1, H] @ HBM, Output projection bias
        quantization_type_out (QuantizationType): Type of quantization for output projection (NONE, STATIC, ROW).
        weight_dequant_scale_out (Optional[nl.ndarray]): Weight dequantization scale for output projection.
            Shape: [PMAX, 1] @ HBM when quantization_type_out is STATIC.
            Shape: [PMAX, H] @ HBM when quantization_type_out is ROW.
            When using FP8 KV cache, this scale should incorporate v_scale (see W_out above).
        input_dequant_scale_out (Optional[nl.ndarray]): Input dequantization scale for output projection.
            Shape: [PMAX, 1] @ HBM when quantization_type_out is STATIC.
        transposed_out (bool): Transpose output layout (requires W_out)
        out_in_sb (bool): Return output in SBUF instead of HBM
        transposed_in (bool): When True, X is in transposed HBM layout
            [H0=pmax, n_prgs, H1_shard, BxS] from a preceding layer's transposed_out.
            (Note that using [H0, n_prgs, BxS, H1_shard] instead would require an extra tensor_copy upstream
            so all-in-all it doesn't save us from doing the equivalent in attention.)
            Requires B*S_tkg <= pmax (small batch path). Default: False.
        sbm (Optional[SbufManager]): SBUF memory manager (otherwise auto-allocated)
        skip_attention (bool): Skip attention computation (for testing)
        is_h_transposed_by_4 (bool): Whether input X and RMSNorm gamma have been pre-shuffled
            along the H dimension for MXFP quantization. Required to be True when
            quantization_type_qkv is MX, STATIC_MX, or ROW_MX. The pre-shuffle reorders
            [B, S, H//512, 128, 4] → [B, S, 4, H//512, 128] so the data matches the layout
            expected by the hardware quantize_mx instruction. Default: False.

        KVDP (int): KV cache data parallelism degree - number of ranks that shard the KV cache
            across the batch dimension (1 = disabled). Each rank processes B/KVDP batches.
        KVDP_replica_group (Optional[ReplicaGroup]): Replica group for collective ops
        KVDP_rank (Optional[nl.ndarray]): [1] @ HBM, uint32. This rank's position within
            its KVDP replica group (0 to KVDP-1). Used for batch/head slicing.
            Required when KVDP > 1.
            TODO: replace with get_rank_id_in_replica_group() when available (see NKI-1964).

        pos_ids (Optional[nl.ndarray]): [B, S_tkg] @ HBM, Absolute sequence position of each
            active token. When provided, the kernel generates the prior causal mask on-chip
            (attending to KV positions < pos_id) instead of loading a full pre-generated mask
            from HBM. In this mode, attention_mask carries only the active portion.
        swa_start_pos_ids (Optional[nl.ndarray]): [B, S_tkg] @ HBM, Per-query SWA window start positions.
            When provided together with pos_ids, generates a banded sliding window attention mask
            where each token attends to positions in [start_pos, pos_id). When None, standard
            causal attention mask is generated (token attends to all positions < pos_id).

        S_ctx (Optional[int]): Explicit context length for flat KV with pos_ids. Required when
            pos_ids is provided and block KV is not used, since S_ctx cannot be derived from
            the attention mask. Must be None otherwise.

        max_context_len (Optional[nl.ndarray]): [1] @ HBM, int32 scalar for dynamic FA early exit.
            When provided, the FA loop exits early after processing ceil(max_context_len / tile_size)
            tiles. Requires block KV cache and pos_ids.

        dtype_mode (DtypeMode): Quantization dtype policy forwarded to all
            FP8-aware subkernels and to the FP8 KV cache when ``K_cache`` /
            ``V_cache`` are allocated with the opaque ``"float8e4"`` sentinel.
            Concrete dtypes are honored as-is.
            - ``DtypeMode.NON_OCP`` (default): ``nl.float8_e4m3`` (max=240).
            - ``DtypeMode.OCP``: ``nl.float8_e4m3fn`` (max=448). TRN3 only.
            - ``DtypeMode.AUTO``: ``nl.float8_e4m3fn`` on TRN3, else ``nl.float8_e4m3``.

    KV Data Parallelism (KVDP > 1):
        KV-DP partitions the KV cache across ranks along the batch dimension. Each rank holds
        B/KVDP batches of the KV cache. Before attention: all_gather Q heads, slice Q/K/V batch.
        After attention: all_gather output batch, slice heads.

        When KV data parallelism is enabled, input/output shapes change:
        - B_attn = B / KVDP (batches per rank for attention)
        - q_heads_attn = q_heads * KVDP (query heads per rank after gather)

        Input shape changes:
        - K_cache, V_cache: [B_attn, ...] instead of [B, ...]
        - attention_mask: [S_ctx, B_attn, q_heads_attn, S_tkg]
        - pos_ids: [B_attn, S_tkg] instead of [B, S_tkg]
        - swa_start_pos_ids: [B_attn, S_tkg] instead of [B, S_tkg]
        - kv_cache_update_idx: [B_attn, 1] (caller must slice per rank)

        Output shape changes (when update_cache=False):
        - K_out: [d_head, B_attn, S_tkg]
        - V_out: [B_attn, 1, S_tkg, d_head]

    Returns:
        out (nl.ndarray): Output tensor with shape depending on projection and output location:
            - Without projection (W_out=None):
                - out_in_sb=False: [B, q_heads, d_head, S_tkg] @ HBM
                - out_in_sb=True: [d_head, B*q_heads*S_tkg] @ SBUF
            - With projection (W_out provided):
                - transposed_out=False, out_in_sb=False: [B*S_tkg, H] @ HBM
                - transposed_out=False, out_in_sb=True: [B*S_tkg, H//lnc] @ SBUF
                - transposed_out=True, out_in_sb=False: [128, lnc, H//lnc//128, B*S_tkg] @ HBM
                - transposed_out=True, out_in_sb=True: [128, H//lnc//128, B*S_tkg] @ SBUF
        K_out (nl.ndarray):
            - If update_cache=True: Updated K cache (shape matches K_cache input)
            - If update_cache=False: New K tokens [d_head, B_attn, S_tkg] @ HBM
        V_out (nl.ndarray):
            - If update_cache=True: Updated V cache (shape matches V_cache input)
            - If update_cache=False: New V tokens [B_attn, 1, S_tkg, d_head] @ HBM

    Notes:
        - Requires NeuronCore v3+
        - d_head must be even
        - H must be multiple of pmax
        - Supports grouped-query attention (GQA) with single key/value head
        - LNC-2 sharding support for KV cache updates

    Pseudocode:
        # Stage 1: QKV Projection
        if rmsnorm_X_enabled:
            X_norm = rms_norm(X, rmsnorm_X_gamma, rmsnorm_X_eps)
        QKV = matmul(X_norm, W_qkv) + bias_qkv

        # Stage 2: Q/K Processing
        Q, K = split_and_transpose(QKV)
        if rmsnorm_QK_pre_rope_enabled:
            Q = rms_norm(Q, rmsnorm_QK_pre_rope_W_Q)
            K = rms_norm(K, rmsnorm_QK_pre_rope_W_K)
        if cos != None and sin != None:
            Q, K = rope(Q, cos, sin), rope(K, cos, sin)
        if rmsnorm_QK_post_rope_enabled:
            Q = rms_norm(Q, rmsnorm_QK_post_rope_W_Q)
            K = rms_norm(K, rmsnorm_QK_post_rope_W_K)
        V = extract_V(QKV)

        # Stage 3: Attention
        Q_scaled = Q / sqrt(d_head)
        attn_out = attention_tkg(Q_scaled, K, V, K_cache, V_cache, attention_mask)

        # Stage 4: KV Cache Update
        if update_cache:
            update_kv_cache(K_cache, V_cache, K, V, kv_cache_update_idx)
            K_out, V_out = K_cache, V_cache  # Return updated caches
        else:
            K_out, V_out = K, V  # Return new tokens

        # Stage 5: Output Projection
        if W_out is not None:
            output = matmul(attn_out, W_out) + bias_out
        else:
            output = attn_out

        return output, K_out, V_out
    """

    # ========== Validation and Setup ==========
    config = _validate_and_extract_config(
        X,
        W_qkv,
        K_cache,
        V_cache,
        attention_mask,
        cos,
        sin,
        rmsnorm_X_gamma,
        K_cache_transposed,
        active_blocks_table,
        W_out,
        k_scale,
        v_scale,
        KVDP,
        KVDP_replica_group,
        KVDP_rank,
        out_in_sb,
        skip_attention,
        transposed_in,
        pos_ids,
        swa_start_pos_ids,
        S_ctx,
        fp8_packed,
        quantization_type_qkv,
        is_h_transposed_by_4,
        dtype_mode,
    )

    B, S_tkg = config['B'], config['S_tkg']
    if not transposed_in and B * S_tkg <= nl.tile_size.pmax:
        print(
            f"WARNING: B*S_tkg={B * S_tkg} <= {nl.tile_size.pmax} and transposed_in=False. "
            f"Consider enabling transposed_in=True for zero-conversion inter-layer data flow."
        )
    d_head, q_heads = config['d_head'], config['q_heads']
    S_ctx, S_max_ctx = config['S_ctx'], config['S_max_ctx']
    is_block_kv, blk_len = config['is_block_kv'], config['blk_len']
    cache_had_head_dim = config['cache_had_head_dim']
    do_out_proj = config['do_out_proj']
    K_cache, V_cache = config['K_cache'], config['V_cache']
    kv_quant = config['kv_quant']
    kv_quant_dtype = config['kv_quant_dtype']
    B, B_attn, q_heads_attn = config['B'], config['B_attn'], config['q_heads_attn']
    is_KVDP = config['is_KVDP']
    use_pos_id = config['use_pos_id']
    n_bxs_tiles = config['n_bxs_tiles']
    bxs_tile = config['bxs_tile']
    kv_heads = 1
    I = d_head * (q_heads + 2 * kv_heads)

    sbm = sbm if sbm != None else create_auto_alloc_manager(logger=get_logger("attn-block-tkg"))
    sbm.open_scope(name="attn-blk-tkg-scope")

    dynamic_KVDP_rank_sb = None
    if is_KVDP:
        dynamic_KVDP_rank_sb = nl.ndarray((1, 1), dtype=KVDP_rank.dtype, buffer=nl.sbuf, name="dynamic_KVDP_rank_sb")
        nisa.dma_copy(dynamic_KVDP_rank_sb, KVDP_rank)

    # ========== QKV Projection ==========
    # Input:  X [B, S_tkg, H] @ HBM
    # Output: QKV [B*S_tkg, I] @ SBUF (small batch, non-MX) or HBM (large batch or MX)
    # where I = d_head * (q_heads + 2)
    # qkv() routes to qkv_tkg (SBUF output) or qkv_cte (HBM output) based on B*S_tkg
    rmsnorm_X_eps = 1e-3 if rmsnorm_X_eps == None else rmsnorm_X_eps
    # MX quantization requires output on HBM (SBUF output not yet supported by qkv_tkg_mx_impl)
    output_in_sbuf = n_bxs_tiles == 1 and not quantization_type_qkv.is_mx()
    QKV_out = qkv(
        input=X,
        fused_qkv_weights=W_qkv,
        output_layout=QKVOutputLayout.BSD,
        bias=bias_qkv,
        fused_norm_type=NormType.RMS_NORM if rmsnorm_X_enabled else NormType.NO_NORM,
        gamma_norm_weights=rmsnorm_X_gamma,
        norm_eps=rmsnorm_X_eps,
        hidden_actual=X_hidden_dim_actual,
        quantization_type=quantization_type_qkv,
        qkv_w_scale=weight_dequant_scale_qkv,
        qkv_in_scale=input_dequant_scale_qkv,
        d_head=d_head,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        store_output_in_sbuf=output_in_sbuf,
        sbm=sbm,
        use_auto_allocation=True,
        transposed_in=transposed_in,
        is_h_dim_4h_transposed=is_h_transposed_by_4,
        dtype_mode=dtype_mode,
    )
    QKV_out = QKV_out.reshape((B * S_tkg, I))

    # ========== Q/K Processing + V extraction: Transpose + RMSNorm pre + RoPE + RMSNorm post, K/V quantization ==========
    # Handles tiling internally: for each tile, loads QKV from HBM → SBUF (large batch)
    # or uses QKV directly from SBUF (small batch), then processes Q/K/V per tile.
    # Input:  QKV_out [B*S_tkg, I] @ SBUF or HBM
    # Output: Q_tkg_sb [d_head, B*q_heads*S_tkg] @ SBUF, K_tkg_sb [d_head, B*S_tkg] @ SBUF,
    #         V_tkg_hbm [B, 1, S_tkg, d_head] @ HBM, V_tkg_sb [B*S_tkg, d_head] @ SBUF (small batch only)
    Q_tkg_sb, K_tkg_sb, V_tkg_hbm, V_tkg_sb = _QKV_processing(
        QKV=QKV_out,
        q_heads=q_heads,
        kv_heads=kv_heads,
        B=B,
        S_tkg=S_tkg,
        d_head=d_head,
        n_bxs_tiles=n_bxs_tiles,
        bxs_tile=bxs_tile,
        rmsnorm_pre_enabled=rmsnorm_QK_pre_rope_enabled,
        rmsnorm_pre_eps=rmsnorm_QK_pre_rope_eps,
        rmsnorm_pre_W_Q=rmsnorm_QK_pre_rope_W_Q,
        rmsnorm_pre_W_K=rmsnorm_QK_pre_rope_W_K,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=rope_contiguous_layout,
        rmsnorm_post_enabled=rmsnorm_QK_post_rope_enabled,
        rmsnorm_post_eps=rmsnorm_QK_post_rope_eps,
        rmsnorm_post_W_Q=rmsnorm_QK_post_rope_W_Q,
        rmsnorm_post_W_K=rmsnorm_QK_post_rope_W_K,
        kv_quant=kv_quant,
        kv_quant_dtype=kv_quant_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
        io_dtype=X.dtype,
        sbm=sbm,
    )

    # ========== KV Data Parallelism: Input Collectives ==========
    if is_KVDP:
        # Gather Q heads, slice Q/K/V batch
        #   B -> B_attn (B/KVDP)
        #   q_heads -> q_heads_attn (q_heads*KVDP)
        # Q: [d, B*q_heads*S] @ SBUF -> [d, B_attn*q_heads_attn*S] @ SBUF
        # K: [d, B*S] @ SBUF -> [d, B_attn*S] @ SBUF
        # V: [B, 1, S, d] @ HBM -> [B_attn, 1, S, d] @ HBM
        Q_tkg_sb, K_tkg_sb, V_tkg_hbm = _KVDP_attention_input_collectives(
            Q_tkg_sb,
            K_tkg_sb,
            V_tkg_hbm,
            q_heads,
            kv_heads,
            d_head,
            KVDP,
            B,
            B_attn,
            S_tkg,
            KVDP_replica_group,
            sbm,
            collective_mode=KVDP_collective_mode,
            dynamic_KVDP_rank_sb=dynamic_KVDP_rank_sb,
        )

    # ========== Attention Computation ==========
    # Input:  Q_tkg_sb [d_head, B_attn*q_heads_attn*S_tkg] @ SBUF
    #         K_tkg_sb [d_head, B_attn*S_tkg] @ SBUF
    #         V_tkg_hbm [B_attn, 1, S_tkg, d_head] @ HBM
    # Output: attn_out [d_head, B_attn*q_heads_attn*S_tkg] @ SBUF or [B_attn, q_heads_attn, d_head, S_tkg] @ HBM
    if skip_attention:
        attn_out = Q_tkg_sb
    else:
        # Scale Q by softmax_scale. When softmax_scale is not explicitly provided,
        # default to 1/sqrt(d_head). When using FP8 KV cache (k_scale provided)
        # without explicit softmax_scale, also divide by k_scale to dequantize
        # the KV cache values in the QK matmul.
        _softmax_scale = softmax_scale if softmax_scale != None else d_head ** (-0.5)
        if softmax_scale == None and kv_quant and k_scale != None:
            # Fuse 1/k_scale into the Q scaling to dequantize KV cache in QK matmul.
            # Load k_scale to SBUF, compute reciprocal, apply both scales in one instruction.
            _q_pdim = Q_tkg_sb.shape[0]
            _k_scale_sb = nl.ndarray(shape=(_q_pdim, 1), dtype=nl.float32, buffer=nl.sbuf)
            if k_scale.shape == (nl.tile_size.pmax, 1):
                nisa.dma_copy(_k_scale_sb, k_scale[0:_q_pdim, 0:1])
            else:
                nisa.dma_copy(_k_scale_sb, TensorView(k_scale).broadcast(dim=0, size=_q_pdim).get_view())
            nisa.reciprocal(_k_scale_sb, _k_scale_sb)
            nisa.tensor_scalar(
                dst=Q_tkg_sb,
                data=Q_tkg_sb,
                op0=nl.multiply,
                operand0=_softmax_scale,
                op1=nl.multiply,
                operand1=_k_scale_sb,
            )
        else:
            nisa.tensor_scalar(dst=Q_tkg_sb, data=Q_tkg_sb, op0=nl.multiply, operand0=_softmax_scale)
        # Allocate attention output buffer
        allocate_attn_out_on_HBM = not do_out_proj and not out_in_sb and not is_KVDP
        if allocate_attn_out_on_HBM:
            attn_out = nl.ndarray(
                (B_attn, q_heads_attn, d_head, S_tkg),
                dtype=X.dtype,
                buffer=nl.shared_hbm,
                name=f"{sbm.get_name_prefix()}attn_v_active_hbm",
            )
        else:  # attn_out @ SBUF
            attn_out = sbm.alloc_stack((d_head, B_attn * q_heads_attn * S_tkg), dtype=X.dtype, buffer=nl.sbuf)

        # Prepare KV cache views for attention
        if is_block_kv:
            k_prior, v_prior = K_cache, V_cache
        else:
            k_shape = (B_attn, 1, d_head, S_max_ctx) if K_cache_transposed else (B_attn, 1, S_max_ctx, d_head)
            k_prior = K_cache.reshape(k_shape)
            v_prior = V_cache.reshape((B_attn, 1, S_max_ctx, d_head))

        attn_cfg = AttnTKGConfig(
            bs=B_attn,
            q_head=q_heads_attn,
            s_active=S_tkg,
            curr_sprior=S_ctx,
            full_sprior=S_max_ctx,
            d_head=d_head,
            block_len=blk_len if is_block_kv else 0,
            # tp_k_prior = "kernel needs to transpose K_prior". K_cache_transposed means
            # K is already transposed in HBM, so the kernel does NOT need to transpose it.
            tp_k_prior=not K_cache_transposed,
            strided_mm1=not is_block_kv,
            use_pos_id=use_pos_id,
            fuse_rope=False,
            use_gpsimd_sb2sb=True,
            qk_in_sb=True,
            k_out_in_sb=False,
            out_in_sb=do_out_proj or out_in_sb or is_KVDP,
            enable_fa_s_prior_tiling=enable_fa_s_prior_tiling,
            fp8_packed=fp8_packed,
        )

        attention_tkg(
            q=Q_tkg_sb,
            k_active=K_tkg_sb,
            v_active=V_tkg_hbm,  # Attention_tkg() wants V @ HBM
            k_prior=k_prior,
            v_prior=v_prior,
            mask=attention_mask,
            out=attn_out,  # OUT
            cfg=attn_cfg,
            sbm=sbm,
            rope_pos_ids=pos_ids,  # attention_tkg uses rope_pos_ids for in-kernel causal mask generation
            start_pos_ids=swa_start_pos_ids,
            sink=sink,
            active_blocks_table=active_blocks_table,
            max_context_len=max_context_len,
            dtype_mode=dtype_mode,
        )

    # ========== KV Data Parallelism: Output Gather ==========
    if is_KVDP:
        # Gather batch, slice heads (restore for output projection):
        #   B_attn (B/KVDP) -> B
        #   q_heads_attn (q_heads*KVDP) -> q_heads
        # attn_out: [d, B_attn*q_heads_attn*S] @ SBUF -> [d, B*q_heads*S] @ SBUF
        # V: [B_attn, 1, S, d] @ HBM -> V_tkg_sb [B_attn*S, d] @ SBUF
        attn_out, V_tkg_sb = _KVDP_attention_output_collectives(
            attn_out,
            V_tkg_hbm,
            KVDP,
            B_attn,
            q_heads,
            d_head,
            S_tkg,
            KVDP_replica_group,
            sbm,
            collective_mode=KVDP_collective_mode,
            dynamic_KVDP_rank_sb=dynamic_KVDP_rank_sb,
        )

    # ========== KV Cache Update ==========
    # Input:  K_tkg_sb [d_head, B_attn*S_tkg] @ SBUF
    #         V_tkg_sb [B_attn*S_tkg, d_head] @ SBUF
    # Output: K_hbm_out, V_hbm_out (updated caches or new tokens) @ HBM
    if update_cache:
        _kv_cache_update(
            K_cache=K_cache,
            V_cache=V_cache,
            K_tkg=K_tkg_sb,
            V_tkg=V_tkg_sb if V_tkg_sb is not None else V_tkg_hbm,
            kv_cache_update_idx=kv_cache_update_idx,
            B=B_attn,
            d_head=d_head,
            S_tkg=S_tkg,
            S_max_ctx=S_max_ctx,
            K_cache_transposed=K_cache_transposed,
            is_block_kv=is_block_kv,
            fp8_packed=fp8_packed,
        )
        K_cache, V_cache = __internal_unsqueeze_head_dim(K_cache, V_cache, cache_had_head_dim)
    else:  # No cache update: return new K/V tokens
        K_tkg_hbm = sbm.alloc((d_head, B_attn, S_tkg), dtype=K_tkg_sb.dtype, buffer=nl.shared_hbm, name="K_hbm")
        nisa.dma_copy(K_tkg_hbm.reshape(K_tkg_sb.shape), K_tkg_sb)

    # ========== Output Projection (Optional) ==========
    # Input:  attn_out [d_head, B, q_heads, S_tkg] @ SBUF/HBM
    # Output: kernel_output layout depends on transposed_out and out_in_sb
    if do_out_proj:
        attn_for_proj = attn_out.reshape((d_head, B, q_heads, S_tkg))
        # MX output projection requires attention input on HBM for reshaping over the partition dimension.
        # We cannot output attention directly to HBM earlier because KVDP requires the input in SBUF.
        # We defer the SBUF-to-HBM copy here, preserving the [D, B, N, S] layout.
        if quantization_type_out.is_mx() and attn_for_proj.buffer == nl.sbuf:
            attn_hbm = nl.ndarray(
                attn_for_proj.shape,
                dtype=attn_for_proj.dtype,
                buffer=nl.shared_hbm,
                name=f"{sbm.get_name_prefix()}attn_for_mx_out_proj",
            )
            nisa.dma_copy(attn_hbm, attn_for_proj)
            attn_for_proj = attn_hbm
        kernel_output = output_projection_tkg(
            attention=attn_for_proj,
            weight=W_out,
            bias=bias_out,
            quantization_type=quantization_type_out,
            weight_scale=weight_dequant_scale_out,
            input_scale=input_dequant_scale_out,
            TRANSPOSE_OUT=transposed_out,
            OUT_IN_SB=out_in_sb,
            sbm=sbm,
            dtype_mode=dtype_mode,
        )
    else:
        kernel_assert(not transposed_out, "transposed_out requires output projection (W_out must be provided)")
        kernel_output = attn_out

    # Copy output to HBM if caller expects it on HBM but it's on SBUF. This is only used for debug when skipping both attention and output-projection.
    if out_in_sb == False and kernel_output.buffer == nl.sbuf:
        kernel_output_hbm = nl.ndarray(
            kernel_output.shape, kernel_output.dtype, nl.shared_hbm, name=f"{sbm.get_name_prefix()}kernel_output_hbm"
        )
        nisa.dma_copy(kernel_output_hbm, kernel_output)
        kernel_output = kernel_output_hbm

    # ========== Cleanup and Return ==========
    sbm.close_scope()
    if update_cache:
        return kernel_output, K_cache, V_cache
    else:
        return kernel_output, K_tkg_hbm, V_tkg_hbm


############### Internal ###############


def _validate_and_extract_config(
    X: nl.ndarray,
    W_qkv: nl.ndarray,
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    attention_mask: nl.ndarray,
    cos: Optional[nl.ndarray],
    sin: Optional[nl.ndarray],
    rmsnorm_X_gamma: Optional[nl.ndarray],
    K_cache_transposed: bool,
    active_blocks_table: Optional[nl.ndarray],
    W_out: Optional[nl.ndarray],
    k_scale: Optional[nl.ndarray],
    v_scale: Optional[nl.ndarray],
    KVDP: int,
    KVDP_replica_group: Optional[ReplicaGroup],
    KVDP_rank: Optional[nl.ndarray],
    out_in_sb: bool,
    skip_attention: bool,
    transposed_in: bool = False,
    pos_ids: Optional[nl.ndarray] = None,
    swa_start_pos_ids: Optional[nl.ndarray] = None,
    S_ctx_param: Optional[int] = None,
    fp8_packed: bool = False,
    quantization_type_qkv: QuantizationType = QuantizationType.NONE,
    is_h_transposed_by_4: bool = False,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
) -> Dict[str, Any]:
    """
    Validate inputs and extract configuration parameters for attention block.

    Args:
        X (nl.ndarray): Input hidden states
        W_qkv (nl.ndarray): QKV projection weights
        K_cache (nl.ndarray): Key cache
        V_cache (nl.ndarray): Value cache
        attention_mask (nl.ndarray): Attention mask
        cos (Optional[nl.ndarray]): RoPE cosine embeddings
        sin (Optional[nl.ndarray]): RoPE sine embeddings
        rmsnorm_X_gamma (Optional[nl.ndarray]): RMSNorm weights
        K_cache_transposed (bool): K cache layout flag
        active_blocks_table (Optional[nl.ndarray]): Block indices for block KV cache
        W_out (Optional[nl.ndarray]): Output projection weights

    Returns:
        Dict[str, Any]: Configuration dictionary with keys: B, S_tkg, H, d_head, half_d,
            q_heads, S_ctx, S_max_ctx, is_block_kv, blk_len, cache_had_head_dim,
            do_out_proj, K_cache, V_cache

    Notes:
        - Validates tensor shapes and dimensions
        - Extracts batch size, sequence lengths, and head dimensions
        - Handles both block and flat KV cache layouts
    """

    kernel_assert(
        nisa.get_nc_version() >= nisa.nc_version.gen3,
        f"Kernel requires nc-version >= gen3, got {nisa.get_nc_version()}",
    )

    kernel_assert(
        not quantization_type_qkv.is_mx() or is_h_transposed_by_4,
        "is_h_transposed_by_4 must be True when using MX quantization for QKV",
    )

    use_pos_id = pos_ids is not None

    is_block_kv = active_blocks_table != None
    if use_pos_id and not is_block_kv:
        kernel_assert(S_ctx_param is not None, "S_ctx is required when using pos_ids with flat KV")
    else:
        kernel_assert(S_ctx_param is None, "S_ctx must be None when not using pos_ids or with block KV")

    if transposed_in:
        # X.shape = (H0, n_prgs, H1_shard, BxS) @ HBM
        kernel_assert(len(X.shape) == 4, "transposed_in X must have 4 dimensions [H0, n_prgs, H1_shard, BxS]")
        kernel_assert(X.shape[0] == nl.tile_size.pmax, f"transposed_in X dim0 must be {nl.tile_size.pmax}")
        kernel_assert(is_hbm_buffer(X), "transposed_in X must be in HBM")
        kernel_assert(X.shape[1] >= 2, f"transposed_in requires LNC>=2, got n_prgs={X.shape[1]}")
        H = X.shape[0] * X.shape[1] * X.shape[2]  # H0 * n_prgs * H1_shard = full H
        BxS = X.shape[3]
        S_tkg = attention_mask.shape[3]  # mask is (_, B_attn, q_heads, S_tkg)
        kernel_assert(BxS % S_tkg == 0, f"transposed_in BxS={BxS} must be divisible by S_tkg={S_tkg}")
        B = BxS // S_tkg
    elif X.buffer == nl.sbuf:
        # X.shape = (pmax, B*S, H // pmax) @ SBUF
        kernel_assert(len(X.shape) == 3, "SBUF input X must have 3 dimensions")
        kernel_assert(X.shape[0] == nl.tile_size.pmax, f"SBUF input X dim0 must be {nl.tile_size.pmax}")
        H = X.shape[2] * nl.tile_size.pmax
        S_tkg = attention_mask.shape[3]  # mask is (_, B_attn, q_heads, S_tkg)
        kernel_assert(X.shape[1] % S_tkg == 0, f"SBUF input X dim1={X.shape[1]} must be divisible by S_tkg={S_tkg}")
        B = X.shape[1] // S_tkg
    else:
        # X.shape = (B,S,H) @ HBM
        kernel_assert(is_hbm_buffer(X), "Input X must be in HBM or SBUF")
        B, S_tkg, H = X.shape

    d_head = V_cache.shape[-1]
    I = W_qkv.shape[1]
    kv_heads = 1

    # This limitation can be relaxed
    is_KVDP = KVDP > 1
    if is_KVDP:
        kernel_assert(KVDP_replica_group != None, "KVDP_replica_group is required when KVDP > 1")
        kernel_assert(KVDP_rank != None, "KVDP_rank tensor is required when KVDP > 1")
        kernel_assert(
            KVDP_rank.shape == (1,),
            f"KVDP_rank must have shape (1,), got {KVDP_rank.shape}",
        )
        kernel_assert(
            KVDP_rank.dtype == nl.uint32,
            f"KVDP_rank must have dtype uint32, got {KVDP_rank.dtype}",
        )

    # Compute tiling for large batch support (B * S_tkg > pmax)
    pmax = nl.tile_size.pmax
    bxs_tile = min(B * S_tkg, (pmax // S_tkg) * S_tkg)  # multiple of S_tkg to allow tiling on batch dim
    n_bxs_tiles = div_ceil(B * S_tkg, bxs_tile)
    # Guards for unsupported combinations when B * S_tkg > pmax
    if n_bxs_tiles > 1:
        kernel_assert(
            not out_in_sb,
            f"out_in_sb is not supported when B * S_tkg > {pmax}, got B * S_tkg = {B * S_tkg}",
        )
        kernel_assert(
            not skip_attention,
            f"skip_attention is not supported when B * S_tkg > {pmax}, got B * S_tkg = {B * S_tkg}",
        )
        kernel_assert(
            X.buffer != nl.sbuf,
            f"SBUF input is not supported when B * S_tkg > {pmax}, got B * S_tkg = {B * S_tkg}",
        )
    kernel_assert(d_head % 2 == 0, f"d_head must be even, got {d_head}")
    kernel_assert(
        d_head > 0 and I % d_head == 0,
        f"QKV weights must be packed as (q_heads + 2) * d_head, got I={I}, d_head={d_head}",
    )

    q_heads = I // d_head - 2 * kv_heads
    half_d = d_head // 2

    # Compute KV data parallelism dimensions early for use in validation
    if is_KVDP:
        B_attn = B // KVDP
        q_heads_attn = q_heads * KVDP
        kernel_assert(B % KVDP == 0, f"B must be divisible by KVDP, got B={B}, KVDP={KVDP}")
    else:
        B_attn = B
        q_heads_attn = q_heads

    # Process KV cache — squeeze kv head dim (axis 1) if present
    K_cache, V_cache, cache_had_head_dim = __internal_squeeze_head_dim(K_cache, V_cache, fp8_packed)

    if is_block_kv:
        blk_len = V_cache.shape[1]
        S_ctx = S_max_ctx = active_blocks_table.shape[1] * blk_len
        if fp8_packed:
            kernel_assert(
                K_cache.shape == (V_cache.shape[0], blk_len // 2, V_cache.shape[2], 2),
                f"Block KV fp8_packed shape mismatch: K={K_cache.shape}, "
                f"expected ({V_cache.shape[0]}, {blk_len // 2}, {V_cache.shape[2]}, 2)",
            )
            kernel_assert(
                is_fp8_e4m3(K_cache.dtype),
                f"Block KV fp8_packed requires float8_e4m3 K_cache, got {K_cache.dtype}",
            )
        else:
            kernel_assert(
                V_cache.shape == K_cache.shape,
                f"Block KV cache shape mismatch: K={K_cache.shape} vs V={V_cache.shape}",
            )
    else:
        S_max_ctx = V_cache.shape[1]
        S_ctx = attention_mask.shape[0] if not use_pos_id else S_ctx_param
        blk_len = 0
        kernel_assert(
            V_cache.shape[0] == B_attn,
            f"V_cache batch mismatch: expected {B_attn}, got {V_cache.shape[0]}",
        )
        expected_K_shape = (B_attn, d_head, S_max_ctx) if K_cache_transposed else (B_attn, S_max_ctx, d_head)
        kernel_assert(
            tuple(K_cache.shape) == expected_K_shape,
            f"K_cache shape mismatch: expected {expected_K_shape}, got {K_cache.shape}",
        )

    # Validate attention mask and pos_ids
    if use_pos_id:
        kernel_assert(
            tuple(pos_ids.shape) == (B_attn, S_tkg),
            f"pos_ids shape mismatch: expected ({B_attn}, {S_tkg}), got {pos_ids.shape}",
        )
    expected_mask_dim0 = S_tkg if use_pos_id else S_ctx
    expected_mask_shape = (expected_mask_dim0, B_attn, q_heads_attn, S_tkg)
    kernel_assert(
        tuple(attention_mask.shape) == expected_mask_shape,
        f"attention_mask shape mismatch: expected {expected_mask_shape}, got {attention_mask.shape}",
    )
    if swa_start_pos_ids is not None:
        kernel_assert(
            use_pos_id,
            "pos_ids is required when swa_start_pos_ids is provided",
        )
        kernel_assert(
            tuple(swa_start_pos_ids.shape) == (B_attn, S_tkg),
            f"swa_start_pos_ids shape mismatch: expected ({B_attn}, {S_tkg}), got {swa_start_pos_ids.shape}",
        )

    # Validate RMSNorm weights
    if rmsnorm_X_gamma != None:
        kernel_assert(
            tuple(rmsnorm_X_gamma.shape) == (1, H),
            f"rmsnorm_X_gamma must be (1, {H}), got {rmsnorm_X_gamma.shape}",
        )

    # Validate RoPE embeddings
    if cos != None and sin != None:
        kernel_assert(
            tuple(cos.shape) == (half_d, B, S_tkg),
            f"cos shape mismatch: expected ({half_d}, {B}, {S_tkg}), got {cos.shape}",
        )
        kernel_assert(
            tuple(sin.shape) == (half_d, B, S_tkg),
            f"sin shape mismatch: expected ({half_d}, {B}, {S_tkg}), got {sin.shape}",
        )

    # KV Quantization
    if k_scale != None and v_scale != None:
        kernel_assert(is_fp8_e4m3(K_cache.dtype), f'KV quantization requires float8_e4m3 K_cache, got {K_cache.dtype}')
        kernel_assert(is_fp8_e4m3(V_cache.dtype), f'KV quantization requires float8_e4m3 V_cache, got {V_cache.dtype}')
        kv_quant = True
        # Use the caller's concrete K_cache dtype when explicit; resolve from
        # dtype_mode only for the opaque "float8e4" sentinel.
        if str(K_cache.dtype) == "float8e4":
            kv_quant_dtype = resolve_fp8_e4m3_dtype(dtype_mode)
        else:
            kv_quant_dtype = K_cache.dtype
    else:
        kv_quant = False
        kv_quant_dtype = None

    return {
        'B': B,
        'S_tkg': S_tkg,
        'H': H,
        'd_head': d_head,
        'half_d': half_d,
        'q_heads': q_heads,
        'S_ctx': S_ctx,
        'S_max_ctx': S_max_ctx,
        'is_block_kv': is_block_kv,
        'blk_len': blk_len,
        'cache_had_head_dim': cache_had_head_dim,
        'do_out_proj': W_out != None,
        'K_cache': K_cache,
        'V_cache': V_cache,
        'kv_quant': kv_quant,
        'kv_quant_dtype': kv_quant_dtype,
        'B': B,
        'B_attn': B_attn,
        'q_heads_attn': q_heads_attn,
        'is_KVDP': is_KVDP,
        'use_pos_id': use_pos_id,
        'n_bxs_tiles': n_bxs_tiles,
        'bxs_tile': bxs_tile,
    }


def __internal_squeeze_head_dim(
    K_cache: nl.ndarray, V_cache: nl.ndarray, fp8_packed: bool = False
) -> Tuple[nl.ndarray, nl.ndarray, bool]:
    """
    Remove head dimension from cache tensors with an extra heads=1 dim at axis 1.

    Args:
        K_cache (nl.ndarray): Key cache
        V_cache (nl.ndarray): Value cache
        fp8_packed (bool): Whether K_cache uses fp8_packed layout (base is 4D not 3D)

    Returns:
        Tuple[nl.ndarray, nl.ndarray, bool]: (K_squeezed, V_squeezed, had_head_dim)

    Notes:
        - Non-packed (base 3D): squeezes when 4D with shape[1]==1
            - Non-block KV: Removes N from BNSd or BNdS
            - Block KV: Removes heads from (blocks, 1, block_len, d_head)
        - fp8_packed (base 4D): squeezes when 5D with shape[1]==1
            - Removes heads from (blocks, 1, block_len//2, d_head, 2)
        - Returns original tensors if head dim is not present
    """
    base_ndim = 4 if fp8_packed else 3
    k_expected_ndim_with_head = base_ndim + 1
    v_expected_ndim_with_head = 4  # V is always 3D base + 1 head dim = 4D when head dim present

    if len(K_cache.shape) != k_expected_ndim_with_head:
        kernel_assert(
            len(K_cache.shape) == base_ndim,
            f"Expecting K_cache to have {base_ndim} or {k_expected_ndim_with_head} dims, got {len(K_cache.shape)}",
        )
        kernel_assert(
            len(V_cache.shape) == 3,
            f"Expecting V_cache to have 3 dims when K_cache has {base_ndim}, got {len(V_cache.shape)}",
        )
        return K_cache, V_cache, False

    head_dim = 1
    kernel_assert(
        len(V_cache.shape) == v_expected_ndim_with_head,
        f"Expecting V_cache to have {v_expected_ndim_with_head} dims when K_cache has {k_expected_ndim_with_head}, got {len(V_cache.shape)}",
    )
    kernel_assert(
        K_cache.shape[head_dim] == V_cache.shape[head_dim] == 1,
        f"Expecting single head for KV at axis 1, got K.shape[1]={K_cache.shape[head_dim]}, V.shape[1]={V_cache.shape[head_dim]}",
    )
    K_shape = list(K_cache.shape[:head_dim]) + list(K_cache.shape[head_dim + 1 :])
    V_shape = list(V_cache.shape[:head_dim]) + list(V_cache.shape[head_dim + 1 :])
    return K_cache.reshape(tuple(K_shape)), V_cache.reshape(tuple(V_shape)), True


def __internal_unsqueeze_head_dim(
    K_cache: nl.ndarray, V_cache: nl.ndarray, cache_had_head_dim: bool
) -> Tuple[nl.ndarray, nl.ndarray]:
    """
    Add back head dimension if cache originally had one.

    Args:
        K_cache (nl.ndarray): Key cache
        V_cache (nl.ndarray): Value cache
        cache_had_head_dim (bool): Whether cache originally had head dimension

    Returns:
        Tuple[nl.ndarray, nl.ndarray]: (K_cache, V_cache) with head dimension restored

    Notes:
        - Inverse operation of __internal_squeeze_head_dim
        - Returns original tensors if cache_had_head_dim is False
    """
    if not cache_had_head_dim:
        return K_cache, V_cache

    head_dim = 1
    K_shape = list(K_cache.shape[:head_dim]) + [1] + list(K_cache.shape[head_dim:])
    V_shape = list(V_cache.shape[:head_dim]) + [1] + list(V_cache.shape[head_dim:])
    return K_cache.reshape(tuple(K_shape)), V_cache.reshape(tuple(V_shape))


def _to_sbuf(buf: nl.ndarray, sbm: SbufManager) -> nl.ndarray:
    """
    Ensure buffer is in SBUF; copy from HBM if needed.

    Args:
        buf (nl.ndarray): Input buffer (HBM or SBUF)
        sbm (SbufManager): SBUF memory manager

    Returns:
        nl.ndarray: Buffer in SBUF

    Notes:
        - Returns original buffer if already in SBUF
        - Allocates and copies if buffer is in HBM
    """
    if buf.buffer == nl.sbuf:
        return buf
    else:
        sb = sbm.alloc_stack(buf.shape, dtype=buf.dtype, buffer=nl.sbuf)
        nisa.dma_copy(sb, buf)
        return sb


def _process_head_group(
    dst_4d: TensorView,
    QKV: nl.ndarray,
    qkv_offset: int,
    rmsnorm_pre_enabled: bool,
    rmsnorm_pre_eps: float,
    rmsnorm_pre_W: Optional[nl.ndarray],
    enable_rope: bool,
    sb_cos: Optional[Union[nl.ndarray, TensorView]],
    sb_sin: Optional[Union[nl.ndarray, TensorView]],
    rope_contiguous_layout: bool,
    rmsnorm_post_enabled: bool,
    rmsnorm_post_eps: float,
    rmsnorm_post_W: Optional[nl.ndarray],
    sbm: SbufManager,
) -> None:
    """
    Process Q or K for a single tile: extract heads into dst_4d with layout [d, B, n_heads, S],
    apply optional RMSNorm pre/post and RoPE in-place on dst_4d.
    For Q: n_heads=q_heads, qkv_offset=0
    For K: n_heads=1, qkv_offset=d*q_heads

    Args:
        dst_4d: [d, B, n_heads, S] @ SBUF — pre-allocated output buffer (or sliced view of
            a larger buffer). Written in place. Must be contiguous along (B, n_heads, S) so
            the flattened (d, B*n_heads*S) view used by RMSNorm is well-formed. Use TensorView
            so slices work properly for AP creation.
        QKV: [B*S, I] @ SBUF where B*S <= pmax.
    """
    d, B, n_heads, S = dst_4d.shape
    # Interleaved RoPE internally reshapes its input to (d, B*n_heads*S) for an nc_matmul,
    # which fails BIR partition-step verification when dst_4d is a sliced view of a larger
    # buffer. In that case, stage RoPE through a fresh contiguous buffer.
    needs_rope_staging = enable_rope and not rope_contiguous_layout

    if needs_rope_staging:
        out = sbm.alloc_stack(shape=(d, B, n_heads, S), dtype=QKV.dtype, buffer=nl.sbuf)
        out_tv = TensorView(out)
    else:
        out = dst_4d.get_view()
        out_tv = dst_4d

    # Transpose heads from [B*S, n_heads*d] into out[:, :, head_idx, :]
    for head_idx in range(n_heads):
        psum = nl.ndarray((d, B * S), dtype=QKV.dtype, buffer=nl.psum)
        nisa.nc_transpose(psum, QKV[:, nl.ds(qkv_offset + head_idx * d, d)])
        nisa.tensor_copy(out[:, :, head_idx, :], psum.reshape((d, B, S)))

    # 2D view for RMSNorm. TensorView.reshape preserves parent strides,
    # generating correct ap patterns even when dst is a slice of a larger buffer.
    out_2d = out_tv.reshape((d, B * n_heads * S)).get_view()

    # Pre-RoPE RMSNorm
    if rmsnorm_pre_enabled:
        _rms_norm_inplace(out_2d, rmsnorm_pre_eps, w=rmsnorm_pre_W, sbm=sbm)

    # RoPE (in-place: x_in_sb == x_out_sb == out)
    if enable_rope:
        RoPE_sbuf(out, sb_cos, sb_sin, out, convert_from_interleaved=not rope_contiguous_layout)

    # Post-RoPE RMSNorm
    if rmsnorm_post_enabled:
        _rms_norm_inplace(out_2d, rmsnorm_post_eps, rmsnorm_post_W, sbm)

    if needs_rope_staging:
        nisa.tensor_copy(dst_4d.get_view(), out)


def _QKV_processing(
    QKV: nl.ndarray,
    q_heads: int,
    kv_heads: int,
    B: int,
    S_tkg: int,
    d_head: int,
    n_bxs_tiles: int,
    bxs_tile: int,
    rmsnorm_pre_enabled: bool,
    rmsnorm_pre_eps: float,
    rmsnorm_pre_W_Q: Optional[nl.ndarray],
    rmsnorm_pre_W_K: Optional[nl.ndarray],
    cos: Optional[nl.ndarray],
    sin: Optional[nl.ndarray],
    rope_contiguous_layout: bool,
    rmsnorm_post_enabled: bool,
    rmsnorm_post_eps: float,
    rmsnorm_post_W_Q: Optional[nl.ndarray],
    rmsnorm_post_W_K: Optional[nl.ndarray],
    kv_quant: bool,
    kv_quant_dtype: Optional[str],
    k_scale: Optional[nl.ndarray],
    v_scale: Optional[nl.ndarray],
    io_dtype,
    sbm: SbufManager,
) -> Tuple[nl.ndarray, nl.ndarray, nl.ndarray, Optional[nl.ndarray]]:
    """
    Unified Q/K/V processing with tiling. For each tile:
      1. Load QKV tile to SBUF (from HBM for large batch, or use QKV directly if already in SBUF)
      2. Process Q and K via _process_head_group (transpose, optional pre-RoPE RMSNorm, optional RoPE, optional post-RoPE RMSNorm)
      3. Extract V and copy to HBM

    Args:
        QKV: [B*S, I] @ SBUF or HBM - concatenated Q/K/V projections where I = d*(q_heads+2)
        q_heads: number of query heads
        kv_heads: number of key/value heads
        B: batch size
        S_tkg: sequence length (tokens per batch)
        d_head: head dimension
        n_bxs_tiles: number of tiles for B*S dimension (1 = small batch, >1 = large batch)
        bxs_tile: tile size for B*S dimension
        rmsnorm_pre_enabled: Apply RMSNorm before RoPE
        rmsnorm_pre_eps: Pre-RoPE RMSNorm epsilon
        rmsnorm_pre_W_Q: Pre-RoPE Q gamma weights (optional)
        rmsnorm_pre_W_K: Pre-RoPE K gamma weights (optional)
        cos: RoPE cosine embeddings (None = skip RoPE)
        sin: RoPE sine embeddings (None = skip RoPE)
        rope_contiguous_layout: True for contiguous halves, False for interleaved
        rmsnorm_post_enabled: Apply RMSNorm after RoPE
        rmsnorm_post_eps: Post-RoPE RMSNorm epsilon
        rmsnorm_post_W_Q: Post-RoPE Q weights (optional)
        rmsnorm_post_W_K: Post-RoPE K weights (optional)
        kv_quant: whether to quantize K/V to FP8
        kv_quant_dtype: target dtype for quantized KV cache
        k_scale: scale for K quantization (optional)
        v_scale: scale for V quantization (optional)
        io_dtype: input/output dtype
        sbm: SBUF memory manager

    Returns:
        Q_sb: [d_head, B*q_heads*S_tkg] @ SBUF
        K_sb: [d_head, B*S_tkg] @ SBUF (fp8 if kv_quant)
        V_hbm: [B, 1, S_tkg, d_head] @ HBM (fp8 if kv_quant)
        V_sb: [B*S_tkg, d_head] @ SBUF for small batch KV cache update, None for large batch
    """
    I = d_head * (q_heads + 2 * kv_heads)

    # Allocate full output buffers
    Q_sb = sbm.alloc_stack((d_head, B * q_heads * S_tkg), dtype=io_dtype, buffer=nl.sbuf)
    K_sb = sbm.alloc_stack((d_head, B * S_tkg), dtype=io_dtype, buffer=nl.sbuf)
    V_hbm = nl.ndarray(
        (B, 1, S_tkg, d_head),
        dtype=kv_quant_dtype if kv_quant else io_dtype,
        buffer=nl.shared_hbm,
        name=f"{sbm.get_name_prefix()}v_attention_hbm",
    )

    enable_rope = cos != None and sin != None

    # Load RoPE embeddings to SBUF if needed
    sb_cos, sb_sin = None, None
    if enable_rope:
        sb_cos = _to_sbuf(cos, sbm)
        sb_sin = _to_sbuf(sin, sbm)

    Q_4d = Q_sb.reshape((d_head, B, q_heads, S_tkg))
    K_4d = K_sb.reshape((d_head, B, 1, S_tkg))

    for tile_idx in range(n_bxs_tiles):
        tile_start = tile_idx * bxs_tile
        tile_size = min(bxs_tile, B * S_tkg - tile_start)
        tile_B = tile_size // S_tkg
        tile_b_start = tile_start // S_tkg

        # Load QKV tile to SBUF if QKV is on HBM, otherwise use directly
        if QKV.buffer != nl.sbuf:
            qkv_sb = sbm.alloc_stack((tile_size, I), dtype=io_dtype, buffer=nl.sbuf)
            qkv_tile = TensorView(QKV).slice(0, start=tile_start, end=tile_start + tile_size)
            nisa.dma_copy(qkv_sb, qkv_tile.get_view())
        else:
            qkv_sb = QKV

        # Slice cos/sin for this tile. Use TensorView so the slice's parent strides survive
        # the rewrap inside RoPE_sbuf; a bare ndarray slice would lose its parent strides.
        tile_cos, tile_sin = None, None
        if enable_rope:
            tile_cos = TensorView(sb_cos).slice(1, start=tile_b_start, end=tile_b_start + tile_B)
            tile_sin = TensorView(sb_sin).slice(1, start=tile_b_start, end=tile_b_start + tile_B)

        # Process Q tile directly into Q_sb's [:, tile_b_start:tile_b_start+tile_B, :, :] slice
        _process_head_group(
            dst_4d=TensorView(Q_4d).slice(1, start=tile_b_start, end=tile_b_start + tile_B),
            QKV=qkv_sb,
            qkv_offset=0,
            rmsnorm_pre_enabled=rmsnorm_pre_enabled,
            rmsnorm_pre_eps=rmsnorm_pre_eps,
            rmsnorm_pre_W=rmsnorm_pre_W_Q,
            enable_rope=enable_rope,
            sb_cos=tile_cos,
            sb_sin=tile_sin,
            rope_contiguous_layout=rope_contiguous_layout,
            rmsnorm_post_enabled=rmsnorm_post_enabled,
            rmsnorm_post_eps=rmsnorm_post_eps,
            rmsnorm_post_W=rmsnorm_post_W_Q,
            sbm=sbm,
        )

        # Process K tile directly into K_sb's [:, tile_b_start:tile_b_start+tile_B, :, :] slice
        _process_head_group(
            dst_4d=TensorView(K_4d).slice(1, start=tile_b_start, end=tile_b_start + tile_B),
            QKV=qkv_sb,
            qkv_offset=d_head * q_heads,
            rmsnorm_pre_enabled=rmsnorm_pre_enabled,
            rmsnorm_pre_eps=rmsnorm_pre_eps,
            rmsnorm_pre_W=rmsnorm_pre_W_K,
            enable_rope=enable_rope,
            sb_cos=tile_cos,
            sb_sin=tile_sin,
            rope_contiguous_layout=rope_contiguous_layout,
            rmsnorm_post_enabled=rmsnorm_post_enabled,
            rmsnorm_post_eps=rmsnorm_post_eps,
            rmsnorm_post_W=rmsnorm_post_W_K,
            sbm=sbm,
        )

        # Extract V from QKV to SBUF, then copy to HBM for attention_tkg
        # attention_tkg expects V input from HBM
        V_tile_sb = sbm.alloc_stack((tile_size, d_head), dtype=io_dtype, buffer=nl.sbuf)
        nisa.tensor_copy(V_tile_sb, qkv_sb[:, nl.ds(d_head * (q_heads + kv_heads), d_head)])
        # Quantize V to FP8 for attention when kv_quant=True
        if kv_quant:
            V_tile_sb = _quantize_to_fp8(V_tile_sb, v_scale, sbm, kv_quant_dtype)
        # Write V tile to HBM
        V_hbm_view = TensorView(V_hbm.reshape((B * S_tkg, d_head))).slice(
            0, start=tile_start, end=tile_start + tile_size
        )
        nisa.dma_copy(V_hbm_view.get_view(), V_tile_sb)

    # Quantize K to FP8 for attention when kv_quant=True
    if kv_quant:
        K_sb = _quantize_to_fp8(K_sb, k_scale, sbm, kv_quant_dtype)

    # V_tile_sb from the last (or only) tile is kept for KV cache update (small batch only).
    # For large batch (n_bxs_tiles > 1), V is only on HBM.
    V_sb = V_tile_sb if n_bxs_tiles == 1 else None

    return Q_sb, K_sb, V_hbm, V_sb


def _rms_norm_inplace(
    x: nl.ndarray, eps: float, w: Optional[nl.ndarray] = None, sbm: Optional[SbufManager] = None
) -> None:
    """
    RMS normalization in-place: x / sqrt(mean(x^2) + eps), optionally scaled by w.
    Computed in fp32, result written back to x in original dtype.

    Args:
        x: [d_head, BnS] @ SBUF - input tensor (d_head must be nl.tile_size.pmax), modified in-place
        eps: epsilon for numerical stability
        w: [d_head, 1] @ HBM - optional scale weights
        sbm: SBUF memory manager
    """
    d_head, BnS = x.shape
    kernel_assert(d_head == nl.tile_size.pmax, f"d_head must be {nl.tile_size.pmax}, got {d_head}")

    # Setup constants
    ones_sb = sbm.alloc_stack((d_head, d_head), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_sb, 1.0)
    eps_sb = sbm.alloc_stack((d_head, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(eps_sb, eps)

    # Compute x^2 in fp32
    x_squared = sbm.alloc_stack((d_head, BnS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(x_squared, x, x, nl.multiply)

    total_free_dim = BnS
    fmax = nl.tile_size.gemm_moving_fmax
    rsqrt_sb = sbm.alloc_stack((d_head, BnS), dtype=nl.float32, buffer=nl.sbuf)
    for t_start in range(0, total_free_dim, fmax):
        t_size = min(fmax, total_free_dim - t_start)
        # Compute sum(x^2) via matmul with all-ones matrix
        psum_sb = nl.ndarray((d_head, t_size), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(psum_sb, stationary=ones_sb, moving=x_squared[:, nl.ds(t_start, t_size)])

        # Compute rsqrt(mean(x^2) + eps)
        nisa.activation(
            dst=rsqrt_sb[:, nl.ds(t_start, t_size)], op=nl.rsqrt, data=psum_sb, bias=eps_sb, scale=1.0 / d_head
        )

    # Normalize: x * rsqrt
    out_sb = sbm.alloc_stack((d_head, BnS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(out_sb, x, rsqrt_sb, nl.multiply)

    # Optional scaling by weights
    if w != None:
        w_sb = sbm.alloc_stack((d_head, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(w_sb, w.reshape((d_head, 1)))
        nisa.tensor_scalar(dst=out_sb, data=out_sb, op0=nl.multiply, operand0=w_sb)

    # Copy result back to x with original dtype
    nisa.tensor_copy(dst=x, src=out_sb)


############################# KV cache update logic #############################


def _kv_cache_update(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    B: int,
    d_head: int,
    S_tkg: int,
    S_max_ctx: int,
    K_cache_transposed: bool,
    is_block_kv: bool,
    fp8_packed: bool = False,
) -> Tuple[nl.ndarray, nl.ndarray]:
    """
    Update KV cache with new tokens for token generation.

    Args:
        K_cache: K cache @ HBM
            - Block KV: [num_blocks, block_len, d_head] (or [num_blocks, block_len // 2, d_head, 2] fp8 when fp8_packed)
            - Flat transposed: [B, d_head, S_max_ctx]
            - Flat: [B, S_max_ctx, d_head]
        V_cache: V cache @ HBM
            - Block KV: [num_blocks, block_len, d_head]
            - Flat: [B, S_max_ctx, d_head]
        K_tkg: [d_head, B*S_tkg] @ SBUF
        V_tkg: [B*S_tkg, d_head] @ SBUF or [B, 1, S_tkg, d_head] @ HBM
        kv_cache_update_idx: [B, S_tkg] for block KV, [B, 1] for flat KV (consecutive tokens assumed)
        B: batch size
        d_head: head dimension
        S_tkg: number of new tokens
        S_max_ctx: max cache sequence length
        K_cache_transposed: K cache layout flag
        is_block_kv: block KV cache flag
        fp8_packed: when True with block KV, K_cache is fp8 [num_blocks, block_len // 2, d_head, 2];
            uses parity-split load-modify-store DMA.

    Returns:
        Updated (K_cache, V_cache) - modified in-place
    """

    # TODO: oob_mode.skip not supported for flat cache. Using oob_mode.skip causes accuracy failures (root cause unknown).

    if is_block_kv:
        kernel_assert(
            kv_cache_update_idx.shape == (B, S_tkg),
            f"kv_cache_update_idx shape mismatch: expected {(B, S_tkg)}, got {kv_cache_update_idx.shape}",
        )
    else:
        kernel_assert(
            kv_cache_update_idx.shape == (B, 1),
            f"kv_cache_update_idx shape mismatch for flat KV: expected {(B, 1)}, got {kv_cache_update_idx.shape}",
        )

    if is_block_kv:
        _update_block_cache(K_cache, V_cache, K_tkg, V_tkg, kv_cache_update_idx, S_tkg, B, fp8_packed=fp8_packed)
    elif S_tkg == 1 and B > 1 and (not K_cache_transposed or B > 16):
        # vector DMA with indirect addressing, tiled over batch dim. Bug for S_tkg > 1.
        _update_flat_cache_batched(
            K_cache,
            V_cache,
            K_tkg,
            V_tkg,
            kv_cache_update_idx,
            S_tkg,
            S_max_ctx,
            B,
            d_head,
            K_cache_transposed=K_cache_transposed,
        )
    else:
        # per-batch scalar DMA
        _update_flat_cache(
            K_cache,
            V_cache,
            K_tkg,
            V_tkg,
            K_cache_transposed,
            kv_cache_update_idx,
            S_tkg,
            S_max_ctx,
            B,
            d_head,
        )


def _update_flat_cache_batched(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    S_tkg: int,
    S_max_ctx: int,
    B: int,
    d_head: int,
    K_cache_transposed: bool = False,
) -> None:
    """
    Update flat (non-block) KV cache with new tokens using batched DMA operations.

    This optimized version writes batches in vector DMA operations using vector_offset
    for indirect addressing.
    Tiles over the batch dimension in chunks of pmax to support B > pmax.
    Currently assuming consecutive tokens due to NKILIB-693

    Args:
        K_cache: [B, S_max_ctx, d_head] or [B, d_head, S_max_ctx] K cache in HBM
        V_cache: [B, S_max_ctx, d_head] V cache in HBM
        K_tkg: [d_head, B*S_tkg] new K tokens in SBUF
        V_tkg: [B*S_tkg, d_head] @ SBUF or [B, 1, S_tkg, d_head] @ HBM
        kv_cache_update_idx: [B, 1] start write position per batch (consecutive tokens assumed)
        S_tkg: number of new tokens per batch (must be 1)
        S_max_ctx: maximum cache sequence length
        B: batch size
        d_head: head dimension
    """
    # Validate sharding configuration
    _, n_prgs, prg_id = get_verified_program_sharding_info("kv_cache update", (0, 1), 2)
    kernel_assert(n_prgs <= 2, f"Expected lnc in [1,2], got {n_prgs}")
    kernel_assert(S_tkg == 1, f"_update_flat_cache_batched() only supports S_tkg=1, got {S_tkg}")

    # Validate tensor shapes
    if K_cache_transposed:
        kernel_assert(
            K_cache.shape == (B, d_head, S_max_ctx),
            f"K_cache shape mismatch: expected {(B, d_head, S_max_ctx)}, got {K_cache.shape}",
        )
    else:
        kernel_assert(
            K_cache.shape == (B, S_max_ctx, d_head),
            f"K_cache shape mismatch: expected {(B, S_max_ctx, d_head)}, got {K_cache.shape}",
        )
    kernel_assert(
        V_cache.shape == (B, S_max_ctx, d_head),
        f"V_cache shape mismatch: expected {(B, S_max_ctx, d_head)}, got {V_cache.shape}",
    )
    kernel_assert(
        K_tkg.shape == (d_head, B * S_tkg),
        f"K_tkg shape mismatch: expected {(d_head, B * S_tkg)}, got {K_tkg.shape}",
    )

    tile_sz = nl.tile_size.pmax
    v_on_hbm = V_tkg.buffer != nl.sbuf

    # token_indices are used by V update and non-transposed K update.
    # When K_cache_transposed=True and lnc=2, lnc=1 computes its own k_token_indices,
    # so token_indices would be unused on lnc=1, causing a compiler error.
    needs_token_indices = n_prgs == 1 or prg_id == 0 or not K_cache_transposed

    for b_start in range(0, B, tile_sz):
        tile_B = min(tile_sz, B - b_start)

        # Compute absolute token indices for this tile:
        #   token_indices[b] = kv_cache_update_idx[b] + b * S_max_ctx
        if needs_token_indices:
            token_indices = nl.ndarray((tile_B, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(token_indices, kv_cache_update_idx[nl.ds(b_start, tile_B), 0])
            batch_offset = nl.ndarray((tile_B, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.iota(batch_offset, [[0, 1]], offset=b_start * S_max_ctx, channel_multiplier=S_max_ctx)
            nisa.tensor_tensor(token_indices, token_indices, batch_offset, nl.add)

        # V tile: Vector DGE requires source in SBUF, so load from HBM if needed
        if v_on_hbm:
            v_tile = nl.ndarray((tile_B, d_head), dtype=V_tkg.dtype, buffer=nl.sbuf)
            nisa.dma_copy(v_tile, V_tkg.reshape((B * S_tkg, d_head))[nl.ds(b_start, tile_B), :])
        else:
            v_tile = V_tkg  # single tile

        # Vector DMA with indirect addressing:
        # - Reshape cache to (B*S_max_ctx, d_head) so each row has stride d_head
        # - vector_offset provides per-batch row indices: token_indices[b]
        # - DMA engine scales token_indices[b] by d_head (stride of indirect_dim=0)

        # Update V_cache on lnc=0
        if n_prgs == 1 or prg_id == 0:
            nisa.dma_copy(
                dst=V_cache.reshape((B * S_max_ctx, d_head)).ap(
                    pattern=[[1, tile_B], [d_head, S_tkg], [1, d_head]],
                    offset=0,
                    vector_offset=token_indices,
                    indirect_dim=0,
                ),
                src=v_tile.ap(pattern=[[S_tkg * d_head, tile_B], [d_head, S_tkg], [1, d_head]]),
            )

        # Update K_cache on lnc=1
        if n_prgs == 1 or prg_id == 1:
            # Transpose K tile: (d_head, tile_B) → (tile_B, d_head)
            K_tile_sb = nl.ndarray((tile_B * S_tkg, d_head), dtype=K_tkg.dtype, buffer=nl.sbuf)
            _transpose_sbuf(K_tkg[:, nl.ds(b_start * S_tkg, tile_B * S_tkg)], K_tile_sb)

            if K_cache_transposed:
                # K_cache [B, d_head, S_max_ctx] — strided scatter
                # k_token_indices[b] = b * d_head * S_max_ctx + idx[b]
                k_token_indices = nl.ndarray((tile_B, 1), dtype=nl.uint32, buffer=nl.sbuf)
                k_batch_offset = nl.ndarray((tile_B, 1), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.iota(
                    k_batch_offset, [[0, 1]], offset=b_start * d_head * S_max_ctx, channel_multiplier=d_head * S_max_ctx
                )
                nisa.dma_copy(k_token_indices, kv_cache_update_idx[nl.ds(b_start, tile_B), 0])
                nisa.tensor_tensor(k_token_indices, k_token_indices, k_batch_offset, nl.add)

                nisa.dma_copy(
                    dst=K_cache.reshape((B * d_head * S_max_ctx,)).ap(
                        pattern=[[1, tile_B], [S_max_ctx, d_head], [1, S_tkg]],
                        offset=0,
                        vector_offset=k_token_indices,
                        indirect_dim=0,
                    ),
                    src=K_tile_sb.ap(pattern=[[d_head, tile_B], [1, d_head], [1, S_tkg]]),
                )
            else:
                # K_cache [B, S_max_ctx, d_head] — same pattern as V
                nisa.dma_copy(
                    dst=K_cache.reshape((B * S_max_ctx, d_head)).ap(
                        pattern=[[1, tile_B], [d_head, S_tkg], [1, d_head]],
                        offset=0,
                        vector_offset=token_indices,
                        indirect_dim=0,
                    ),
                    src=K_tile_sb.ap(pattern=[[S_tkg * d_head, tile_B], [d_head, S_tkg], [1, d_head]]),
                )


def _update_flat_cache(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    K_cache_transposed: bool,
    kv_cache_update_idx: nl.ndarray,
    S_tkg: int,
    S_max_ctx: int,
    B: int,
    d_head: int,
) -> None:
    """
    Update flat (non-block) KV cache with new tokens using per-batch scalar_offset.

    This version iterates over batches and uses scalar_offset for indirect addressing.
    Supports any B*S_tkg via tiled K transpose. When V_tkg is on HBM, V is loaded
    per-batch.

    Args:
        K_cache: [B, d_head, S_max_ctx] if transposed else [B, S_max_ctx, d_head] @ HBM
        V_cache: [B, S_max_ctx, d_head] @ HBM
        K_tkg: [d_head, B*S_tkg] @ SBUF
        V_tkg: [B*S_tkg, d_head] @ SBUF or [B, 1, S_tkg, d_head] @ HBM
        K_cache_transposed: K cache layout flag
        kv_cache_update_idx: [B, 1] start write position per batch (consecutive tokens assumed)
        S_tkg: number of new tokens per batch
        S_max_ctx: maximum cache sequence length
        B: batch size
        d_head: head dimension
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("kv_cache update", (0, 1), 2)
    kernel_assert(n_prgs <= 2, f"Expected lnc in [1,2], got {n_prgs}")

    v_on_hbm = V_tkg.buffer != nl.sbuf

    # Validate tensor shapes
    kernel_assert(
        V_cache.shape == (B, S_max_ctx, d_head),
        f"V_cache shape mismatch: expected {(B, S_max_ctx, d_head)}, got {V_cache.shape}",
    )
    kernel_assert(
        K_tkg.shape == (d_head, B * S_tkg),
        f"K_tkg shape mismatch: expected {(d_head, B * S_tkg)}, got {K_tkg.shape}",
    )

    # Tiled K transpose for non-transposed K cache
    if not K_cache_transposed and (n_prgs == 1 or prg_id == 1):
        K_transposed_sb, tile_sz = _tiled_k_transpose(K_tkg, B, S_tkg)

    # Update V_cache on lnc=0
    if n_prgs == 1 or prg_id == 0:
        start_position = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        for batch_idx in range(B):
            nisa.dma_copy(start_position, kv_cache_update_idx[batch_idx, 0])
            if v_on_hbm:
                v_src = V_tkg.reshape((B * S_tkg, d_head))[nl.ds(batch_idx * S_tkg, S_tkg), :]
            else:
                v_src = V_tkg[nl.ds(batch_idx * S_tkg, S_tkg), :]
            nisa.dma_copy(
                dst=V_cache.ap(
                    pattern=[[d_head, S_tkg], [1, d_head]],
                    offset=batch_idx * S_max_ctx * d_head,
                    scalar_offset=start_position,
                    indirect_dim=1,
                ),
                src=v_src,
            )

    # Update K_cache on lnc=1
    if n_prgs == 1 or prg_id == 1:
        if K_cache_transposed:
            kernel_assert(
                K_cache.shape == (B, d_head, S_max_ctx),
                f"K_cache shape mismatch: expected {(B, d_head, S_max_ctx)}, got {K_cache.shape}",
            )
            # K_tkg is already in correct layout [d_head, B*S_tkg]
            start_position = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
            for batch_idx in range(B):
                nisa.dma_copy(start_position, kv_cache_update_idx[batch_idx, 0])
                nisa.dma_copy(
                    dst=K_cache.ap(
                        pattern=[[S_max_ctx, d_head], [1, S_tkg]],
                        offset=batch_idx * d_head * S_max_ctx,
                        scalar_offset=start_position,
                        indirect_dim=2,
                    ),
                    src=K_tkg[:, nl.ds(batch_idx * S_tkg, S_tkg)],
                )
        else:
            kernel_assert(
                K_cache.shape == (B, S_max_ctx, d_head),
                f"K_cache shape mismatch: expected {(B, S_max_ctx, d_head)}, got {K_cache.shape}",
            )
            # Write transposed K to cache — index into tiled K_transposed_sb
            start_position = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
            for batch_idx in range(B):
                nisa.dma_copy(start_position, kv_cache_update_idx[batch_idx, 0])
                k_src = _get_k_transposed_slice(K_transposed_sb, tile_sz, batch_idx, S_tkg)
                nisa.dma_copy(
                    dst=K_cache.ap(
                        pattern=[[d_head, S_tkg], [1, d_head]],
                        offset=batch_idx * S_max_ctx * d_head,
                        scalar_offset=start_position,
                        indirect_dim=1,
                    ),
                    src=k_src,
                )


def _update_block_cache(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    S_tkg: int,
    B: int,
    fp8_packed: bool = False,
) -> None:
    """
    Update block KV cache with new tokens.

    Delegates to vectorized scatter DMA implementation.

    Args:
        K_cache: [num_blocks, block_len, d_head] (or [num_blocks, block_len // 2, d_head, 2] fp8 when fp8_packed)
        V_cache: [num_blocks, block_len, d_head]
        K_tkg: [d_head, B*S_tkg]
        V_tkg: [B*S_tkg, d_head]
        kv_cache_update_idx: [B, S_tkg] per-token slot indices for cache update (uint32 max = skip)
        S_tkg: number of new tokens
        B: batch size
        fp8_packed: when True, K_cache packs two fp8 rows into one bf16 row using parity-split scatter DMA.
    """
    _update_block_cache_vectorized(K_cache, V_cache, K_tkg, V_tkg, kv_cache_update_idx, S_tkg, B, fp8_packed=fp8_packed)


def _update_block_cache_vectorized(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    S_tkg: int,
    B: int,
    fp8_packed: bool = False,
) -> None:
    """Update block KV cache with new tokens using vectorized scatter DMA.

    When fp8_packed=False, K and V are both scattered directly.
    When fp8_packed=True, K uses parity-split load-modify-store: two sequential
    passes (even then odd) are required because each pass does gather-modify-scatter
    on full packed rows. The odd pass must gather rows after the even pass has
    scattered them back, otherwise it would overwrite the even byte that was just updated.

    Args:
        K_cache: [num_blocks, block_len, d_head] or [num_blocks, block_len//2, d_head, 2] fp8 when fp8_packed
        V_cache: [num_blocks, block_len, d_head]
        K_tkg: [d_head, B*S_tkg]
        V_tkg: [B*S_tkg, d_head] @ SBUF or [B, 1, S_tkg, d_head] @ HBM
        kv_cache_update_idx: [B, S_tkg] per-token slot indices for cache update (uint32 max = skip)
        S_tkg: number of new tokens
        B: batch size
        fp8_packed: when True, K_cache is fp8 packed layout
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("kv_cache update", (0, 1), 2)
    kernel_assert(n_prgs <= 2, f"Expected lnc in [1,2], got {n_prgs}")

    v_on_hbm = V_tkg.buffer != nl.sbuf

    num_blocks = V_cache.shape[0]
    blk_len = V_cache.shape[1]
    d_head = V_cache.shape[2]
    BxS = B * S_tkg

    if not fp8_packed:
        kernel_assert(
            K_cache.shape == V_cache.shape,
            f"K/V cache shape mismatch: K={K_cache.shape} vs V={V_cache.shape}",
        )
    kernel_assert(
        K_tkg.shape == (d_head, BxS),
        f"K_tkg shape mismatch: expected {(d_head, BxS)}, got {K_tkg.shape}",
    )

    tile_sz = nl.tile_size.pmax

    # Flatten positions: (B, S_tkg) -> (B*S_tkg, 1) for per-token scatter
    idx_flat = kv_cache_update_idx.reshape((BxS, 1))

    # fp8_packed: reinterpret K_cache as bf16 for DMA
    if fp8_packed:
        num_packed_rows = K_cache.shape[0] * K_cache.shape[1]
        K_cache_bf16 = (
            TensorView(K_cache.reshape((num_packed_rows, d_head * 2))).reinterpret_cast(nl.bfloat16).get_view()
        )
        ones_tile = nl.ndarray((tile_sz, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.iota(ones_tile, [[0, 1]], offset=1)

    for b_start in range(0, BxS, tile_sz):
        tile_B = min(tile_sz, BxS - b_start)

        idx_tile = nl.ndarray((tile_B, 1), dtype=kv_cache_update_idx.dtype, buffer=nl.sbuf)
        nisa.dma_copy(idx_tile, idx_flat[nl.ds(b_start, tile_B)])

        if v_on_hbm:
            v_tile = nl.ndarray((tile_B, d_head), dtype=V_tkg.dtype, buffer=nl.sbuf)
            nisa.dma_copy(v_tile, V_tkg.reshape((BxS, d_head))[nl.ds(b_start, tile_B), :])
        else:
            v_tile = V_tkg[nl.ds(b_start, tile_B), :]

        # V update (core 0)
        if n_prgs == 1 or prg_id == 0:
            nisa.dma_copy(
                dst=V_cache.reshape((num_blocks * blk_len, d_head)).ap(
                    pattern=[[d_head, tile_B], [1, d_head]],
                    offset=0,
                    vector_offset=idx_tile,
                    indirect_dim=0,
                ),
                src=v_tile,
                oob_mode=oob_mode.skip,
            )

        # K update (core 1)
        if n_prgs == 1 or prg_id == 1:
            K_tile_sb = nl.ndarray((tile_B, d_head), dtype=K_tkg.dtype, buffer=nl.sbuf)
            _transpose_sbuf(K_tkg[:, nl.ds(b_start, tile_B)], K_tile_sb)

            if fp8_packed:
                _k_update_fp8_packed(K_cache_bf16, K_tile_sb, idx_tile, ones_tile, tile_B, d_head)
            else:
                nisa.dma_copy(
                    dst=K_cache.reshape((num_blocks * blk_len, d_head)).ap(
                        pattern=[[d_head, tile_B], [1, d_head]],
                        offset=0,
                        vector_offset=idx_tile,
                        indirect_dim=0,
                    ),
                    src=K_tile_sb,
                    oob_mode=oob_mode.skip,
                )


def _k_update_fp8_packed(
    K_cache_bf16: nl.ndarray,
    K_tile_sb: nl.ndarray,
    idx_tile: nl.ndarray,
    ones_tile: nl.ndarray,
    tile_B: int,
    d_head: int,
) -> None:
    """Parity-split load-modify-store for fp8_packed K cache update."""
    # Packed row index: slot_idx >> 1
    packed_row_idx = nl.ndarray((tile_B, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(packed_row_idx, idx_tile, nl.right_shift, 1)

    # Parity: slot_idx & 1 (0=even/low byte, 1=odd/high byte)
    parity_mask = nl.ndarray((tile_B, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(parity_mask, idx_tile, nl.bitwise_and, 1)

    # Even-parity row indices (odd partitions get 0xFFFFFFFF -> OOB skip)
    zero_tile = nl.ndarray((tile_B, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.iota(zero_tile, [[0, 1]], offset=0)
    even_oob_mask = nl.ndarray((tile_B, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_tensor(even_oob_mask, zero_tile, parity_mask, nl.subtract)
    packed_rows_even = nl.ndarray((tile_B, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_tensor(packed_rows_even, packed_row_idx, even_oob_mask, nl.bitwise_or)

    # Odd-parity row indices (even partitions get 0xFFFFFFFF -> OOB skip)
    odd_oob_mask = nl.ndarray((tile_B, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_tensor(odd_oob_mask, parity_mask, ones_tile[nl.ds(0, tile_B), :], nl.subtract)
    packed_rows_odd = nl.ndarray((tile_B, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_tensor(packed_rows_odd, packed_row_idx, odd_oob_mask, nl.bitwise_or)

    # Pass 1 (even): gather packed rows, write fp8 into low bytes, scatter back
    row_buf_even = nl.ndarray((tile_B, d_head), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=row_buf_even,
        src=K_cache_bf16.ap(
            pattern=[[d_head, tile_B], [1, d_head]],
            offset=0,
            vector_offset=packed_rows_even,
            indirect_dim=0,
        ),
        oob_mode=oob_mode.skip,
    )
    row_buf_even_fp8 = TensorView(row_buf_even).reinterpret_cast(nl.float8_e4m3)
    nisa.tensor_copy(
        dst=row_buf_even_fp8.slice(1, start=0, end=d_head * 2, step=2).get_view(),
        src=K_tile_sb,
    )
    nisa.dma_copy(
        dst=K_cache_bf16.ap(
            pattern=[[d_head, tile_B], [1, d_head]],
            offset=0,
            vector_offset=packed_rows_even,
            indirect_dim=0,
        ),
        src=row_buf_even,
        oob_mode=oob_mode.skip,
    )

    # Pass 2 (odd): gather packed rows (with updated even bytes), write fp8 into high bytes, scatter back
    row_buf_odd = nl.ndarray((tile_B, d_head), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=row_buf_odd,
        src=K_cache_bf16.ap(
            pattern=[[d_head, tile_B], [1, d_head]],
            offset=0,
            vector_offset=packed_rows_odd,
            indirect_dim=0,
        ),
        oob_mode=oob_mode.skip,
    )
    row_buf_odd_fp8 = TensorView(row_buf_odd).reinterpret_cast(nl.float8_e4m3)
    nisa.tensor_copy(
        dst=row_buf_odd_fp8.slice(1, start=1, end=d_head * 2, step=2).get_view(),
        src=K_tile_sb,
    )
    nisa.dma_copy(
        dst=K_cache_bf16.ap(
            pattern=[[d_head, tile_B], [1, d_head]],
            offset=0,
            vector_offset=packed_rows_odd,
            indirect_dim=0,
        ),
        src=row_buf_odd,
        oob_mode=oob_mode.skip,
    )


############################# FP8 Quantization Helpers #############################


def _quantize_to_fp8(tensor, scale, sbm, dtype):
    """
    Quantize a tensor to FP8 E4M3 format using a single scalar scale.

    Computes: output = cast_to_fp8(clip(tensor * scale, [-max, max]))

    The scale must represent a single scalar value. Two shapes are supported for
    compatibility with different APIs:
    - (1, 1): scalar, broadcast to partition dim
    - (PMAX, 1): assumed to contain identical values, copied directly

    Args:
        tensor: Input tensor in SBUF, shape (P, F), dtype bf16 or f32
        scale: Scale tensor in HBM, shape (PMAX, 1) or (1, 1).
               Must contain a single scalar value (broadcast or replicated).
               Supported dtypes: float32, float16, bfloat16.
        sbm: SbufManager for allocations
        dtype: Target quantized dtype (e.g. nl.float8_e4m3)

    Returns:
        FP8 E4M3 quantized tensor in SBUF, same shape as input
    """
    kernel_assert(tensor.buffer == nl.sbuf, "quantize_to_fp8 requires tensor in SBUF")
    kernel_assert(not is_fp8_e4m3(tensor.dtype), f"quantize_to_fp8 input already FP8: {tensor.dtype}")

    fp8_max = get_max_positive_value_for_dtype(dtype)
    fp8_min = -fp8_max

    partition_dim = tensor.shape[0]

    # Copy scale to SBUF
    # ndarray avoids anti-dependency with other stack values
    scale_sb = nl.ndarray(shape=(partition_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
    if scale.shape == (nl.tile_size.pmax, 1):
        nisa.dma_copy(dst=scale_sb, src=scale[0:partition_dim, :])
    else:
        kernel_assert(scale.shape == (1, 1), f"scale must be (pmax, 1) or (1, 1), got {scale.shape}")
        nisa.dma_copy(dst=scale_sb, src=TensorView(scale).broadcast(dim=0, size=partition_dim).get_view())

    # Scale: multiply by scale
    tensor_scaled = sbm.alloc_stack(tensor.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(tensor_scaled, tensor, nl.multiply, scale_sb)

    # Clip to FP8 range and cast
    tensor_fp8 = sbm.alloc_stack(tensor.shape, dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(tensor_fp8, tensor_scaled, nl.minimum, fp8_max, op1=nl.maximum, operand1=fp8_min)

    return tensor_fp8


def _transpose_sbuf(src, dst):
    """
    Transpose tensor from SBUF to SBUF via PSUM.

    For FP8: nc_transpose doesn't support FP8, so we cast to bf16, transpose, cast back.

    Args:
        src: Source tensor in SBUF (P, F)
        dst: Destination tensor in SBUF (F, P) - must be pre-allocated
    """
    if is_fp8_e4m3(src.dtype):
        # FP8 workaround: cast to bf16, transpose, cast back
        src_bf16 = nl.ndarray(src.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(src_bf16, src)
        psum = nl.ndarray(dst.shape, dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=psum, data=src_bf16)
        nisa.tensor_copy(dst=dst, src=psum)
    else:
        kernel_assert(
            src.dtype in (nl.bfloat16, nl.float16),
            f"_transpose_sbuf only supports bf16, fp16, or fp8, got {src.dtype}",
        )
        psum = nl.ndarray(dst.shape, dtype=src.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=psum, data=src)
        nisa.tensor_copy(dst=dst, src=psum)


def _tiled_k_transpose(K_tkg: nl.ndarray, B: int, S_tkg: int) -> Tuple[nl.ndarray, int]:
    """
    Transpose K_tkg from (d_head, B*S_tkg) to tiled (tile_sz, n_tiles, d_head) in SBUF.

    Tiles in chunks of tile_sz (multiple of S_tkg, <= pmax) so batch boundaries
    align with tile boundaries. Index result as in _get_k_transposed_slice.

    Args:
        K_tkg: [d_head, B*S_tkg] @ SBUF
        B: batch size
        S_tkg: tokens per batch

    Returns:
        K_transposed_sb: [tile_sz, n_tiles, d_head] @ SBUF
        tile_sz: tile size used (multiple of S_tkg)
    """
    total_bxs = B * S_tkg
    tile_sz = min(total_bxs, (nl.tile_size.pmax // S_tkg) * S_tkg)
    n_k_tiles = div_ceil(total_bxs, tile_sz)
    d_head = K_tkg.shape[0]
    K_transposed_sb = nl.ndarray((tile_sz, n_k_tiles, d_head), K_tkg.dtype, nl.sbuf)
    for t_idx in range(n_k_tiles):
        t_start = t_idx * tile_sz
        t_size = min(tile_sz, total_bxs - t_start)
        k_dst = TensorView(K_transposed_sb).select(1, t_idx).slice(0, start=0, end=t_size).get_view()
        _transpose_sbuf(K_tkg[:, nl.ds(t_start, t_size)], k_dst)
    return K_transposed_sb, tile_sz


def _get_k_transposed_slice(K_transposed_sb: nl.ndarray, tile_sz: int, batch_idx: int, S_tkg: int):
    """Index into tiled K transpose buffer for a given batch.

    Args:
        K_transposed_sb: [tile_sz, n_tiles, d_head] from _tiled_k_transpose
        tile_sz: tile size used (returned by _tiled_k_transpose)
        batch_idx: batch index
        S_tkg: tokens per batch

    Returns:
        View of (S_tkg, d_head) for this batch's K data
    """
    flat_idx = batch_idx * S_tkg
    tile_idx = flat_idx // tile_sz
    tile_off = flat_idx % tile_sz
    return TensorView(K_transposed_sb).select(1, tile_idx).slice(0, start=tile_off, end=tile_off + S_tkg).get_view()
