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
This kernel implements attention specifically optimized for Token Generation (TKG, also known as Decode)
scenarios where the active sequence length is small (typically 8 or smaller).
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import nki.isa as nisa
import nki.language as nl
import numpy as np
from nki.isa import dge_mode, dma_engine, oob_mode

from ..utils.allocator import SbufManager, sizeinbytes
from ..utils.common_types import DtypeMode
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, resolve_fp8_e4m3_dtype
from ..utils.stream_shuffle_broadcast import (
    stream_shuffle_broadcast,
)
from ..utils.tensor_view import TensorView
from ..utils.tp_broadcast import tp_broadcast
from .attention_tkg_utils import (
    AttnTKGConfig,
    TileConstants,
    is_batch_sharded,
    is_fp8_e4m3,
    is_fp8_e5m2,
    is_s_prior_sharded,
    resize_cache_block_len_for_attention_tkg_kernel,
    uses_batch_tiling,
    uses_flash_attention,
)
from .gen_mask_tkg import gen_mask_tkg

_MAX_D_HEAD = 128
_MAX_S_PRIOR_ACCURATE_ROPE = 2**17

_MIN_FLOAT32 = float(np.finfo(np.float32).min)
_MAX_FLOAT32 = float(np.finfo(np.float32).max)

# Sentinel value for inactive block slots in active_blocks_table.
# A batch only needs ceil((prior_tokens + active_tokens) / block_len) blocks;
# remaining ABT slots use this value. Indirect DMA loads of the KV cache use
# oob_mode.skip to skip these entries (since -1 is out of bounds), avoiding
# wasted memory bandwidth. The attention mask ensures they don't contribute
# to the output.
INACTIVE_BLOCK_IDX = np.int32(-1)


def attention_tkg(
    q: nl.ndarray,
    k_active: nl.ndarray,
    v_active: nl.ndarray,
    k_prior: nl.ndarray,
    v_prior: nl.ndarray,
    mask: nl.ndarray,
    out: nl.ndarray,
    cfg: AttnTKGConfig,
    sbm: SbufManager,
    inv_freqs: Optional[nl.ndarray] = None,
    rope_pos_ids: Optional[nl.ndarray] = None,
    start_pos_ids: Optional[nl.ndarray] = None,
    sink: Optional[nl.ndarray] = None,
    active_blocks_table: Optional[nl.ndarray] = None,
    k_out: Optional[nl.ndarray] = None,
    DBG_TENSORS: Optional[tuple] = None,
    max_context_len: Optional[nl.ndarray] = None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
) -> Tuple[nl.ndarray, Optional[nl.ndarray]]:
    """Attention specifically optimized for token-gen (where s_active is small). Can optionally fuse RoPE at the start.
    Please refer to attention_tkg_torch.attention_tkg_torch for an equivalent torch implementation.

    NOTE: KV cache can have a batch size larger than B when kernel caller decides to add an extra buffer batch
    to KV cache to write garbage data. This is irrelevant to kernel impl which strictly uses the first B batches from
    KV cache in all cases. This is denoted as B+ in the shapes below.

    Dimensions:
        B: Batch size
        H: Number of query heads
        d: Head dimension
        s_active: Active sequence length (current tokens being processed)
        s_prior: Prior sequence length (KV cache length)
        block_len: Block length for block KV cache
        block_count: Number of blocks in KV cache

    Args:
      q: Query tensor. NOTE: Q is scaled with 1/sqrt(d_head) iff. cfg.fuse_rope!
        Shape: if cfg.qk_in_sb:
                [d, B * H * s_active] (indexing: [d, b * H * s_active+ h * s_active + s])
              else:
                [B, d, H, s_active]
      k_active: Active key tensor.
        Shape:  if cfg.qk_in_sb:
                  [d, B * s_active] (indexing: [d, b * s_active + s])
                else:
                  [B, d, s_active]
      v_active: Active value tensor. Shape [B, 1, s_active, d]
      k_prior: Prior key tensor from KV cache. Shape [B+, 1, s_prior, d] if cfg.tp_k_prior else [B+, 1, d, s_prior].
               For block KV cache, shape is [B+ * block_count, block_len, d] (indexing: [b * block_count + blk, block_len, d])
               For block KV cache with fp8_packed, shape is [B+ * block_count, block_len // 2, d, 2] fp8
      v_prior: Prior value tensor from KV cache. Shape [B+, 1, s_prior, d].
               For block KV cache, shape is [B+ * block_count, block_len, d] (indexing: [b * block_count + blk, block_len, d])
      mask: Attention mask. Shape [s_active, B, H, s_active] if cfg.use_pos_id else [s_prior, B, H, s_active]
      out: Output tensor.
        Shape: if cfg.out_in_sb:
                [d, B * H * s_active] (indexing: [d, b * H * s_active+ h * s_active + s])
              else:
                [B, H, d, s_active]
      cfg: Kernel configuration with shapes and performance flags. See `AttnTKGConfig`
      sbm: SBUF memory manager for allocating temporary buffers
      inv_freqs: Inverse frequencies for RoPE. Shape [d // 2, 1]. Required when cfg.fuse_rope is True
      rope_pos_ids: Position IDs for RoPE. Shape [B, s_active]. Required when cfg.fuse_rope or cfg.use_pos_id is True
      start_pos_ids: Per-query SWA window start positions. Shape [B, s_active]. Optional.
                    When None, standard attention (full context) is used.
                    When provided, per-query sliding window attention mask is generated.
      sink: Sink attention tokens. Shape [H, 1] for streaming attention sink tokens
      active_blocks_table: Table of active blocks for block KV cache. Shape [B, num_blocks].
                          Required when using block KV cache
      k_out: Output key tensor after RoPE. Populated when cfg.fuse_rope is True, stores k_active after applying RoPE
        Shape: if cfg.k_out_in_sb:
                [d, B * s_active] (indexing [d, b * s_active + s])
              else:
                [B, 1, d, s_active].
      DBG_TENSORS: Optional tuple of 4-5 debug tensors with shared HBM type for intermediate value inspection.
                  Expects:
                    - QK: Result of Q@K^T.
                    - QK_MAX: Result of max reduction of QK.
                    - QK_EXP: Result of exp(QK).
                    - EXP_SUM: Result of sum(exp(QK)).
                    - ACTIVE_TABLE: (only use with block KV) Result after loading the active blocks table.
                  See implementation for shapes of these tensors.
      max_context_len: Optional scalar tensor for dynamic FA early exit. Shape [1], dtype int32.
                      When provided, the FA loop exits early after processing ceil(max_context_len / tile_size)
                      tiles instead of all tiles. Requires block KV cache and use_pos_id=True.
      dtype_mode: Quantization dtype policy for the SBUF K/V tile allocations.
                  When ``k_prior.dtype`` / ``v_prior.dtype`` is concrete
                  (``nl.float8_e4m3`` or ``nl.float8_e4m3fn``), the kernel uses
                  it directly. When the caller leaves the dtype as the opaque
                  ``"float8e4"`` sentinel, ``dtype_mode`` selects the variant:
                  ``NON_OCP`` → ``nl.float8_e4m3``, ``OCP`` → ``nl.float8_e4m3fn``,
                  ``AUTO`` → ``nl.float8_e4m3fn`` on TRN3 else ``nl.float8_e4m3``.
                  Compiler enforces a single E4M3 variant per traced module
                  (``EOCP001``); pick one variant for the whole call graph.

    Returns:
      out: Attention output tensor.
        Shape: if cfg.out_in_sb:
                [d, B * H * s_active] (indexing: [d, b * H * s_active+ h * s_active + s])
              else:
                [B, H, d, s_active]
      k_out: Key output tensor.
        Shape: if cfg.k_out_in_sb:
                [d, B * s_active] (indexing [d, b * s_active + s])
              else:
                [B, 1, d, s_active]

    FEATURES:

    1. Flexible Tensor Placement:
      - q, k, k_out, and out tensors can be placed in either SBUF or HBM
      - When qk_in_sb=True, q and k tensors are pre-loaded in SBUF (required for block KV cache)
      - out_in_sb and k_out_in_sb flags control output tensor placement for reduced memory transfers
      - Use this feature for performance improvement when integrating this kernel into a larger kernel

    2. Adaptive LNC2 Sharding:
      - Automatically selects sharding strategy based on tensor dimensions
      - Batch sharding: Used when batch is even AND (s_prior < 256 OR b*q_head*s_active > 128)
      - Sequence sharding: Used when s_prior >= 256 and batch sharding criteria not met
      - Balances computation across 2 NeuronCores for improved throughput

    3. Mask Generation:
      - use_pos_id=False: Pre-generated mask loaded from HBM
      - use_pos_id=True: Mask generated in-kernel from position IDs
      - In-kernel generation reduces memory bandwidth but requires position ID input

    4. Fused RoPE (Rotary Position Embedding):
      - fuse_rope integrates RoPE computation directly into the attention kernel
      - Applies rotary embeddings to Q and K tensors, scaling Q by 1/sqrt(d_head)
      - Reduces memory traffic by avoiding separate RoPE passes

    5. Block KV Cache:
      - Supports block-sparse KV cache with configurable block_len
      - Uses active_blocks_table to track which cache blocks are active per batch
      - Enables efficient long-context inference with sparse memory access patterns

    6. K_prior Transpose Handling:
      - tp_k_prior flag indicates whether the kernel needs to transpose K_prior during load
      - Flat KV:
        - True: K_prior is [B, 1, s_prior, d] in HBM, kernel transposes to [d, s_prior] in SBUF
        - False: K_prior is [B, 1, d, s_prior] in HBM, kernel loads directly (already transposed)
      - Block KV: must be True. K_prior is [num_blocks, block_len, d], kernel always transposes during block loading

    7. Strided Memory Access (strided_mm1):
      - Enables strided read patterns for K in first matmul
      - When enabled, allows MM2 to use sequential V reads for better DMA throughput
      - Trades off MM1 memory access for MM2 optimization

    8. Attention Sink:
      - Supports streaming attention with sink tokens for infinite context
      - Sink tokens maintain fixed attention scores across all positions
      - Integrated into softmax reduction for minimal overhead

    9. GPSIMD SBUF-to-SBUF Transfers:
      - use_gpsimd_sb2sb enables high-performance GPSIMD instructions for inter-core communication
      - Optimizes LNC2 sharding by using extended instructions for SBUF-to-SBUF data transfers

    10. Context Length Management:
      - curr_sprior: Current prior sequence length (actual KV cache content for this invocation)
      - full_sprior: Full prior sequence length (maximum KV cache capacity allocated)
      - Allows progressive filling of KV cache during autoregressive generation

    11. Stack-based SBUF Allocation:
      - Uses SbufManager for efficient on-chip memory management
      - Hierarchical scoping with interleave_degree for multi-bank utilization
      - Automatic alignment and temporary buffer lifecycle management

    IMPLEMENTATION DETAILS:

      The kernel goes through the following steps:
        -1. Setup of intermediate buffers, mask, block KV, and debug tensors.
         0. Perform rope if fuse_rope is set
         1. Performs the KQ^T computation.
          - Loop over each batch
          - Load the current chunk of K based on configuration (block KV, transpose, etc.)
          - Tile over the multiplication of K and Q in groups of 4k size
         2. Compute the max reduction of KQ^T computation.
          - Compute the max in tiles of size 128 over bs * q_head * s_active
          - Prepare the sink if used
          - Transpose and broadcast along the partition dimension
         3. Compute Exp(KQ^T - max(KQ^T))
          - Add/subtract the max based on whether it was negated
          - Apply the exponentiation activation
         4. Compute sum reduction of the exponentiation result
          - Compute the sum in tiles of size 128 over bs * q_head * s_active
          - Perform additional reductions based on sink or other optimization flags
          - Compute the reciprocal with the same tiling scheme, and then broadcast
         5. Compute the product of the above and V and store the result
          - Loop over each batch
          - Load the current chunk of V based on configuration (same as step 1)
          - Perform the matmul over sprior tiles
          - If needed, copy information over core boundaries or to HBM

    INTENDED USAGE:

      This kernel is optimized for cases when there are few active tokens.
      Use with s_active <= 7, and with d_head <= 128.

    Notes:
        - KV cache can have batch size larger than B (denoted B+) for garbage data buffering
        - Q is scaled with 1/sqrt(d_head) only when cfg.fuse_rope is True
        - Block KV cache requires qk_in_sb=True
        - LNC2 sharding automatically selected based on tensor dimensions
        - Extended GPSIMD instructions require 16-partition alignment

    Pseudocode:
        # Setup
        TC = get_tile_constants()
        atp = compute_tile_params(cfg, TC, q, active_blocks_table)
        bufs = allocate_internal_buffers()

        # Step 0: Optional RoPE
        if cfg.fuse_rope:
            q_sb, k_active_sb = apply_rope(q, k_active, inv_freqs, rope_pos_ids)

        loop over flash_attention_tile_idx:
            # Step 1: Compute KQ^T
            for batch_idx in range(bs):
                k_sb = load_k_prior_and_active(k_prior, k_active, batch_idx)
                qk[batch_idx] = matmul(k_sb, q_sb[batch_idx])  # Tiled in 4k groups

            # Step 2: Max reduction
            qk_max = reduce_max(qk, axis=s_prior)  # Cascaded reduction
            if sprior_n_prgs > 1:
                qk_max = sendrecv_and_reduce(qk_max)
            if sink is not None:
                qk_max = reduce_with_sink(qk_max, sink)

            # Step 3: Compute exp(QK - max)
            qk_exp = exp(qk - qk_max)

            # Step 4: Sum reduction and reciprocal
            exp_sum = reduce_sum(qk_exp, axis=s_prior)  # Cascaded reduction
            if sprior_n_prgs > 1:
                exp_sum = sendrecv_and_add(exp_sum)
            if sink is not None:
                exp_sum = add_sink_contribution(exp_sum, sink, qk_max)
            exp_sum_recip = reciprocal(exp_sum)

            # Step 5: Compute (exp @ V)^T
            for batch_idx in range(bs):
                v_sb = load_v_prior_and_active(v_prior, v_active, batch_idx)
                exp_v[batch_idx] = matmul(v_sb, qk_exp[batch_idx]) * exp_sum_recip[batch_idx]

            if sprior_n_prgs > 1:
                exp_v = sendrecv_and_add(exp_v)

        finalize_flash_attention_and_store_output(exp_v, out)
    """

    TC = TileConstants.get_tile_constants()
    atp = _compute_tile_params(cfg, TC, q, k_prior, v_prior, k_active, v_active, active_blocks_table, dtype_mode)
    # Initialize batch-dependent fields with full per-NC batch (before any batch tiling)
    # This is used to set up the debug tensor shapes.
    _update_atp_for_batch_tile(atp, atp.bs_per_nc, TC)
    bufs = AttnInternalBuffers()

    if atp.is_block_kv:
        _setup_block_kv_cache(
            k_prior,
            v_prior,
            k_active,
            v_active,
            active_blocks_table,
            atp,
            cfg,
            TC,
            sbm,
            bufs,
        )

    if DBG_TENSORS:
        _setup_debug_tensors(DBG_TENSORS, atp, TC, bufs)

    bufs.one_vec = sbm.alloc_stack(
        (TC.p_max, 1), dtype=atp.io_type, buffer=nl.sbuf, align=4
    )  # align to 4 bytes to prevent race condition (TODO: fix properly in NKILIB-876)
    nisa.memset(bufs.one_vec, value=1.0)

    # Load position IDs (needed for RoPE and mask generation)
    _load_position_ids(rope_pos_ids, start_pos_ids, atp, cfg, TC, sbm, bufs)

    # Step 0. Optional RoPE
    if cfg.fuse_rope:
        _perform_rope(q, k_active, inv_freqs, k_out, atp, cfg, TC, sbm, bufs)
    else:
        kernel_assert(
            cfg.qk_in_sb,
            "Currently only suppport skipping fusing RoPE when QK is in SBUF (qk_in_sb==True).",
        )
        bufs.q_sb = q
        bufs.k_active_sb = k_active

    # Compute batch tiling parameters
    _, batch_tile_size = uses_batch_tiling(
        atp.bs_per_nc,
        cfg.q_head,
        cfg.s_active,
        atp.fa_tile_s_prior,
        sbm.is_auto_alloc(),
        dtype_size=sizeinbytes(atp.io_type),
    )
    num_batch_tiles = div_ceil(atp.bs_per_nc, batch_tile_size)

    # Pre-compute dynamic FA trip count (outside batch loop — doesn't depend on batch tile)
    num_non_last_tiles_reg = None
    if max_context_len is not None and atp.use_fa:
        kernel_assert(
            atp.is_block_kv,
            "Dynamic FA early exit (max_context_len) requires block KV cache. Flat KV is not supported.",
        )
        kernel_assert(
            cfg.use_pos_id,
            "Dynamic FA early exit (max_context_len) requires use_pos_id=True for in-kernel mask generation.",
        )
        kernel_assert(
            atp.s_prior % atp.fa_tile_s_prior == 0,
            f"Dynamic FA early exit (max_context_len) requires s_prior ({atp.s_prior}) to be a multiple of "
            f"fa_tile_size ({atp.fa_tile_s_prior}).",
        )
        max_ctx_sbuf = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=max_ctx_sbuf, src=max_context_len)
        if atp.interleaved_fa_tiles:
            nisa.tensor_scalar(dst=max_ctx_sbuf, data=max_ctx_sbuf, op0=nl.right_shift, operand0=1)
        fa_tile_shift = int(math.log2(atp.fa_tile_s_prior))
        # Compute: num_non_last_tiles = max(ceil(mcl / tile_size) - 1, 0)
        #   Step 1: mcl += tile_size - 1          (prepare for integer ceil division)
        #   Step 2: mcl >>= log2(tile_size)       (= ceil(mcl / tile_size))
        #   Step 3: mcl = max(mcl - 1, 0)         (subtract 1 for non-last count, clamp to 0)
        nisa.tensor_scalar(dst=max_ctx_sbuf, data=max_ctx_sbuf, op0=nl.add, operand0=atp.fa_tile_s_prior - 1)
        nisa.tensor_scalar(dst=max_ctx_sbuf, data=max_ctx_sbuf, op0=nl.right_shift, operand0=fa_tile_shift)
        nisa.tensor_scalar(dst=max_ctx_sbuf, data=max_ctx_sbuf, op0=nl.add, operand0=-1, op1=nl.maximum, operand1=0)
        num_non_last_tiles_reg = nisa.register_alloc()
        nisa.register_load(num_non_last_tiles_reg, max_ctx_sbuf)

    # Batch outer loop - tiles the batch dimension to fit within SBUF memory budget
    # When batch_tile_size == atp.bs_per_nc, num_batch_tiles=1 so this is a single iteration
    for batch_tile_idx in range(num_batch_tiles):
        tile_batch_offset = batch_tile_idx * batch_tile_size
        tile_bs = min(batch_tile_size, atp.bs_per_nc - tile_batch_offset)
        btc = BatchTileContext(
            batch_tile_idx=batch_tile_idx,
            tile_bs=tile_bs,
            tile_batch_offset=tile_batch_offset,
            global_batch_offset=atp.bs_prg_id * atp.bs_per_nc + tile_batch_offset,
        )
        _update_atp_for_batch_tile(atp, tile_bs, TC)

        # Open scope for this batch tile's buffers (FA running buffers, etc.)
        sbm.open_scope()

        if atp.use_online_softmax:
            _allocate_online_softmax_buffers(atp, cfg, sbm, bufs, is_dynamic=(num_non_last_tiles_reg is not None))

        # Flash Attention loop - iterates over tiles of s_prior
        # When use_fa=False, num_fa_tiles=1 so this is a single iteration
        if num_non_last_tiles_reg is not None:
            # Dynamic FA early exit: split into N-1 dynamic iterations + 1 static last tile.
            # Trip count pre-computed in num_non_last_tiles_reg above.

            # Reset tile offset counters for this batch tile
            # Interleaved LNC2 (sprior-sharded): NC0 starts at offset 0, NC1 at tile_size.
            # When batch-sharded, interleaved_fa_tiles=False so nc_start_offset=0.
            nc_start_offset = atp.sprior_prg_id * atp.fa_tile_s_prior if atp.interleaved_fa_tiles else 0
            dynamic_tile_offset_sbuf = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.memset(dynamic_tile_offset_sbuf, value=nc_start_offset)
            TC = TileConstants.get_tile_constants()
            dynamic_tile_offset_f32 = nl.ndarray((TC.p_max, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dynamic_tile_offset_f32, value=float(nc_start_offset))

            # Dynamic loop: tiles 0 through N-2 (all full-sized, is_last_fa_tile=False)
            for _ in nl.dynamic_range(0, num_non_last_tiles_reg):
                fa_ctx = FATileContext(
                    fa_tile_idx=0,  # placeholder, not used for branching in dynamic path
                    tile_s_prior=atp.fa_tile_s_prior,
                    tile_n_sprior=atp.fa_n_sprior_tile,
                    tile_offset=0,  # placeholder, dynamic_tile_offset_sbuf used instead
                    is_last_fa_tile=False,
                    dynamic_tile_offset_sbuf=dynamic_tile_offset_sbuf,
                    dynamic_tile_offset_f32=dynamic_tile_offset_f32,
                )

                sbm.open_scope()
                _execute_fa_tile_body(
                    active_blocks_table,
                    mask,
                    k_prior,
                    v_prior,
                    v_active,
                    out,
                    sink,
                    DBG_TENSORS,
                    atp,
                    cfg,
                    TC,
                    sbm,
                    bufs,
                    fa_ctx,
                    btc,
                )
                sbm.close_scope()

                # Increment tile offset (2 counters: int32 for DMA, float32 for iota)
                # Interleaved LNC2: stride by 2 tiles (skip the other NC's tile)
                tile_stride = atp.fa_tile_s_prior * atp.sprior_n_prgs
                nisa.tensor_scalar(
                    dst=dynamic_tile_offset_sbuf,
                    data=dynamic_tile_offset_sbuf,
                    op0=nl.add,
                    operand0=tile_stride,
                )
                nisa.tensor_scalar(
                    dst=dynamic_tile_offset_f32,
                    data=dynamic_tile_offset_f32,
                    op0=nl.add,
                    operand0=float(tile_stride),
                )

            # Static last tile
            last_fa_ctx = _compute_fa_tile_context(atp.num_fa_tiles - 1, atp, TC)
            last_fa_ctx.dynamic_tile_offset_sbuf = dynamic_tile_offset_sbuf
            last_fa_ctx.dynamic_tile_offset_f32 = dynamic_tile_offset_f32

            sbm.open_scope()
            _execute_fa_tile_body(
                active_blocks_table,
                mask,
                k_prior,
                v_prior,
                v_active,
                out,
                sink,
                DBG_TENSORS,
                atp,
                cfg,
                TC,
                sbm,
                bufs,
                last_fa_ctx,
                btc,
            )
            sbm.close_scope()
        else:
            # Original static FA loop
            for fa_tile_idx in range(atp.num_fa_tiles):
                fa_ctx = _compute_fa_tile_context(fa_tile_idx, atp, TC)

                sbm.open_scope()
                _execute_fa_tile_body(
                    active_blocks_table,
                    mask,
                    k_prior,
                    v_prior,
                    v_active,
                    out,
                    sink,
                    DBG_TENSORS,
                    atp,
                    cfg,
                    TC,
                    sbm,
                    bufs,
                    fa_ctx,
                    btc,
                )
                sbm.close_scope()

        # Final normalization and store for the online-softmax path (FA or sharded non-FA)
        if atp.use_online_softmax:
            _finalize_and_store(sink, out, atp, cfg, TC, sbm, bufs, btc, DBG_TENSORS=DBG_TENSORS)

        # Close scope for this batch tile's buffers
        sbm.close_scope()

    return out, k_out


def _execute_fa_tile_body(
    active_blocks_table,
    mask,
    k_prior,
    v_prior,
    v_active,
    out,
    sink,
    DBG_TENSORS,
    atp,
    cfg,
    TC,
    sbm,
    bufs,
    fa_ctx,
    btc,
):
    """Execute one FA tile iteration (the function calls inside the FA loop)."""
    # Load active blocks table for this FA tile and batch tile (block KV only)
    if atp.is_block_kv:
        _load_and_reshape_active_blk_table(active_blocks_table, atp, sbm, bufs, btc, fa_ctx)
    # Allocate QK and mask buffers
    _allocate_qk_buffers(atp, TC, sbm, bufs, fa_ctx)
    # Load mask for this FA tile
    _load_mask(mask, atp, cfg, TC, sbm, bufs, fa_ctx, btc)
    # Step 1. Matmult 1 of KQ^T (and optional K_prior transpose)
    _compute_qk_matmul(k_prior, DBG_TENSORS, atp, cfg, TC, sbm, bufs, fa_ctx, btc)
    # Step 2. Cascaded max reduce of KQ^T (includes FA running max update)
    _cascaded_max_reduce(sink, DBG_TENSORS, atp, cfg, TC, sbm, bufs, fa_ctx, btc)
    # Step 3. Exp(KQ^T - max(KQ^T))
    _compute_exp_qk(DBG_TENSORS, atp, cfg, TC, sbm, bufs, fa_ctx, btc)
    # Step 4. Cascaded sum reduction of exp
    _cascaded_sum_reduction(sink, DBG_TENSORS, atp, cfg, TC, sbm, bufs, fa_ctx, btc)
    # Step 5. Matmult 2 of (exp @ V)^T and store output
    _compute_pv_matmul_and_store(v_prior, v_active, out, atp, cfg, TC, sbm, bufs, fa_ctx, btc)


OOB_MODE_SKIP = nisa.oob_mode.skip  # FIXME: needs to be instantiated externally from kernel


@dataclass
class AttnTileParams(nl.NKIObject):
    """Computed tiling and dimension parameters for the attention kernel.

    This dataclass holds all the computed parameters needed for tiling the attention
    computation, including data types, sharding information, dimension calculations,
    and flash attention parameters.

    Fields are grouped into:
    - Global parameters: fixed for the entire kernel invocation
    - Per-batch-tile parameters: recomputed by _update_atp_for_batch_tile for each batch tile
    - Softmax reduction parameters
    """

    # ========== Global parameters (fixed for entire kernel invocation) ==========

    # Data types
    io_type = None
    """Data type for input/output tensors (e.g., bfloat16, float32). Derived from query tensor dtype."""

    inter_type = None
    """Data type for intermediate computations (typically float32 for numerical stability)."""

    k_prior_load_type = None
    """Data type used when loading k_prior. For FP8 KV cache, this is bfloat16; otherwise matches k_prior.dtype."""

    # Block KV cache flag
    is_block_kv: bool = None
    """Whether block KV cache is being used (True when active_blocks_table is provided)."""

    # KV FP8 Quantization Flag
    is_fp8_kv: bool = None
    """Whether FP8 quantization is used for KV cache tensors. When True, all KV
    tensors must have the same FP8 E4M3 dtype (``nl.float8_e4m3`` or
    ``nl.float8_e4m3fn``)."""

    kv_e4m3_tile_dtype: Any = None
    """Concrete FP8 E4M3 dtype for SBUF K/V tiles when ``is_fp8_kv`` is True.
    Uses caller's concrete ``k_prior.dtype`` when explicit; resolves from
    ``dtype_mode`` only when the caller passed the opaque ``"float8e4"``
    sentinel. ``None`` when ``is_fp8_kv`` is False."""

    # DMA transpose optimization flag
    use_dma_transpose: bool = None
    """Whether to use DMA transpose for block KV loading. True when d_head==128 and dtype is 2 bytes."""

    # Sharding parameters
    sprior_n_prgs: int = None
    """Number of programs (NeuronCores) that s_prior is sharded across. Either 1 or 2 for LNC2."""

    sprior_prg_id: int = None
    """Program ID for s_prior sharding (0 or 1). Each program handles AttnTileParams.s_prior // AttnTileParams.sprior_n_prgs elements."""

    bs_n_prgs: int = None
    """Number of programs (NeuronCores) that batch dimension is sharded across. Either 1 or 2 for LNC2."""

    bs_prg_id: int = None
    """Program ID for batch sharding (0 or 1). Each program handles AttnTileParams.bs // AttnTileParams.bs_n_prgs batches."""

    n_prgs: int = None
    """Total number of programs. Equals AttnTileParams.sprior_n_prgs * AttnTileParams.bs_n_prgs (either 1 or 2)."""

    # Batch size parameters
    bs_full: int = None
    """Full batch size before any sharding is applied. Equals AttnTKGConfig.bs."""

    bs_per_nc: int = None
    """Full batch size per NC before batch tiling. Equals AttnTileParams.bs_full // AttnTileParams.bs_n_prgs."""

    # Sequence and head dimensions (not batch-dependent)
    s_prior: int = None
    """Prior sequence length per program after sharding. Equals AttnTKGConfig.curr_sprior // AttnTileParams.sprior_n_prgs."""

    s_active_qh: int = None
    """Flattened dimension of [q_head, s_active]. Equals AttnTKGConfig.s_active * AttnTKGConfig.q_head."""

    n_sprior_tile: int = None
    """Total number of TileConstants.p_max-sized tiles across the post-sharded s_prior dimension. Equals ceil(AttnTileParams.s_prior / TileConstants.p_max)."""

    # Block KV cache parameters (if used)
    block_len: int = None
    """Block length for block KV cache. 0 for flat KV cache, or adjusted AttnTKGConfig.block_len after resizing."""

    blk_cache_resize_factor: int = None
    """Resize factor for block KV cache. Original block_len / resized block_len. 1 when no resize needed."""

    use_v_dma_skipping: bool = False
    """Whether to use DMA skipping for V load (block KV). DMA skipping currently doesn't help since we become gpsimd/compute bound
    in the majority of configurations, and it adds an extra memset. In addition, DMA batching does not currently support DMA skipping
    (uCode-407). In future we can flip this switch.
    """

    # Flash attention parameters
    use_fa: bool = None
    """Whether flash attention tiling is enabled. True when AttnTileParams.s_prior > FA_TILE_SIZE (8K)."""

    use_online_softmax: bool = None
    """Whether to use the online softmax running buffers (running_max/sum/output/correction_factor)
    and defer the cross-NC softmax sync to _finalize_and_store. True when AttnTileParams.use_fa is
    True (multi-tile accumulation needs running buffers) OR s_prior is sharded across NCs (running
    state must persist past the per-FA-tile scope so the cross-NC sync can run in
    _finalize_and_store). False for single-NC non-FA, where the classical per-tile softmax path
    is used."""

    num_fa_tiles: int = None
    """Number of flash attention tiles to iterate over. 1 if not using FA, otherwise ceil(AttnTileParams.s_prior / fa_tile_size)."""

    fa_tile_s_prior: int = None
    """Size of each flash attention tile in s_prior dimension. Equals FA_TILE_SIZE (8K) when FA enabled."""

    fa_n_sprior_tile: int = None
    """Number of TileConstants.p_max-sized tiles within each FA tile. Equals ceil(AttnTileParams.fa_tile_s_prior / TileConstants.p_max)."""

    interleaved_fa_tiles: bool = False
    """Whether FA tiles are interleaved across NCs for block KV DMA skipping load balancing.
    False for flat KV or batch sharding (uses sequential local offsets)."""

    # ========== Per-batch-tile parameters (recomputed by _update_atp_for_batch_tile) ==========

    bs: int = None
    """Batch size for the current batch tile. May be smaller than bs_per_nc for the last tile."""

    s_active_bqh: int = None
    """Flattened dimension of [bs, q_head, s_active] for the current batch tile. Equals AttnTileParams.bs * AttnTileParams.s_active_qh."""

    s_active_bqh_remainder: int = None
    """Remainder when AttnTileParams.s_active_bqh doesn't evenly divide into TileConstants.p_max tiles. Equals AttnTileParams.s_active_bqh % TileConstants.p_max."""

    n_bsq_full_tiles: int = None
    """Number of full TileConstants.p_max-sized tiles that fit in AttnTileParams.s_active_bqh. Equals AttnTileParams.s_active_bqh // TileConstants.p_max."""

    n_bsq_tiles: int = None
    """Total number of tiles needed for AttnTileParams.s_active_bqh (including partial). Equals AttnTileParams.n_bsq_full_tiles + (1 if remainder > 0)."""

    s_active_bqh_tile: int = None
    """Size of each BSQ tile. Equals TileConstants.p_max if multiple tiles needed, otherwise AttnTileParams.s_active_bqh."""

    batch_interleave_degree: int = None
    """Degree of interleaving across batches for PSUM bank utilization. Min of AttnTileParams.bs and TileConstants.psum_b_max (8)."""

    # Softmax reduction parameters ==========

    num_folds_per_batch: int = None
    """Number of 128-block folds for the current FA tile in block KV cache. Set by _load_and_reshape_active_blk_table.
    Equals (blocks_per_batch * resize_factor) / TileConstants.p_max."""

    softmax_final_reduction_length: int = None
    """Number of elements in final softmax reduction = 1 (local) + (sink slot if per-tile softmax sync
    is used — see AttnTileParams.sync_softmax_per_fa_tile). Sharded cases defer the cross-NC sync to
    _finalize_and_store; sink is loaded locally inside _finalize_and_store in that case."""

    softmax_final_reduction_local_idx: int = None
    """Index in reduction buffer for local NC's result. Always 0."""

    softmax_final_reduction_sink_idx: int = None
    """Index in reduction buffer for sink contribution. None unless sink is staged into qk_max_buf for
    per-tile softmax sync; else AttnTileParams.softmax_final_reduction_length - 1."""

    sync_softmax_per_fa_tile: bool = None
    """Whether to synchronize softmax statistics (max/sum) across NeuronCores after each FA tile.
    True when not sharded on s_prior. When False (sharded), the cross-NC sync is deferred to
    _finalize_and_store so sendrecv does not block GPSIMD V prefetches.
    """

    max_negated: bool = None
    """Whether the max values are stored negated (for exp computation optimization with sink)."""


@dataclass
class AttnInternalBuffers(nl.NKIObject):
    """Internal SBUF buffers needed across multiple steps of the attention kernel.

    This dataclass holds all temporary SBUF buffers that are allocated during
    kernel execution and shared across different computation steps.
    """

    # Core attention tensors
    qk: nl.ndarray = None
    """QK^T result buffer. Shape [TileConstants.p_max, FATileContext.tile_n_sprior * AttnTileParams.s_active_bqh]. Filled with -inf initially for masking."""

    qk_io_type: nl.ndarray = None
    """QK buffer in AttnTileParams.io_type (e.g., bfloat16) for matmuls. Same shape as qk, stores exp(QK - max) after softmax."""

    qk_max: nl.ndarray = None
    """Per-position max of QK^T for softmax stability. Shape [TileConstants.p_max, AttnTileParams.s_active_bqh]."""

    qk_max_buf: nl.ndarray = None
    """Buffer for max reduction across tiles and LNC2 cores. Shape [AttnTileParams.s_active_bqh_tile, AttnTileParams.n_bsq_tiles * AttnTileParams.softmax_final_reduction_length]."""

    exp_sum: nl.ndarray = None
    """Sum of exp(QK - max) for softmax normalization. Shape [AttnTileParams.s_active_bqh_tile, AttnTileParams.n_bsq_tiles * AttnTileParams.softmax_final_reduction_length]."""

    exp_sum_recip: nl.ndarray = None
    """Reciprocal of exp_sum, broadcasted for final normalization. Shape [TileConstants.p_max, AttnTileParams.s_active_bqh]. Not used when FA enabled."""

    exp_v: nl.ndarray = None
    """Result of softmax(QK) @ V matmul. Shape [AttnTKGConfig.d_head, AttnTileParams.bs, AttnTileParams.s_active_qh]. Contains unnormalized output for FA."""

    # Preprocessed inputs
    q_sb: nl.ndarray = None
    """Query tensor in SBUF after optional RoPE. Shape [AttnTKGConfig.d_head, AttnTileParams.bs_full * AttnTileParams.s_active_qh]. Scaled by 1/sqrt(AttnTKGConfig.d_head) if fuse_rope."""

    k_active_sb: nl.ndarray = None
    """Active key tensor in SBUF after optional RoPE. Shape [AttnTKGConfig.d_head, AttnTileParams.bs_full * AttnTKGConfig.s_active]."""

    pos_ids_sb: nl.ndarray = None
    """Position IDs broadcasted to all partitions for RoPE/mask generation. Shape [TileConstants.p_max, AttnTileParams.bs_per_nc * AttnTKGConfig.s_active]."""

    start_pos_sb: nl.ndarray = None
    """Per-query SWA window start positions broadcasted to all partitions. Shape [TileConstants.p_max, AttnTileParams.bs * AttnTKGConfig.s_active]. None when SWA is disabled."""

    mask_sb: nl.ndarray = None
    """Attention mask in SBUF. Shape [TileConstants.p_max, FATileContext.tile_n_sprior * AttnTileParams.s_active_bqh]. Values: 1 for valid, 0 for masked."""

    one_vec: nl.ndarray = None
    """Vector of ones for sum reduction via matmul. Shape [TileConstants.p_max, 1]."""

    # Block KV cache buffers
    active_blocks_sb: nl.ndarray = None
    """Active block indices in SBUF for block KV cache. Shape [TileConstants.p_max, AttnTileParams.num_folds_per_batch * AttnTileParams.bs].
    Loaded per FA tile and batch tile by _load_and_reshape_active_blk_table. Contains block indices."""

    active_blocks_sb_u32: nl.ndarray = None
    """Pre-cast uint32 copy of active_blocks_sb for DMA transpose path. Avoids per-fold int32→uint32 cast in hot loop."""

    v_active_reshaped: nl.ndarray = None
    """Reshaped v_active for block KV loading. Shape [AttnTKGConfig.bs, AttnTKGConfig.s_active * AttnTKGConfig.d_head]."""

    k_prior_reshaped: nl.ndarray = None
    """Reshaped k_prior cache for block-sparse access.
    Shape [num_blocks * resize_factor, block_len * d_head] (or [num_blocks * resize_factor, block_len//2 * d_head * 2] fp8 when fp8_packed)."""

    v_prior_reshaped: nl.ndarray = None
    """Reshaped v_prior cache for block-sparse access. Shape [num_blocks * resize_factor, AttnTileParams.block_len * AttnTKGConfig.d_head]."""

    # Debug tensors (reshaped from DBG_TENSORS)
    DBG_QK: nl.ndarray = None
    """Debug tensor for QK^T results. Shape [TileConstants.p_max, AttnTileParams.sprior_n_prgs, AttnTileParams.n_sprior_tile, AttnTileParams.bs_n_prgs, AttnTileParams.s_active_bqh]."""

    DBG_QK_MAX: nl.ndarray = None
    """Debug tensor for QK max values. Shape [AttnTileParams.bs_n_prgs, AttnTileParams.n_bsq_tiles, AttnTileParams.s_active_bqh_tile]."""

    DBG_QK_EXP: nl.ndarray = None
    """Debug tensor for exp(QK - max). Shape [TileConstants.p_max, AttnTileParams.sprior_n_prgs, AttnTileParams.n_sprior_tile, AttnTileParams.bs_n_prgs, AttnTileParams.s_active_bqh]."""

    DBG_EXP_SUM: nl.ndarray = None
    """Debug tensor for exp sum values. Shape [AttnTileParams.bs_n_prgs, AttnTileParams.n_bsq_tiles, AttnTileParams.s_active_bqh_tile]."""

    DBG_ACTIVE_TABLE: nl.ndarray = None
    """Debug tensor for active blocks table (block KV only). Shape [TileConstants.p_max, AttnTileParams.num_folds_per_batch * AttnTileParams.sprior_n_prgs, AttnTileParams.bs_full]."""

    # Flash attention buffers
    running_max: nl.ndarray = None
    """Running max across FA tiles for online softmax. Shape [AttnTileParams.s_active_bqh_tile, AttnTileParams.n_bsq_tiles]. Updated each FA tile."""

    running_sum: nl.ndarray = None
    """Running sum of exp values across FA tiles. Shape [AttnTileParams.s_active_bqh_tile, AttnTileParams.n_bsq_tiles]. Accumulated each FA tile."""

    correction_factor: nl.ndarray = None
    """Correction factor exp(prev_max - curr_max) for rescaling. Shape [AttnTileParams.s_active_bqh_tile, AttnTileParams.n_bsq_tiles]."""

    running_output: nl.ndarray = None
    """Running accumulated PV output across FA tiles. Shape [AttnTKGConfig.d_head, AttnTileParams.s_active_bqh]. Normalized at end."""


@dataclass
class FATileContext(nl.NKIObject):
    """Context for the current FA tile being processed.

    This holds tile-specific parameters that vary per FA tile iteration.
    Functions inside the FA loop should use these values instead of
    AttnTileParams.fa_tile_s_prior / AttnTileParams.fa_n_sprior_tile which are max values.

    Flash attention tiles process s_prior in chunks of fa_tile_size (8K).
    The last tile may be smaller than fa_tile_size if s_prior is not evenly divisible.
    """

    fa_tile_idx: int
    """Which FA tile is being processed (0-indexed). Ranges from 0 to AttnTileParams.num_fa_tiles - 1."""

    tile_s_prior: int
    """Actual s_prior for this tile. Equals fa_tile_size (8K) for all but last tile; last tile may be smaller."""

    tile_n_sprior: int
    """Number of TileConstants.p_max-sized tiles within this FA tile. Equals ceil(FATileContext.tile_s_prior / TileConstants.p_max)."""

    tile_offset: int
    """Global s_prior offset where this tile starts. Includes NC base offset."""

    is_last_fa_tile: bool
    """True if this is the final FA tile. Used to determine when to load k_active/v_active and finalize output."""

    dynamic_tile_offset_sbuf: nl.ndarray = None
    """SBUF (1,1) int32 scalar holding the dynamic tile offset (fa_tile_idx * fa_tile_s_prior).
    Non-None only in the dynamic FA loop path. When set, functions should use this
    instead of tile_offset for runtime-dependent offset computations."""

    dynamic_tile_offset_f32: nl.ndarray = None
    """SBUF (P_MAX, 1) float32 version of dynamic_tile_offset_sbuf for gen_mask_tkg iota bias."""


def _compute_fa_tile_context(fa_tile_idx: int, atp: AttnTileParams, TC: TileConstants) -> FATileContext:
    """Compute the context for a specific FA tile.

    For block KV, tile_offset is always a global s_prior offset (works for both interleaved
    and contiguous sharding). For flat KV, tile_offset is local to the NC's portion.
    """
    is_last_fa_tile = fa_tile_idx == atp.num_fa_tiles - 1

    # Compute tile offset (always global)
    if atp.interleaved_fa_tiles:
        tile_offset = _get_interleaved_fa_tile_offset(fa_tile_idx, atp)
    else:
        tile_offset = atp.sprior_prg_id * atp.s_prior + fa_tile_idx * atp.fa_tile_s_prior

    # Compute tile size (last tile may be smaller)
    if is_last_fa_tile and atp.use_fa:
        tile_s_prior = atp.s_prior - fa_tile_idx * atp.fa_tile_s_prior
        tile_s_prior = min(atp.fa_tile_s_prior, tile_s_prior)
        tile_n_sprior = div_ceil(tile_s_prior, TC.p_max)
    else:
        tile_s_prior = atp.fa_tile_s_prior
        tile_n_sprior = atp.fa_n_sprior_tile

    return FATileContext(
        fa_tile_idx=fa_tile_idx,
        tile_s_prior=tile_s_prior,
        tile_n_sprior=tile_n_sprior,
        tile_offset=tile_offset,
        is_last_fa_tile=is_last_fa_tile,
    )


@dataclass
class BatchTileContext(nl.NKIObject):
    """Context for the current batch tile being processed.

    When the full per-NC batch size is too large to fit in SBUF, the batch dimension
    is tiled. This dataclass tracks the current batch tile's parameters.
    """

    batch_tile_idx: int
    """Which batch tile is being processed (0-indexed)."""

    tile_bs: int
    """Batch size for this tile. May be smaller than batch_tile_size for the last tile."""

    tile_batch_offset: int
    """Offset within the NC's batch portion where this tile starts. Equals batch_tile_idx * batch_tile_size."""

    global_batch_offset: int
    """Offset into the full (all-NC) batch dimension. Equals bs_prg_id * bs_per_nc + tile_batch_offset.
    Use as: global_batch_offset + i_b to index into full-batch tensors (q_sb, k_prior, v_prior, etc.)."""


def _update_atp_for_batch_tile(atp: AttnTileParams, tile_bs: int, TC: TileConstants):
    """Recompute batch-dependent atp fields for a given batch tile size.

    These fields depend on the current batch tile's bs and must be recomputed
    each time the batch tile changes. When num_batch_tiles == 1, tile_bs == bs_per_nc
    and these are equivalent to the original (pre-tiling) values.
    """
    atp.bs = tile_bs
    atp.s_active_bqh = atp.bs * atp.s_active_qh  # flattened dim of [bs, q_heads, s_active] for this batch tile
    atp.s_active_bqh_remainder = atp.s_active_bqh % TC.p_max
    atp.n_bsq_full_tiles = atp.s_active_bqh // TC.p_max
    atp.n_bsq_tiles = atp.n_bsq_full_tiles + (atp.s_active_bqh_remainder > 0)
    atp.s_active_bqh_tile = TC.p_max if atp.n_bsq_tiles > 1 else atp.s_active_bqh
    atp.batch_interleave_degree = min(atp.bs, TC.psum_b_max)  # PSUM bank interleaving across batches


"""
Initialization functions
"""


def _compute_tile_params(
    cfg: AttnTKGConfig,
    TC: TileConstants,
    q,
    k_prior,
    v_prior,
    k_active,
    v_active,
    active_blocks_table,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
) -> AttnTileParams:
    """Compute tiling and dimension parameters from configuration."""
    atp = AttnTileParams()

    atp.is_block_kv = active_blocks_table is not None
    atp.block_len = 0  # Default for flat KV cache; overwritten in _setup_block_kv_cache for block KV

    # Determine FP8 KV status
    k_prior_fp8 = is_fp8_e4m3(k_prior.dtype)
    v_prior_fp8 = is_fp8_e4m3(v_prior.dtype)
    k_active_fp8 = is_fp8_e4m3(k_active.dtype)
    v_active_fp8 = is_fp8_e4m3(v_active.dtype)
    any_fp8 = k_prior_fp8 or v_prior_fp8 or k_active_fp8 or v_active_fp8
    all_fp8 = k_prior_fp8 and v_prior_fp8 and k_active_fp8 and v_active_fp8
    atp.is_fp8_kv = all_fp8

    kernel_assert(
        not cfg.fp8_packed or all_fp8,
        f"fp8_packed requires all KV tensors to be FP8. Got k_prior_fp8={k_prior_fp8}, "
        f"v_prior_fp8={v_prior_fp8}, k_active_fp8={k_active_fp8}, v_active_fp8={v_active_fp8}.",
    )

    # Resolve FP8 E4M3 dtype for SBUF K/V tile allocations: use caller's
    # concrete k_prior.dtype when explicit; resolve from dtype_mode only when
    # the caller passed the opaque "float8e4" sentinel.
    if all_fp8:
        if str(k_prior.dtype) == "float8e4":
            atp.kv_e4m3_tile_dtype = resolve_fp8_e4m3_dtype(dtype_mode)
        else:
            atp.kv_e4m3_tile_dtype = k_prior.dtype
    else:
        atp.kv_e4m3_tile_dtype = None

    # ========== Input validation (kernel asserts) ==========
    # Basic shape constraints
    kernel_assert(
        0 < cfg.bs,
        f"Batch size must be strictly positive, got bs={cfg.bs}.",
    )
    kernel_assert(
        0 < cfg.q_head,
        f"Number of Q heads must be strictly positive, got q_head={cfg.q_head}.",
    )
    kernel_assert(
        0 < cfg.s_active,
        f"Number of decode tokens must be strictly positive, got s_active={cfg.s_active}.",
    )
    kernel_assert(
        0 < cfg.curr_sprior <= cfg.full_sprior,
        f"curr_sprior must be <= full_sprior. Got curr_sprior={cfg.curr_sprior}, full_sprior={cfg.full_sprior}.",
    )
    kernel_assert(
        0 < cfg.d_head <= _MAX_D_HEAD,
        f"Unsupported d_head. Got d_head={cfg.d_head}, must be between 1 and {_MAX_D_HEAD}, inclusive.",
    )

    # FP8 dtype validation
    kernel_assert(
        not any_fp8 or all_fp8,
        f"FP8 KV cache requires all KV tensors to have the same FP8 E4M3 dtype "
        f"(nl.float8_e4m3 or nl.float8_e4m3fn). "
        f"Got k_prior.dtype={k_prior.dtype}, v_prior.dtype={v_prior.dtype}, "
        f"k_active.dtype={k_active.dtype}, v_active.dtype={v_active.dtype}.",
    )
    kernel_assert(
        not is_fp8_e5m2(k_prior.dtype)
        and not is_fp8_e5m2(v_prior.dtype)
        and not is_fp8_e5m2(k_active.dtype)
        and not is_fp8_e5m2(v_active.dtype),
        f"nl.float8_e5m2 is not supported for KV tensors. "
        f"Got k_prior.dtype={k_prior.dtype}, v_prior.dtype={v_prior.dtype}, "
        f"k_active.dtype={k_active.dtype}, v_active.dtype={v_active.dtype}.",
    )

    # FP8 KV configuration constraints
    kernel_assert(
        not atp.is_fp8_kv or not cfg.fuse_rope,
        f"fuse_rope must be False when using FP8 KV cache. Got fuse_rope={cfg.fuse_rope}.",
    )
    kernel_assert(
        not atp.is_fp8_kv or q.dtype != nl.float32,
        f"float32 query dtype is not supported with FP8 KV cache. Got q.dtype={q.dtype}. Use nl.bfloat16 instead.",
    )
    kernel_assert(
        not atp.is_fp8_kv or cfg.qk_in_sb,
        f"qk_in_sb must be True when using FP8 KV cache. Got qk_in_sb={cfg.qk_in_sb}.",
    )

    # assign to object (can't directly because of NKI limitation)
    sprior_n_prgs, sprior_prg_id, bs_n_prgs, bs_prg_id = _get_lnc_sharding(cfg)
    atp.sprior_n_prgs = sprior_n_prgs
    atp.sprior_prg_id = sprior_prg_id
    atp.bs_n_prgs = bs_n_prgs
    atp.bs_prg_id = bs_prg_id
    atp.n_prgs = atp.sprior_n_prgs * atp.bs_n_prgs

    # Get shapes and dtypes
    atp.k_prior_load_type = nl.bfloat16 if atp.is_fp8_kv else k_prior.dtype
    atp.use_dma_transpose = (sizeinbytes(atp.k_prior_load_type) == 2) and (
        not atp.is_fp8_kv or cfg.fp8_packed
    )  # use_dma_transpose may be further disabled in _setup_block_kv_cache for small block_len configs
    atp.io_type = q.dtype
    atp.inter_type = nl.float32
    atp.bs_full = cfg.bs
    atp.bs_per_nc = atp.bs_full // atp.bs_n_prgs  # full per-NC batch size before batch tiling
    atp.s_prior = cfg.curr_sprior // atp.sprior_n_prgs  # shard prior seqlen onto each prg
    atp.s_active_qh = cfg.s_active * cfg.q_head  # flattened dim of [q_heads, s_active]
    atp.n_sprior_tile = div_ceil(atp.s_prior, TC.p_max)  # total number of p_max-tiles across full s_prior

    # ========== Derived parameter validation ==========
    kernel_assert(
        atp.s_prior % TC.p_max == 0,
        f"Sharded s_prior must be divisible by p_max. Got sharded s_prior={atp.s_prior}, p_max={TC.p_max}.",
    )

    kernel_assert(
        not cfg.fuse_rope or atp.bs_n_prgs == 1,
        f"Fuse rope requires batch to not be sharded. See `is_batch_sharded`.",
    )
    kernel_assert(
        not cfg.fuse_rope or cfg.bs * cfg.q_head * cfg.s_active <= TC.p_max,
        f"Fuse rope requires batch * q_head * s_active to be fit on the partition dimension, got {cfg.bs * atp.s_active_qh}.",
    )

    # Flash attention parameters
    # Enable FA when s_prior exceeds the tile size threshold
    use_fa, fa_tile_size = uses_flash_attention(cfg.enable_fa_s_prior_tiling, atp.s_prior)
    atp.use_fa = use_fa
    if atp.use_fa:
        atp.num_fa_tiles = div_ceil(atp.s_prior, fa_tile_size)
        atp.fa_tile_s_prior = fa_tile_size
        atp.fa_n_sprior_tile = div_ceil(fa_tile_size, TC.p_max)
        # Last FA tile must be able to hold s_active (k_active is loaded at tile end)
        last_tile_s_prior = atp.s_prior % fa_tile_size if atp.s_prior % fa_tile_size != 0 else fa_tile_size
        kernel_assert(
            last_tile_s_prior >= cfg.s_active,
            f"Last FA tile size ({last_tile_s_prior}) must be >= s_active ({cfg.s_active})",
        )
    else:
        atp.num_fa_tiles = 1
        atp.fa_tile_s_prior = atp.s_prior
        atp.fa_n_sprior_tile = atp.n_sprior_tile

    # For block KV with LNC2 s_prior sharding, enable interleaved FA tile assignment.
    # Instead of each NC owning a contiguous half of s_prior, tiles alternate between NCs
    # for better load balancing when cache is partially filled (DMA skipping).
    if atp.is_block_kv and atp.sprior_n_prgs == 2:
        atp.interleaved_fa_tiles = True

    # Online softmax (running max/sum/output buffers) is used whenever:
    #   - FA tiling is active (running state must persist across FA tiles), OR
    #   - s_prior is sharded across NCs (the running state carries local values past the per-FA-tile
    #     scope into _finalize_and_store, where the cross-NC sync runs).
    # Only the classical single-NC non-FA path skips them.
    atp.use_online_softmax = atp.use_fa or atp.sprior_n_prgs > 1

    # Whether softmax sync is complete within the _cascaded_* path for this tile.
    #   - single-NC: True on the (only) tile — there is nothing cross-NC to do.
    #   - sharded (FA or non-FA): False — cross-NC sync happens in _finalize_and_store via the
    #     running buffers.
    atp.sync_softmax_per_fa_tile = atp.sprior_n_prgs == 1

    return atp


def _setup_block_kv_cache(
    k_prior,
    v_prior,
    k_active,
    v_active,
    active_blocks_table,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """
    Setup block KV cache by validating shapes and reshaping tensors for block-sparse access.

    Validates that k_prior and v_prior have correct block_len and d_head dimensions, then reshapes
    v_active and cache tensors to enable efficient block-sparse memory access. When blocks per batch
    is less than 128, resizes block_len to make blocks per batch a multiple of 128 for optimal performance.
    Loads and reshapes the active blocks table to track which cache blocks are active per batch.
    """
    # Active blocks table is int32 with INACTIVE_BLOCK_IDX (-1) for invalid/padding blocks.
    # When use_dma_transpose is enabled, we convert to uint32
    # When use_dma_transpose is disabled, int32 indices with -1 enable oob_mode.skip in dma_copy.
    if active_blocks_table.dtype != nl.int32:
        print(
            f"WARNING: active_blocks_table dtype is {active_blocks_table.dtype}, not int32. "
            f"Use int32 with -1 for OOB indices to take advantage of DMA skipping."
        )
    # Check shapes
    kernel_assert(not cfg.strided_mm1, f"Block KV requires MM1 to not be strided.")
    kernel_assert(cfg.tp_k_prior, f"Block KV requires k_prior to not be transposed.")
    if cfg.fp8_packed:
        kernel_assert(
            cfg.block_len % 2 == 0,
            f"fp8_packed requires block_len to be even, got {cfg.block_len}.",
        )
        kernel_assert(
            k_prior.shape == (v_prior.shape[0], cfg.block_len // 2, cfg.d_head, 2),
            f"Block KV fp8_packed requires k_prior shape (*, {cfg.block_len // 2}, {cfg.d_head}, 2), "
            f"got {k_prior.shape}.",
        )
    else:
        kernel_assert(
            k_prior.shape[1] == cfg.block_len,
            f"Block KV requires k_prior input must be reshaped to have block_len as the second dimension, expected k_prior.shape[1]={cfg.block_len}, got {k_prior.shape[1]}.",
        )
        kernel_assert(
            k_prior.shape[2] == cfg.d_head,
            f"Block KV requires k_prior input must be reshaped to have d_head as the third dimension, expected k_prior.shape[2]={cfg.d_head}, got {k_prior.shape[2]}.",
        )
    kernel_assert(
        v_prior.shape[1:] == (cfg.block_len, cfg.d_head),
        f"Block KV requires v_prior shape (*, {cfg.block_len}, {cfg.d_head}), got v_prior.shape={v_prior.shape}.",
    )
    if cfg.fp8_packed:
        kernel_assert(
            k_prior.shape[0] == v_prior.shape[0],
            f"Block KV requires k_prior and v_prior to have the same number of blocks, "
            f"got k_prior.shape[0]={k_prior.shape[0]}, v_prior.shape[0]={v_prior.shape[0]}.",
        )
    else:
        kernel_assert(
            k_prior.shape == v_prior.shape,
            f"Block KV requires k_prior and v_prior shapes to match, got {k_prior.shape=}, {v_prior.shape=}",
        )
    kernel_assert(
        cfg.qk_in_sb,
        "Block KV loading from k_active is currently only supported when qk is in SBUF (qk_in_sb==True)",
    )
    kernel_assert(
        k_active.shape == (cfg.d_head, cfg.bs * cfg.s_active),
        f"Block KV requires k_active has the shape (d_head, bs * s_active), expected {(cfg.d_head, cfg.bs * cfg.s_active)}, got {k_active.shape}.",
    )  # This is equivalent to qk_in_sb, but just in case
    kernel_assert(
        active_blocks_table.shape[0] == cfg.bs,
        f"Block KV requires active_blocks_table has the shape (bs, num_blocks_per_batch), expected active_blocks_table.shape[0]={cfg.bs}, got {active_blocks_table.shape[0]}",
    )
    kernel_assert(
        active_blocks_table.shape[1] * cfg.block_len == cfg.curr_sprior,
        f"Block KV requires the number of blocks per batch times the number of blocks to match the current context length, expected active_blocks_table.shape[1] * cfg.block_len={cfg.curr_sprior}, got {active_blocks_table.shape[1] * cfg.block_len}",
    )

    # Reshape before performing modifications on the dimensions
    bufs.v_active_reshaped = v_active.reshape((cfg.bs, cfg.s_active * cfg.d_head))

    # For block cache support, the kernel requires the number of blocks per batch to be a multiple of 128.
    # When S_ctx is small and blocks per batch < 128, we will "resize" blocks to make blocks per batch a multiple of 128.
    n_prgs = nl.num_programs(0)
    block_len, blk_cache_resize_factor = resize_cache_block_len_for_attention_tkg_kernel(
        num_blocks_per_batch=active_blocks_table.shape[1],
        block_len=cfg.block_len,
        lnc=n_prgs,
        p_max=TC.p_max,
        bs=cfg.bs,
        q_head=cfg.q_head,
        s_active=cfg.s_active,
        full_sprior=cfg.full_sprior,
        enable_fa_s_prior_tiling=cfg.enable_fa_s_prior_tiling,
        fuse_rope=cfg.fuse_rope,
    )

    # fp8_packed: resized block_len must be >= 2 to keep packed pairs intact
    kernel_assert(
        not cfg.fp8_packed or block_len >= 2,
        f"fp8_packed requires resized block_len >= 2, got {block_len}. "
        f"Increase S_ctx so that blocks_per_batch >= 128 without shrinking block_len below 2.",
    )

    # Disable dma_transpose when block_len is very small, or block_len, d_head, and batches per core are all small on LNC2.
    # For these configs the DMA transpose overhead outweighs the benefits because there are too few batches to pipeline DMA bursts
    # and small block_len leads to too many folds per batch.
    # fp8_packed always uses DMA transpose (no PE transpose fallback for packed layout).
    if not cfg.fp8_packed:
        if (block_len < 8) or (block_len <= 16 and cfg.d_head <= 16 and atp.bs < 4 and atp.n_prgs == 2):
            atp.use_dma_transpose = False

    # assign to atp (can't do directly because function return value cannot be assigned to object)
    atp.block_len = block_len
    atp.blk_cache_resize_factor = blk_cache_resize_factor

    k_new_cache_shape = (k_prior.shape[0] * blk_cache_resize_factor, atp.block_len * cfg.d_head)
    bufs.k_prior_reshaped = k_prior.reshape(k_new_cache_shape)

    v_new_cache_shape = (
        v_prior.shape[0] * blk_cache_resize_factor,
        atp.block_len * cfg.d_head,
    )
    bufs.v_prior_reshaped = v_prior.reshape(v_new_cache_shape)

    # if using flash attention verify the fa tile size is divisible by atp.block_len * TC.p_max
    # since that is assumed during KV load. Last tile can be smaller. Note that the current resize
    # logic doesn't account for flash attention so it is possible below assertion breaks when the
    # flash attention tile size is too small.
    if atp.use_fa:
        kernel_assert(
            atp.fa_tile_s_prior % (atp.block_len * TC.p_max) == 0,
            f"Block KV requires the Flash attention tile size to be divisible by product of resized block len and max partitions, got {atp.fa_tile_s_prior=}, {atp.block_len=}, {TC.p_max=}",
        )
        # check last tile is also divisible
        if atp.s_prior % atp.fa_tile_s_prior != 0:
            last_tile_s_prior = atp.s_prior % atp.fa_tile_s_prior
            kernel_assert(
                last_tile_s_prior % (atp.block_len * TC.p_max) == 0,
                f"Block KV requires the Flash attention tile size to be divisible by product of resized block len and max partitions, got {last_tile_s_prior=}, {atp.block_len=}, {TC.p_max=}",
            )


def _setup_debug_tensors(DBG_TENSORS, atp: AttnTileParams, TC: TileConstants, bufs: AttnInternalBuffers):
    """Setup debug tensor references."""
    kernel_assert(
        len(DBG_TENSORS) == 4 + (1 if atp.is_block_kv else 0),
        f"Received {len(DBG_TENSORS)} debug tensors, when 4 are expected (or 5 if block KV is used)",
    )
    # Intermediate values for debugging.
    bufs.DBG_QK = DBG_TENSORS[0].reshape(
        (
            TC.p_max,
            atp.sprior_n_prgs,
            atp.n_sprior_tile,
            atp.bs_n_prgs,
            atp.s_active_bqh,
        )
    )
    bufs.DBG_QK_MAX = DBG_TENSORS[1].reshape((atp.bs_n_prgs, atp.n_bsq_tiles, atp.s_active_bqh_tile))
    bufs.DBG_QK_EXP = DBG_TENSORS[2].reshape(
        (
            TC.p_max,
            atp.sprior_n_prgs,
            atp.n_sprior_tile,
            atp.bs_n_prgs,
            atp.s_active_bqh,
        )
    )
    bufs.DBG_EXP_SUM = DBG_TENSORS[3].reshape((atp.bs_n_prgs, atp.n_bsq_tiles, atp.s_active_bqh_tile))
    if atp.is_block_kv:
        bufs.DBG_ACTIVE_TABLE = DBG_TENSORS[4]
        # DBG_ACTIVE_TABLE shape validation — compute full num_folds_per_batch from atp fields
        full_num_folds_per_batch = atp.s_prior // (atp.block_len * TC.p_max)
        kernel_assert(
            bufs.DBG_ACTIVE_TABLE.shape[1] == full_num_folds_per_batch * atp.sprior_n_prgs,
            "Active table debug tensor second dimension incorrect (needs to have shape (P_MAX, curr_sprior // block_len, batch_size)), "
            f"expected DBG_ACTIVE_TABLE.shape[1]={full_num_folds_per_batch * atp.sprior_n_prgs}, got {bufs.DBG_ACTIVE_TABLE.shape[1]}",
        )
        kernel_assert(
            bufs.DBG_ACTIVE_TABLE.shape[2] == atp.bs_full,
            "Active table debug tensor third dimension incorrect (needs to have shape (P_MAX, curr_sprior // block_len, batch_size))"
            f"expected DBG_ACTIVE_TABLE.shape[2]={atp.bs_full}, got {bufs.DBG_ACTIVE_TABLE.shape[2]}",
        )
        # Note: DBG_ACTIVE_TABLE store is done incrementally inside _load_and_reshape_active_blk_table


def _store_dbg_qk_max(
    src: nl.ndarray,
    max_is_negated: bool,
    name_suffix: str,
    atp: AttnTileParams,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """Transpose a max tensor with shape [s_active_bqh_tile, n_bsq_tiles] into a
    [n_bsq_tiles, s_active_bqh_tile] block in DBG_QK_MAX, un-negating if needed.

    Only writes the current NC's slice. Also pads the remainder columns with zeros when
    s_active_bqh is not a multiple of p_max. Caller must ensure atp.bs == atp.bs_per_nc
    (i.e. no batch tiling) because the offset-based write assumes full-batch layout.

    Args:
      src: Source tensor with shape [s_active_bqh_tile, n_bsq_tiles].
      max_is_negated: Whether `src` holds negated max values (requires multiply by -1 on dump).
      name_suffix: Unique suffix for the DMA ops' names.
    """
    sbm.open_scope()
    qk_max_dbg_psum = nl.ndarray(
        (atp.n_bsq_tiles, atp.s_active_bqh_tile),
        dtype=src.dtype,
        buffer=nl.psum,
        address=None if sbm.is_auto_alloc() else (0, 0),
    )
    qk_max_dbg = sbm.alloc_stack((atp.n_bsq_tiles, atp.s_active_bqh_tile), dtype=src.dtype)
    nisa.nc_transpose(qk_max_dbg_psum, src[: atp.s_active_bqh_tile, : atp.n_bsq_tiles])
    if max_is_negated:
        nisa.tensor_copy(qk_max_dbg, qk_max_dbg_psum)
    else:
        # Multiply by -1 so DBG_QK_MAX always stores negated values (consistent with the
        # legacy per-tile classical path, which stored from the negated qk_max_buf).
        nisa.tensor_scalar(qk_max_dbg, qk_max_dbg_psum, op0=nl.multiply, operand0=-1)

    dbg_qk_max_view = TensorView(bufs.DBG_QK_MAX).select(0, atp.bs_prg_id)
    nisa.dma_copy(
        dbg_qk_max_view.get_view(),
        qk_max_dbg,
        dge_mode=dge_mode.none,
        name=f"dbg_qk_max_store_{name_suffix}",
    )

    # Pad remainder-tile region with zeros (the last BSQ tile is smaller when s_active_bqh
    # is not a multiple of p_max; the remainder columns need defined values).
    if atp.n_bsq_full_tiles > 0 and atp.s_active_bqh_remainder > 0:
        zeros = sbm.alloc_stack((1, atp.s_active_bqh_tile - atp.s_active_bqh_remainder), dtype=src.dtype)
        nisa.memset(zeros, 0)
        nisa.dma_copy(
            dbg_qk_max_view.select(0, atp.n_bsq_full_tiles)
            .expand_dim(0)
            .slice(1, atp.s_active_bqh_remainder, atp.s_active_bqh_tile)
            .get_view(),
            zeros,
            dge_mode=dge_mode.none,
            name=f"dbg_qk_max_store_zeros_{name_suffix}",
        )
    sbm.close_scope()


def _store_dbg_exp_sum(
    src: nl.ndarray,
    name_suffix: str,
    atp: AttnTileParams,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """Transpose a sum tensor with shape [s_active_bqh_tile, n_bsq_tiles] into a
    [n_bsq_tiles, s_active_bqh_tile] block in DBG_EXP_SUM. See _store_dbg_qk_max for the
    full-batch / batch-tiling contract."""
    sbm.open_scope()
    exp_sum_dbg_psum = nl.ndarray(
        (atp.n_bsq_tiles, atp.s_active_bqh_tile),
        dtype=src.dtype,
        buffer=nl.psum,
        address=None if sbm.is_auto_alloc() else (0, 0),
    )
    exp_sum_dbg = sbm.alloc_stack((atp.n_bsq_tiles, atp.s_active_bqh_tile), dtype=src.dtype)
    nisa.nc_transpose(exp_sum_dbg_psum, src[: atp.s_active_bqh_tile, : atp.n_bsq_tiles])
    nisa.tensor_copy(exp_sum_dbg, exp_sum_dbg_psum)

    dbg_exp_sum_view = TensorView(bufs.DBG_EXP_SUM).select(0, atp.bs_prg_id)
    nisa.dma_copy(
        dst=dbg_exp_sum_view.get_view(),
        src=exp_sum_dbg,
        dge_mode=dge_mode.none,
        name=f"dbg_exp_sum_store_{name_suffix}",
    )

    if atp.n_bsq_full_tiles > 0 and atp.s_active_bqh_remainder > 0:
        zeros = sbm.alloc_stack((1, atp.s_active_bqh_tile - atp.s_active_bqh_remainder), dtype=src.dtype)
        nisa.memset(zeros, 0)
        nisa.dma_copy(
            dbg_exp_sum_view.select(0, atp.n_bsq_full_tiles)
            .expand_dim(0)
            .slice(1, atp.s_active_bqh_remainder, atp.s_active_bqh_tile)
            .get_view(),
            zeros,
            dge_mode=dge_mode.none,
            name=f"dbg_exp_sum_store_zeros_{name_suffix}",
        )
    sbm.close_scope()


def _store_dbg_qk_max_zeros_full_batch(atp: AttnTileParams, sbm: SbufManager, bufs: AttnInternalBuffers):
    """Fallback zero-fill for DBG_QK_MAX when batch tiling is active (full-batch offset writes
    aren't reliable). Writes once on the first batch tile using the debug tensor's full shape."""
    sbm.open_scope()
    dbg_qk_max_view = TensorView(bufs.DBG_QK_MAX).select(0, atp.bs_prg_id)
    dbg_zero = sbm.alloc_stack((dbg_qk_max_view.shape[0], 1), dtype=bufs.DBG_QK_MAX.dtype, buffer=nl.sbuf)
    nisa.memset(dbg_zero, 0.0)
    nisa.dma_copy(
        dbg_qk_max_view.get_view(),
        TensorView(dbg_zero).broadcast(1, dbg_qk_max_view.shape[1]).get_view(),
        dge_mode=dge_mode.none,
        name="dbg_qk_max_store_zeros_batch_tiling",
    )
    sbm.close_scope()


def _store_dbg_exp_sum_zeros_full_batch(atp: AttnTileParams, sbm: SbufManager, bufs: AttnInternalBuffers):
    """Fallback zero-fill for DBG_EXP_SUM when batch tiling is active."""
    sbm.open_scope()
    dbg_exp_sum_view = TensorView(bufs.DBG_EXP_SUM).select(0, atp.bs_prg_id)
    dbg_zero = sbm.alloc_stack((dbg_exp_sum_view.shape[0], 1), dtype=bufs.DBG_EXP_SUM.dtype, buffer=nl.sbuf)
    nisa.memset(dbg_zero, 0.0)
    nisa.dma_copy(
        dbg_exp_sum_view.get_view(),
        TensorView(dbg_zero).broadcast(1, dbg_exp_sum_view.shape[1]).get_view(),
        dge_mode=dge_mode.none,
        name="dbg_exp_sum_store_zeros_batch_tiling",
    )
    sbm.close_scope()


def _allocate_qk_buffers(
    atp: AttnTileParams, TC: TileConstants, sbm: SbufManager, bufs: AttnInternalBuffers, fa_ctx: FATileContext
):
    """Allocate core QK buffers for the current FA tile.

    Create KQ^T result mloc for all batches (filled with -INF for masking)
    The tensor has shape [p_max, tile_n_sprior * bs * s_active_qh], where on the free dimension, there are
      tile_n_sprior tiles, each tile contains bs number of subtiles, and each subtile is s_active_qh in length.
      I.e., the s_active_qh tiles are interleaved by batch on the free dimension.
    The cascaded max reduce later on will do a strided access on the free dimension.

    Uses fa_ctx.tile_n_sprior which is the actual tile size (may be smaller for last FA tile).
    """

    bufs.qk = sbm.alloc_stack(
        (TC.p_max, fa_ctx.tile_n_sprior * atp.s_active_bqh),
        dtype=atp.inter_type,
    )

    bufs.qk_io_type = sbm.alloc_stack(bufs.qk.shape, dtype=atp.io_type)  # for matmults

    # Allocate mask buffer with same shape as qk
    bufs.mask_sb = sbm.alloc_stack(bufs.qk.shape, dtype=nl.uint8, buffer=nl.sbuf)


def _allocate_online_softmax_buffers(
    atp: AttnTileParams, cfg: AttnTKGConfig, sbm: SbufManager, bufs: AttnInternalBuffers, is_dynamic: bool = False
):
    """Allocate the online-softmax running-statistics buffers (running max/sum/output and the
    correction-factor staging buffer).

    Only called when atp.use_online_softmax is True (i.e. atp.use_fa or atp.sprior_n_prgs > 1).
    When is_dynamic=True, buffers are initialized to identity values so the update math works
    correctly on the first dynamic iteration without a special case.
    """
    # Running max - same shape as qk_max_buf[:, :n_bsq_tiles]
    bufs.running_max = sbm.alloc_stack((atp.s_active_bqh_tile, atp.n_bsq_tiles), dtype=atp.inter_type, buffer=nl.sbuf)
    if is_dynamic:
        nisa.memset(bufs.running_max, value=-np.inf)

    # Running sum - same shape as exp_sum[:, :n_bsq_tiles]
    bufs.running_sum = sbm.alloc_stack((atp.s_active_bqh_tile, atp.n_bsq_tiles), dtype=atp.inter_type, buffer=nl.sbuf)
    if is_dynamic:
        nisa.memset(bufs.running_sum, value=0)

    # Correction factor exp(prev_max - curr_max) - same shape as running_max
    bufs.correction_factor = sbm.alloc_stack(
        (atp.s_active_bqh_tile, atp.n_bsq_tiles), dtype=atp.inter_type, buffer=nl.sbuf
    )
    if is_dynamic:
        nisa.memset(bufs.correction_factor, value=1.0)

    # Running output - accumulates PV results across tiles
    # Allocate with flat shape [d_head, s_active_bqh] to match exp_v layout
    bufs.running_output = sbm.alloc_stack((cfg.d_head, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf)
    if is_dynamic:
        nisa.memset(bufs.running_output, value=0)


def _update_correction_factor(
    atp: AttnTileParams, correction_factor: nl.ndarray, curr_running_max: nl.ndarray, prev_running_max: nl.ndarray
):
    """Updates correction_factor to exp(prev_running_max - curr_running_max)"""
    for i_bsq_tile in range(atp.n_bsq_tiles):
        nisa.activation(
            correction_factor[:, i_bsq_tile],
            nl.exp,
            (prev_running_max if atp.max_negated else curr_running_max)[:, i_bsq_tile],
            bias=(curr_running_max if atp.max_negated else prev_running_max)[:, i_bsq_tile],
            scale=-1.0,
        )


def _gather_and_compute_global_running_max(
    atp: AttnTileParams,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    prev_running_max: nl.ndarray,
    sink_values: Optional[nl.ndarray] = None,
):
    """Fold sink (if present) and remote-NC running max into running_max to produce the global
    max, then update correction_factor = exp(prev_running_max - global_max).

    Called from _finalize_and_store (sharded path). When `sink_values` is provided, fold it
    locally so the sendrecv'd max includes the sink term.
    """
    kernel_assert(not atp.max_negated, "Unexpected atp.max_negated=True when computing cross-NC max")

    if sink_values is not None:
        # Fold sink into running_max locally so the sendrecv'd max includes the sink term.
        nisa.tensor_tensor(
            dst=bufs.running_max,
            data1=bufs.running_max,
            data2=sink_values,
            op=nl.maximum,
        )

    sbm.open_scope()
    remote_running_max = sbm.alloc_stack(bufs.running_max.shape, dtype=atp.inter_type, buffer=nl.sbuf)
    nisa.sendrecv(
        src=bufs.running_max,
        dst=remote_running_max,
        send_to_rank=(1 - atp.sprior_prg_id),
        recv_from_rank=(1 - atp.sprior_prg_id),
        pipe_id=0,
    )
    # global_max = max(local+sink, remote)
    nisa.tensor_tensor(
        dst=bufs.running_max,
        data1=bufs.running_max,
        data2=remote_running_max,
        op=nl.maximum,
    )
    # correction_factor = exp(prev_running_max - global_max)
    _update_correction_factor(atp, bufs.correction_factor, bufs.running_max, prev_running_max)
    sbm.close_scope()


def _update_running_max(
    atp: AttnTileParams,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
):
    """Update flash attention running max after computing tile max.

    Called inside _cascaded_max_reduce after step 2.3 for each FA tile.
    Updates:
    - running_max: running max across tiles, shape (s_active_bqh_tile, n_bsq_tiles)
    - correction_factor: exp(prev_max - curr_max) for rescaling, same shape

    The tile max is in bufs.qk_max_buf[:, :n_bsq_tiles] after reduction.
    When max_negated=True, values are negated (so min gives true max).
    When max_negated=False, values are not negated (so max gives true max).

    Cross-NC sync (when sharded) is deferred to _finalize_and_store; running_max stays local
    here.
    """
    # Get current tile max from qk_max_buf (first n_bsq_tiles columns after reduction)
    tile_max = bufs.qk_max_buf[: atp.s_active_bqh_tile, : atp.n_bsq_tiles]

    is_dynamic = fa_ctx.dynamic_tile_offset_sbuf is not None
    if not is_dynamic and fa_ctx.fa_tile_idx == 0:
        # First tile (static path): just copy tile max, correction = 1.0
        nisa.tensor_copy(bufs.running_max, tile_max)
        nisa.memset(bufs.correction_factor, value=1.0)
    else:
        # Update running max and compute correction factor.
        # In dynamic path, running_max is identity-init (-inf) so first tile works correctly.
        sbm.open_scope()
        # Save previous running max
        prev_running_max = sbm.alloc_stack(bufs.running_max.shape, dtype=bufs.running_max.dtype)
        nisa.tensor_copy(prev_running_max, bufs.running_max)

        # Update running max: min if negated, max if not negated
        nisa.tensor_tensor(
            bufs.running_max, bufs.running_max, tile_max, op=(nl.minimum if atp.max_negated else nl.maximum)
        )

        # Local correction only. Global sync (for sharded) happens in _finalize_and_store.
        _update_correction_factor(atp, bufs.correction_factor, bufs.running_max, prev_running_max)

        sbm.close_scope()


def _gather_and_compute_global_running_sum(
    atp: AttnTileParams,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    sink_values: Optional[nl.ndarray] = None,
):
    """Gather remote running sum into running_sum and add sink_exp (if present) to produce the
    global running sum.

    Called from _finalize_and_store (sharded path). When `sink_values` is provided, the raw sink
    buffer is overwritten in-place with sink_exp = exp(sink - global_max) and added to running_sum.

    Precondition: running_sum holds the local contribution already scaled to the global max via
    the correction factor applied before this helper.
    """
    kernel_assert(not atp.max_negated, "Unexpected atp.max_negated=True when computing cross-NC sum")

    sbm.open_scope()
    remote_running_sum = sbm.alloc_stack(bufs.running_sum.shape, dtype=atp.inter_type, buffer=nl.sbuf)
    nisa.sendrecv(
        src=bufs.running_sum,
        dst=remote_running_sum,
        send_to_rank=(1 - atp.sprior_prg_id),
        recv_from_rank=(1 - atp.sprior_prg_id),
        pipe_id=0,
    )
    # global_sum (without sink) = local_running_sum + remote_running_sum
    nisa.tensor_tensor(
        dst=bufs.running_sum,
        data1=bufs.running_sum,
        data2=remote_running_sum,
        op=nl.add,
    )
    sbm.close_scope()

    if sink_values is not None:
        # sink_exp = exp(sink_raw - global_max) in-place, then running_sum += sink_exp.
        for i_bsq_tile in range(atp.n_bsq_tiles):
            nisa.activation(
                dst=sink_values[:, i_bsq_tile],
                op=nl.exp,
                data=bufs.running_max[: atp.s_active_bqh_tile, i_bsq_tile],
                bias=sink_values[:, i_bsq_tile],
                scale=-1.0,
            )
        nisa.tensor_tensor(
            dst=bufs.running_sum,
            data1=bufs.running_sum,
            data2=sink_values,
            op=nl.add,
        )


def _update_running_sum(
    atp: AttnTileParams,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
):
    """Update flash attention running sum after computing tile exp sum.

    Called in _cascaded_sum_reduction for each FA tile.
    Updates running_sum = running_sum * correction_factor + tile_sum

    All tensors have shape (s_active_bqh_tile, n_bsq_tiles).
    """
    # exp_sum has shape [s_active_bqh_tile, n_bsq_tiles * softmax_final_reduction_length]
    # After reduction, tile sum is in exp_sum[:, 0:n_bsq_tiles]
    tile_sum = bufs.exp_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles]

    is_dynamic = fa_ctx.dynamic_tile_offset_sbuf is not None
    if not is_dynamic and fa_ctx.fa_tile_idx == 0:
        # First tile (static path): just copy. Dynamic path uses identity-init (0).
        nisa.tensor_copy(bufs.running_sum, tile_sum)
    else:
        # running_sum = running_sum * correction_factor + tile_sum
        # Step 1: running_sum *= correction_factor
        nisa.tensor_tensor(
            bufs.running_sum,
            bufs.running_sum,
            bufs.correction_factor,
            op=nl.multiply,
        )
        # Step 2: running_sum += tile_sum
        nisa.tensor_tensor(
            bufs.running_sum,
            bufs.running_sum,
            tile_sum,
            op=nl.add,
        )


def _accumulate_output(
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
):
    """Accumulate PV output for flash attention.

    Called after PV matmul for each FA tile.
    Updates running_output = running_output * correction_factor + tile_output

    Note: exp_v has shape [d_head, bs, s_active_qh] and contains unnormalized PV output.
    running_output has shape [d_head, s_active_bqh] (flat).
    For FA, we don't multiply by exp_sum_recip here - that's done in finalize.
    """
    # Reshape exp_v to flat [d_head, s_active_bqh]
    exp_v_flat = bufs.exp_v.reshape((cfg.d_head, atp.s_active_bqh))

    is_dynamic = fa_ctx.dynamic_tile_offset_sbuf is not None
    if not is_dynamic and fa_ctx.fa_tile_idx == 0:
        # First tile (static path): just copy. Dynamic path uses identity-init (0).
        nisa.tensor_copy(bufs.running_output, exp_v_flat)
    else:
        # running_output = running_output * correction_factor + tile_output
        # correction_factor has shape [s_active_bqh_tile, n_bsq_tiles]
        # running_output has shape [d_head, s_active_bqh]
        #
        # Transpose correction_factor to [1, s_active_bqh] then broadcast to [d_head, s_active_bqh]
        sbm.open_scope()
        # Broadcasted correction factor - shape [d_head, s_active_bqh] for element-wise ops with running_output
        correction_factor_bc = sbm.alloc_stack((cfg.d_head, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf)
        _s_active_bqh_tile_transpose_broadcast(bufs.correction_factor, correction_factor_bc, atp, TC)

        # Now apply: running_output = running_output * correction_factor_bc + exp_v
        nisa.tensor_tensor(
            bufs.running_output,
            bufs.running_output,
            correction_factor_bc,
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            bufs.running_output,
            bufs.running_output,
            exp_v_flat,
            op=nl.add,
        )
        sbm.close_scope()


def _finalize_and_store(
    sink,
    out: nl.ndarray,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    btc: BatchTileContext,
    DBG_TENSORS=None,
):
    """Finalize flash attention output: sync softmax across NCs (if sharded), normalize by running
    sum and store to HBM.

    After all FA tiles are processed, for each NeuronCore we have:
    - running_max: local max over QK across all FA tiles (no sink, no remote).
    - running_sum: sum over local tiles of exp(qk - local_max).
    - running_output: sum over local tiles of exp(qk - local_max) @ V.

    When s_prior is sharded across NCs, we run the cross-NC softmax sync here (after the last PV
    matmul) so sendrecv does not block GPSIMD V prefetches on the last FA tile. The sync:
        1. Save local_max to a scope-local buffer (used as prev for the correction factor).
        2. If sink is used: load sink into a scope-local buffer via _prep_sink.
        3. Call _gather_and_compute_global_running_max(prev_running_max=local_max, sink_values=...)
           which folds sink + sendrecv max + updates correction_factor to exp(local - global).
        4. Rescale running_sum and running_output by correction_factor (output is
           transpose_broadcast to [d_head, s_active_bqh]).
        5. Call _gather_and_compute_global_running_sum(sink_values=...) which sendrecv's sum,
           adds remote, and adds exp(sink - global_max).

    Then normalize running_output by reciprocal(running_sum) and store.

    running_output has shape [d_head, s_active_bqh] (flat).
    running_sum has shape [s_active_bqh_tile, n_bsq_tiles].
    """
    sbm.open_scope()

    if atp.sprior_n_prgs > 1:
        kernel_assert(not atp.max_negated, "Unexpected atp.max_negated=True when deferring softmax gather")

        sbm.open_scope()

        # Scope-local sink buffer for the cross-NC sync; sink is consumed here rather than in
        # _cascaded_max_reduce / _cascaded_sum_reduction.
        sink_values = None
        if sink is not None:
            sink_values = sbm.alloc_stack(
                (atp.s_active_bqh_tile, atp.n_bsq_tiles), dtype=atp.inter_type, buffer=nl.sbuf
            )
            _prep_sink(sink, sink_values, atp, cfg, TC, sbm, btc)

        # 1. Save local_max (used as prev when computing c_local = exp(local_max - global_max)
        #    and for rescaling running_sum / running_output to the global scale).
        local_running_max = sbm.alloc_stack(bufs.running_max.shape, dtype=atp.inter_type, buffer=nl.sbuf)
        nisa.tensor_copy(local_running_max, bufs.running_max)

        # 2. Fold sink locally + sendrecv remote + update correction_factor = exp(local - global).
        _gather_and_compute_global_running_max(atp, sbm, bufs, local_running_max, sink_values=sink_values)

        # 3. Rescale running_sum by c_local (so the local contribution is at global-max scale).
        nisa.tensor_tensor(
            bufs.running_sum,
            bufs.running_sum,
            bufs.correction_factor,
            op=nl.multiply,
        )

        # 4. Rescale running_output by c_local (broadcast to [d_head, s_active_bqh]).
        correction_factor_bc = sbm.alloc_stack((cfg.d_head, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf)
        _s_active_bqh_tile_transpose_broadcast(bufs.correction_factor, correction_factor_bc, atp, TC)
        nisa.tensor_tensor(
            bufs.running_output,
            bufs.running_output,
            correction_factor_bc,
            op=nl.multiply,
        )

        # 5. Sendrecv running_sum + add remote + add sink_exp.
        _gather_and_compute_global_running_sum(atp, sbm, bufs, sink_values=sink_values)

        sbm.close_scope()

    # Debug tensor writes for the online-softmax path (sharded; FA single-NC). Dumping from
    # running_max / running_sum captures the final global values. Batch-tiling case
    # (bs != bs_per_nc) is handled by the zero-fill fallback in _cascaded_*.
    if DBG_TENSORS is not None and atp.bs == atp.bs_per_nc:
        _store_dbg_qk_max(bufs.running_max, atp.max_negated, "finalize", atp, TC, sbm, bufs)
        _store_dbg_exp_sum(bufs.running_sum, "finalize", atp, TC, sbm, bufs)

    # Compute reciprocal of running sum in-place
    nisa.reciprocal(
        bufs.running_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
        bufs.running_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
    )

    # Transpose and broadcast sum_recip to [d_head, s_active_bqh] for final normalization

    sum_recip_bc = sbm.alloc_stack((cfg.d_head, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf)
    _s_active_bqh_tile_transpose_broadcast(bufs.running_sum, sum_recip_bc, atp, TC)

    # Normalize: running_output *= sum_recip_bc
    nisa.tensor_tensor(
        bufs.running_output,
        bufs.running_output,
        sum_recip_bc,
        op=nl.multiply,
    )
    sbm.close_scope()
    exp_v_sendrecv_gpsimd = (
        cfg.use_gpsimd_sb2sb and atp.sprior_n_prgs > 1 and atp.bs * atp.s_active_qh <= 128 and cfg.d_head % 16 == 0
    )
    _gather_and_store_output(out, bufs.running_output, exp_v_sendrecv_gpsimd, atp, cfg, sbm, btc)


def _load_and_broadcast_pos_ids(pos_ids, atp, cfg, TC, sbm, name):
    """Load position IDs and broadcast onto all 128 partitions (for TensorScalarPtr).

    Shared helper for loading both rope_pos_ids and start_pos_ids, which follow
    the same pattern: reshape → alloc_stack → dma_copy → alloc_stack → stream_shuffle_broadcast.
    """
    pos_ids_sb = sbm.alloc_stack((TC.p_max, atp.bs_per_nc * cfg.s_active), dtype=pos_ids.dtype)
    pos_ids = pos_ids.reshape([atp.bs_n_prgs, atp.bs_per_nc * cfg.s_active])

    sbm.open_scope()
    pos_ids_loaded = sbm.alloc_stack((1, atp.bs_per_nc * cfg.s_active), dtype=pos_ids.dtype, align=4)
    nisa.dma_copy(pos_ids_loaded, pos_ids[atp.bs_prg_id, :], dge_mode=dge_mode.none, name=name)
    stream_shuffle_broadcast(src=pos_ids_loaded, dst=pos_ids_sb)
    sbm.close_scope()
    return pos_ids_sb


def _load_position_ids(
    rope_pos_ids,
    start_pos_ids,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """Load position IDs and optional SWA start positions."""
    bufs.pos_ids_sb = None
    if rope_pos_ids is None:
        # only two components that use pos ids
        kernel_assert(
            not cfg.use_pos_id and not cfg.fuse_rope,
            "To generate mask or fuse rope, rope_pos_ids tensor must be provided",
        )
    else:
        bufs.pos_ids_sb = _load_and_broadcast_pos_ids(rope_pos_ids, atp, cfg, TC, sbm, "rope_pos_ids_load")

    bufs.start_pos_sb = None
    if start_pos_ids is not None:
        bufs.start_pos_sb = _load_and_broadcast_pos_ids(start_pos_ids, atp, cfg, TC, sbm, "start_pos_ids_load")


def _load_mask(
    mask,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Load mask for the current FA tile and batch tile."""
    fa_tile_n_sprior = fa_ctx.tile_n_sprior
    fa_tile_offset = fa_ctx.tile_offset
    is_dynamic_mask = fa_ctx.dynamic_tile_offset_sbuf is not None

    # If we don't use pos_id, mask is already generated outside of kernel. Otherwise, generate prior mask in kernel and
    # load active mask at the end of the generated mask.
    if not cfg.use_pos_id:
        # Reshape mask with full per-NC bqh, then slice to batch tile
        full_s_active_bqh = atp.bs_per_nc * atp.s_active_qh

        # Compute source offset including FA tile offset and batch tile offset
        bqh_offset = btc.tile_batch_offset * atp.s_active_qh

        # Reshape mask as full s_prior and slice by global tile_offset
        mask = mask.reshape((cfg.curr_sprior, atp.bs_n_prgs, full_s_active_bqh))
        mask_hbm_view = (
            TensorView(mask)
            .select(dim=1, index=atp.bs_prg_id)
            .slice(dim=0, start=fa_tile_offset, end=fa_tile_offset + fa_ctx.tile_s_prior)
            .slice(dim=1, start=bqh_offset, end=bqh_offset + atp.s_active_bqh)
        )

        # gen_mask_tkg_hbm stores in n_sprior_tile-major layout:
        # [n_sprior_tile, P_MAX, ...]. After flatten to [s_prior, ...], the
        # load must undo the tiling to recover [P_MAX, n_sprior_tile] in SBUF.
        #
        # TODO: The strided_mm1 flat-KV branch below uses reshape_dim(0,
        # [P_MAX, n_sprior_tile]) which assumes P_MAX-major order. This is
        # inconsistent with the n_sprior_tile-major HBM layout and should
        # use the else path (reshape + permute) like all other cases.
        # Kept as-is pending end-to-end validation; tracked for follow-up.
        if cfg.strided_mm1 and not atp.is_block_kv:
            mask_hbm_view = mask_hbm_view.reshape_dim(0, [TC.p_max, fa_tile_n_sprior])
        else:
            mask_hbm_view = mask_hbm_view.reshape_dim(0, [fa_tile_n_sprior, TC.p_max]).permute([1, 0, 2])

        mask_sb_view = TensorView(bufs.mask_sb).reshape_dim(1, [fa_tile_n_sprior, atp.s_active_bqh])
        nisa.dma_copy(
            dst=mask_sb_view.get_view(),
            src=mask_hbm_view.get_view(),
            dge_mode=dge_mode.none,
            name=f"{sbm.get_name_prefix()}mask_load_pregenerated_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}",
        )
    else:
        # In-kernel mask generation supports both flat and block KV cache
        # atp.block_len is 0 for flat KV cache, adjusted block_len for block KV cache
        bufs.mask_sb = bufs.mask_sb.reshape((TC.p_max, fa_tile_n_sprior, atp.bs, cfg.q_head, cfg.s_active))

        # For FA, only load active mask on the last FA tile and last NC
        # For non-FA, load active mask on the last NC (sprior_prg_id == sprior_n_prgs - 1)
        load_active_mask = (atp.sprior_prg_id == atp.sprior_n_prgs - 1) and fa_ctx.is_last_fa_tile

        # Slice pos_ids_sb to the batch tile's portion
        pos_ids_offset = btc.tile_batch_offset * cfg.s_active
        pos_ids_tile = bufs.pos_ids_sb[:, nl.ds(pos_ids_offset, atp.bs * cfg.s_active)]

        start_pos_tile = None
        if bufs.start_pos_sb is not None:
            start_pos_tile = bufs.start_pos_sb[:, nl.ds(pos_ids_offset, atp.bs * cfg.s_active)]

        sbm_prefix = sbm.get_name_prefix()
        sbm.set_name_prefix(f"{sbm_prefix}_bt{btc.batch_tile_idx}_")
        # tile_offset is always global — so we pass full s_prior_per_shard
        # and is_s_prior_sharded=False.
        gen_mask_tkg(
            pos_ids=pos_ids_tile,
            mask_out=bufs.mask_sb,
            bs=atp.bs,
            q_head=cfg.q_head,
            s_active=cfg.s_active,
            s_prior_per_shard=cfg.curr_sprior,
            start_pos=start_pos_tile,
            s_prior_offset=fa_tile_offset,
            block_len=atp.block_len,
            strided_mm1=cfg.strided_mm1,
            active_mask=mask if load_active_mask else None,
            sbm=sbm,
            is_batch_sharded=atp.bs_n_prgs > 1,
            is_s_prior_sharded=False,
            batch_offset=btc.tile_batch_offset,
            dynamic_s_prior_offset=fa_ctx.dynamic_tile_offset_f32 if is_dynamic_mask else None,
        )
        sbm.set_name_prefix(sbm_prefix)
        bufs.mask_sb = bufs.mask_sb.reshape(bufs.qk.shape)


def _perform_rope(
    q,
    k_active,
    inv_freqs,
    k_out,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """Step 0. Optional RoPE"""
    kernel_assert(
        cfg.curr_sprior <= _MAX_S_PRIOR_ACCURATE_ROPE,
        f"Rope requires modulo, which for s_prior={cfg.curr_sprior} > {_MAX_S_PRIOR_ACCURATE_ROPE} is innacurate due to float32 error build-up.",
    )

    # If we fuse rope, Q and K_active would need be processed by RoPE first then be stored in Q_sb.
    bufs.q_sb = sbm.alloc_stack(
        (cfg.d_head, atp.bs_n_prgs * atp.bs_per_nc * atp.s_active_qh),
        dtype=atp.io_type,
        buffer=nl.sbuf,
    )
    bufs.k_active_sb = (
        k_out
        if cfg.k_out_in_sb
        else sbm.alloc_stack(
            (cfg.d_head, atp.bs_n_prgs * atp.bs_per_nc * cfg.s_active),
            dtype=atp.io_type,
            buffer=nl.sbuf,
        )
    )

    # Load inv_freqs
    sbm.open_scope()
    inv_freqs_sb = sbm.alloc_stack(inv_freqs.shape, dtype=inv_freqs.dtype, buffer=nl.sbuf)
    nisa.dma_copy(inv_freqs_sb, inv_freqs, dge_mode=dge_mode.none, name="inv_freqs_load")

    # Compute RoPE coefficients, then apply (while loading) onto Q and K_active (only last NC handles K_active)
    cos, sin = _rope(
        inv_freqs_sb,
        bufs.pos_ids_sb,
        bs=atp.bs_per_nc,
        s_a=cfg.s_active,
        d_head=cfg.d_head,
        sbm=sbm,
    )
    _apply_rope(q, cos, sin, bufs.q_sb, cfg, sbm=sbm, name_suffix="q")
    if cfg.k_out_in_sb or (atp.sprior_prg_id == atp.sprior_n_prgs - 1):
        _apply_rope(k_active, cos, sin, bufs.k_active_sb, cfg, ignore_heads=True, sbm=sbm, name_suffix="k_active")
        # Store K to the second output if not kOutInSB; otherwise we already write to it via name alias to k_active_sb
        if not cfg.k_out_in_sb and k_out is not None:
            k_active_sb_view = TensorView(bufs.k_active_sb).reshape_dim(1, [cfg.bs, cfg.s_active])
            k_out_hbm_view = TensorView(k_out).squeeze_dim(1).permute([1, 0, 2])
            nisa.dma_copy(
                src=k_active_sb_view.get_view(),
                dst=k_out_hbm_view.get_view(),
                dge_mode=dge_mode.none,
                name="k_out_store_after_rope",
            )

    nisa.activation(bufs.q_sb, op=nl.copy, data=bufs.q_sb, scale=1 / math.sqrt(cfg.d_head))
    sbm.close_scope()


"""
Main computation blocks
"""


def _compute_qk_matmul(
    k_prior,
    DBG_TENSORS,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Step 1. Matmult 1 of KQ^T (and optional K_prior transpose)"""
    fa_tile_s_prior = fa_ctx.tile_s_prior
    fa_tile_n_sprior = fa_ctx.tile_n_sprior
    fa_tile_offset = fa_ctx.tile_offset
    is_last_fa_tile = fa_ctx.is_last_fa_tile

    # Use per-tile s_prior for batch interleave calculation
    # For block KV with PE transpose path, each batch also accumulates k_loaded buffers across folds.
    sbuf_usage_per_batch = fa_tile_s_prior * sizeinbytes(k_prior.dtype)
    if atp.is_block_kv:
        # For FA, compute which folds correspond to this tile
        # Each fold covers block_len * 128 elements of s_prior
        fold_s_prior = atp.block_len * TC.p_max
        fold_start = fa_tile_offset // fold_s_prior
        fold_end = div_ceil(fa_tile_offset + fa_tile_s_prior, fold_s_prior)
        num_folds_this_tile = fold_end - fold_start

        if not atp.use_dma_transpose:
            sbuf_usage_per_batch += (
                num_folds_this_tile * atp.block_len * cfg.d_head * sizeinbytes(atp.k_prior_load_type)
            )
    elif cfg.tp_k_prior and atp.is_fp8_kv:
        sbuf_usage_per_batch += TC.psum_b_max * cfg.d_head * sizeinbytes(atp.k_prior_load_type)
    batch_interleave_degree_safe = _get_safe_batch_interleave_degree(
        sbuf_usage_per_batch,
        atp.batch_interleave_degree,
        sbm,
    )

    # Maximum multi-buffer degree inside a batch is 8 (banks) // bs (multi-buffer degree on the current scope)
    per_batch_interleave_degree = math.floor(float(TC.psum_b_max) / batch_interleave_degree_safe)
    sbm.open_scope(interleave_degree=batch_interleave_degree_safe, name="qk_matmul")
    for i_b in range(atp.bs):
        # Load entire K_prior for current batch into sbuf (tile portion for FA)
        # FP8 KV: use the resolved FP8 dtype (caller's concrete dtype, or
        # dtype_mode resolution for opaque "float8e4"). Otherwise pass through k_prior.dtype.
        _k_sb_dtype = atp.kv_e4m3_tile_dtype if atp.is_fp8_kv else k_prior.dtype

        # For fp8_packed: allocate as bf16 with half the seq length (each bf16 slot holds 2 fp8 values)
        if cfg.fp8_packed:
            k_sb = sbm.alloc_stack((cfg.d_head, fa_tile_s_prior // 2), dtype=nl.bfloat16, buffer=nl.sbuf, align=32)
        else:
            k_sb = sbm.alloc_stack((cfg.d_head, fa_tile_s_prior), dtype=_k_sb_dtype, buffer=nl.sbuf, align=32)
        if atp.is_block_kv:
            # Pre-compute for dma_transpose path
            if atp.use_dma_transpose:
                if cfg.fp8_packed:
                    # Reinterpret fp8 [N, elems_fp8] as bf16 [N, elems_fp8 // 2]
                    k_prior_bf16 = TensorView(bufs.k_prior_reshaped).reinterpret_cast(nl.bfloat16).get_view()
                    k_block_len_dma = bufs.k_prior_reshaped.shape[1] // (cfg.d_head * 2)
                    k_prior_4d = k_prior_bf16.reshape((bufs.k_prior_reshaped.shape[0], 1, k_block_len_dma, cfg.d_head))
                else:
                    k_block_len_dma = atp.block_len
                    k_prior_4d = bufs.k_prior_reshaped.reshape(
                        (bufs.k_prior_reshaped.shape[0], 1, k_block_len_dma, cfg.d_head)
                    )

            sbm.open_scope()
            for i_fold_rel in range(num_folds_this_tile):
                i_fold = fold_start + i_fold_rel
                batch_pos = i_fold_rel * atp.bs + i_b
                cur_blks = TensorView(bufs.active_blocks_sb).slice(dim=1, start=batch_pos, end=batch_pos + 1).get_view()
                kernel_assert(
                    cur_blks.shape == (TC.p_max, 1),
                    f"Internal error: unexpected shape error after loading current blocks, expected {(TC.p_max, 1)}, got {cur_blks.shape}.",
                )

                if atp.use_dma_transpose:
                    # DMA transpose path: single indirect DMA transpose per fold.
                    # dma_transpose requires uint32 indices, but active_blocks_sb is kept
                    # as int32 so V's dma_copy can use oob_mode.skip with -1 sentinels.
                    # Use pre-computed uint32 copy (int32(-1) → float32(-1.0) → uint32(0))
                    # via the vector engine cast done once upfront.
                    cur_blks_u32 = (
                        TensorView(bufs.active_blocks_sb_u32)
                        .slice(dim=1, start=batch_pos, end=batch_pos + 1)
                        .get_view()
                    )
                    # Note that indirect dma_transpose requires src to be a 4-d tile (in addition to other constraints for src and the indices)
                    nisa.dma_transpose(
                        dst=(
                            TensorView(k_sb)
                            .reshape_dim(1, (num_folds_this_tile, k_block_len_dma, TC.p_max))
                            .select(1, i_fold_rel)
                            .expand_dim(1)
                            .get_view()
                        ),
                        # TODO: Port to TensorView once dynamic vector_offset is supported
                        src=k_prior_4d.ap(
                            [
                                [k_block_len_dma * cfg.d_head, TC.p_max],
                                [1, 1],
                                [cfg.d_head, k_block_len_dma],
                                [1, cfg.d_head],
                            ],
                            offset=0,
                            vector_offset=cur_blks_u32,
                            indirect_dim=0,
                        ),
                        axes=(3, 1, 2, 0),
                        dge_mode=dge_mode.swdge,
                    )
                else:
                    # PE transpose path: indirect DMA load + PE transposes per fold
                    k_loaded = sbm.alloc_stack(
                        (TC.p_max, atp.block_len * cfg.d_head),
                        dtype=atp.k_prior_load_type,
                        buffer=nl.sbuf,
                    )
                    # Memset K to 0 for easier accuracy debug: not strictly necessary since runtime does not
                    # throw NaN errors by default, and the QK result for skipped blocks is masked
                    # to -inf by the attention mask (so NaN from stale K data doesn't propagate).
                    # Can be removed for max perf since the NaN result is not copied back, and
                    # the SBUF value stays as -inf after softmax masking. Having memset helps debug if
                    # we enable NaN runtime error for issues elsewhere in graph.
                    # NOTE: Commented out for performance, re-enable for NaN debug purposes
                    # nisa.memset(k_loaded, value=0)
                    nisa.dma_copy(
                        dst=k_loaded,
                        # TODO: Port to TensorView once dynamic vector_offset is supported
                        src=bufs.k_prior_reshaped.ap(
                            [
                                [atp.block_len * cfg.d_head, TC.p_max],
                                [1, atp.block_len * cfg.d_head],
                            ],
                            offset=0,
                            vector_offset=cur_blks,
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.skip,
                        name=f"k_prior_block_load_indirect_fa{fa_ctx.fa_tile_idx}_b{i_b}_f{i_fold}_bt{btc.batch_tile_idx}",
                    )

                    # Transpose to [d_head, blk_len * 128blks]
                    # Explicitly group transposes that can share a single psum bank to allow compiler to fuse to a 1024-free-dim PSUM.
                    transpose_grp_size = min(
                        8, atp.block_len
                    )  # FIXME: parameterize this value 8 to psum free_dim size // data size
                    kernel_assert(
                        atp.block_len % transpose_grp_size == 0,
                        (
                            "Internal error: If block length is greater than 8, then it needs to be a multiple of 8 to allow tiling transpose. "
                            f"Instead got block length of {atp.block_len}."
                        ),
                    )
                    num_transpose_grps = atp.block_len // transpose_grp_size
                    for tp_grp_i in range(num_transpose_grps):
                        for tp_j_in_grp in range(transpose_grp_size):
                            blk_len_i = tp_grp_i * transpose_grp_size + tp_j_in_grp
                            tp_psum = nl.ndarray(
                                (cfg.d_head, TC.p_max),
                                dtype=atp.k_prior_load_type,
                                buffer=nl.psum,
                                address=None
                                if sbm.is_auto_alloc()
                                else (
                                    0,
                                    (tp_j_in_grp % per_batch_interleave_degree) * TC.psum_f_max_bytes,
                                ),
                            )
                            nisa.nc_transpose(
                                tp_psum,
                                k_loaded[:, nl.ds(blk_len_i * cfg.d_head, cfg.d_head)],
                            )

                            # Balance psum->sbuf copies across vector and scalar engines
                            cur_idx = i_fold_rel * atp.block_len + blk_len_i
                            if cur_idx % 2 == 0:
                                engine = nisa.vector_engine
                            else:
                                engine = nisa.scalar_engine

                            nisa.tensor_copy(k_sb[:, nl.ds(cur_idx * 128, 128)], tp_psum, engine=engine)
            sbm.close_scope()
        elif not cfg.tp_k_prior:
            kernel_assert(
                k_prior.shape[1:] == (1, cfg.d_head, cfg.full_sprior),
                f"k_prior[1:] expected to have shape {(1, cfg.d_head, cfg.full_sprior)=}, received {k_prior.shape[1:]=}",
            )
            # k_prior shape: [B+, 1, d, full_sprior]
            # K_prior is already transposed, insert flat load
            s_prior_pos = fa_tile_offset
            k_prior_view = (
                TensorView(k_prior)
                .select(0, btc.global_batch_offset + i_b)
                .squeeze_dim(0)
                .slice(1, start=s_prior_pos, end=s_prior_pos + fa_tile_s_prior)
            )
            nisa.dma_copy(
                k_sb,
                k_prior_view.get_view(),
                dge_mode=dge_mode.none,
                name=f"{sbm.get_name_prefix()}k_prior_flat_load_transposed_fa{fa_ctx.fa_tile_idx}_b{i_b}_bt{btc.batch_tile_idx}",
            )
        else:
            kernel_assert(
                k_prior.shape[1:] == (1, cfg.full_sprior, cfg.d_head),
                f"k_prior[1:] expected to have shape {(1, cfg.full_sprior, cfg.d_head)=}, received {k_prior.shape[1:]=}",
            )

            if atp.is_fp8_kv:
                # Can't do DMA transpose for FP8, so load as BF16, transpose via PSUM, copy to k_sb (casts to FP8)
                sbm.open_scope(interleave_degree=TC.psum_b_max)
                for tp_grp_i in range(fa_tile_n_sprior):
                    tile_start = tp_grp_i * TC.p_max
                    tile_size = min(TC.p_max, fa_tile_s_prior - tile_start)
                    k_loaded = sbm.alloc_stack((TC.p_max, cfg.d_head), dtype=atp.k_prior_load_type, buffer=nl.sbuf)
                    s_prior_pos = fa_tile_offset + tile_start
                    k_prior_view = (
                        TensorView(k_prior)
                        .select(0, btc.global_batch_offset + i_b)
                        .squeeze_dim(0)
                        .slice(0, start=s_prior_pos, end=s_prior_pos + tile_size)
                    )
                    nisa.dma_copy(
                        dst=k_loaded[:tile_size, :],
                        src=k_prior_view.get_view(),
                    )
                    tp_psum = nl.ndarray(
                        (cfg.d_head, TC.p_max),
                        dtype=atp.k_prior_load_type,
                        buffer=nl.psum,
                        address=None if sbm.is_auto_alloc() else (0, (tp_grp_i % TC.psum_b_max) * TC.psum_f_max_bytes),
                    )
                    nisa.nc_transpose(tp_psum[:, :tile_size], k_loaded[:tile_size, :])
                    nisa.tensor_copy(k_sb[:, nl.ds(tile_start, tile_size)], tp_psum[:, :tile_size])
                    sbm.increment_section()
                sbm.close_scope()

            else:
                # FIXME: 4d reshape_dim required here, while simple slicing should suffice
                k_sb_view = TensorView(k_sb).reshape_dim(1, [1, 1, fa_tile_s_prior])
                s_prior_pos = fa_tile_offset
                k_prior_view = (
                    TensorView(k_prior)
                    .select(0, btc.global_batch_offset + i_b)
                    .squeeze_dim(0)
                    .slice(0, start=s_prior_pos, end=s_prior_pos + fa_tile_s_prior)
                    .reshape_dim(1, [1, 1, cfg.d_head])
                )
                nisa.dma_transpose(k_sb_view.get_view(), k_prior_view.get_view())

        # If on final NC and last FA tile, add K_active to the end of k_sb
        if atp.sprior_prg_id == atp.sprior_n_prgs - 1 and is_last_fa_tile:
            if atp.is_block_kv:
                # For block KV with FA, use tile-relative fold count
                num_blks_covering_s_active = div_ceil(cfg.s_active, atp.block_len)
                extra_covered = num_blks_covering_s_active * atp.block_len - cfg.s_active

                if cfg.fp8_packed:
                    # fp8_packed k_active stitching:
                    # k_sb is bf16 [d_head, fa_tile_s_prior // 2]. Reinterpret as fp8 to get
                    # [d_head, fa_tile_s_prior] with interleaved layout.
                    kernel_assert(
                        num_blks_covering_s_active <= TC.p_max,
                        f"fp8_packed k_active stitching requires all active blocks fit in one fold (p_max={TC.p_max}), "
                        f"got num_blks_covering_s_active={num_blks_covering_s_active}.",
                    )
                    k_sb_fp8_stitch = TensorView(k_sb).reinterpret_cast(_k_sb_dtype)
                    k_sb_fp8_4d = k_sb_fp8_stitch.reshape_dim(
                        1, [num_folds_this_tile * (atp.block_len // 2), TC.p_max * 2]
                    ).reshape_dim(2, [TC.p_max, 2])
                    k_sb_fp8_perm = k_sb_fp8_4d.permute([0, 2, 1, 3])

                    k_active_batch = (
                        TensorView(bufs.k_active_sb)
                        .reshape_dim(1, [atp.bs_full, cfg.s_active])
                        .select(1, btc.global_batch_offset + i_b)
                    )

                    last_fold_blk_half_offset = (num_folds_this_tile - 1) * (atp.block_len // 2)
                    first_block_partition = TC.p_max - num_blks_covering_s_active

                    for i_active in range(cfg.s_active):
                        pos_in_blocks = extra_covered + i_active
                        blk_idx = pos_in_blocks // atp.block_len
                        seq_in_blk = pos_in_blocks % atp.block_len
                        blk_half = seq_in_blk // 2
                        parity = seq_in_blk % 2

                        partition_idx = first_block_partition + blk_idx
                        bh_idx = last_fold_blk_half_offset + blk_half

                        dst_view = (
                            k_sb_fp8_perm.slice(1, start=partition_idx, end=partition_idx + 1)
                            .slice(2, start=bh_idx, end=bh_idx + 1)
                            .slice(3, start=parity, end=parity + 1)
                        )
                        src_view = k_active_batch.slice(1, start=i_active, end=i_active + 1)

                        nisa.tensor_copy(
                            dst=dst_view.get_view(),
                            src=src_view.get_view(),
                        )
                else:
                    # Need to mask as dim_1 * blk_len + dim_2 >= extra_covered
                    # Solving the above inequality with 0 <= dim_1 < num_blks_covering_s_active and 0 <= dim_2 < blk_len
                    # we get (dim_1, dim_2) in {(0,[extra_covered, blk_len)) and ([1, num_blks_covering_s_active), [0, blk_len))}
                    # Thus, if extra_covered != 0, we do an access pattern for dim_1 == 0 and dim_2 in [extra_covered, blk_len)
                    # and for main copy we don't need any restrictions
                    if extra_covered > 0:
                        if atp.block_len > extra_covered:
                            dst_offset = (
                                (num_folds_this_tile - 1) * atp.block_len * TC.p_max
                                + TC.p_max
                                - num_blks_covering_s_active
                            )

                            start = dst_offset + TC.p_max * extra_covered

                            size = atp.block_len - extra_covered
                            end = start + size * TC.p_max
                            k_sb_view = TensorView(k_sb).slice(1, start=start, end=end, step=TC.p_max)

                            # k_active shape: [cfg.d_head, B * s_active]
                            k_active_view = (
                                TensorView(bufs.k_active_sb)
                                .reshape_dim(1, [atp.bs_full, cfg.s_active])
                                .select(1, btc.global_batch_offset + i_b)
                                .slice(1, start=0, end=atp.block_len - extra_covered)
                            )
                            nisa.tensor_copy(
                                dst=k_sb_view.get_view(),
                                src=k_active_view.get_view(),
                            )
                        if num_blks_covering_s_active > 1:
                            k_sb_start_1 = TC.p_max - num_blks_covering_s_active + 1
                            k_sb_start_2 = (num_folds_this_tile - 1) * atp.block_len

                            k_sb_view = (
                                TensorView(k_sb)
                                .reshape_dim(1, [fa_tile_n_sprior, TC.p_max])
                                .permute([0, 2, 1])
                                .slice(1, start=k_sb_start_1, end=k_sb_start_1 + num_blks_covering_s_active - 1)
                                .slice(2, start=k_sb_start_2, end=k_sb_start_2 + atp.block_len)
                            )

                            k_active_start_1 = atp.block_len - extra_covered
                            k_active_view = (
                                TensorView(bufs.k_active_sb)
                                .reshape_dim(1, [atp.bs_full, cfg.s_active])
                                .select(1, btc.global_batch_offset + i_b)
                                .slice(
                                    1,
                                    start=k_active_start_1,
                                    end=k_active_start_1 + (num_blks_covering_s_active - 1) * atp.block_len,
                                )
                                .reshape_dim(1, [num_blks_covering_s_active - 1, atp.block_len])
                            )

                            nisa.tensor_copy(
                                dst=k_sb_view.get_view(),
                                src=k_active_view.get_view(),
                            )
                    else:
                        k_sb_start_1 = TC.p_max - num_blks_covering_s_active
                        k_sb_start_2 = (num_folds_this_tile - 1) * atp.block_len

                        k_sb_view = (
                            TensorView(k_sb)
                            .reshape_dim(1, [fa_tile_n_sprior, TC.p_max])
                            .permute([0, 2, 1])
                            .slice(1, start=k_sb_start_1, end=k_sb_start_1 + num_blks_covering_s_active)
                            .slice(2, start=k_sb_start_2, end=k_sb_start_2 + atp.block_len)
                        )

                        k_active_view = (
                            TensorView(bufs.k_active_sb)
                            .reshape_dim(1, [atp.bs_full, cfg.s_active])
                            .select(1, btc.global_batch_offset + i_b)
                            .slice(
                                1,
                                start=0,
                                end=num_blks_covering_s_active * atp.block_len,
                            )
                            .reshape_dim(1, [num_blks_covering_s_active, atp.block_len])
                        )
                        nisa.tensor_copy(
                            dst=k_sb_view.get_view(),
                            src=k_active_view.get_view(),
                        )
            else:
                nisa.tensor_copy(
                    k_sb[:, fa_tile_s_prior - cfg.s_active : fa_tile_s_prior],
                    bufs.k_active_sb[
                        :,
                        nl.ds((btc.global_batch_offset + i_b) * cfg.s_active, cfg.s_active),
                    ],
                    engine=nisa.scalar_engine,
                )

        # Do MM1 in grps (default 4k grp size), make sure appropriate group size is selected s.t. psum free < hw limit
        mm1_grp_sz = 4 * 1024
        if (mm1_grp_sz // TC.p_max) * atp.s_active_qh > TC.psum_f_max:
            mm1_grp_sz = (TC.psum_f_max // atp.s_active_qh) * TC.p_max
        n_mm1_per_grp = mm1_grp_sz // TC.p_max

        # For fp8_packed, create the fp8 reinterpreted view once outside the matmul loop
        k_sb_fp8 = TensorView(k_sb).reinterpret_cast(_k_sb_dtype) if cfg.fp8_packed else None

        """
        Tiling Strategy for MM1 (KQ^T computation):
        - K stationary: [d_head, s_prior] loaded per batch into k_sb
        - Q moving: [d_head, s_active_qh] per batch from q_sb
        - Tile size: mm1_grp_sz (default 4096 = 4k) to balance PSUM usage
        - PSUM allocation: [P_MAX, n_mm1_per_grp * s_active_qh]
          where n_mm1_per_grp = mm1_grp_sz / P_MAX
        - PSUM constraint: (mm1_grp_sz / P_MAX) * s_active_qh < psum_f_max
        - Output: qk [P_MAX, n_sprior_tile * s_active_bqh] with batch interleaving
        - Memory: Each tile processes P_MAX rows of K against full Q per batch
        """

        for i_mm1_grp in range(div_ceil(fa_tile_s_prior, mm1_grp_sz)):
            # Perform MM for this (4k) tile in MMs of p_max (limited to p_max as K is stationary)
            # The psum can store entire output for a tile which only needs (grp_sz / p_max) * s_active_qh free dim
            qk_psum = nl.ndarray(
                (TC.p_max, n_mm1_per_grp * atp.s_active_qh),
                dtype=nl.float32,
                buffer=nl.psum,
                address=None
                if sbm.is_auto_alloc()
                else (
                    0,
                    (i_mm1_grp % per_batch_interleave_degree) * TC.psum_f_max_bytes,
                ),
            )

            # Do (mm1_grp_sz / p_max) matmults, note mm1_grp_sz is divisible by p_max
            for i_mm1 in range(n_mm1_per_grp):
                if (
                    cfg.strided_mm1
                ):  # optionally use strided read to K s.t. MM2 can also be strided with sequential read to V
                    k_tile_offset = i_mm1_grp * n_mm1_per_grp + i_mm1
                    num_acc = min(
                        TC.p_max,
                        (fa_tile_s_prior - 1 - k_tile_offset) // fa_tile_n_sprior + 1,
                    )
                    if num_acc <= 0:
                        break  # k_tile_offset is strictly increasing
                    k_tile = (
                        TensorView(k_sb)
                        .slice(
                            1,
                            start=k_tile_offset,
                            end=k_tile_offset + num_acc * fa_tile_n_sprior,
                            step=fa_tile_n_sprior,
                        )
                        .get_view()
                    )
                else:
                    k_tile_offset = i_mm1_grp * mm1_grp_sz + i_mm1 * TC.p_max
                    num_acc = min(TC.p_max, fa_tile_s_prior - k_tile_offset)
                    if num_acc <= 0:
                        break  # k_tile_offset is strictly increasing
                    if cfg.fp8_packed:
                        # fp8_packed: k_sb is bf16 [d_head, fa_tile_s_prior // 2].
                        # Reinterpret as fp8 gives [d_head, fa_tile_s_prior] with interleaved layout.
                        logical_tile_idx = k_tile_offset // TC.p_max
                        blk_half = logical_tile_idx // 2
                        parity = logical_tile_idx % 2
                        phys_start = blk_half * 2 * TC.p_max + parity
                        k_tile = k_sb_fp8.slice(1, start=phys_start, end=phys_start + num_acc * 2, step=2).get_view()
                    else:
                        k_tile = k_sb[0 : cfg.d_head, nl.ds(k_tile_offset, num_acc)]

                qk_psum_view = (
                    TensorView(qk_psum)
                    .reshape_dim(1, [n_mm1_per_grp, atp.s_active_qh])
                    .select(1, i_mm1)
                    .slice(0, start=0, end=num_acc)
                )

                q_sb_view = (
                    TensorView(bufs.q_sb)
                    .reshape_dim(1, [atp.bs_full, atp.s_active_qh])
                    .select(1, (btc.global_batch_offset + i_b))
                )
                nisa.nc_matmul(
                    qk_psum_view.get_view(),
                    stationary=k_tile,  # mask k_sb
                    moving=q_sb_view.get_view(),
                )

            # Flush psum -> sb, the write to sb needs to be strided for batch interleaving
            num_acc_cpy = min(n_mm1_per_grp, fa_tile_s_prior // TC.p_max - i_mm1_grp * n_mm1_per_grp)

            if num_acc_cpy <= 0:
                break  # i_mm1_grp * n_mm1_per_grp is strictly increasing

            qk_psum_view = (
                TensorView(qk_psum).reshape_dim(1, [n_mm1_per_grp, atp.s_active_qh]).slice(1, start=0, end=num_acc_cpy)
            )

            sprior_tile_pos = i_mm1_grp * n_mm1_per_grp
            qk_sb_view = (
                TensorView(bufs.qk)
                .reshape_dim(1, [fa_tile_n_sprior, atp.bs, atp.s_active_qh])
                .slice(1, start=sprior_tile_pos, end=sprior_tile_pos + num_acc_cpy)
                .select(2, i_b)
            )

            mask_sb_view = (
                TensorView(bufs.mask_sb)
                .reshape_dim(1, [fa_tile_n_sprior, atp.bs, atp.s_active_qh])
                .slice(1, start=sprior_tile_pos, end=sprior_tile_pos + num_acc_cpy)
                .select(2, i_b)
            )
            if num_acc_cpy * atp.s_active_qh == 1:
                # Use tensor_copy_predicated due to select_reduce bug with free dim 1
                # Tracked in NKI-2209

                # Memset QK to -inf
                # This is necessary because tensor_copy_predicated only copies positions where mask=1,
                # leaving positions where mask=0 with stale values
                nisa.memset(qk_sb_view.get_view(), value=-np.inf)
                nisa.tensor_copy_predicated(
                    src=qk_psum_view.get_view(),
                    dst=qk_sb_view.get_view(),
                    predicate=mask_sb_view.get_view(),
                )
            else:
                nisa.select_reduce(
                    dst=qk_sb_view.get_view(),
                    predicate=mask_sb_view.get_view(),
                    on_true=qk_psum_view.get_view(),
                    on_false=-np.inf,
                )
        sbm.increment_section()
    sbm.close_scope()

    if DBG_TENSORS:
        if cfg.strided_mm1 and (atp.use_fa or atp.bs != atp.bs_per_nc):
            # strided_mm1 + FA has complex K column remapping — write zeros so the tensor is defined.
            # strided_mm1 + batch tiling: batch and sprior tiles are interleaved in QK buffer,
            # so per-tile slices don't concatenate to match the full-batch layout.
            sbm.open_scope()
            dbg_tile_offset = fa_ctx.fa_tile_idx * atp.fa_n_sprior_tile
            bqh_offset = btc.tile_batch_offset * atp.s_active_qh
            dbg_zero = sbm.alloc_stack((TC.p_max, 1), dtype=bufs.qk.dtype, buffer=nl.sbuf)
            nisa.memset(dbg_zero, 0.0)
            dbg_zero_bc = (
                TensorView(dbg_zero)
                .reshape_dim(1, [1, 1, 1, 1])
                .broadcast(2, fa_tile_n_sprior)
                .broadcast(4, atp.s_active_bqh)
            )
            nisa.dma_copy(
                bufs.DBG_QK[
                    :,
                    atp.sprior_prg_id,
                    dbg_tile_offset : dbg_tile_offset + fa_tile_n_sprior,
                    atp.bs_prg_id,
                    bqh_offset : bqh_offset + atp.s_active_bqh,
                ],
                dbg_zero_bc.get_view(),
                dge_mode=dge_mode.none,
                name=f"dbg_qk_store_zeros_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}",
            )
            sbm.close_scope()
        else:
            # For FA, copy to the slice corresponding to this FA tile
            dbg_tile_offset = fa_ctx.fa_tile_idx * atp.fa_n_sprior_tile
            bqh_offset = btc.tile_batch_offset * atp.s_active_qh
            nisa.dma_copy(
                bufs.DBG_QK[
                    :,
                    atp.sprior_prg_id,
                    dbg_tile_offset : dbg_tile_offset + fa_tile_n_sprior,
                    atp.bs_prg_id,
                    bqh_offset : bqh_offset + atp.s_active_bqh,
                ],
                bufs.qk.reshape((TC.p_max, 1, fa_tile_n_sprior, 1, atp.s_active_bqh)),
                dge_mode=dge_mode.none,
                name=f"dbg_qk_store_mm1_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}",
            )


def _cascaded_max_reduce(
    sink,
    DBG_TENSORS,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Step 2. Cascaded max reduce of KQ^T"""
    fa_tile_n_sprior = fa_ctx.tile_n_sprior

    bufs.qk_max = sbm.alloc_stack((TC.p_max, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf, align=4)

    # Step 2.1. Strided reduce from [p_max, tile_n_sprior * bs * s_active_qh] -> [p_max, bs * s_active_qh]
    if nisa.get_nc_version() <= nisa.nc_version.gen3:
        # This is small (e.g. if n=2, s_a=6, s_p=8192, then free dim is 64*12=768), reasonable to be done with one inst
        qk_view = TensorView(bufs.qk).reshape_dim(1, [fa_tile_n_sprior, atp.s_active_bqh]).permute([0, 2, 1])
        nisa.tensor_reduce(
            dst=bufs.qk_max, op=nl.maximum, data=qk_view.get_view(), axis=[2], keepdims=False
        )  # The axis is modified here
    else:
        # Split across DVE (first half) and ACT/Scalar Engine (second half) for parallelism
        s_active_bqh_half = atp.s_active_bqh // 2
        s_active_bqh_first_half = atp.s_active_bqh - s_active_bqh_half  # ceiling half handles odd s_active_bqh
        s_active_bqh_second_half = s_active_bqh_half

        # First half with DVE (tensor_reduce)
        qk_view = TensorView(bufs.qk).reshape_dim(1, [fa_tile_n_sprior, atp.s_active_bqh]).permute([0, 2, 1])
        qk_view_first_half = qk_view.slice(1, start=0, end=s_active_bqh_first_half)
        nisa.tensor_reduce(
            dst=bufs.qk_max[:, :s_active_bqh_first_half],
            op=nl.maximum,
            data=qk_view_first_half.get_view(),
            axis=[2],
            keepdims=False,
        )

        # Second half with ACT (activate2 with reduction accumulator)
        for i_s_active_bqh_half in nl.affine_range(s_active_bqh_second_half):
            qk_i_s_view = qk_view.select(1, s_active_bqh_first_half + i_s_active_bqh_half)
            nisa.activate2(
                dst=qk_i_s_view.get_view(),
                op=nl.copy,
                data=qk_i_s_view.get_view(),
                imm0=1.0,
                imm1=0.0,
                op0=nl.multiply,
                op1=nl.add,
                reduce_op=nl.max,
                reduce_res=bufs.qk_max[:, s_active_bqh_first_half + i_s_active_bqh_half],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )

    # Sink prep placement: only the classical per-tile-sync path stages sink into qk_max_buf on the
    # first FA tile (single-NC) and folds it into tile_max during the final reduction below.
    # Sharded paths defer cross-NC sync (and sink fold) to _finalize_and_store.
    should_prep_sink_in_cascade = sink is not None and atp.sync_softmax_per_fa_tile and fa_ctx.fa_tile_idx == 0

    # The free-dim length reserves slots in qk_max_buf (and exp_sum) for:
    #   - 1: local reduction result (always).
    #   - +1 for sink: only when sink is staged in _cascaded_* (per-tile path).
    atp.softmax_final_reduction_length = 1 + should_prep_sink_in_cascade
    atp.softmax_final_reduction_local_idx = 0  # The reduction result from local qk goes to 1st entry.
    atp.softmax_final_reduction_sink_idx = (
        atp.softmax_final_reduction_length - 1 if should_prep_sink_in_cascade else None
    )

    if cfg.use_gpsimd_sb2sb and atp.sprior_n_prgs > 1:
        # Extended instructions require input/output tensors have multiple of 16 partitions
        padded_qk_max_pdim = pad_partitions_for_ext_inst(atp.s_active_bqh_tile)
    else:
        padded_qk_max_pdim = atp.s_active_bqh_tile

    bufs.qk_max_buf = sbm.alloc_stack(
        (padded_qk_max_pdim, atp.n_bsq_tiles * atp.softmax_final_reduction_length),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )

    # Step 2.2 Transpose to psum -> [bs * s_active_qh, p_max]
    sbm.open_scope()
    for i_bsq_tile in range(atp.n_bsq_full_tiles):
        _transpose_max_psum(i_bsq_tile, atp.s_active_bqh_tile, atp, TC, bufs, sbm)

    if atp.s_active_bqh_remainder > 0:
        _transpose_max_psum(atp.n_bsq_full_tiles, atp.s_active_bqh_remainder, atp, TC, bufs, sbm)
    sbm.close_scope()

    # Step 2.3.1  If there is sink, load with the right layout.
    if should_prep_sink_in_cascade:
        # Stage sink into qk_max_buf for the per-tile final reduction below (single-NC path).
        sink_offset = atp.n_bsq_tiles * atp.softmax_final_reduction_sink_idx
        _prep_sink(
            sink,
            bufs.qk_max_buf[: atp.s_active_bqh_tile, nl.ds(sink_offset, atp.n_bsq_tiles)],
            atp,
            cfg,
            TC,
            sbm,
            btc,
        )
    # Step 2.3.3  Do the final reduction (2 or 3 reduce to 1) -> [bs * s_active_qh, 1]
    #             Negate if we are doing the reduction to save one op for sink exponential.
    # Do this only if syncing softmax per tile (not deferring to FA finalization)
    atp.max_negated = False
    tile_max = bufs.qk_max_buf[: atp.s_active_bqh_tile, : atp.n_bsq_tiles]
    if atp.softmax_final_reduction_length > 1 and atp.sync_softmax_per_fa_tile:
        atp.max_negated = True
        for i_bsq_tile in range(atp.n_bsq_tiles):
            qk_max_buf_view = (
                TensorView(bufs.qk_max_buf)
                .slice(0, start=0, end=atp.s_active_bqh_tile)
                .reshape_dim(1, [atp.softmax_final_reduction_length, atp.n_bsq_tiles])
                .select(2, i_bsq_tile)
            )
            nisa.tensor_reduce(
                tile_max[:, i_bsq_tile],
                data=qk_max_buf_view.get_view(),
                op=nl.maximum,
                axis=1,
                negate=True,
            )
    elif sink is not None and atp.use_fa and atp.sprior_n_prgs == 1:
        # need to negate in tile > 0 for consistency with 0th tile even though no sink
        atp.max_negated = True
        nisa.tensor_scalar(tile_max, tile_max, op0=nl.multiply, operand0=-1)

    _clamp_max_to_finite(dst=tile_max, src=tile_max, max_negated=atp.max_negated)

    # Step 2.3.4 Update running max (if online softmax is used — FA or sharded)
    if atp.use_online_softmax:
        _update_running_max(atp, sbm, bufs, fa_ctx)

    # Step 2.4. Tranpose and broadcast along pdim -> [128, bs * s_active_qh]
    # (Either running_max or qk_max_buf depending on whether online softmax is used)
    for i_bsq_tile in range(atp.n_bsq_full_tiles):
        _transpose_broadcast_max(i_bsq_tile, atp.s_active_bqh_tile, atp, TC, sbm, bufs)

    if atp.s_active_bqh_remainder > 0:
        _transpose_broadcast_max(atp.n_bsq_full_tiles, atp.s_active_bqh_remainder, atp, TC, sbm, bufs)

    if DBG_TENSORS and not atp.use_online_softmax and atp.bs == atp.bs_per_nc:
        # Non-online-softmax path (single-NC non-FA). qk_max_buf holds the final max; dump it
        # directly. Online-softmax paths dump from _finalize_and_store instead.
        local_slice = bufs.qk_max_buf[
            : atp.s_active_bqh_tile,
            nl.ds(atp.softmax_final_reduction_local_idx * atp.n_bsq_tiles, atp.n_bsq_tiles),
        ]
        _store_dbg_qk_max(local_slice, atp.max_negated, "cascaded", atp, TC, sbm, bufs)
    elif DBG_TENSORS and fa_ctx.is_last_fa_tile and btc.batch_tile_idx == 0 and atp.bs != atp.bs_per_nc:
        # Batch tiling active (offset-based writes into the debug tensor unreliable): write zeros
        # once on the first batch tile using the debug tensor's full-batch shape. The
        # online-softmax + full-batch case is handled by _finalize_and_store.
        _store_dbg_qk_max_zeros_full_batch(atp, sbm, bufs)


def _transpose_max_psum(
    index: int,
    tile_size: int,
    atp: AttnTileParams,
    TC: TileConstants,
    bufs: AttnInternalBuffers,
    sbm: SbufManager,
):
    """
    Step 2.2 Transpose to psum -> [bs * s_active_qh, p_max]
    Step 2.3.0 Reduce the new 128 fdim while copying to sbuf -> [bs * s_active_qh, 1]
    """
    # Step 2.2
    qk_max_psum = nl.ndarray(
        (tile_size, TC.p_max),
        dtype=atp.inter_type,
        buffer=nl.psum,
        address=None if sbm.is_auto_alloc() else (0, (index % TC.psum_b_max) * TC.psum_f_max_bytes),
    )
    nisa.nc_transpose(
        qk_max_psum,
        bufs.qk_max[:, nl.ds(index * atp.s_active_bqh_tile, tile_size)],
    )

    # Step 2.3.0
    nisa.tensor_reduce(
        bufs.qk_max_buf[
            :tile_size,
            atp.n_bsq_tiles * atp.softmax_final_reduction_local_idx + index,
        ],
        op=nl.maximum,
        data=qk_max_psum,
        axis=1,
        keepdims=True,
    )


def _transpose_broadcast_max(
    index,
    tile_size,
    atp: AttnTileParams,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """Step 2.4. Tranpose and broadcast along pdim -> [128, bs * s_active_qh]"""
    sbm.open_scope()
    qk_max_copy = sbm.alloc_stack((TC.p_max, tile_size), dtype=bufs.qk_max.dtype)

    # Use running max when online softmax is active (FA or sharded), else tile max
    if atp.use_online_softmax:
        max_src_tensor = bufs.running_max[:tile_size]
    else:
        max_src_tensor = bufs.qk_max_buf[:tile_size]

    tp_broadcast(
        src=max_src_tensor, dst=qk_max_copy, src_offset=index, psum_address=None if sbm.is_auto_alloc() else (0, 0)
    )
    nisa.tensor_copy(
        bufs.qk_max[:, nl.ds(index * atp.s_active_bqh_tile, tile_size)],
        qk_max_copy,
    )
    sbm.close_scope()


def _compute_exp_qk(
    DBG_TENSORS,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Step 3. Exp(KQ^T - max(KQ^T))"""
    fa_tile_n_sprior = fa_ctx.tile_n_sprior

    # Instruction startup time on TRN2 does not outweight pipelining advantages
    if nisa.get_nc_version() >= nisa.nc_version.gen4:
        for i_s_prior in range(fa_tile_n_sprior):
            qk_view = (
                TensorView(bufs.qk)
                .reshape_dim(1, [fa_tile_n_sprior, atp.s_active_bqh])
                .slice(1, start=i_s_prior, end=i_s_prior + 1)
            )

            nisa.tensor_tensor(
                qk_view.get_view(), qk_view.get_view(), bufs.qk_max, op=(nl.add if atp.max_negated else nl.subtract)
            )

            qk_io_type_view = (
                TensorView(bufs.qk_io_type)
                .reshape_dim(1, [fa_tile_n_sprior, atp.s_active_bqh])
                .slice(1, start=i_s_prior, end=i_s_prior + 1)
            )
            nisa.activation(qk_io_type_view.get_view(), op=nl.exp, data=qk_view.get_view())
    else:
        qk_max_view = TensorView(bufs.qk_max).expand_dim(1).broadcast(1, fa_tile_n_sprior)

        nisa.tensor_tensor(
            bufs.qk,
            bufs.qk,
            qk_max_view.get_view(),
            op=(nl.add if atp.max_negated else nl.subtract),
        )
        nisa.activation(bufs.qk_io_type, op=nl.exp, data=bufs.qk)

    if DBG_TENSORS and not atp.use_online_softmax and not (cfg.strided_mm1 and atp.bs != atp.bs_per_nc):
        # Skip when online softmax is used (exp is relative to a per-tile/local max, not a stable
        # global max) and for strided_mm1 + batch tiling (same interleaving issue as DBG_QK).
        bqh_offset = btc.tile_batch_offset * atp.s_active_qh
        nisa.dma_copy(
            bufs.DBG_QK_EXP[
                :,
                atp.sprior_prg_id,
                :,
                atp.bs_prg_id,
                bqh_offset : bqh_offset + atp.s_active_bqh,
            ],
            bufs.qk_io_type.reshape((TC.p_max, 1, fa_tile_n_sprior, 1, atp.s_active_bqh)),
            dge_mode=dge_mode.none,
            name=f"dbg_qk_exp_store_bt{btc.batch_tile_idx}",
        )
    elif DBG_TENSORS and fa_ctx.is_last_fa_tile and btc.batch_tile_idx == 0:
        # Online softmax uses running max so qk_io_type values aren't meaningful, but the debug tensor
        # must still be written to avoid a compiler error. Only write on first batch tile; use debug
        # tensor's full-batch bqh dimension.
        sbm.open_scope()
        full_bqh = bufs.DBG_QK_EXP.shape[-1]
        dbg_zero = sbm.alloc_stack((TC.p_max, 1), dtype=bufs.qk_io_type.dtype, buffer=nl.sbuf)
        nisa.memset(dbg_zero, 0.0)
        dbg_zero_bc = (
            TensorView(dbg_zero).reshape_dim(1, [1, 1, 1, 1]).broadcast(2, atp.n_sprior_tile).broadcast(4, full_bqh)
        )
        nisa.dma_copy(
            bufs.DBG_QK_EXP[:, atp.sprior_prg_id, :, atp.bs_prg_id, :],
            dbg_zero_bc.get_view(),
            dge_mode=dge_mode.none,
            name="dbg_qk_exp_store_zeros",
        )
        sbm.close_scope()


def _cascaded_sum_reduction(
    sink,
    DBG_TENSORS,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Step 4. Cascaded sum reduction of exp"""
    fa_tile_n_sprior = fa_ctx.tile_n_sprior

    if cfg.use_gpsimd_sb2sb and atp.sprior_n_prgs > 1:
        # Extended instructions require input/output tensors have multiple of 16 partitions
        padded_exp_sum_pdim = pad_partitions_for_ext_inst(atp.s_active_bqh_tile)
    else:
        padded_exp_sum_pdim = atp.s_active_bqh_tile
    bufs.exp_sum = sbm.alloc_stack(
        (padded_exp_sum_pdim, atp.n_bsq_tiles * atp.softmax_final_reduction_length),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )
    if not atp.use_online_softmax:
        # When online softmax is active, reciprocal is applied in _finalize_and_store
        bufs.exp_sum_recip = sbm.alloc_stack((TC.p_max, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf)

    sbm.open_scope()
    for i_bsq_tile in range(atp.n_bsq_full_tiles):
        _tile_sum_reduction(i_bsq_tile, atp.s_active_bqh_tile, fa_tile_n_sprior, atp, TC, bufs, sbm)
    sbm.close_scope()

    if atp.s_active_bqh_remainder > 0:
        sbm.open_scope()
        _tile_sum_reduction(atp.n_bsq_full_tiles, atp.s_active_bqh_remainder, fa_tile_n_sprior, atp, TC, bufs, sbm)
        sbm.close_scope()

    if sink is not None and atp.sync_softmax_per_fa_tile and fa_ctx.fa_tile_idx == 0:
        kernel_assert(
            atp.max_negated,
            "Internal error: Unexpectedly found that maximum has not been negated when using sink",
        )
        kernel_assert(
            atp.softmax_final_reduction_sink_idx is not None,
            "Internal error: Unexpectedly found that softmax_final_reduction_sink_idx is None",
        )
        reduction_offset = atp.n_bsq_tiles * atp.softmax_final_reduction_sink_idx
        for i_bsq_tile in range(atp.n_bsq_tiles):
            # Use running max when online softmax is active (FA single-NC), else tile max from qk_max_buf
            max_buf_for_sink = bufs.running_max if atp.use_online_softmax else bufs.qk_max_buf
            nisa.tensor_scalar(
                bufs.qk_max_buf[: atp.s_active_bqh_tile, reduction_offset + i_bsq_tile],
                bufs.qk_max_buf[: atp.s_active_bqh_tile, reduction_offset + i_bsq_tile],
                nl.add,
                max_buf_for_sink[: atp.s_active_bqh_tile, i_bsq_tile],
            )
            nisa.activation(
                bufs.exp_sum[: atp.s_active_bqh_tile, reduction_offset + i_bsq_tile],
                nl.exp,
                bufs.qk_max_buf[: atp.s_active_bqh_tile, reduction_offset + i_bsq_tile],
            )

    if atp.softmax_final_reduction_length > 1 and atp.sync_softmax_per_fa_tile:
        for i_bsq_tile in range(atp.n_bsq_tiles):
            exp_sum_view = (
                TensorView(bufs.exp_sum)
                .slice(0, start=0, end=atp.s_active_bqh_tile)
                .reshape_dim(1, [atp.softmax_final_reduction_length, atp.n_bsq_tiles])
                .select(2, i_bsq_tile)
            )
            nisa.tensor_reduce(
                bufs.exp_sum[: atp.s_active_bqh_tile, i_bsq_tile],
                data=exp_sum_view.get_view(),
                op=nl.add,
                axis=1,
            )

    if atp.use_online_softmax:
        _update_running_sum(atp, sbm, bufs, fa_ctx)

    if DBG_TENSORS and fa_ctx.is_last_fa_tile and atp.bs == atp.bs_per_nc and not atp.use_online_softmax:
        # Non-online-softmax path (single-NC non-FA). exp_sum holds the final sum; dump it directly.
        # Online-softmax paths dump from _finalize_and_store where running_sum is the
        # globally-synced sum.
        local_slice = bufs.exp_sum[
            : atp.s_active_bqh_tile,
            nl.ds(atp.softmax_final_reduction_local_idx * atp.n_bsq_tiles, atp.n_bsq_tiles),
        ]
        _store_dbg_exp_sum(local_slice, "cascaded", atp, TC, sbm, bufs)
    elif DBG_TENSORS and fa_ctx.is_last_fa_tile and btc.batch_tile_idx == 0 and atp.bs != atp.bs_per_nc:
        # Batch tiling active: write zeros with full-batch shape so the tensor is defined.
        _store_dbg_exp_sum_zeros_full_batch(atp, sbm, bufs)

    # Skip reciprocal when online softmax is active — reciprocal is applied in _finalize_and_store.
    if atp.use_online_softmax:
        return

    # Take sum recip, transpose and broadcast on pdim
    nisa.reciprocal(
        bufs.exp_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
        bufs.exp_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
    )

    _s_active_bqh_tile_transpose_broadcast(bufs.exp_sum, bufs.exp_sum_recip, atp, TC)


def _tile_sum_reduction(
    index, tile_size, tile_n_sprior, atp: AttnTileParams, TC: TileConstants, bufs: AttnInternalBuffers, sbm: SbufManager
):
    """
    Step 4.1. Each of the tile_n_sprior matmult reduces one tile of qk[128(P), 1, s] -> [s, 1]
    Step 4.2. Copy partial reduce output from psum -> sb while reducing the free dim (num_sprior_t128)
    tile_size is either atp.s_active_bqh_tile or atp.s_active_bqh_remainder
    """
    sum_reduce_psum = nl.ndarray(
        (tile_size, tile_n_sprior),
        dtype=nl.float32,
        buffer=nl.psum,
        address=None if sbm.is_auto_alloc() else (0, (index % TC.psum_b_max) * TC.psum_f_max_bytes),
    )

    # Step 4.1. Each of the tile_n_sprior matmult reduces one tile of qk[128(P), 1, s] -> [s, 1]
    for i_exp_reduce in range(tile_n_sprior):
        sum_reduce_psum_view = TensorView(sum_reduce_psum).slice(1, start=i_exp_reduce, end=i_exp_reduce + 1)
        s_active_bqh_pos = index * atp.s_active_bqh_tile
        qk_io_type_view = (
            TensorView(bufs.qk_io_type)
            .reshape_dim(1, [tile_n_sprior, atp.s_active_bqh])
            .select(1, i_exp_reduce)
            .slice(1, start=s_active_bqh_pos, end=s_active_bqh_pos + tile_size)
        )

        nisa.nc_matmul(
            sum_reduce_psum_view.get_view(),
            stationary=qk_io_type_view.get_view(),
            moving=bufs.one_vec,
        )

    # Step 4.2. Copy partial reduce output from psum -> sb while reducing the free dim (num_sprior_t128)
    nisa.tensor_reduce(
        bufs.exp_sum[
            :tile_size,
            atp.n_bsq_tiles * atp.softmax_final_reduction_local_idx + index,
        ],
        op=nl.add,
        data=sum_reduce_psum,
        axis=1,
    )


def _column_tile_transpose(src, dst, index, tile_size, tile_stride, TC: TileConstants):
    """Transpose a column tile to a row and place it at the correct offset in dst.

    Transposes src[0:tile_size, index:index+1] to dst[0:1, base_offset:base_offset+tile_size]
    where base_offset = index * tile_stride.

    Args:
        src: Source tensor with shape [tile_stride, num_tiles]
        dst: Destination tensor with shape [1, total_size] or broadcastable
        index: Which column tile to transpose (0-indexed)
        tile_size: Number of elements in this tile (may be less than tile_stride for remainder)
        tile_stride: Stride between tiles in the output
    """
    base_offset = index * tile_stride
    for quadrant_idx in range(div_ceil(tile_size, TC.sbuf_quadrant_size)):
        offset = quadrant_idx * TC.sbuf_quadrant_size
        full_offset = base_offset + offset
        tp_size = min(TC.sbuf_quadrant_size, tile_size - offset)
        # Even though TP on vector engine is slower, the vector engine is not busy while the tensor engine is
        nisa.nc_transpose(
            dst[:1, full_offset : full_offset + tp_size],
            src[offset : offset + tp_size, index : index + 1],
            engine=nisa.vector_engine,
        )


def _s_active_bqh_tile_transpose_broadcast(src, dst, atp: AttnTileParams, TC: TileConstants):
    """Transpose all tiles from src and broadcast to dst.

    Transposes src with shape [s_active_bqh_tile, n_bsq_tiles] to [1,s_active_bqh]
    and then broadcast to dst with shape [d_head, s_active_bqh].

    Args:
        src: Source tensor with shape [s_active_bqh_tile, n_bsq_tiles]
        dst: Destination tensor with shape [d_head, s_active_bqh]
    """
    for i_bsq_tile in range(atp.n_bsq_full_tiles):
        _column_tile_transpose(src, dst, i_bsq_tile, atp.s_active_bqh_tile, atp.s_active_bqh_tile, TC)
    if atp.s_active_bqh_remainder > 0:
        _column_tile_transpose(src, dst, atp.n_bsq_full_tiles, atp.s_active_bqh_remainder, atp.s_active_bqh_tile, TC)
    stream_shuffle_broadcast(src=dst[:1, : atp.s_active_bqh], dst=dst)


def _compute_pv_matmul_and_store(
    v_prior,
    v_active,
    out,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Step 5. Matmult 2 of (exp @ V)^T and store output"""
    fa_tile_s_prior = fa_ctx.tile_s_prior
    fa_tile_n_sprior = fa_ctx.tile_n_sprior
    fa_tile_offset = fa_ctx.tile_offset
    is_last_fa_tile = fa_ctx.is_last_fa_tile

    exp_v_sendrecv_gpsimd = cfg.use_gpsimd_sb2sb and atp.sprior_n_prgs > 1 and atp.bs * atp.s_active_qh <= 128
    if exp_v_sendrecv_gpsimd:
        # Extended instructions require input/output tensors have multiple of 16 partitions
        padded_exp_v_pdim = pad_partitions_for_ext_inst(cfg.d_head)
    else:
        padded_exp_v_pdim = cfg.d_head
    bufs.exp_v = sbm.alloc_stack(
        (padded_exp_v_pdim, atp.bs, atp.s_active_qh),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )

    batch_interleave_degree_safe = _get_safe_batch_interleave_degree(
        cfg.d_head * fa_tile_n_sprior * sizeinbytes(v_prior.dtype), atp.batch_interleave_degree, sbm
    )

    """
    Tiling Strategy for MM2 ((exp @ V)^T computation and output):
    - V stationary: [s_prior, d_head] loaded per batch into v_sb as [P_MAX, n_sprior_tile * d_head]
    - exp(QK) moving: [P_MAX, s_active_bqh] from qk_io_type (already computed and normalized)
    - Output: exp_v [d_head, bs, s_active_qh] accumulated in PSUM then copied to SBUF
    - PSUM allocation: [d_head, s_active_qh] per batch
    - Memory layout: V loaded horizontally tiled (strided if strided_mm1=False, sequential if True)
    - Batch interleaving: Uses batch_interleave_degree_safe for DMA/compute overlap
    - Final output: Gathered across cores if sprior_n_prgs > 1, then stored to HBM or kept in SBUF
    """

    sbm.open_scope(interleave_degree=batch_interleave_degree_safe, name="pv_matmul")
    for i_b in range(atp.bs):
        # Load V_prior from HBM [s_prior, d_head] into SB [128, (tile_s_prior / 128) * d_head]
        # Do strided load (horizontal tile) if not strided_mm1, otherwise load sequentially for better DMA throughput
        # FP8 KV: use the resolved FP8 dtype (caller's concrete dtype, or
        # dtype_mode resolution for opaque "float8e4"). Otherwise pass through v_prior.dtype.
        _v_sb_dtype = atp.kv_e4m3_tile_dtype if atp.is_fp8_kv else v_prior.dtype
        v_sb = sbm.alloc_stack(
            (TC.p_max, cfg.d_head * fa_tile_n_sprior),
            dtype=_v_sb_dtype,
            buffer=nl.sbuf,
        )
        v_sb_view = TensorView(v_sb).reshape_dim(1, [fa_tile_n_sprior, cfg.d_head])
        if atp.is_block_kv:
            # For FA, compute which folds correspond to this tile
            fold_s_prior = atp.block_len * TC.p_max
            fold_start = fa_tile_offset // fold_s_prior
            fold_end = div_ceil(fa_tile_offset + fa_tile_s_prior, fold_s_prior)
            num_folds_this_tile = fold_end - fold_start

            if atp.use_v_dma_skipping:
                # This memset is required for oob skip to prevent uninitialized NaNs from corrupting results
                nisa.memset(v_sb, value=0)

            sbm.open_scope()
            for i_fold_rel in range(num_folds_this_tile):
                i_fold = fold_start + i_fold_rel
                # FIXME: This should be a slice, but needs to be an AP because of indirect DMA lowering
                cur_blks = (
                    TensorView(bufs.active_blocks_sb if atp.use_v_dma_skipping else bufs.active_blocks_sb_u32)
                    .reshape_dim(1, [atp.num_folds_per_batch, atp.bs])
                    .select(1, i_fold_rel)
                    .slice(1, start=i_b, end=i_b + 1)
                    .get_view()
                )
                kernel_assert(
                    cur_blks.shape == (TC.p_max, 1),
                    f"Internal error: unexpected shape error after loading current blocks, expected {(TC.p_max, 1)}, got {cur_blks.shape}.",
                )
                nisa.dma_copy(
                    dst=v_sb[
                        :,
                        nl.ds(
                            i_fold_rel * atp.block_len * cfg.d_head,
                            atp.block_len * cfg.d_head,
                        ),
                    ],
                    # TODO: Port to TensorView once dynamic vector_offset is supported
                    src=bufs.v_prior_reshaped.ap(
                        [
                            [atp.block_len * cfg.d_head, TC.p_max],
                            [1, atp.block_len * cfg.d_head],
                        ],
                        offset=0,
                        vector_offset=cur_blks,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.skip if atp.use_v_dma_skipping else oob_mode.error,
                    name=f"v_prior_block_load_indirect_fa{fa_ctx.fa_tile_idx}_b{i_b}_f{i_fold}_bt{btc.batch_tile_idx}",
                )
            sbm.close_scope()
        elif cfg.strided_mm1:
            s_prior_pos = fa_tile_offset
            v_prior_view = (
                TensorView(v_prior)
                .select(0, btc.global_batch_offset + i_b)
                .squeeze_dim(0)
                .slice(0, start=s_prior_pos, end=s_prior_pos + (TC.p_max * fa_tile_n_sprior))
                .reshape_dim(0, [TC.p_max, fa_tile_n_sprior])
            )
            nisa.dma_copy(
                v_sb_view.get_view(),
                v_prior_view.get_view(),
                dge_mode=dge_mode.none,
                name=f"{sbm.get_name_prefix()}v_prior_load_strided_mm1_fa{fa_ctx.fa_tile_idx}_b{i_b}_bt{btc.batch_tile_idx}",
            )
        else:
            s_prior_pos = fa_tile_offset
            v_prior_view = (
                TensorView(v_prior)
                .select(0, btc.global_batch_offset + i_b)
                .squeeze_dim(0)
                .slice(0, start=s_prior_pos, end=s_prior_pos + (TC.p_max * fa_tile_n_sprior))
                .reshape_dim(0, [fa_tile_n_sprior, TC.p_max])
                .permute((1, 0, 2))
            )
            nisa.dma_copy(
                dst=v_sb_view.get_view(),
                src=v_prior_view.get_view(),
                dge_mode=dge_mode.none,
                name=f"{sbm.get_name_prefix()}v_prior_load_sequential_fa{fa_ctx.fa_tile_idx}_b{i_b}_bt{btc.batch_tile_idx}",
            )

        # Load V_active to the last portion if needed (only on last FA tile)
        if atp.sprior_prg_id == atp.sprior_n_prgs - 1 and is_last_fa_tile:
            if atp.is_block_kv:
                num_blks_covering_s_active = div_ceil(cfg.s_active, atp.block_len)
                extra_covered = num_blks_covering_s_active * atp.block_len - cfg.s_active

                v_sb_partition_base = TC.p_max - num_blks_covering_s_active
                v_sb_s_prior_base = (num_folds_this_tile - 1) * atp.block_len

                v_active_reshaped_batch_pos = btc.global_batch_offset + i_b

                # Need to mask as dim_0 * blk_len + dim_1 >= extra_covered
                # Solving the above inequality with 0 <= dim_0 < num_blks_covering_s_active and 0 <= dim_1 < blk_len
                # we get (dim_0, dim_1) in {(0,[extra_covered, blk_len)) and ([1, num_blks_covering_s_active), [0, blk_len))}
                # Thus, if extra_covered != 0, we do an access pattern for dim_0 == 0 and dim_1 in [extra_covered, blk_len)
                # and for main copy we don't need any restrictions
                if extra_covered > 0:
                    if atp.block_len > extra_covered:
                        v_sb_view = (
                            TensorView(v_sb)
                            .slice(0, start=v_sb_partition_base, end=v_sb_partition_base + 1)
                            .reshape_dim(1, [fa_tile_n_sprior, cfg.d_head])
                            .slice(1, start=v_sb_s_prior_base + extra_covered, end=v_sb_s_prior_base + atp.block_len)
                        )

                        v_active_reshaped_view = (
                            TensorView(bufs.v_active_reshaped)
                            .slice(0, start=v_active_reshaped_batch_pos, end=v_active_reshaped_batch_pos + 1)
                            .reshape_dim(1, [cfg.s_active, cfg.d_head])
                            .slice(1, start=0, end=atp.block_len - extra_covered)
                        )
                        nisa.dma_copy(
                            dst=v_sb_view.get_view(),
                            src=v_active_reshaped_view.get_view(),
                            dge_mode=dge_mode.none,
                            name=f"v_active_block_load_partial_rows_b{i_b}_bt{btc.batch_tile_idx}",
                        )
                    if num_blks_covering_s_active > 1:
                        v_sb_view = (
                            TensorView(v_sb)
                            .slice(
                                0, start=v_sb_partition_base + 1, end=v_sb_partition_base + num_blks_covering_s_active
                            )
                            .reshape_dim(1, [fa_tile_n_sprior, cfg.d_head])
                            .slice(1, start=v_sb_s_prior_base, end=v_sb_s_prior_base + atp.block_len)
                        )

                        s_active_pos = atp.block_len - extra_covered
                        v_active_reshaped_view = (
                            TensorView(bufs.v_active_reshaped)
                            .select(
                                0,
                                v_active_reshaped_batch_pos,
                            )
                            .reshape_dim(0, [cfg.s_active, cfg.d_head])
                            .slice(
                                0,
                                start=s_active_pos,
                                end=s_active_pos + atp.block_len * (num_blks_covering_s_active - 1),
                            )
                            .reshape_dim(0, [(num_blks_covering_s_active - 1), atp.block_len])
                        )
                        nisa.dma_copy(
                            dst=v_sb_view.get_view(),
                            src=v_active_reshaped_view.get_view(),
                            dge_mode=dge_mode.none,
                            name=f"v_active_block_load_remaining_blocks_b{i_b}_bt{btc.batch_tile_idx}",
                        )
                else:
                    v_sb_view = (
                        TensorView(v_sb)
                        .slice(0, start=v_sb_partition_base, end=v_sb_partition_base + num_blks_covering_s_active)
                        .reshape_dim(1, [fa_tile_n_sprior, cfg.d_head])
                        .slice(1, start=v_sb_s_prior_base, end=v_sb_s_prior_base + atp.block_len)
                    )

                    v_active_reshaped_view = (
                        TensorView(bufs.v_active_reshaped)
                        .select(
                            0,
                            v_active_reshaped_batch_pos,
                        )
                        .reshape_dim(0, [cfg.s_active, cfg.d_head])
                        .slice(0, start=0, end=atp.block_len * num_blks_covering_s_active)
                        .reshape_dim(0, [num_blks_covering_s_active, atp.block_len])
                    )

                    nisa.dma_copy(
                        dst=v_sb_view.get_view(),
                        src=v_active_reshaped_view.get_view(),
                        dge_mode=dge_mode.none,
                        name=f"v_active_block_load_full_b{i_b}_bt{btc.batch_tile_idx}",
                    )
            elif cfg.strided_mm1:
                # Need to load V_active in a strided manner across the entire free dim, this requires two loads because we need
                # to load s_active rows of d_head into v_sb, which has free dim of (tile_n_sprior * d_head).
                load1_nrows = cfg.s_active % fa_tile_n_sprior
                load2_nrows = cfg.s_active - load1_nrows

                # Load 1. Load the first (s_active % tile_n_sprior) rows of V_active (less than one row in v_sb)
                if load1_nrows > 0:
                    load1_pidx = TC.p_max - (load2_nrows // fa_tile_n_sprior) - 1

                    v_sb_view = (
                        TensorView(v_sb)
                        .slice(0, start=load1_pidx, end=load1_pidx + 1)
                        .reshape_dim(1, [fa_tile_n_sprior, cfg.d_head])
                        .slice(1, start=fa_tile_n_sprior - load1_nrows, end=fa_tile_n_sprior)
                    )

                    v_active_view = (
                        TensorView(v_active).select(0, btc.global_batch_offset + i_b).slice(1, start=0, end=load1_nrows)
                    )

                    nisa.dma_copy(
                        v_sb_view.get_view(),
                        v_active_view.get_view(),
                        dge_mode=dge_mode.none,
                        name=f"{sbm.get_name_prefix()}v_active_strided_load_partial_b{i_b}_bt{btc.batch_tile_idx}",
                    )

                # Load 2. Load the remaining rows
                if load2_nrows > 0:
                    load2_pidx = TC.p_max - (load2_nrows // fa_tile_n_sprior)

                    v_sb_view = (
                        TensorView(v_sb)
                        .slice(0, start=load2_pidx, end=load2_pidx + load2_nrows // fa_tile_n_sprior)
                        .reshape_dim(1, [fa_tile_n_sprior, cfg.d_head])
                    )

                    v_active_view = (
                        TensorView(v_active)
                        .select(0, btc.global_batch_offset + i_b)
                        .squeeze_dim(0)
                        .slice(0, start=load1_nrows, end=load1_nrows + load2_nrows)
                        .reshape_dim(0, [load2_nrows // fa_tile_n_sprior, fa_tile_n_sprior])
                    )

                    nisa.dma_copy(
                        v_sb_view.get_view(),
                        v_active_view.get_view(),
                        dge_mode=dge_mode.none,
                        name=f"{sbm.get_name_prefix()}v_active_strided_load_remaining_b{i_b}_bt{btc.batch_tile_idx}",
                    )
            else:
                v_active_view = TensorView(v_active).select(0, btc.global_batch_offset + i_b).squeeze_dim(0)
                # Load to the bottom right part of last chunk of v_sb: [s_active, d_head]
                nisa.dma_copy(
                    v_sb[TC.p_max - cfg.s_active :, v_sb.shape[1] - cfg.d_head :],
                    v_active_view.get_view(),
                    dge_mode=dge_mode.none,
                    name=f"v_active_load_sequential_b{i_b}_bt{btc.batch_tile_idx}",
                )

        # Perform V^T @ exp^T, which equals to (exp @ V)^T. Recall mm1 output is transposed - KQ^T
        exp_v_psum = nl.ndarray(
            (cfg.d_head, atp.s_active_qh),
            dtype=nl.float32,
            buffer=nl.psum,
            address=None if sbm.is_auto_alloc() else (0, (i_b % batch_interleave_degree_safe) * TC.psum_f_max_bytes),
        )
        for i_t in range(fa_tile_n_sprior):
            v_sb_view = TensorView(v_sb).reshape_dim(1, [fa_tile_n_sprior, cfg.d_head]).select(1, i_t)
            batch_s_active_qh_pos = i_b * atp.s_active_qh
            qk_io_type_view = (
                TensorView(bufs.qk_io_type)
                .reshape_dim(1, [fa_tile_n_sprior, atp.s_active_bqh])
                .select(1, i_t)
                .slice(1, start=batch_s_active_qh_pos, end=batch_s_active_qh_pos + atp.s_active_qh)
            )
            nisa.nc_matmul(
                exp_v_psum,
                stationary=v_sb_view.get_view(),
                moving=qk_io_type_view.get_view(),
            )

        # Copy mm2 output from psum -> sb while multiplying recip(sum)
        # When online softmax is active, we don't multiply by recip(sum) here — that's done in finalize.
        # exp_sum_recip = exp_sum_recip.reshape((p_max, bs, s_active_qh))
        exp_v_view = TensorView(bufs.exp_v).select(1, i_b).slice(0, start=0, end=cfg.d_head)
        if atp.use_online_softmax:
            # Online softmax: just copy without multiplying by recip(sum)
            nisa.tensor_copy(exp_v_view.get_view(), exp_v_psum)
        else:
            exp_sum_recip_view = (
                TensorView(bufs.exp_sum_recip)
                .reshape_dim(1, [atp.bs, atp.s_active_qh])
                .select(1, i_b)
                .slice(0, start=0, end=cfg.d_head)
            )
            # FIXME: SCAN ALL SLICING!!!
            nisa.tensor_tensor(exp_v_view.get_view(), exp_v_psum, exp_sum_recip_view.get_view(), op=nl.multiply)
        sbm.increment_section()
    sbm.close_scope()

    # When online softmax is active, accumulate the tile output into running_output and defer the
    # cross-NC gather+store to _finalize_and_store. Otherwise, the per-tile reciprocal above has
    # already normalized exp_v — gather-and-store it directly.
    if atp.use_online_softmax:
        _accumulate_output(atp, cfg, TC, sbm, bufs, fa_ctx)
    else:
        _gather_and_store_output(out, bufs.exp_v, exp_v_sendrecv_gpsimd, atp, cfg, sbm, btc)


def _gather_and_store_output(
    out: nl.ndarray,
    res: nl.ndarray,
    exp_v_sendrecv_gpsimd: bool,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    sbm: SbufManager,
    btc: BatchTileContext,
):
    """Gather partial results from other NC if sharded, then store output to HBM/SBUF.

    Args:
        out: Output tensor in HBM/SBUF
        res: Result tensor in SBUF with shape [d_head, s_active_bqh]
        btc: Batch tile context for correct output offset
    """
    sbm.open_scope()
    # Gather and add partial results from other NC if sprior is sharded
    if atp.sprior_n_prgs > 1:
        res_recv = sbm.alloc_stack(res.shape, res.dtype, buffer=nl.sbuf)
        nisa.sendrecv(
            src=res,
            dst=res_recv,
            send_to_rank=(1 - atp.sprior_prg_id),
            recv_from_rank=(1 - atp.sprior_prg_id),
            pipe_id=0,
            dma_engine=dma_engine.gpsimd_dma if exp_v_sendrecv_gpsimd else dma_engine.dma,
        )
        # Only NC0 adds partial results, unless we have out_in_sb then both cores will obtain the result
        if cfg.out_in_sb or (atp.sprior_prg_id == 0):
            nisa.tensor_tensor(res, res, res_recv, op=nl.add)

    # Store to output
    if cfg.out_in_sb:
        # exp_v and out may have different dtype, easier to just always keep this tensor copy to do the conversion
        res_reshaped = res.reshape((cfg.d_head, atp.s_active_bqh))

        out_offset = btc.global_batch_offset * atp.s_active_qh
        nisa.tensor_copy(out[0 : cfg.d_head, nl.ds(out_offset, atp.s_active_bqh)], src=res_reshaped)
        if atp.bs_n_prgs > 1:
            dst_bs_offset = ((1 - atp.bs_prg_id) * atp.bs_per_nc + btc.tile_batch_offset) * atp.s_active_qh
            nisa.sendrecv(
                src=out[0 : cfg.d_head, nl.ds(out_offset, atp.s_active_bqh)],
                dst=out[0 : cfg.d_head, nl.ds(dst_bs_offset, atp.s_active_bqh)],
                send_to_rank=(1 - atp.bs_prg_id),
                recv_from_rank=(1 - atp.bs_prg_id),
                pipe_id=0,
                dma_engine=dma_engine.gpsimd_dma if exp_v_sendrecv_gpsimd else dma_engine.dma,
            )
    else:
        # Save exp_v (output) into DRAM (each NC writes its own batches in case of batch sharded)
        # This needs to be strided save due to different layout in SBUF and DRAM:
        #   SBUF: [d_head, s_active * n_qhead_per_kvhead]
        #   DRAM: [n_qhead_per_kvhead, d_head, s_active]
        if atp.sprior_prg_id == 0:
            res_reshaped = res.reshape((cfg.d_head, atp.bs, cfg.q_head, cfg.s_active))
            batch_pos = btc.global_batch_offset
            out_view = (
                TensorView(out)  # [B, H, d, S_active]
                .slice(0, start=batch_pos, end=batch_pos + atp.bs)
                .permute((2, 0, 1, 3))
            )

            nisa.dma_copy(
                dst=out_view.get_view(),
                src=res_reshaped,
                dge_mode=dge_mode.none,
                name=f"out_store_hbm_bt{btc.batch_tile_idx}",
            )
    sbm.close_scope()


"""
Sharding Logic
"""


def _get_lnc_sharding(cfg: AttnTKGConfig) -> Tuple[int, int, int, int]:
    """
    Returns sharding parameters for context length (s_prior) and batch (bs) based on configuration.
    """
    n_prgs, prg_id = nl.num_programs(0), nl.program_id(0)
    kernel_assert(
        n_prgs <= 2,
        f"Attention cascaded supports unsharded or LNC2 sharded; but got a spmd grid size of {n_prgs}",
    )

    sprior_n_prgs, sprior_prg_id, bs_n_prgs, bs_prg_id = (1, 0, 1, 0)
    if n_prgs > 1:
        TILE_CONSTANTS = TileConstants.get_tile_constants()
        if is_batch_sharded(cfg.bs, cfg.q_head, cfg.s_active, cfg.curr_sprior, TILE_CONSTANTS.p_max, cfg.fuse_rope):
            bs_n_prgs, bs_prg_id = (n_prgs, prg_id)
        elif is_s_prior_sharded(
            cfg.bs, cfg.q_head, cfg.s_active, cfg.curr_sprior, TILE_CONSTANTS.p_max, cfg.fuse_rope
        ):  # If s_prior is small, and batch is not divisible by lnc
            sprior_n_prgs, sprior_prg_id = (n_prgs, prg_id)

    return sprior_n_prgs, sprior_prg_id, bs_n_prgs, bs_prg_id


def _get_interleaved_fa_tile_offset(fa_tile_idx: int, atp: AttnTileParams) -> int:
    """Compute global s_prior offset for an interleaved FA tile.

    FA tiles alternate between NC0 and NC1 for better DMA skipping load balance.
    Example with s_prior=40K, fa_tile_size=8K (each NC gets 20K = 3 tiles of 8K, 8K, 4K):

        NC0: tiles [0-8K], [16K-24K], [32K-36K]
        NC1: tiles [8K-16K], [24K-32K], [36K-40K]

    NC1 always gets the last global tile (which loads active tokens).

    Non-last tiles: offset = (2 * local_idx + prg_id) * fa_tile_size
    Last tile: offset = (num_tiles - 1) * 2 * fa_tile_size + prg_id * last_tile_size
    """
    if fa_tile_idx < atp.num_fa_tiles - 1:
        return (2 * fa_tile_idx + atp.sprior_prg_id) * atp.fa_tile_s_prior
    else:
        last_tile_size = atp.s_prior - (atp.num_fa_tiles - 1) * atp.fa_tile_s_prior
        return (atp.num_fa_tiles - 1) * 2 * atp.fa_tile_s_prior + atp.sprior_prg_id * last_tile_size


"""
RoPE
"""


def _apply_rope(
    x_inp,
    cos,
    sin,
    x_embed,
    cfg: AttnTKGConfig,
    ignore_heads: bool = False,
    sbm: SbufManager = None,
    name_suffix: str = "",
):
    """Applies rotary embedding for x following this algorithm:
      def _rotate_half(x) -> Tensor:
        '''Rotates half the hidden dims of the input.'''
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

      x_embed = (x * cos) + (_rotate_half(x) * sin)

    Args:
      x_input: HBM input memloc, shape [bs, n_head, s_active, d_head]
      cos: SB input memloc, shape [par(d_head), bs * s_active]
      sin: SB input memloc, shape [par(d_head), bs * s_active]
      cfg: attention tokengen config, including shapes and optimization configs
      ignore_heads: For some inputs (e.g. K active), ignore the head dim in cfg
      sbm: SbufManager of calling kernel

    Returns:
      x_embed: SB output memloc, shape [par(d_head), bs * n_head * s_active]
    """
    # Get basic shapes
    bs, s_active, d_head = cfg.bs, cfg.s_active, cfg.d_head
    n_head = x_inp.shape[1] if not cfg.qk_in_sb else x_inp.shape[1] // (bs * s_active)
    n_head = 1 if ignore_heads else n_head

    x_f = bs * n_head * s_active  # free dim of x after load + transpose
    x = x_inp if cfg.qk_in_sb else sbm.alloc_stack((d_head, x_f), dtype=nl.float32, buffer=nl.sbuf)
    x_shape_expanded = (d_head, bs, n_head, s_active)

    # Load and transpose x_inp from BNSd to BNdS
    if not cfg.qk_in_sb:
        x_inp = x_inp.reshape((x_f, d_head))
        x_pre_tp = sbm.alloc_stack((x_f, d_head), dtype=x_inp.dtype, buffer=nl.sbuf)
        nisa.dma_copy(x_pre_tp, x_inp, dge_mode=dge_mode.none, name=f"rope_x_inp_load_{name_suffix}")
        # FIXME: Add arch check
        tp_dtype = x_pre_tp.dtype
        x_tp_psum = nl.ndarray(
            (d_head, x_f), dtype=tp_dtype, buffer=nl.psum, address=None if sbm.is_auto_alloc() else (0, 0)
        )
        nisa.nc_transpose(x_tp_psum, x_pre_tp)
        nisa.tensor_copy(x, x_tp_psum)

    # Compute x * cos, the read to cos is repeated n_head times
    x_cos = sbm.alloc_stack(
        x_shape_expanded, dtype=nl.float32, buffer=nl.sbuf
    )  # expand dims for more convenient indexing
    # cos[i_d, s_active*i_B + i_S]
    cos_view = TensorView(cos).reshape_dim(1, [bs, s_active]).expand_dim(2).broadcast(2, size=n_head)
    nisa.tensor_tensor(
        dst=x_cos,
        data1=x.reshape(x_shape_expanded),
        data2=cos_view.get_view(),
        op=nl.multiply,
    )
    x_cos = x_cos.reshape(x.shape)

    # Compute _rotate_half(x)
    rotated_x = sbm.alloc_stack(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(rotated_x[d_head // 2 :, :], x[: d_head // 2, :])
    nisa.tensor_scalar(rotated_x[: d_head // 2, :], x[d_head // 2 :, :], op0=nl.multiply, operand0=-1.0)

    # Compute _rotate_half(x) * sin
    rotated_x = rotated_x.reshape(x_shape_expanded)
    sin_view = TensorView(sin).reshape_dim(1, [bs, s_active]).expand_dim(2).broadcast(2, size=n_head)
    nisa.tensor_tensor(
        dst=rotated_x,
        data1=rotated_x,
        data2=sin_view.get_view(),
        op=nl.multiply,
    )
    rotated_x = rotated_x.reshape(x.shape)

    # Add two intermediates
    nisa.tensor_tensor(x_embed, x_cos, rotated_x, op=nl.add)


def _rope(inv_freqs, pos_ids, bs: int, s_a: int, d_head: int, sbm: SbufManager):
    """Computes rotary embedding for current pos_ids following this algorithm:
      freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
      emb = torch.cat((freqs, freqs), dim=-1)
      cos = emb.cos()
      sin = emb.sin()

    All inputs and outputs to this function are assumed to be in sbuf.

    Args:
      inv_freqs: input ndarray, shape [par(d_head // 2), 1]
      pos_ids: input ndarray, shape [par(p_max), bs * s_active], par(p_max) is broadcasted
      bs: batch size
      s_a: active seqeunce length
      d_head: head dimension

    Returns:
      cos: output ndarray, shape [par(d_head), bs * s_active]
      sin: output ndarray, shape [par(d_head), bs * s_active]
    """
    # Most of the computation handles half of d_head at a time.
    # [d_head_half : d_head_half + d_head_half] requires d_head_half to be a multiple of 32, which means d_head is a multiple of 64
    kernel_assert(d_head % 64 == 0, f"RoPE expects head dim ({d_head}) to be divisible by 64")
    d_head_half = d_head // 2

    # Create outputs
    cos = sbm.alloc_stack((d_head, bs * s_a), dtype=nl.float32, buffer=nl.sbuf, name="name_cos")
    sin = sbm.alloc_stack((d_head, bs * s_a), dtype=nl.float32, buffer=nl.sbuf, name="sin_rope")

    # Compute freqs = dot(inv_freqs, pos_ids), can be simplified to elem-wise multiply
    emb = sbm.alloc_stack((d_head_half, bs * s_a), dtype=nl.float32, buffer=nl.sbuf, name="emb_rope")
    nisa.tensor_scalar(emb, pos_ids[0:d_head_half, :], op0=nl.multiply, operand0=inv_freqs)

    # Compute ((emb + π) % 2π) - π, note that sin(θ) = sin((θ + π) % 2π - π)
    # This is to reduce emb to [-π, π] which is the restriction for Sine on ACT engine
    emb4sin = sbm.alloc_stack((d_head, bs * s_a), dtype=nl.float32, buffer=nl.sbuf, name="eb4sin_rope")
    nisa.tensor_scalar(emb4sin[0:d_head_half, :], emb, op0=nl.add, operand0=math.pi)
    _modulo(x=emb4sin, y=2.0 * math.pi, out=emb4sin, sbm=sbm)
    nisa.tensor_scalar(
        emb4sin[0:d_head_half, :],
        emb4sin[0:d_head_half, :],
        op0=nl.add,
        operand0=-math.pi,
    )

    # Compute sin = sin(torch.cat((freqs, freqs), dim=-1))
    nisa.tensor_copy(emb4sin[d_head_half : d_head_half + d_head_half, :], emb4sin[0:d_head_half, :])
    nisa.activation(sin, op=nl.sin, data=emb4sin)

    # Compute ((emb + π/2 + π) % 2π) - π, note that cos(θ) = sin((θ + π/2 + π) % 2π) - π).
    # This is to reduce emb to [-π, π] which is the legal restriction for Act Sine (and we dont have Act Cosine).
    emb4cos = sbm.alloc_stack((d_head, bs * s_a), dtype=nl.float32, buffer=nl.sbuf, name="emb4cos_rope")
    nisa.tensor_scalar(emb4cos[0:d_head_half, :], emb, op0=nl.add, operand0=1.5 * math.pi)
    _modulo(x=emb4cos, y=2.0 * math.pi, out=emb4cos, sbm=sbm)
    nisa.tensor_scalar(
        emb4cos[0:d_head_half, :],
        emb4cos[0:d_head_half, :],
        op0=nl.add,
        operand0=-math.pi,
    )

    # Compute cos = cos(torch.cat((freqs, freqs), dim=-1))
    nisa.tensor_copy(emb4cos[d_head_half : d_head_half + d_head_half, :], emb4cos[0:d_head_half, :])
    nisa.activation(cos, op=nl.sin, data=emb4cos)

    return cos, sin


"""
Other utilities
"""


def _modulo(x, y: float, out, sbm=None):
    """Computes modulo with the following algorithm:
      q = round(x/y - 0.5)
      res = x - q * y

    All inputs and outputs to this function are assumed to be in sbuf.
    This requires both x and y to be positive.

    Args:
      x: 2D input tensor
      y: input scalar

    Returns:
      out: output sbuf tensor of the same shape as x
    """
    kernel_assert(len(x.shape) == 2, "Expect 2D input x for modulo kernel.")
    p, f = x.shape

    # Compute q = round(x/y - 0.5)
    q_f32 = sbm.alloc_stack((p, f), dtype=nl.float32, buffer=nl.sbuf)
    q_i32 = sbm.alloc_stack((p, f), dtype=nl.int32, buffer=nl.sbuf)

    nisa.tensor_scalar(q_f32, x, nl.multiply, 1.0 / y, False, nl.add, -0.5, False)
    nisa.tensor_copy(q_i32, q_f32)

    # Compute q * y
    qy = sbm.alloc_stack((p, f), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(qy, q_i32, nl.multiply, y)

    # Compute x - (q * y)
    # out = x if in_place else nl.ndarray((p, f), dtype=nl.float32, buffer=nl.sbuf, name='modulo_out')
    nisa.tensor_tensor(out, x, qy, nl.subtract)

    return out


### Sink
def _prep_sink(
    sink_hbm,
    result,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    btc: BatchTileContext,
):
    """
    Helper function that loads sink and replicate/transpose it from [1, H] to [B * H * S_active, 1].
    If B * H * S_active needs tiling, then to [p_max, ceil(B * H * S_active / p_max)]
    """
    sbm.open_scope()

    sink_sb = sbm.alloc_stack((1, cfg.q_head), dtype=sink_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        sink_sb, sink_hbm.reshape((1, cfg.q_head)), dge_mode=dge_mode.none, name=f"sink_load_bt{btc.batch_tile_idx}"
    )

    # Access pattern to 1) repeat_interleave sink for s_active times, 2) and repeat (non-interleave) for bs times.
    sink_repeated = sbm.alloc_stack((1, atp.s_active_bqh), buffer=nl.sbuf, dtype=sink_hbm.dtype)
    sink_repeated_view = TensorView(sink_repeated).reshape_dim(1, [atp.bs, cfg.q_head, cfg.s_active])
    sink_sb_view = (
        TensorView(sink_sb).expand_dim(2).broadcast(2, size=cfg.s_active).expand_dim(1).broadcast(1, size=atp.bs)
    )
    nisa.tensor_copy(
        dst=sink_repeated_view.get_view(),
        src=sink_sb_view.get_view(),
    )

    # And transpose.
    sink_tp_repeated = sbm.alloc_stack((atp.s_active_bqh_tile, atp.n_bsq_tiles), buffer=nl.sbuf, dtype=sink_hbm.dtype)

    for i_bsq_tile in range(atp.n_bsq_full_tiles):
        _tile_sink_transpose(
            i_bsq_tile,
            atp.s_active_bqh_tile,
            sink_hbm,
            sink_repeated,
            sink_tp_repeated,
            atp,
            TC,
            sbm,
        )

    if atp.s_active_bqh_remainder > 0:
        _tile_sink_transpose(
            atp.n_bsq_full_tiles, atp.s_active_bqh_remainder, sink_hbm, sink_repeated, sink_tp_repeated, atp, TC, sbm
        )

    nisa.tensor_copy(result, sink_tp_repeated)

    sbm.close_scope()


def _tile_sink_transpose(
    index,
    tile_size,
    sink_hbm,
    sink_repeated,
    sink_tp_repeated,
    atp: AttnTileParams,
    TC: TileConstants,
    sbm: SbufManager,
):
    sink_tp_psum = nl.ndarray(
        (tile_size, 1),
        buffer=nl.psum,
        dtype=sink_hbm.dtype,
        address=None
        if sbm.is_auto_alloc()
        else (
            0,
            (index % TC.psum_b_max) * TC.psum_f_max_bytes,
        ),
    )
    nisa.nc_transpose(
        sink_tp_psum,
        sink_repeated[:, nl.ds(index * atp.s_active_bqh_tile, tile_size)],
    )
    nisa.tensor_copy(sink_tp_repeated[:tile_size, index], sink_tp_psum)


def _load_and_reshape_active_blk_table(
    active_blk_table,
    atp: AttnTileParams,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    btc: BatchTileContext,
    fa_ctx: FATileContext,
):
    """
    Load active blocks table into SB for the current FA tile and batch tile.
    Only loads the folds corresponding to the current FA tile and batches for the current batch tile.
    Put every 128 consecutive blocks on the same column, spread along the partition dimension.
    If blocks per batch < 128, reduce block_len to increase blocks per batch to 128.
    Sets bufs.active_blocks_sb and atp.num_folds_per_batch.
    """
    TC = TileConstants.get_tile_constants()
    resize_factor = atp.blk_cache_resize_factor
    n_prgs = atp.sprior_n_prgs
    prg_id = atp.sprior_prg_id

    num_active_blks = active_blk_table.shape[1] * resize_factor
    kernel_assert(
        num_active_blks % (TC.p_max * n_prgs) == 0,
        (
            f"Block KV requires the number of active blocks per batch to be a multiple of (p_max * n_prgs). "
            f"Got {num_active_blks} with {n_prgs} shards. Consider using resize_cache_block_len_for_attention_tkg_kernel to get the correct resize_factor."
        ),
    )

    # Compute which folds correspond to this FA tile
    fold_s_prior = atp.block_len * TC.p_max
    is_dynamic = fa_ctx.dynamic_tile_offset_sbuf is not None

    if is_dynamic:
        # Dynamic path: num_folds_this_tile is compile-time (all non-last tiles are full-sized),
        # but fold_start is runtime. We use scalar_offset on the DMA to shift dynamically.
        num_folds_this_tile = fa_ctx.tile_s_prior // fold_s_prior
        fold_start = 0  # placeholder for TensorView; actual offset via scalar_offset
        # Compute dynamic fold start: dynamic_tile_offset_sbuf / fold_s_prior
        fold_s_prior_shift = int(math.log2(fold_s_prior))
        dynamic_fold_start_sbuf = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=dynamic_fold_start_sbuf,
            data=fa_ctx.dynamic_tile_offset_sbuf,
            op0=nl.right_shift,
            operand0=fold_s_prior_shift,
        )
    else:
        fold_start = fa_ctx.tile_offset // fold_s_prior
        fold_end = div_ceil(fa_ctx.tile_offset + fa_ctx.tile_s_prior, fold_s_prior)
        num_folds_this_tile = fold_end - fold_start

    batch_start = btc.global_batch_offset
    batch_size = atp.bs

    # Set atp.num_folds_per_batch to the per-tile value
    atp.num_folds_per_batch = num_folds_this_tile

    """
  Say active_blks has shape (B=2, blks_per_batch=4), with a reshape factor = 128/4 = 32
  [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]  # Note in reality block indices aren't sequential.

  We could load to SBUF as follows, and then do `blk_idx_sbuf * resize_factor + arange(resize_factor)`.
    par[ 0: 32]-> [0, 4,  8, 12]
    par[32: 64]-> [1, 5,  9, 13]
    par[64: 96]-> [2, 6, 10, 14]
    par[96:128]-> [3, 7, 11, 15]
  However we cannot use an affine expression with two indices on the partition dimension,
  so we cannot easily get to this state in SBUF.

  The alternative is to load to SBUF as shape (4, 128),
    [0,   0, ...,  0,   1,  1, ...,  1,   2,  2, ...,  2,   3,  3, ...,  3]
    [4,   4, ...,  4,   5,  5, ...,  5,   6,  6, ...,  6,   7,  7, ...,  7]
    [8,   8, ...,  8,   9,  9, ...,  9,  10, 10, ..., 10,  11, 11, ..., 11]
    [12, 12, ..., 12,  13, 13, ..., 13,  14, 14, ..., 14,  15, 15, ..., 15]
  And then transpose to the above desired shape.
  """
    if resize_factor == 1:
        # The code below is semantically correct for resize_factor > 1 but we cannot use
        # an affine expression with two indices on the partition dimension today.
        full_num_folds_total = num_active_blks // TC.p_max
        partition_resize = TC.p_max // resize_factor

        active_blk_table_sb = sbm.alloc_stack(
            (TC.p_max, num_folds_this_tile * batch_size),
            dtype=active_blk_table.dtype,
            buffer=nl.sbuf,
        )

        active_blk_table_sb_tv = (
            TensorView(active_blk_table_sb)
            # .reshape_dim(0, [partition_resize, resize_factor]) # Semantically correct, if two indices on partition were allowed
            .reshape_dim(1, [num_folds_this_tile, batch_size])
        )

        if is_dynamic:
            # Dynamic path: global fold indexing with scalar_offset for runtime fold start.
            dynamic_fold_offset_elems = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=dynamic_fold_offset_elems, data=dynamic_fold_start_sbuf, op0=nl.multiply, operand0=partition_resize
            )

            active_blk_table_tv = (
                TensorView(active_blk_table)
                .reshape_dim(1, [full_num_folds_total, partition_resize])
                .slice(0, batch_start, batch_start + batch_size)
                .slice(1, 0, num_folds_this_tile)
                .permute([2, 1, 0])
                .expand_dim(1)
                .broadcast(1, resize_factor)
                ._copy(scalar_offset=dynamic_fold_offset_elems, indirect_dim=1)
            )
            nisa.dma_copy(
                src=active_blk_table_tv.get_view(),
                dst=active_blk_table_sb_tv.get_view(),
                dge_mode=dge_mode.hwdge,
            )
        else:
            # Static path: tile_offset is global, fold indices are global.
            active_blk_table_tv = (
                TensorView(active_blk_table)
                .reshape_dim(1, [full_num_folds_total, partition_resize])
                .slice(0, batch_start, batch_start + batch_size)
                .slice(1, fold_start, fold_start + num_folds_this_tile)
                .permute([2, 1, 0])
                .expand_dim(1)
                .broadcast(1, resize_factor)
            )
            nisa.dma_copy(
                src=active_blk_table_tv.get_view(),
                dst=active_blk_table_sb_tv.get_view(),
                dge_mode=dge_mode.none,
                name=f"active_blk_table_load_resize1_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}",
            )
    else:
        # We need to "resize" the cache blocks.
        # Only load the original blocks needed for this FA tile's folds.
        # Each expanded fold covers P_MAX sub-blocks = P_MAX/resize_factor original blocks.
        orig_blk_start = fold_start * TC.p_max // resize_factor
        orig_blk_end = (fold_start + num_folds_this_tile) * TC.p_max // resize_factor
        orig_blks_this_tile = orig_blk_end - orig_blk_start

        # tile_offset is always global for block KV, so orig_blk indices are global.

        active_blk_table_sb = sbm.alloc_stack(
            (TC.p_max, batch_size * num_folds_this_tile),
            dtype=active_blk_table.dtype,
            buffer=nl.sbuf,
            align=4,
        )

        sbm.open_scope()
        # Process in batch chunks of p_max to keep partition dim <= p_max
        batch_chunk = min(batch_size, TC.p_max)
        for batch_offset in range(0, batch_size, batch_chunk):
            cur_batch_chunk_sz = min(batch_chunk, batch_size - batch_offset)

            active_blk_pre_reshape = sbm.alloc_stack(
                (cur_batch_chunk_sz, orig_blks_this_tile), dtype=active_blk_table.dtype, buffer=nl.sbuf
            )
            active_blk_table_slice = (
                TensorView(active_blk_table)
                .slice(0, start=batch_start + batch_offset, end=batch_start + batch_offset + cur_batch_chunk_sz)
                .slice(1, start=orig_blk_start, end=orig_blk_end)
            )
            nisa.dma_copy(
                dst=active_blk_pre_reshape,
                src=active_blk_table_slice.get_view(),
                dge_mode=dge_mode.none,
                name=f"active_blk_table_load_pre_reshape_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}_b{batch_offset}",
            )

            # Now update the active blocks table with.  New active blocks table will be:
            #   old_blk_idx * resize_factor + arange(resize_factor)
            reshape_arange = sbm.alloc_stack(
                (cur_batch_chunk_sz, resize_factor), dtype=active_blk_table.dtype, buffer=nl.sbuf
            )
            nisa.iota(dst=reshape_arange, pattern=[[1, resize_factor]], offset=0)

            active_blk_reshaped = sbm.alloc_stack(
                (cur_batch_chunk_sz, orig_blks_this_tile, resize_factor), dtype=nl.float32, buffer=nl.sbuf
            )
            active_blk_pre_reshape_view = (
                TensorView(active_blk_pre_reshape).expand_dim(2).broadcast(2, size=resize_factor)
            )
            reshape_arange_view = TensorView(reshape_arange).expand_dim(1).broadcast(1, size=orig_blks_this_tile)
            nisa.scalar_tensor_tensor(
                dst=active_blk_reshaped,
                data=active_blk_pre_reshape_view.get_view(),
                op0=nl.multiply,
                operand0=float(resize_factor),
                op1=nl.add,
                operand1=reshape_arange_view.get_view(),
            )

            # Reshaped to flat sub-blocks: num_folds_this_tile * P_MAX sub-blocks
            active_blk_reshaped = active_blk_reshaped.reshape((cur_batch_chunk_sz, num_folds_this_tile * TC.p_max))

            # Transpose each fold into the output
            sbm.open_scope()
            for fold_rel_idx in range(num_folds_this_tile):
                active_blk_transposed = nl.ndarray(
                    (TC.p_max, cur_batch_chunk_sz),
                    dtype=active_blk_reshaped.dtype,
                    buffer=nl.psum,
                    address=None if sbm.is_auto_alloc() else (0, (fold_rel_idx % TC.psum_b_max) * TC.psum_f_max_bytes),
                )
                nisa.nc_transpose(
                    active_blk_transposed,
                    active_blk_reshaped[:, nl.ds(fold_rel_idx * TC.p_max, TC.p_max)],
                )
                nisa.tensor_copy(
                    active_blk_table_sb[:, nl.ds(fold_rel_idx * batch_size + batch_offset, cur_batch_chunk_sz)],
                    src=active_blk_transposed,
                    engine=nisa.vector_engine,
                )
            sbm.close_scope()
        sbm.close_scope()

    bufs.active_blocks_sb = active_blk_table_sb

    if atp.use_dma_transpose or not atp.use_v_dma_skipping:
        # Pre-compute active_blocks_sb_u32 to avoid casting in hot loop.
        bufs.active_blocks_sb_u32 = sbm.alloc_stack(
            (TC.p_max, batch_size * num_folds_this_tile),
            dtype=nl.uint32,
            buffer=nl.sbuf,
        )
        nisa.tensor_copy(bufs.active_blocks_sb_u32, active_blk_table_sb, engine=nisa.vector_engine)

    # Store to debug tensor if available (incrementally per FA tile and batch tile)
    if bufs.DBG_ACTIVE_TABLE is not None:
        dbg_fold_offset = atp.sprior_prg_id * (atp.s_prior // (atp.block_len * TC.p_max)) + fold_start
        bufs.active_blocks_sb = active_blk_table_sb.reshape((TC.p_max, num_folds_this_tile, batch_size))
        nisa.dma_copy(
            bufs.DBG_ACTIVE_TABLE[
                :,
                nl.ds(dbg_fold_offset, num_folds_this_tile),
                nl.ds(batch_start, batch_size),
            ],
            bufs.active_blocks_sb,
            dge_mode=dge_mode.none,
            name=f"dbg_active_blocks_table_store_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}",
        )
        bufs.active_blocks_sb = active_blk_table_sb.reshape((TC.p_max, num_folds_this_tile * batch_size))


### Other Helpers
def _get_safe_batch_interleave_degree(space_needed_per_batch: int, max_batch_interleave_degree: int, sbm: SbufManager):
    """
    Compute the batch interleave degree that will not overflow memory given the current memory available.
    space_needed_per_batch is in bytes (interleaved across batches).

    For auto_alloc sbm, return 1 (interleave degree is ignored in this case).
    """
    if sbm.is_auto_alloc():
        return 1
    # use 1 if auto alloc since otherwise it can fail despite enough space available
    space_available = sbm.get_free_space()
    kernel_assert(max_batch_interleave_degree > 0, "batch_interleave_degree must be greater than 0")

    result = min(max_batch_interleave_degree, space_available // space_needed_per_batch)

    kernel_assert(
        result > 0,
        (
            f"Insufficient memory to run batch loop, even at interleave_degree=1."
            f"Need {space_needed_per_batch} bytes, but only have {space_available} bytes available."
        ),
    )

    return result


# Extended instructions require input/output tensors have multiple of 16 partitions
# This is temporary, will go away once gpsimd sb2sb moves from extended isa to its final isa
def pad_partitions_for_ext_inst(partitions):
    PARTITIONS_PER_GPSIMD_CORE = 16
    return (partitions + PARTITIONS_PER_GPSIMD_CORE - 1) // PARTITIONS_PER_GPSIMD_CORE * PARTITIONS_PER_GPSIMD_CORE


def _clamp_max_to_finite(dst: nl.ndarray, src: nl.ndarray, max_negated: bool = False):
    """Clamp infinite max values to finite bounds to prevent NaN in exp(a - b).

    Fully masked tiles produce -Inf (or +Inf when negated) as the softmax max.
    When computing exp(prev_max - curr_max), -Inf - (-Inf) = NaN per IEEE 754.
    Clamping to a finite bound ensures exp produces 0 instead.

    When max_negated=False: -Inf -> _MIN_FLOAT32 (clamp up via maximum)
    When max_negated=True:  +Inf -> _MAX_FLOAT32 (clamp down via minimum)
    """
    op, bound = (nl.minimum, _MAX_FLOAT32) if max_negated else (nl.maximum, _MIN_FLOAT32)
    nisa.tensor_scalar(dst, src, op0=op, operand0=bound)
