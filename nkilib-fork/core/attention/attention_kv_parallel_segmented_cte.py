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
"""KV-parallel segmented prefill attention kernel.

This kernel enables context parallelism by distributing the KV cache across multiple
ranks. Each rank computes attention over its local KV shard, then results are merged
using online softmax.

See README.md for detailed documentation.
"""

from typing import Optional

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil
from ..utils.modular_allocator import ModularAllocator
from .attention_segmented_cte import attention_segmented_cte

P_MAX = nl.tile_size.pmax


@nki.jit
def attention_kv_parallel_segmented_cte(
    q: nl.ndarray,
    k_cache: nl.ndarray,
    v_cache: nl.ndarray,
    block_tables: nl.ndarray,
    kvp_q_offset: nl.ndarray,
    replica_groups: ReplicaGroup,
    group_size: int,
    block_size: int,
    seg_size: int,
    scale: float = 1.0,
    global_q_offset: int = 0,
    tp_out: bool = False,
    sliding_window: int = 0,
    kvp_rank_id: Optional[nl.ndarray] = None,
    kvp_group_size: int = 0,
) -> nl.ndarray:
    """
    KV-parallel segmented prefill attention.

    Distributes attention computation across ranks, where each rank holds a shard
    of the KV cache. Uses online softmax to merge partial results.

    Dimensions:
        BS: Batch size (lnc_degree = Q heads per physical rank)
        S: Sequence length
        D: Head dimension
        G: Group size (number of ranks per replica group)
        R: Number of physical ranks per group (group_size // lnc_degree)

    Args:
        q (nl.ndarray): [BS, S, D], This rank's Q heads (BS = lnc_degree).
        k_cache (nl.ndarray): [num_blocks, num_kv_heads, block_size, D], Local KV cache (K).
        v_cache (nl.ndarray): [num_blocks, num_kv_heads, block_size, D], Local KV cache (V).
        block_tables (nl.ndarray): [1, max_blocks] int32, Block indices for paged KV.
        kvp_q_offset (nl.ndarray): [1, 1] int32, Causal mask offset =
            -rank_id * local_kv_len + global_q_offset.
            For round-robin KV distribution, set to just
            global_q_offset since the global-to-local bound conversion handles K-side positioning.
        replica_groups (ReplicaGroup): ReplicaGroup for collective operations.
        group_size (int): Number of ranks in the replica group.
        block_size (int): KV cache block size.
        seg_size (int): Segment size for attention iteration.
        scale (float): Attention scale factor (default 1.0).
        global_q_offset (int): Global token position of Q token 0 (default 0). Used to compute
            how many prior KV tokens exist within this rank's shard for each Q chunk.
        tp_out (bool): If True, output is transposed to [BS, D, S] (default False).
        sliding_window (int): Sliding window size for attention (0 = disabled).
        kvp_rank_id (nl.ndarray): [1, 1] int32, This rank's index within the KV-parallel group.
            Required for interleaved (round-robin) KV distribution to convert global K
            positions to segment-local positions.
        kvp_group_size (int): Number of ranks sharing the KV cache in round-robin fashion.
            When > 0, enables interleaved KV mode where rank r holds global blocks
            r, r+R, r+2R, ... (R = kvp_group_size).

    Returns:
        out (nl.ndarray): [BS, S, D], Merged attention output for this rank's Q heads.

    Pseudocode:
        # Step 1: All-gather Q across ranks
        q_full = all_gather(q)  # [group_size, S, D]

        # Step 2: For each Q chunk, compute local attention
        for q_chunk_idx in range(num_q_chunks):
            q_chunk = q_full[shard_id, q_start:q_end, :]
            chunk_out, chunk_neg_max, chunk_sum_recip = attention_segmented_cte(q_chunk, k_cache, v_cache)
            partial_out[shard_id, q_start:q_end] = chunk_out

        # Step 3: Pack softmax stats + partial outputs, exchange via all-to-all
        send_packed = pack(partial_out, neg_max, sum_recip)
        recv_packed = all_to_all(send_packed)

        # Step 4: Merge partials using online softmax
        for tile_idx in range(num_tiles):
            global_neg_max = min(recv_packed[:, neg_max_channel])
            factors = exp(global_neg_max - neg_max_per_rank) / sum_recip_per_rank
            factors = factors / sum(factors)
            out[shard_id, tile] = sum(factors * recv_packed[:, out_channel])
    """
    bs, seq_len, head_dim = q.shape
    num_q_chunks = seq_len // seg_size

    shard_id = nl.program_id(0)

    # Input validation
    kernel_assert(seq_len % seg_size == 0, f"seq_len ({seq_len}) must be divisible by seg_size ({seg_size})")
    kernel_assert(group_size % bs == 0, f"group_size ({group_size}) must be divisible by lnc_degree ({bs})")
    kernel_assert(
        q.shape[2] == k_cache.shape[3],
        f"head_dim mismatch: q has {q.shape[2]}, k_cache has {k_cache.shape[3]}",
    )
    kernel_assert(
        k_cache.shape[2] == block_size,
        f"k_cache block_size dim ({k_cache.shape[2]}) must match block_size ({block_size})",
    )

    # bs = lnc_degree = Q heads per physical rank = NCs per physical rank
    lnc_degree = bs
    num_physical_ranks = group_size // lnc_degree
    heads_per_nc = group_size // lnc_degree
    q_heads_per_rank = lnc_degree

    # All-gather Q across ranks.
    # Collectives cannot read/write I/O tensors directly, so each NC DMAs its slice into shared_hbm first.
    q_src = nl.ndarray((lnc_degree, seq_len, head_dim), dtype=q.dtype, buffer=nl.shared_hbm, name="q_src")
    for nc_idx in range(lnc_degree):
        if shard_id == nc_idx:
            nisa.dma_copy(dst=q_src[nc_idx, :, :], src=q[nc_idx, :, :])

    q_full = nl.ndarray(
        (group_size, seq_len, head_dim),
        dtype=q.dtype,
        buffer=nl.shared_hbm,
        name="q_full",
    )
    ncc.all_gather(dsts=[q_full], srcs=[q_src], replica_group=replica_groups, collective_dim=0)

    partial_out = nl.ndarray(
        (group_size, seq_len, head_dim), dtype=nl.float32, buffer=nl.shared_hbm, name="partial_out"
    )
    neg_max = nl.ndarray((group_size, seq_len), dtype=nl.float32, buffer=nl.shared_hbm, name="neg_max")
    sum_recip = nl.ndarray((group_size, seq_len), dtype=nl.float32, buffer=nl.shared_hbm, name="sum_recip")

    chunk_allocator = ModularAllocator()
    kvp_offset_chunk_sbuf = chunk_allocator.alloc_sbuf_tensor((1, 1), nl.int32)
    kvp_offset_chunk_hbm = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.private_hbm, name="kvp_offset_chunk_hbm")
    prior_tokens_chunk_sbuf = chunk_allocator.alloc_sbuf_tensor((1, 1), nl.int32)
    prior_tokens_chunk_hbm = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.private_hbm, name="prior_tokens_chunk_hbm")

    num_kv_blocks = k_cache.shape[0]
    local_kv_len = num_kv_blocks * block_size
    max_prior_tokens = local_kv_len - seg_size

    # Compute local attention for each Q chunk.
    # Each NC processes heads_per_nc Q heads starting at shard_id * heads_per_nc.
    for q_chunk_idx in range(num_q_chunks):
        q_start = q_chunk_idx * seg_size
        q_end = q_start + seg_size

        q_chunk = nl.ndarray(
            (heads_per_nc, seg_size, head_dim), dtype=q.dtype, buffer=nl.private_hbm, name=f"q_chunk_{q_chunk_idx}"
        )
        nisa.dma_copy(
            dst=q_chunk,
            src=q_full[nl.ds(shard_id * heads_per_nc, heads_per_nc), q_start:q_end, :],
        )

        prior_tokens_for_chunk = min(q_start + global_q_offset, max_prior_tokens)

        # For interleaved KV: set active_block_offset so the active segment covers all
        # local blocks that Q can see. Prior segments become fully visible (no masking).
        kvp_prior_load_blocks = 0  # 0 = use default (num_blocks_per_seg)
        if kvp_group_size > 0:
            stride = num_physical_ranks * block_size
            kvp_cp_offset_int = q_start + global_q_offset
            num_blocks_per_seg = seg_size // block_size
            num_active_blocks = seg_size // block_size
            blocks_per_k_tile = 512 // block_size  # _K_TILE_SZ // block_size
            # Last local block Q can see (any rank)
            last_needed_block = (kvp_cp_offset_int + seg_size - 1) // stride
            # Active must contain last_needed_block
            min_active_start = max(0, last_needed_block - num_active_blocks + 1)
            # Round UP to K-tile boundary so partial prior fills complete tiles
            active_block_offset_int = (
                (min_active_start + blocks_per_k_tile - 1) // blocks_per_k_tile
            ) * blocks_per_k_tile
            # Ensure at least 1 full prior segment (required for reduce_one_batch path)
            active_block_offset_int = max(active_block_offset_int, num_blocks_per_seg)
            # Verify prior is fully visible: max global in prior < kvp_cp_offset_int
            # max_global_in_prior = active_block_offset_int * stride - 1
            prior_fully_visible = (active_block_offset_int * stride - 1) < kvp_cp_offset_int
            # Verify active covers all needed blocks
            covers_needed = (active_block_offset_int + num_active_blocks - 1) >= last_needed_block
            # Only apply if both constraints are met and it reduces prior_tokens
            if prior_fully_visible and covers_needed and active_block_offset_int < prior_tokens_for_chunk // block_size:
                # Compute partial prior load count
                num_full_prior_segs = active_block_offset_int // num_blocks_per_seg
                partial_blocks = active_block_offset_int - num_full_prior_segs * num_blocks_per_seg
                if partial_blocks > 0:
                    kvp_prior_load_blocks = partial_blocks
                prior_tokens_for_chunk = active_block_offset_int * block_size

        nisa.dma_copy(dst=kvp_offset_chunk_sbuf, src=kvp_q_offset)
        nisa.tensor_scalar(dst=kvp_offset_chunk_sbuf, data=kvp_offset_chunk_sbuf, op0=nl.add, operand0=q_start)
        nisa.dma_copy(dst=kvp_offset_chunk_hbm, src=kvp_offset_chunk_sbuf)
        nisa.memset(prior_tokens_chunk_sbuf[...], value=prior_tokens_for_chunk)
        nisa.dma_copy(dst=prior_tokens_chunk_hbm, src=prior_tokens_chunk_sbuf)

        chunk_out, chunk_neg_max, chunk_sum_recip = attention_segmented_cte(
            q=q_chunk,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            prior_tokens=prior_tokens_chunk_hbm,
            block_size=block_size,
            prior_seg_size=seg_size,
            scale=scale,
            tp_q=True,
            tp_out=False,  # Always False internally; tp_out handled at final output write
            kvp_q_offset=kvp_offset_chunk_hbm,
            kvp_rank_id=kvp_rank_id,
            kvp_group_size=kvp_group_size,
            sliding_window=sliding_window,
            kvp_cp_offset_int=q_start + global_q_offset,
            kvp_seg_block_offset_int=prior_tokens_for_chunk // block_size,
            kvp_prior_load_blocks=kvp_prior_load_blocks,
            kvp_prior_fully_visible=prior_fully_visible if kvp_group_size > 0 else False,
        )

        nisa.dma_copy(
            dst=partial_out[nl.ds(shard_id * heads_per_nc, heads_per_nc), q_start:q_end, :],
            src=chunk_out,
        )
        nisa.dma_copy(
            dst=neg_max[nl.ds(shard_id * heads_per_nc, heads_per_nc), q_start:q_end],
            src=chunk_neg_max,
        )
        nisa.dma_copy(
            dst=sum_recip[nl.ds(shard_id * heads_per_nc, heads_per_nc), q_start:q_end],
            src=chunk_sum_recip,
        )
    # Exchange partial outputs and softmax stats via coalesced all-to-all (3 separate tensors).
    recv_out = nl.ndarray(
        (num_physical_ranks, q_heads_per_rank, seq_len, head_dim),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
        name="recv_out",
    )
    recv_neg_max = nl.ndarray(
        (num_physical_ranks, q_heads_per_rank, seq_len),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
        name="recv_neg_max",
    )
    recv_sum_recip = nl.ndarray(
        (num_physical_ranks, q_heads_per_rank, seq_len),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
        name="recv_sum_recip",
    )
    ncc.all_to_all(
        dsts=[recv_out, recv_neg_max, recv_sum_recip],
        srcs=[partial_out, neg_max, sum_recip],
        replica_group=replica_groups,
        collective_dim=0,
    )

    # Merge partial attention outputs using online softmax.
    allocator = ModularAllocator()
    out = nl.ndarray(
        (bs, head_dim, seq_len) if tp_out else (bs, seq_len, head_dim), dtype=q.dtype, buffer=nl.shared_hbm, name="out"
    )
    _merge_partial_attention_outputs(
        recv_out,
        recv_neg_max,
        recv_sum_recip,
        out,
        shard_id,
        num_physical_ranks,
        q_heads_per_rank,
        seq_len,
        head_dim,
        q.dtype,
        allocator,
        tp_out,
    )
    return out


def _merge_partial_attention_outputs(
    recv_out: nl.ndarray,
    recv_neg_max: nl.ndarray,
    recv_sum_recip: nl.ndarray,
    out: nl.ndarray,
    shard_id: int,
    num_physical_ranks: int,
    q_heads_per_rank: int,
    seq_len: int,
    head_dim: int,
    out_dtype,
    allocator: ModularAllocator,
    tp_out: bool = False,
) -> None:
    """
    Merge partial attention outputs from all ranks using online softmax rescaling.

    recv_out: [num_physical_ranks, q_heads_per_rank, seq_len, head_dim]
    recv_neg_max: [num_physical_ranks, q_heads_per_rank, seq_len]
    recv_sum_recip: [num_physical_ranks, q_heads_per_rank, seq_len]
    """
    neg_max_sbuf = allocator.alloc_sbuf_tensor((P_MAX, num_physical_ranks), nl.float32)
    sum_recip_sbuf = allocator.alloc_sbuf_tensor((P_MAX, num_physical_ranks), nl.float32)
    global_neg_max = allocator.alloc_sbuf_tensor((P_MAX, 1), nl.float32)
    factors = allocator.alloc_sbuf_tensor((P_MAX, num_physical_ranks), nl.float32)
    exp_term = allocator.alloc_sbuf_tensor((P_MAX, num_physical_ranks), nl.float32)
    recip = allocator.alloc_sbuf_tensor((P_MAX, num_physical_ranks), nl.float32)
    factor_sum = allocator.alloc_sbuf_tensor((P_MAX, 1), nl.float32)
    factor_sum_recip = allocator.alloc_sbuf_tensor((P_MAX, 1), nl.float32)
    out_tile = allocator.alloc_sbuf_tensor((P_MAX, head_dim), nl.float32)
    partial_tile = allocator.alloc_sbuf_tensor((P_MAX, head_dim), nl.float32)
    scaled = allocator.alloc_sbuf_tensor((P_MAX, head_dim), nl.float32)
    out_tile_cast = allocator.alloc_sbuf_tensor((P_MAX, head_dim), out_dtype)
    # tp_out: extra SBUF buffers for transposing merged tile (P_MAX, head_dim) → (head_dim, P_MAX).
    out_tile_tp_psum = nl.ndarray((head_dim, P_MAX), dtype=nl.float32, buffer=nl.psum) if tp_out else None
    out_tile_tp_sbuf = allocator.alloc_sbuf_tensor((head_dim, P_MAX), out_dtype) if tp_out else None

    pos = shard_id  # this NC's position within q_heads_per_rank (0 or 1 for LNC=2)

    num_tiles = div_ceil(seq_len, P_MAX)
    for tile_idx in range(num_tiles):
        tile_start = tile_idx * P_MAX
        tile_end = min(tile_start + P_MAX, seq_len)
        tile_size = tile_end - tile_start

        # Load neg_max and sum_recip for all ranks — contiguous layout.
        # recv_neg_max: [num_physical_ranks, q_heads_per_rank, seq_len]
        # Want: neg_max_sbuf[tile, rank] — partition=tile (P_MAX), free=rank
        stats_p_stride = 1  # consecutive tokens
        stats_f_stride = q_heads_per_rank * seq_len  # stride between ranks
        stats_offset = pos * seq_len + tile_start
        stats_ap = [[stats_p_stride, P_MAX], [stats_f_stride, num_physical_ranks]]
        nisa.dma_copy(dst=neg_max_sbuf, src=recv_neg_max.ap(pattern=stats_ap, offset=stats_offset))
        nisa.dma_copy(dst=sum_recip_sbuf, src=recv_sum_recip.ap(pattern=stats_ap, offset=stats_offset))

        # Compute per-rank rescaling factors using online softmax, then normalize.
        # exp(global_neg_max - neg_max_sbuf) = exp(-neg_max_sbuf + global_neg_max)
        nisa.tensor_reduce(
            dst=global_neg_max[:tile_size, :], op=nl.minimum, data=neg_max_sbuf[:tile_size, :], axis=[1], keepdims=True
        )
        nisa.activation(
            dst=exp_term[:tile_size, :],
            op=nl.exp,
            data=neg_max_sbuf[:tile_size, :],
            scale=-1.0,
            bias=global_neg_max[:tile_size, :],
        )
        nisa.activation(dst=recip[:tile_size, :], op=nl.reciprocal, data=sum_recip_sbuf[:tile_size, :])
        nisa.tensor_tensor(
            dst=factors[:tile_size, :], data1=exp_term[:tile_size, :], data2=recip[:tile_size, :], op=nl.multiply
        )
        nisa.tensor_reduce(
            dst=factor_sum[:tile_size, :], op=nl.add, data=factors[:tile_size, :], axis=[1], keepdims=True
        )
        nisa.activation(dst=factor_sum_recip[:tile_size, :], op=nl.reciprocal, data=factor_sum[:tile_size, :])
        nisa.activation(
            dst=factors[:tile_size, :], op=nl.copy, data=factors[:tile_size, :], scale=factor_sum_recip[:tile_size, :]
        )

        # Weighted sum: out = sum_r(factors[:, r] * partial[:, r, :]).
        # With contiguous recv_out, each rank's partial is a simple slice — no AP needed.
        nisa.memset(out_tile, 0)
        for rank_idx in range(num_physical_ranks):
            nisa.dma_copy(dst=partial_tile[:tile_size, :], src=recv_out[rank_idx, pos, tile_start:tile_end, :])
            nisa.tensor_scalar(
                dst=scaled[:tile_size, :],
                data=partial_tile[:tile_size, :],
                op0=nl.multiply,
                operand0=factors[:tile_size, rank_idx : rank_idx + 1],
            )
            nisa.tensor_tensor(
                dst=out_tile[:tile_size, :],
                data1=out_tile[:tile_size, :],
                data2=scaled[:tile_size, :],
                op=nl.add,
            )

        nisa.tensor_copy(dst=out_tile_cast[:tile_size, :], src=out_tile[:tile_size, :])
        if tp_out:
            nisa.nc_transpose(out_tile_tp_psum[:, :tile_size], out_tile[:tile_size, :])
            nisa.tensor_copy(dst=out_tile_tp_sbuf[:, :tile_size], src=out_tile_tp_psum[:, :tile_size])
            nisa.dma_copy(dst=out[shard_id, :, tile_start:tile_end], src=out_tile_tp_sbuf[:, :tile_size])
        else:
            nisa.dma_copy(dst=out[shard_id, tile_start:tile_end, :], src=out_tile_cast[:tile_size, :])
