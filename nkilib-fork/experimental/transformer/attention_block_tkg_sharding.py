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
KV Data Parallel (KVDP) helpers for attention_block_tkg.

KVDP partitions the KV cache across ranks along the batch dimension. Each rank holds
B/KVDP batches of the KV cache. Before attention: redistribute Q heads/batch via
collective (all_to_all or all_gather+slice), slice K/V batch.
After attention: redistribute output heads/batch via collective.

See attention_block_tkg_sharding_design_spec.md for detailed design documentation.
"""

from enum import Enum

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ...core.utils.allocator import SbufManager
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.tensor_view import TensorView


class KVDPCollectiveMode(Enum):
    """Collective operation mode for KVDP input/output redistribution.

    ALL_TO_ALL: single all_to_all collective (default, runtime requires ≥4 ranks lnc2 or ≥8 ranks lnc1)
    ALL_GATHER_SLICE: all_gather on heads/batch + KVDP_rank slice (works with any rank count)
    """

    ALL_TO_ALL = 1
    ALL_GATHER_SLICE = 0


def _KVDP_attention_input_collectives(
    Q_tkg_sb: nl.ndarray,
    K_tkg_sb: nl.ndarray,
    V_tkg_hbm: nl.ndarray,
    q_heads: int,
    kv_heads: int,
    d_head: int,
    KVDP: int,
    B: int,
    B_attn: int,
    S_tkg: int,
    replica_group: ReplicaGroup,
    sbm: SbufManager,
    collective_mode: KVDPCollectiveMode,
    dynamic_KVDP_rank_sb: nl.ndarray,
):
    """Input collectives for KV data parallelism.

    Dispatches Q redistribution to the appropriate helper based on collective_mode,
    then slices K/V batch for this rank.

    Pseudocode:

        # Q: (d, B*q*S) @SBUF -> _KVDP_Q_input_{mode} -> (d, B_attn*KVDP*q*S) @SBUF
        # K: (d, B*S) @SBUF -> dynamic_KVDP_rank_sb slice batch -> (d, B_attn*S) @SBUF
        # V: (B, kv, S, d) @HBM -> dynamic_KVDP_rank_sb slice batch -> (B_attn, kv, S, d) @HBM

    Example: TP64 QKV projection → TP8 KVDP8 attention for GPT-OSS (64 q_heads, 8 k_heads) for B=16
        - 64 ranks compute QKV projection, each with q_heads=1, B=16
        - Q redistribution within each KVDP group: 1 q_head x KVDP8 → 8 q_heads for B/KVDP=2 batches
        - Each rank now has 8 Q heads for 2 batches for 1 K head (TP8)

    Args:
        Q_tkg_sb (nl.ndarray): [d_head, B * q_heads * S_tkg] @ SBUF - Q after RoPE
        K_tkg_sb (nl.ndarray): [d_head, B * S_tkg] @ SBUF - K after RoPE
        V_tkg_hbm (nl.ndarray): [B, kv_heads=1, S_tkg, d_head] @ HBM - V from QKV extraction
        q_heads (int): Number of query heads per rank (before gather)
        kv_heads (int): Number of KV heads
        d_head (int): Head dimension
        KVDP (int): KV data parallelism degree (number of ranks)
        B (int): Total batch size across all ranks
        B_attn (int): Batch size per rank for attention (B / KVDP)
        S_tkg (int): Token generation sequence length
        replica_group (ReplicaGroup): Replica group for collective ops
        sbm: SBUF memory manager
        dynamic_KVDP_rank_sb (nl.ndarray): [1, 1] @ SBUF, this rank's position within its KVDP replica group (0 to KVDP-1).

    Returns:
        Q_tkg_sb (nl.ndarray): [d_head, B_attn * q_heads * KVDP * S_tkg] @ SBUF - gathered Q
        K_tkg_sb (nl.ndarray): [d_head, B_attn * S_tkg] @ SBUF - sliced K
        V_tkg_hbm (nl.ndarray): [B_attn, kv_heads, S_tkg, d_head] @ HBM - sliced V for attention_tkg

    Notes:
        - V is returned in HBM because attention_tkg loads V tile-by-tile during P*V matmul
        - Batch selection uses reshape to (KVDP, rest) + select with dynamic_KVDP_rank_sb
    """
    dtype = Q_tkg_sb.dtype
    kv_dtype = K_tkg_sb.dtype
    kernel_assert(K_tkg_sb.dtype == V_tkg_hbm.dtype, f"K/V dtype mismatch: {K_tkg_sb.dtype} != {V_tkg_hbm.dtype}")

    # Shape assertions
    kernel_assert(
        Q_tkg_sb.shape == (d_head, B * q_heads * S_tkg),
        f"Q_tkg_sb shape mismatch: {Q_tkg_sb.shape} != {(d_head, B * q_heads * S_tkg)}",
    )
    kernel_assert(
        K_tkg_sb.shape == (d_head, B * S_tkg),
        f"K_tkg_sb shape mismatch: {K_tkg_sb.shape} != {(d_head, B * S_tkg)}",
    )
    kernel_assert(
        V_tkg_hbm.shape == (B, kv_heads, S_tkg, d_head),
        f"V_tkg_hbm shape mismatch: {V_tkg_hbm.shape} != {(B, kv_heads, S_tkg, d_head)}",
    )

    # ========== Q: collective redistribution ==========

    if collective_mode == KVDPCollectiveMode.ALL_TO_ALL:
        Q_tkg_sb_out = _KVDP_Q_input_all_to_all(
            Q_tkg_sb,
            q_heads,
            d_head,
            KVDP,
            B,
            B_attn,
            S_tkg,
            dtype,
            replica_group,
            sbm,
        )
    else:
        Q_tkg_sb_out = _KVDP_Q_input_all_gather_slice(
            Q_tkg_sb,
            q_heads,
            d_head,
            KVDP,
            B,
            B_attn,
            S_tkg,
            dtype,
            replica_group,
            sbm,
            dynamic_KVDP_rank_sb,
        )

    # ========== K: slice batch ==========
    # Compiler restriction (NCC_ILDM008): dynamic-AP DMA can only access DRAM, so route
    # through HBM. Stage K to HBM once, then do a single dynamic-AP DMA from the strided
    # HBM view directly into SBUF
    K_full = nl.ndarray((d_head, B * S_tkg), dtype=kv_dtype, buffer=nl.shared_hbm, name="K_full")
    nisa.dma_copy(K_full, K_tkg_sb)

    K_full_view = TensorView(K_full.reshape((d_head, KVDP, B_attn * S_tkg))).select(dim=1, index=dynamic_KVDP_rank_sb)
    K_tkg_sb_out = sbm.alloc_stack((d_head, B_attn * S_tkg), dtype=kv_dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=K_tkg_sb_out, src=K_full_view.get_view())

    # ========== V: slice batch ==========
    # (B, kv_heads=1, S_tkg, d_head) -> (KVDP, B_attn, kv_heads=1, S_tkg, d_head) -> select -> (B_attn, kv_heads=1, S_tkg, d_head)
    V_tkg_hbm_batch_slice_view = TensorView(V_tkg_hbm.reshape((KVDP, B_attn, kv_heads, S_tkg, d_head))).select(
        dim=0, index=dynamic_KVDP_rank_sb
    )
    V_tkg_hbm_out = nl.ndarray(
        V_tkg_hbm_batch_slice_view.shape, dtype=kv_dtype, buffer=nl.shared_hbm, name="V_tkg_hbm_out"
    )
    nisa.dma_copy(dst=V_tkg_hbm_out, src=V_tkg_hbm_batch_slice_view.get_view())

    return Q_tkg_sb_out, K_tkg_sb_out, V_tkg_hbm_out


def _KVDP_attention_output_collectives(
    attn_sb: nl.ndarray,
    V_tkg_hbm: nl.ndarray,
    KVDP: int,
    B_attn: int,
    q_heads: int,
    d_head: int,
    S_tkg: int,
    replica_group: ReplicaGroup,
    sbm: SbufManager,
    collective_mode: KVDPCollectiveMode,
    dynamic_KVDP_rank_sb: nl.ndarray,
):
    """Output collectives for KV data parallelism.

    Dispatches attention output redistribution to the appropriate helper based on
    collective_mode, then copies V from HBM to SBUF for KV cache update.

    Pseudocode:

        # attn: (d, B_attn*KVDP*q*S) @SBUF -> _KVDP_attn_output_{mode} -> (d, B*q*S) @SBUF
        # V: (B_attn, kv, S, d) @HBM -> copy to SBUF -> (B_attn*S, d) @SBUF

    Args:

        attn_sb (nl.ndarray): [d_head, B_attn * q_heads * KVDP * S_tkg] @ SBUF - attention output
        V_tkg_hbm (nl.ndarray): [B_attn, kv_heads=1, S_tkg, d_head] @ HBM - V for this rank's batch slice
        KVDP (int): KV data parallelism degree (number of ranks)
        B_attn (int): Batch size per rank for attention (B / KVDP)
        q_heads (int): Number of query heads per rank (after slice)
        d_head (int): Head dimension
        S_tkg (int): Token generation sequence length
        replica_group (ReplicaGroup): Replica group for collective ops
        sbm: SBUF memory manager
        dynamic_KVDP_rank_sb (nl.ndarray): [1, 1] @ SBUF, this rank's position within its KVDP replica group (0 to KVDP-1).
            Unused when collective_mode is ALL_TO_ALL.

    Returns:
        attn_out (nl.ndarray): [d_head, B * q_heads * S_tkg] @ SBUF - gathered attention output
        V_tkg_sb (nl.ndarray): [B_attn * S_tkg, d_head] @ SBUF - V for KV cache update

    Notes:
        - V is copied back to SBUF for KV cache update which requires SBUF input
        - Each rank gets its own q_heads slice of the gathered attention output
    """

    # Shape assertions
    q_heads_attn = q_heads * KVDP
    kv_heads = V_tkg_hbm.shape[1]
    kernel_assert(
        attn_sb.shape == (d_head, B_attn * q_heads_attn * S_tkg),
        f"attn_sb shape mismatch: {attn_sb.shape} != {(d_head, B_attn * q_heads_attn * S_tkg)}",
    )
    kernel_assert(
        V_tkg_hbm.shape == (B_attn, kv_heads, S_tkg, d_head),
        f"V_tkg_hbm shape mismatch: {V_tkg_hbm.shape} != {(B_attn, kv_heads, S_tkg, d_head)}",
    )

    # ========== Attention output: collective redistribution ==========

    if collective_mode == KVDPCollectiveMode.ALL_TO_ALL:
        attn_final_sb = _KVDP_attn_output_all_to_all(
            attn_sb,
            q_heads,
            d_head,
            KVDP,
            B_attn,
            S_tkg,
            replica_group,
            sbm,
        )
    else:
        attn_final_sb = _KVDP_attn_output_all_gather_slice(
            attn_sb,
            q_heads,
            d_head,
            KVDP,
            B_attn,
            S_tkg,
            replica_group,
            sbm,
            dynamic_KVDP_rank_sb,
        )

    # ========== V: copy from HBM to SBUF for KV cache update ==========
    # When B_attn*S_tkg > pmax, V stays on HBM (cache update handles this via V_tkg_hbm)
    if B_attn * S_tkg <= nl.tile_size.pmax:
        V_tkg_sb = nl.ndarray((B_attn * S_tkg, d_head), dtype=V_tkg_hbm.dtype, buffer=nl.sbuf)
        kernel_assert(kv_heads == 1, f"kv_heads must be 1 for V reshape, got {kv_heads}")
        nisa.dma_copy(V_tkg_sb, V_tkg_hbm.reshape((B_attn * S_tkg, d_head)))
    else:
        V_tkg_sb = None

    return attn_final_sb, V_tkg_sb


def _KVDP_Q_input_all_to_all(Q_tkg_sb, q_heads, d_head, KVDP, B, B_attn, S_tkg, dtype, replica_group, sbm: SbufManager):
    """Q input redistribution using all_to_all.

    Rearranges Q to put KVDP batch groups on dim 0, then exchanges chunks so each rank
    gets all heads for its local batch slice.

    Pseudocode:

        if q_heads > 1:
            1. tensor_copy rearrange: (d, B, q, S) @SBUF -> (d, KVDP, q, B_attn, S) @SBUF
            2. Tiled nc_transpose: (d, KVDP*q, B_attn, S) @SBUF -> (KVDP*q, B_attn, S, d) @SBUF

        3. dma_copy: (KVDP*q, B_attn, S, d) @SBUF -> @HBM
        4. all_to_all dim=0: (KVDP*q, B_attn, S, d) @HBM -> (KVDP*q, B_attn, S, d) @HBM
        5. dma_copy: (KVDP*q, B_attn, S, d) @HBM -> @SBUF

        if q_heads > 1:
            6. Tiled nc_transpose: (KVDP*q, B_attn, S, d) @SBUF -> (d, q_attn, B_attn, S) @SBUF
            7. tensor_copy rearrange: (d, q_attn, B_attn, S) @SBUF -> (d, B_attn, q_attn, S) @SBUF

    When q_heads==1, skips steps 1, 2, 6, 7 (no tensor_copy or nc_transpose needed):
    dma_copy to HBM, dma_copy rearrange in HBM, all_to_all, dma_copy rearrange to SBUF.

    Args:
        Q_tkg_sb (nl.ndarray): [d_head, B * q_heads * S_tkg] @ SBUF

    Returns:
        Q_tkg_sb_out (nl.ndarray): [d_head, B_attn * q_heads * KVDP * S_tkg] @ SBUF
    """
    q_heads_attn = q_heads * KVDP
    nBS = q_heads_attn * B_attn * S_tkg
    # all_to_all requires Mesh algorithm (≥4 ranks with lnc2 or ≥8 ranks with lnc1).
    kernel_assert(KVDP >= 4, f"ALL_TO_ALL requires KVDP >= 4, got {KVDP}")
    Q_tkg_sb_out = sbm.alloc_stack((d_head, nBS), dtype=dtype, buffer=nl.sbuf)
    Q_src_hbm = nl.ndarray((q_heads_attn, d_head, B_attn, S_tkg), dtype=dtype, buffer=nl.shared_hbm, name="Q_a2a_src")

    # Step 1-2: rearrange + tiled transpose (q_heads > 1 only)
    if q_heads > 1:
        kernel_assert(d_head <= nl.tile_size.pmax, f"d_head must be <= {nl.tile_size.pmax}, got {d_head}")
        # Rearrange: (d, B, q, S) -> (d, KVDP, q, B_attn, S)
        # tensor_copy materializes the rearranged view into contiguous SBUF memory for nc_transpose
        Q_tkg_sb_rearranged = nl.ndarray((d_head, q_heads * B * S_tkg), dtype=dtype, buffer=nl.sbuf)
        Q_tkg_sb_view = TensorView(Q_tkg_sb.reshape((d_head, KVDP, B_attn, q_heads, S_tkg))).rearrange(
            ("d", "KVDP", "B", "n", "S"), ("d", "KVDP", "n", "B", "S"), {}
        )
        nisa.tensor_copy(Q_tkg_sb_rearranged, Q_tkg_sb_view.get_view())
        # Tiled transpose + dma_copy: (d, KVDP*q*B_attn*S) -> (KVDP*q*B_attn*S, d) -> HBM
        tile_sz = nl.tile_size.pmax
        Q_src_hbm_flat = Q_src_hbm.reshape((nBS, d_head))
        for t_start in range(0, nBS, tile_sz):
            t_size = min(tile_sz, nBS - t_start)
            Q_psum = nl.ndarray((t_size, d_head), dtype=dtype, buffer=nl.psum)
            nisa.nc_transpose(Q_psum, Q_tkg_sb_rearranged[:, nl.ds(t_start, t_size)])
            Q_tile_sb = nl.ndarray((t_size, d_head), dtype=dtype, buffer=nl.sbuf)
            # Copy PSUM -> SBUF
            nisa.tensor_copy(Q_tile_sb, Q_psum)
            nisa.dma_copy(Q_src_hbm_flat[nl.ds(t_start, t_size), :], Q_tile_sb)

    # Step 3: dma_copy SBUF -> HBM (q_heads == 1 only; q>1 fused in tiled loop above)
    if q_heads == 1:
        Q_src_hbm_view = (
            TensorView(Q_src_hbm.reshape((KVDP, d_head, B_attn * S_tkg)))
            .rearrange(("KVDP", "d", "B"), ("d", "KVDP", "B"), {})
            .get_view()
        )
        nisa.dma_copy(Q_src_hbm_view, Q_tkg_sb.reshape((d_head, KVDP, B_attn * S_tkg)))

    # Step 4: all_to_all
    Q_dst_hbm = nl.ndarray(Q_src_hbm.shape, dtype=dtype, buffer=nl.shared_hbm, name="Q_a2a_dst")
    ncc.all_to_all(dsts=[Q_dst_hbm], srcs=[Q_src_hbm], replica_group=replica_group, collective_dim=0)

    # Step 5: dma_copy HBM -> SBUF
    if q_heads == 1:
        # Load Q_dst_hbm into SBUF in native (d, q_attn, B_attn, S) layout where
        # B_attn is innermost-contiguous on both sides, then rearrange in SBUF to
        # the (d, B_attn, q_attn, S) layout the downstream kernel expects.
        Q_tkg_sb_qBS = sbm.alloc_stack((d_head, nBS), dtype=dtype, buffer=nl.sbuf)
        Q_hbm_native = (
            TensorView(Q_dst_hbm.reshape((q_heads_attn, d_head, B_attn, S_tkg)))
            .rearrange(("n", "d", "B", "S"), ("d", "n", "B", "S"), {})
            .get_view()
        )
        nisa.dma_copy(Q_tkg_sb_qBS.reshape((d_head, q_heads_attn, B_attn, S_tkg)), Q_hbm_native)

        Q_sbuf_rearranged = (
            TensorView(Q_tkg_sb_qBS.reshape((d_head, q_heads_attn, B_attn, S_tkg)))
            .rearrange(("d", "n", "B", "S"), ("d", "B", "n", "S"), {})
            .get_view()
        )
        nisa.tensor_copy(Q_tkg_sb_out, Q_sbuf_rearranged)
    else:
        # Step 6-7: dma_copy + tiled transpose + rearrange (q_heads > 1)
        Q_hbm_src = Q_dst_hbm.reshape((nBS, d_head))
        tile_sz = nl.tile_size.pmax
        Q_transposed = sbm.alloc_stack((d_head, nBS), dtype=dtype, buffer=nl.sbuf)
        for t_start in range(0, nBS, tile_sz):
            t_size = min(tile_sz, nBS - t_start)
            Q_tile_sb = nl.ndarray((t_size, d_head), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(Q_tile_sb, Q_hbm_src[nl.ds(t_start, t_size), :])
            Q_psum = nl.ndarray((d_head, t_size), dtype=dtype, buffer=nl.psum)
            nisa.nc_transpose(Q_psum, Q_tile_sb)
            nisa.tensor_copy(Q_transposed[:, nl.ds(t_start, t_size)], Q_psum)
        # Rearrange: (d, q_attn, B_attn, S) -> (d, B_attn, q_attn, S)
        Q_view = TensorView(Q_transposed.reshape((d_head, q_heads_attn, B_attn, S_tkg))).rearrange(
            ('d', 'n', 'B', 'S'), ('d', 'B', 'n', 'S'), {}
        )
        nisa.tensor_copy(Q_tkg_sb_out, Q_view.get_view())

    return Q_tkg_sb_out


def _KVDP_Q_input_all_gather_slice(
    Q_tkg_sb, q_heads, d_head, KVDP, B, B_attn, S_tkg, dtype, replica_group, sbm: SbufManager, dynamic_KVDP_rank_sb
):
    """Q input redistribution using all_gather + KVDP_rank slice.

    Pseudocode:

        if q_heads > 1:
            1. tensor_copy rearrange: (d, B, q, S) @SBUF -> (d, q, B, S) @SBUF
            2. Tiled nc_transpose: (d, q, B, S) @SBUF -> (q, B, S, d) @SBUF

        3. dma_copy: (q, B, S, d) @SBUF -> @HBM
        4. all_gather dim=0: (q, B, S, d) @HBM -> (KVDP*q, B, S, d) @HBM
        5. dma_copy KVDP_rank slice: (KVDP*q, B, S, d) @HBM -> (KVDP*q, B_attn, S, d) @HBM

        if q_heads > 1:
            6. dma_copy: (KVDP*q, B_attn, S, d) @HBM -> @SBUF
            7. Tiled nc_transpose: (KVDP*q, B_attn, S, d) @SBUF -> (d, q_attn, B_attn, S) @SBUF
            8. tensor_copy rearrange: (d, q_attn, B_attn, S) @SBUF -> (d, B_attn, q_attn, S) @SBUF

    When q_heads==1, skips steps 1, 2, 6, 7, 8 (no tensor_copy or nc_transpose needed):
    dma_copy to HBM, all_gather on d dim, dma_copy KVDP_rank slice, dma_copy rearrange to SBUF.

    Args:
        Q_tkg_sb (nl.ndarray): [d_head, B * q_heads * S_tkg] @ SBUF
        dynamic_KVDP_rank_sb (nl.ndarray): [1, 1] @ SBUF, this rank's position within its KVDP replica group (0 to KVDP-1).

    Returns:
        Q_tkg_sb_out (nl.ndarray): [d_head, B_attn * q_heads * KVDP * S_tkg] @ SBUF
    """
    q_heads_attn = q_heads * KVDP
    nBS = q_heads_attn * B_attn * S_tkg
    if q_heads == 1:
        # Optimized path for q_heads=1: no transpose needed
        # SBUF -> HBM for collectives
        Q_hbm = nl.ndarray((d_head, B * q_heads * S_tkg), dtype=dtype, buffer=nl.shared_hbm, name="Q_hbm")
        nisa.dma_copy(Q_hbm, Q_tkg_sb)
        # all_gather on dim=0: (d_head, B*S_tkg) -> (q_heads_attn*d_head, B*S_tkg) where q_heads_attn=KVDP*1
        Q_gathered_hbm = nl.ndarray(
            (q_heads_attn * d_head, B * S_tkg), dtype=dtype, buffer=nl.shared_hbm, name="Q_gathered_hbm"
        )
        ncc.all_gather(dsts=[Q_gathered_hbm], srcs=[Q_hbm], replica_group=replica_group, collective_dim=0)

        # Slice Q batch + rearrange into SBUF in one DMA.
        # (q_heads_attn, d_head, KVDP, B_attn, S_tkg) -> select(dim=2) -> (q_heads_attn, d_head, B_attn, S_tkg)
        #                                                                  -> rearrange -> (d_head, B_attn, q_heads_attn, S_tkg)
        Q_gathered_view = (
            TensorView(Q_gathered_hbm.reshape((q_heads_attn, d_head, KVDP, B_attn, S_tkg)))
            .select(dim=2, index=dynamic_KVDP_rank_sb)
            .rearrange(("n", "d", "B", "S"), ("d", "B", "n", "S"), {})
        )
        Q_tkg_sb_out = sbm.alloc_stack((d_head, B_attn * q_heads_attn * S_tkg), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(Q_tkg_sb_out.reshape((d_head, B_attn, q_heads_attn, S_tkg)), Q_gathered_view.get_view())
    else:
        # General path: transpose to get q_heads on dim=0 for all_gather
        # Transpose Q to HBM: (d_head, B*q_heads*S) -> (q_heads, B, S_tkg, d_head)
        # Need q_heads on dim=0 for all_gather on Q heads
        kernel_assert(d_head <= nl.tile_size.pmax, f"d_head must be <= {nl.tile_size.pmax}, got {d_head}")
        # Rearrange q_heads to first dim of free dim before transpose:
        #   (d_head, B, q_heads, S) -> (d_head, q_heads, B, S)
        Q_tkg_sb_rearranged = nl.ndarray((d_head, q_heads * B * S_tkg), dtype=dtype, buffer=nl.sbuf)
        Q_tkg_sb_view = TensorView(Q_tkg_sb.reshape((d_head, B, q_heads, S_tkg))).rearrange(
            ("d", "B", "n", "S"), ("d", "n", "B", "S"), {}
        )
        nisa.tensor_copy(Q_tkg_sb_rearranged, Q_tkg_sb_view.get_view())
        # Tiled transpose + dma_copy: (d_head, q_heads*B*S) -> (q_heads*B*S, d_head) -> HBM
        tile_sz = nl.tile_size.pmax
        Q_hbm = nl.ndarray((q_heads, B, S_tkg, d_head), dtype=dtype, buffer=nl.shared_hbm, name="Q_hbm")
        Q_hbm_flat = Q_hbm.reshape((nBS, d_head))
        for t_start in range(0, nBS, tile_sz):
            t_size = min(tile_sz, nBS - t_start)
            Q_psum = nl.ndarray((t_size, d_head), dtype=dtype, buffer=nl.psum)
            nisa.nc_transpose(Q_psum, Q_tkg_sb_rearranged[:, nl.ds(t_start, t_size)])
            Q_tile_sb = nl.ndarray((t_size, d_head), dtype=dtype, buffer=nl.sbuf)
            # Copy PSUM -> SBUF
            nisa.tensor_copy(Q_tile_sb, Q_psum)
            # SBUF -> HBM with target shape (q_heads, B, S_tkg, d_head) for collectives
            nisa.dma_copy(Q_hbm_flat[nl.ds(t_start, t_size), :], Q_tile_sb)

        # all_gather Q on head dim: (q_heads, B, S_tkg, d_head) -> (KVDP*q_heads, B, S_tkg, d_head)
        Q_gathered_hbm = nl.ndarray(
            (q_heads_attn, B, S_tkg, d_head), dtype=dtype, buffer=nl.shared_hbm, name="Q_gathered_hbm"
        )
        ncc.all_gather(dsts=[Q_gathered_hbm], srcs=[Q_hbm], replica_group=replica_group, collective_dim=0)

        # Slice Q batch
        # Reshape to (q_heads_attn, KVDP, B_attn, S_tkg, d_head)
        #                            ^---- select batch on dim=1
        #
        # Why we need two dma_copy below: 1) HBM->HBM batch slice 2) HBM->SBUF transpose
        #
        # Step  Tensor              Shape                    Physical Strides                          Contiguous?
        # ----  ------              -----                    ----------------                          -----------
        # 1     Q_gathered_hbm      (H, B, S, d)             (B·S·d, S·d, d, 1)                        Yes
        # 2     reshape             (H, KVDP, B_attn, S, d)  (KVDP·B_attn·S·d, B_attn·S·d, S·d, d, 1)  Yes
        # 3     select(dim=1)       (H, B_attn, S, d)        (KVDP·B_attn·S·d, S·d, d, 1)              No - stride[0] has KVDP gap
        # 4     Q_sliced_hbm (DMA)  (H, B_attn, S, d)        (B_attn·S·d, S·d, d, 1)                   Yes
        # 5     reshape for SBUF    (H·B_attn·S, d)          (d, 1)                                    Yes
        #
        # After select (step 3), stride[0] still has the KVDP factor, creating gaps where other ranks' data lives.
        # We can't flatten this non-contiguous view to 2D for SBUF load.
        # Additionally, dynamic DMA requires src/dst to have same number of dimensions (4D view -> 2D SBUF fails).
        # The DMA to Q_sliced_hbm materializes the slice into contiguous memory, enabling the reshape in step 5.
        Q_gathered_hbm_batch_slice_view = TensorView(
            Q_gathered_hbm.reshape((q_heads_attn, KVDP, B_attn, S_tkg, d_head))
        ).select(dim=1, index=dynamic_KVDP_rank_sb)
        Q_sliced_hbm = nl.ndarray(
            Q_gathered_hbm_batch_slice_view.shape, dtype=dtype, buffer=nl.shared_hbm, name="Q_sliced_hbm"
        )
        nisa.dma_copy(dst=Q_sliced_hbm, src=Q_gathered_hbm_batch_slice_view.get_view())

        # dma_copy + tiled transpose: (q_heads_attn, B_attn, S_tkg, d_head) -> (d_head, B_attn*q_heads_attn*S_tkg)
        tile_sz = nl.tile_size.pmax
        Q_tkg_sb_out = sbm.alloc_stack((d_head, B_attn * q_heads_attn * S_tkg), dtype=dtype, buffer=nl.sbuf)
        Q_sliced_flat = Q_sliced_hbm.reshape((nBS, d_head))
        for t_start in range(0, nBS, tile_sz):
            t_size = min(tile_sz, nBS - t_start)
            Q_tile_sb = nl.ndarray((t_size, d_head), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(Q_tile_sb, Q_sliced_flat[nl.ds(t_start, t_size), :])
            Q_psum = nl.ndarray((d_head, t_size), dtype=dtype, buffer=nl.psum)
            nisa.nc_transpose(Q_psum, Q_tile_sb)
            # PSUM -> SBUF
            nisa.tensor_copy(Q_tkg_sb_out[:, nl.ds(t_start, t_size)], Q_psum)
        # Rearrange: (d_head, q_heads_attn, B_attn, S_tkg) -> (d_head, B_attn, q_heads_attn, S_tkg)
        Q_tkg_sb_rearranged_out = sbm.alloc_stack((d_head, B_attn * q_heads_attn * S_tkg), dtype=dtype, buffer=nl.sbuf)
        Q_out_view = TensorView(Q_tkg_sb_out.reshape((d_head, q_heads_attn, B_attn, S_tkg))).rearrange(
            ('d', 'n', 'B', 'S'), ('d', 'B', 'n', 'S'), {}
        )
        nisa.tensor_copy(Q_tkg_sb_rearranged_out, Q_out_view.get_view())
        Q_tkg_sb_out = Q_tkg_sb_rearranged_out

    return Q_tkg_sb_out


def _KVDP_attn_output_all_to_all(attn_sb, q_heads, d_head, KVDP, B_attn, S_tkg, replica_group, sbm: SbufManager):
    """Attention output redistribution using all_to_all.

    Pseudocode:

        1. tensor_copy rearrange: (d, B_attn, q_attn, S) @SBUF -> (d, q_attn, B_attn, S) @SBUF
        2. Tiled nc_transpose: (d, q_attn, B_attn, S) @SBUF -> (q_attn, B_attn, d, S) @SBUF
        3. dma_copy: (q_attn, B_attn, d, S) @SBUF -> @HBM
        4. all_to_all dim=0: (q_attn, B_attn, d, S) @HBM -> (q_attn, B_attn, d, S) @HBM
        5. dma_copy: (q_attn, B_attn, d, S) @HBM -> @SBUF
        6. Tiled nc_transpose: (q_attn, B_attn, d, S) @SBUF -> (d, KVDP, q, B_attn, S) @SBUF
        7. tensor_copy rearrange: (d, KVDP, q, B_attn, S) @SBUF -> (d, B, q, S) @SBUF

    Args:
        attn_sb (nl.ndarray): [d_head, B_attn * q_heads * KVDP * S_tkg] @ SBUF

    Returns:
        attn_final_sb (nl.ndarray): [d_head, B * q_heads * S_tkg] @ SBUF
    """
    q_heads_attn = q_heads * KVDP
    nBS = q_heads_attn * B_attn * S_tkg
    B = B_attn * KVDP
    dtype = attn_sb.dtype
    # all_to_all uses Mesh algorithm which requires ≥4 ranks (lnc2) or ≥8 ranks (lnc1)
    kernel_assert(KVDP >= 4, f"ALL_TO_ALL requires KVDP >= 4, got {KVDP}")
    # Rearrange in SBUF: (d, B_attn, q_heads_attn, S) -> (d, q_heads_attn, B_attn, S)
    attn_sb_rearranged = nl.ndarray((d_head, B_attn * q_heads_attn * S_tkg), dtype=dtype, buffer=nl.sbuf)
    attn_sb_view = TensorView(attn_sb.reshape((d_head, B_attn, q_heads_attn, S_tkg))).rearrange(
        ("d", "B", "n", "S"), ("d", "n", "B", "S"), {}
    )
    nisa.tensor_copy(attn_sb_rearranged, attn_sb_view.get_view())

    # Tiled transpose + dma_copy to HBM: (d, q_attn*B_attn*S) -> (q_attn*B_attn*S, d) -> HBM
    tile_sz = nl.tile_size.pmax
    attn_src = nl.ndarray((q_heads_attn, B_attn, d_head, S_tkg), dtype=dtype, buffer=nl.shared_hbm, name="attn_a2a_src")
    attn_src_flat = attn_src.reshape((nBS, d_head))
    for t_start in range(0, nBS, tile_sz):
        t_size = min(tile_sz, nBS - t_start)
        psum_t = nl.ndarray((t_size, d_head), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(psum_t, attn_sb_rearranged[:, nl.ds(t_start, t_size)])
        attn_tile_sb = nl.ndarray((t_size, d_head), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_copy(attn_tile_sb, psum_t)
        nisa.dma_copy(attn_src_flat[nl.ds(t_start, t_size), :], attn_tile_sb)

    # all_to_all on dim 0
    attn_dst = nl.ndarray((q_heads_attn, B_attn, d_head, S_tkg), dtype=dtype, buffer=nl.shared_hbm, name="attn_a2a_dst")
    ncc.all_to_all(dsts=[attn_dst], srcs=[attn_src], replica_group=replica_group, collective_dim=0)

    # dma_copy + tiled transpose back to SBUF
    attn_transposed_back = sbm.alloc_stack((d_head, nBS), dtype=dtype, buffer=nl.sbuf)
    attn_dst_flat = attn_dst.reshape((q_heads_attn * B_attn * S_tkg, d_head))
    for t_start in range(0, nBS, tile_sz):
        t_size = min(tile_sz, nBS - t_start)
        attn_tile = nl.ndarray((t_size, d_head), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(attn_tile, attn_dst_flat[nl.ds(t_start, t_size), :])
        psum = nl.ndarray((d_head, t_size), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(psum, attn_tile)
        nisa.tensor_copy(attn_transposed_back[:, nl.ds(t_start, t_size)], psum)

    # Rearrange: (d, KVDP, q_heads, B_attn, S) -> (d, KVDP, B_attn, q_heads, S) = (d, B, q_heads, S)
    attn_final_sb = nl.ndarray((d_head, nBS), dtype=dtype, buffer=nl.sbuf)
    attn_view = TensorView(attn_transposed_back.reshape((d_head, KVDP, q_heads, B_attn, S_tkg))).rearrange(
        ('d', 'KVDP', 'n', 'B', 'S'), ('d', 'KVDP', 'B', 'n', 'S'), {}
    )
    nisa.tensor_copy(attn_final_sb, attn_view.get_view())
    return attn_final_sb


def _KVDP_attn_output_all_gather_slice(
    attn_sb,
    q_heads,
    d_head,
    KVDP,
    B_attn,
    S_tkg,
    replica_group,
    sbm: SbufManager,
    dynamic_KVDP_rank_sb,
):
    """Attention output redistribution using all_gather + KVDP_rank slice.

    Pseudocode:

        1. Tiled nc_transpose: (d, B_attn*q_attn*S) @SBUF -> (B_attn, q_attn, d, S) @SBUF
        2. dma_copy: (B_attn, q_attn, d, S) @SBUF -> @HBM
        3. all_gather dim=0: (B_attn, q_attn, d, S) @HBM -> (B, q_attn, d, S) @HBM
        4. dma_copy KVDP_rank slice: (B, q_attn, d, S) @HBM -> (B, q, d, S) @HBM
        5. dma_copy: (B, q, d, S) @HBM -> @SBUF
        6. Tiled nc_transpose: (B, q, d, S) @SBUF -> (d, B*q*S) @SBUF

    Args:
        attn_sb (nl.ndarray): [d_head, B_attn * q_heads * KVDP * S_tkg] @ SBUF
        dynamic_KVDP_rank_sb (nl.ndarray): [1, 1] @ SBUF, this rank's position within its KVDP replica group (0 to KVDP-1).

    Returns:
        attn_final_sb (nl.ndarray): [d_head, B * q_heads * S_tkg] @ SBUF
    """
    q_heads_attn = q_heads * KVDP
    nBS = q_heads_attn * B_attn * S_tkg
    B = B_attn * KVDP
    dtype = attn_sb.dtype
    # Tiled transpose + dma_copy to HBM: (d, B_attn*q_attn*S) -> (B_attn*q_attn*S, d) -> HBM
    tile_sz = nl.tile_size.pmax
    attn_hbm = nl.ndarray(
        (B_attn, q_heads_attn, d_head, S_tkg), dtype=dtype, buffer=nl.shared_hbm, name="attn_pre_gather"
    )
    attn_hbm_flat = attn_hbm.reshape((nBS, d_head))
    for t_start in range(0, nBS, tile_sz):
        t_size = min(tile_sz, nBS - t_start)
        psum_t = nl.ndarray((t_size, d_head), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(psum_t, attn_sb[:, nl.ds(t_start, t_size)])
        attn_tile_sb = nl.ndarray((t_size, d_head), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_copy(attn_tile_sb, psum_t)
        nisa.dma_copy(attn_hbm_flat[nl.ds(t_start, t_size), :], attn_tile_sb)

    # all_gather on batch dim
    attn_gathered = nl.ndarray(
        (B, q_heads_attn, d_head, S_tkg), dtype=dtype, buffer=nl.shared_hbm, name="attn_gathered"
    )
    ncc.all_gather(dsts=[attn_gathered], srcs=[attn_hbm], replica_group=replica_group, collective_dim=0)

    # Slice heads with dynamic_KVDP_rank_sb
    attn_gathered_head_slice_view = TensorView(attn_gathered.reshape((B, KVDP, q_heads, d_head, S_tkg))).select(
        dim=1, index=dynamic_KVDP_rank_sb
    )
    attn_sliced = nl.ndarray(attn_gathered_head_slice_view.shape, dtype=dtype, buffer=nl.shared_hbm, name="attn_sliced")
    nisa.dma_copy(dst=attn_sliced, src=attn_gathered_head_slice_view.get_view())

    # dma_copy + tiled transpose back to SBUF: (B*q*S, d) -> (d, B*q*S)
    attn_final_sb = sbm.alloc_stack((d_head, nBS), dtype=dtype, buffer=nl.sbuf)
    attn_sliced_flat = attn_sliced.reshape((nBS, d_head))
    for t_start in range(0, nBS, tile_sz):
        t_size = min(tile_sz, nBS - t_start)
        attn_tile = nl.ndarray((t_size, d_head), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(attn_tile, attn_sliced_flat[nl.ds(t_start, t_size), :])
        psum = nl.ndarray((d_head, t_size), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(psum, attn_tile)
        nisa.tensor_copy(attn_final_sb[:, nl.ds(t_start, t_size)], psum)
    return attn_final_sb
