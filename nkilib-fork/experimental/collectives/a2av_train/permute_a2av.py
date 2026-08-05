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
"""All-to-all-v dispatch kernel for MoE training.

Permutes tokens by destination EP rank and exchanges them via ``ncc.all_to_all_v``.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup
from nki.isa.constants import oob_mode

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from .a2av_train_utils import (
    _exclusive_cumsum_u32,
    _validate_a2av_indices_counts,
    _write_a2av_v_metadata,
)

TILE_SIZE = 128


@nki.jit
def permute_a2av(
    hidden_states: nl.ndarray,
    send_indices: nl.ndarray,
    send_counts: nl.ndarray,
    replica_group: ReplicaGroup,
) -> tuple[nl.ndarray, nl.ndarray]:
    """Permute tokens by destination EP rank and dispatch via all-to-all-v.

    Gathers tokens per destination EP rank into a packed send buffer at
    ``cumsum(send_counts)`` row offsets, builds the ``(4, EP)`` uint32
    metadata tensor, then exchanges via ``ncc.all_to_all_v``. The receive-side
    layout is determined by the runtime (packed cumsum of recv_counts).

    TODO: Specify intended usage range (e.g., T per rank, H, EP degree).

    Dimensions:
        T:  number of local tokens (SP sharded).
        H:  hidden dimension.
        EP: number of expert-parallel ranks (== replica_group size).

    Args:
        hidden_states (nl.ndarray): [T, H]@HBM. Input tokens (bf16/fp16/fp32).
        send_indices (nl.ndarray): [T, EP]@HBM, int32. Source-row indices per
            destination rank. Entries with value = T are skipped (via
            ``oob_mode.skip``), so slots beyond ``send_counts[d]`` for
            destination ``d`` are no-ops.
        send_counts (nl.ndarray): [1, EP]@HBM, int32/uint32. Tokens sent per
            dest rank.
        replica_group (ReplicaGroup): EP replica group.

    Returns:
        recv_data (nl.ndarray): [EP*T, H]@HBM. Received tokens packed by the
            runtime at ``cumsum(recv_counts)`` offsets.
        metadata (nl.ndarray): [4, EP]@HBM, uint32. After the collective, row 2
            contains the runtime-populated ``recv_counts * H``.

    Pseudocode:
        sdispls = exclusive_cumsum(send_counts)        # [EP]
        send_hbm = zeros((EP*T, H))
        for ep_dst in range(EP):
            for tile_idx in range(div_ceil(T, TILE_SIZE)):
                tile_start = tile_idx * TILE_SIZE
                tile_end = min(tile_start + TILE_SIZE, T)
                # Gather source rows for this destination (OOB slots skipped).
                gathered = hidden_states[send_indices[tile_start:tile_end, ep_dst]]
                # Scatter into packed send buffer at sdispls offsets.
                dst_rows = arange(tile_start, tile_end) + sdispls[ep_dst]
                send_hbm[dst_rows] = gathered
        metadata = build_metadata(send_counts, sdispls, H, EP)   # [4, EP]
        recv_hbm = all_to_all_v(send_hbm, metadata, replica_group)
        return recv_hbm, metadata
    """
    T, H = hidden_states.shape
    EP = send_indices.shape[1]
    dtype = hidden_states.dtype
    NUM_TILES = div_ceil(T, TILE_SIZE)

    # Shape / dtype validation.
    _validate_a2av_indices_counts(send_indices, send_counts, T, EP)
    kernel_assert(
        tuple(hidden_states.shape) == (T, H),
        f"hidden_states must be (T={T}, H={H}), got {tuple(hidden_states.shape)}",
    )

    # Step 1: exclusive cumsum of send_counts -> scatter displacements.
    sc_sb, sdispls_sb, sdispls_hbm = _exclusive_cumsum_u32(send_counts, EP)

    # Step 2: gather tokens into packed send_hbm at cumsum offsets.
    send_hbm = nl.ndarray((EP * T, H), dtype=dtype, buffer=nl.shared_hbm, name="disp_send_pk")
    sdispls_hbm_1d = sdispls_hbm.reshape((EP,))

    for ep_dst in nl.affine_range(EP):
        for tile_idx in nl.affine_range(NUM_TILES):
            ts = tile_idx * TILE_SIZE
            te = min(ts + TILE_SIZE, T)
            tsize = te - ts

            # Load gather indices for this destination. Pre-fill with sentinel
            # T so the tail rows `[tsize:TILE_SIZE]` of the last (partial) tile
            # fall outside the valid source range `[0, T)` and the gather
            # `oob_mode.skip` masks them off.
            idx_sb = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.memset(idx_sb, T)
            nisa.dma_copy(
                dst=idx_sb[0:tsize, 0:1],
                src=send_indices[ts:te, ep_dst : ep_dst + 1],
            )

            # Gather rows from hidden_states via indirect DMA.
            tile_sb = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=tile_sb,
                src=hidden_states.ap(
                    pattern=[[T, TILE_SIZE], [1, H]],
                    offset=0,
                    vector_offset=idx_sb,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
            )

            # Destination rows: iota(TILE_SIZE) + ts + sdispls[ep_dst].
            disp_tile = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=disp_tile,
                src=sdispls_hbm_1d.ap(pattern=[[0, TILE_SIZE], [1, 1]], offset=ep_dst),
            )
            iota_sb = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.iota(dst=iota_sb, pattern=[[0, 1]], offset=0, channel_multiplier=1)
            row_idx = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(data=iota_sb, op0=nl.add, operand0=ts, dst=row_idx)
            nisa.tensor_tensor(row_idx, row_idx, disp_tile, op=nl.add)

            # Scatter tile into packed send buffer.
            nisa.dma_copy(
                dst=send_hbm.ap(
                    pattern=[[EP * T, TILE_SIZE], [1, H]],
                    offset=0,
                    vector_offset=row_idx,
                    indirect_dim=0,
                ),
                src=tile_sb,
                oob_mode=oob_mode.skip,
            )

    # Step 3: build metadata. Rows 2-3 zero; runtime fills
    # row 2 with recv_counts since recv_counts_known=False.
    metadata = nl.ndarray((4, EP), dtype=nl.uint32, buffer=nl.shared_hbm, name="disp_meta")
    _write_a2av_v_metadata(metadata, sc_sb, sdispls_sb, H, EP)

    # Step 4: all-to-all-v. Runtime computes recv_counts and recv_displs.
    recv_hbm = nl.ndarray((EP * T, H), dtype=dtype, buffer=nl.shared_hbm, name="disp_recv_pk")
    ncc.all_to_all_v(
        srcs=[send_hbm],
        dsts=[recv_hbm],
        replica_group=replica_group,
        metadata_tensor=metadata,
        recv_counts_known=False,
        has_rdispls=False,
    )

    recv_data = nl.ndarray((EP * T, H), dtype=dtype, buffer=nl.shared_hbm, name="disp_recv_data")
    metadata_out = nl.ndarray((4, EP), dtype=nl.uint32, buffer=nl.shared_hbm, name="disp_meta_out")
    nisa.dma_copy(dst=recv_data, src=recv_hbm)
    nisa.dma_copy(dst=metadata_out, src=metadata)

    return recv_data, metadata_out
