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
Multi-scale deformable attention kernel for NeuronCore.

This kernel implements multi-scale deformable attention using indirect DMA transpose for efficient
gathering of values from multiple feature pyramid levels.
"""

from dataclasses import dataclass

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager, sizeinbytes
from ...core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ...core.utils.logging import get_logger


@nki.jit
def ms_deformable_attention(
    value: nl.ndarray,
    spatial_shapes: tuple,
    level_start_index: tuple,
    sampling_locations: nl.ndarray,
    attention_weights: nl.ndarray,
    value_layout: str = "BLNC",
    sampling_locations_layout: str = "BQHLP2",
    align_corners: bool = False,
    padding_mode: str = "zeros",
) -> nl.ndarray:
    """
    Multi-scale deformable attention kernel that uses indirect DMA transpose.

    Dimensions:
        B: Batch size
        N_q: Number of queries
        N_h: Number of attention heads
        C_h: Channels per head
        N_l: Number of feature pyramid levels
        N_p: Number of sampling points per query per head per level
        L: Total flattened spatial dimension (sum of H_i * W_i across all levels)
        H_i: Height of feature map at level i
        W_i: Width of feature map at level i

    Args:
        value (nl.ndarray): Value tensor in HBM. Shape depends on value_layout:
            - If value_layout="BLNC": (B, L, N_h, C_h)
            - If value_layout="BNLC": (B, N_h, L, C_h)
        spatial_shapes (tuple): Tuple of (H_i, W_i) tuples specifying spatial dimensions for each level
        level_start_index (tuple): Tuple of start indices for each level in the flattened L dimension
        sampling_locations (nl.ndarray): Normalized sampling coordinates in HBM. Shape depends on layout:
            - If sampling_locations_layout="BQHLP2": (B, N_q, N_h, N_l, N_p, 2)
            - If sampling_locations_layout="B2QHLP": (B, 2, N_q, N_h, N_l, N_p)
        attention_weights (nl.ndarray): Attention weights in HBM, shape (B, N_q, N_h, N_l, N_p)
        value_layout (str): Layout of value tensor, either "BLNC" or "BNLC". Default: "BLNC"
        sampling_locations_layout (str): Layout of sampling_locations, either "BQHLP2" or "B2QHLP". Default: "BQHLP2"
        align_corners (bool): If True, coordinates map [0,1] to [0, H-1]. If False, map to [-0.5, H-0.5]. Default: False
        padding_mode (str): Padding mode for out-of-bounds coordinates, either "zeros" or "border". Default: "zeros"

    Returns:
        output (nl.ndarray): Attention output in HBM, shape (B, N_q, N_h * C_h)

    Pseudocode:
        # For each batch, query, and head:
        for b in range(B):
            for q in range(N_q):
                for h in range(N_h):
                    output[b, q, h*C_h:(h+1)*C_h] = 0
                    # For each level and sampling point:
                    for l in range(N_l):
                        for p in range(N_p):
                            # Get normalized coordinates and scale to pixel space
                            x, y = sampling_locations[b, q, h, l, p]
                            x_scaled = x * (H_l - 1) if align_corners else x * H_l - 0.5
                            y_scaled = y * (W_l - 1) if align_corners else y * W_l - 0.5

                            # Compute bilinear interpolation corners
                            x0, y0 = floor(x_scaled), floor(y_scaled)
                            x1, y1 = x0 + 1, y0 + 1

                            # Clamp to valid range
                            x0_c, y0_c = clamp(x0, 0, H_l-1), clamp(y0, 0, W_l-1)
                            x1_c, y1_c = clamp(x1, 0, H_l-1), clamp(y1, 0, W_l-1)

                            # Compute bilinear weights
                            dx, dy = x_scaled - x0, y_scaled - y0
                            w_00 = (1 - dx) * (1 - dy)
                            w_01 = (1 - dx) * dy
                            w_10 = dx * (1 - dy)
                            w_11 = dx * dy

                            # Apply padding mode (zeros: mask OOB, border: use clamped)
                            if padding_mode == "zeros":
                                # Zero out weights for out-of-bounds corners
                                w_00 *= (x0 == x0_c) * (y0 == y0_c)
                                w_01 *= (x0 == x0_c) * (y1 == y1_c)
                                w_10 *= (x1 == x1_c) * (y0 == y0_c)
                                w_11 *= (x1 == x1_c) * (y1 == y1_c)

                            # Gather values and accumulate weighted sum
                            attn_w = attention_weights[b, q, h, l, p]
                            output[b, q, h*C_h:(h+1)*C_h] += (
                                w_00 * attn_w * value[b, level_start_index[l] + x0_c*W_l + y0_c, h, :] +
                                w_01 * attn_w * value[b, level_start_index[l] + x0_c*W_l + y1_c, h, :] +
                                w_10 * attn_w * value[b, level_start_index[l] + x1_c*W_l + y0_c, h, :] +
                                w_11 * attn_w * value[b, level_start_index[l] + x1_c*W_l + y1_c, h, :]
                            )
    """
    # LNC sharding on Q dimension
    _, lnc, shard_id = get_verified_program_sharding_info("ms_deformable_attention", (0, 1))

    # Build config
    cfg = _build_config(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        value_layout,
        sampling_locations_layout,
        padding_mode,
        lnc,
        shard_id,
    )

    # Extract config values
    P_MAX = cfg.P_MAX
    B, N_q, N_h, C_h = cfg.B, cfg.N_q, cfg.N_h, cfg.C_h
    N_l, N_p = cfg.N_l, cfg.N_p
    L = cfg.L
    spatial_shapes = cfg.spatial_shapes
    level_start_index = cfg.level_start_index
    Q_tile = cfg.Q_tile
    H_tile = cfg.H_tile
    num_q_tiles = cfg.num_q_tiles
    num_h_tiles = cfg.num_h_tiles
    num_C_h_tiles = cfg.num_C_h_tiles
    dtype = cfg.dtype
    h_interleave = cfg.h_interleave
    gather_interleave = cfg.gather_interleave
    total_sbuf_req = cfg.total_sbuf
    q_start_global = cfg.q_start_global
    local_N_q = cfg.local_N_q

    # Allocate output in HBM
    output = nl.ndarray((B, N_q, N_h * C_h), dtype=dtype, buffer=nl.shared_hbm)

    # Initialize SBUF manager
    logger = get_logger("ms_deformable_attention")
    sbm = SbufManager(0, total_sbuf_req, logger=logger)

    # Open H-tile scope
    sbm.open_scope(interleave_degree=h_interleave, name="ms_deformable_attention_h_scope")

    # Loop over batches
    for batch_idx in range(B):
        # Loop over queries
        for q_tile_idx in range(num_q_tiles):
            q_start_local = q_tile_idx * Q_tile
            q_end_local = min(q_start_local + Q_tile, local_N_q)
            q_actual = q_end_local - q_start_local

            # Calculate global query indices for reading from HBM
            q_start = q_start_global + q_start_local
            q_end = q_start_global + q_end_local

            # Loop over head tiles
            for h_tile_idx in range(num_h_tiles):
                h_start = h_tile_idx * H_tile
                h_end = min(h_start + H_tile, N_h)
                h_actual = h_end - h_start

                # ====================================================================
                # Step 1: load x, y, attn_w from HBM
                # ====================================================================
                x = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf)
                y = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf)
                attn_w = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=dtype, buffer=nl.sbuf)

                # Load x and y
                if cfg.sampling_locations_layout == "BQHLP2":
                    # sampling_locations: (B, N_q, N_h, N_l, N_p, 2)
                    nisa.dma_copy(
                        dst=x,
                        src=sampling_locations[batch_idx, q_start:q_end, h_start:h_end, :, :, 0],
                    )
                    nisa.dma_copy(
                        dst=y,
                        src=sampling_locations[batch_idx, q_start:q_end, h_start:h_end, :, :, 1],
                    )
                else:
                    # sampling_locations: (B, 2, N_q, N_h, N_l, N_p)
                    nisa.dma_copy(
                        dst=x,
                        src=sampling_locations[batch_idx, 0, q_start:q_end, h_start:h_end, :, :],
                    )
                    nisa.dma_copy(
                        dst=y,
                        src=sampling_locations[batch_idx, 1, q_start:q_end, h_start:h_end, :, :],
                    )

                # Load attn_w
                nisa.dma_copy(
                    dst=attn_w,
                    src=attention_weights[batch_idx, q_start:q_end, h_start:h_end, :, :],
                )

                # ====================================================================
                # Step 2: Scale coordinates by spatial dimensions based on align_corners
                # ====================================================================
                for h_idx in range(h_actual):
                    for level_idx in range(N_l):
                        H_l, W_l = spatial_shapes[level_idx]
                        if align_corners:
                            # align_corners=True: [0,1] -> [0, W_l-1]
                            nisa.tensor_scalar(
                                dst=x[:, h_idx, level_idx, :],
                                data=x[:, h_idx, level_idx, :],
                                op0=nl.multiply,
                                operand0=float(W_l - 1),
                                engine=nisa.engine.scalar,
                            )
                            nisa.tensor_scalar(
                                dst=y[:, h_idx, level_idx, :],
                                data=y[:, h_idx, level_idx, :],
                                op0=nl.multiply,
                                operand0=float(H_l - 1),
                                engine=nisa.engine.scalar,
                            )
                        else:
                            # align_corners=False: [0,1] -> [-0.5, W_l-0.5]
                            nisa.tensor_scalar(
                                dst=x[:, h_idx, level_idx, :],
                                data=x[:, h_idx, level_idx, :],
                                op0=nl.multiply,
                                operand0=float(W_l),
                                op1=nl.add,
                                operand1=-0.5,
                                engine=nisa.engine.scalar,
                            )
                            nisa.tensor_scalar(
                                dst=y[:, h_idx, level_idx, :],
                                data=y[:, h_idx, level_idx, :],
                                op0=nl.multiply,
                                operand0=float(H_l),
                                op1=nl.add,
                                operand1=-0.5,
                                engine=nisa.engine.scalar,
                            )

                # ====================================================================
                # Step 3: Compute bilinear coordinates x0, y0, x1, y1
                # ====================================================================
                x0_unclamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                y0_unclamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                x1_unclamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)
                y1_unclamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)

                x0_unclamped[0:q_actual, :, :, :] = nl.floor(x=x[0:q_actual, :, :, :], dtype=nl.int32)
                y0_unclamped[0:q_actual, :, :, :] = nl.floor(x=y[0:q_actual, :, :, :], dtype=nl.int32)

                nisa.tensor_scalar(
                    dst=x1_unclamped,
                    data=x0_unclamped,
                    op0=nl.add,
                    operand0=1,
                    engine=nisa.engine.scalar,
                )
                nisa.tensor_scalar(
                    dst=y1_unclamped,
                    data=y0_unclamped,
                    op0=nl.add,
                    operand0=1,
                    engine=nisa.engine.scalar,
                )

                x0_clamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.uint32, buffer=nl.sbuf)
                y0_clamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.uint32, buffer=nl.sbuf)
                x1_clamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.uint32, buffer=nl.sbuf)
                y1_clamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.uint32, buffer=nl.sbuf)

                # Clamp coordinates per level
                for level_idx in range(N_l):
                    H_l, W_l = spatial_shapes[level_idx]

                    # x0, x1
                    nisa.tensor_scalar(
                        dst=x0_clamped[:, :, level_idx, :],
                        data=x0_unclamped[:, :, level_idx, :],
                        op0=nl.minimum,
                        operand0=float(W_l - 1),
                        op1=nl.maximum,
                        operand1=0.0,
                    )
                    nisa.tensor_scalar(
                        dst=x1_clamped[:, :, level_idx, :],
                        data=x1_unclamped[:, :, level_idx, :],
                        op0=nl.minimum,
                        operand0=float(W_l - 1),
                        op1=nl.maximum,
                        operand1=0.0,
                    )

                    # y0, y1
                    nisa.tensor_scalar(
                        dst=y0_clamped[:, :, level_idx, :],
                        data=y0_unclamped[:, :, level_idx, :],
                        op0=nl.minimum,
                        operand0=float(H_l - 1),
                        op1=nl.maximum,
                        operand1=0.0,
                    )
                    nisa.tensor_scalar(
                        dst=y1_clamped[:, :, level_idx, :],
                        data=y1_unclamped[:, :, level_idx, :],
                        op0=nl.minimum,
                        operand0=float(H_l - 1),
                        op1=nl.maximum,
                        operand1=0.0,
                    )

                # ====================================================================
                # Step 4: Compute fractional contributions: dx, dy, 1-dx, 1-dy
                # ====================================================================
                dx_dy = sbm.alloc_stack((q_actual, 2, h_actual, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                one_minus_dx_dy = sbm.alloc_stack((q_actual, 2, h_actual, N_l, N_p), dtype=dtype, buffer=nl.sbuf)

                nisa.tensor_tensor(
                    dst=dx_dy[:, 0, :, :, :],
                    data1=x,
                    data2=x0_unclamped,
                    op=nl.subtract,
                )
                nisa.tensor_tensor(
                    dst=dx_dy[:, 1, :, :, :],
                    data1=y,
                    data2=y0_unclamped,
                    op=nl.subtract,
                )

                dx_dy_2d = dx_dy.reshape((q_actual, 2 * h_actual * N_l * N_p))
                one_minus_dx_dy_2d = one_minus_dx_dy.reshape((q_actual, 2 * h_actual * N_l * N_p))

                nisa.tensor_scalar(
                    dst=one_minus_dx_dy_2d[0:q_actual, :],
                    data=dx_dy_2d[0:q_actual, :],
                    op0=nl.multiply,
                    operand0=-1.0,
                    op1=nl.add,
                    operand1=1.0,
                    engine=nisa.engine.scalar,
                )

                # ====================================================================
                # Step 5: Compute bilinear weights: w_00, w_01, w_10, w_11
                # ====================================================================
                combined_weights = sbm.alloc_stack((q_actual, 4, h_actual, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 0],
                    data1=one_minus_dx_dy[0:q_actual, 0],
                    data2=one_minus_dx_dy[0:q_actual, 1],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 1],
                    data1=one_minus_dx_dy[0:q_actual, 1],
                    data2=dx_dy[0:q_actual, 0],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 2],
                    data1=dx_dy[0:q_actual, 1],
                    data2=one_minus_dx_dy[0:q_actual, 0],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 3],
                    data1=dx_dy[0:q_actual, 0],
                    data2=dx_dy[0:q_actual, 1],
                    op=nl.multiply,
                )

                # ====================================================================
                # Step 6: Compute flat indexes
                # ====================================================================
                flat_idx_all = sbm.alloc_stack((q_actual, h_actual, N_l, N_p, 4), dtype=nl.uint32, buffer=nl.sbuf)

                for h_idx in range(h_actual):
                    h_global = h_start + h_idx
                    for level_idx in range(N_l):
                        H_l, W_l = spatial_shapes[level_idx]
                        level_offset = level_start_index[level_idx]

                        # Compute flat index based on layout
                        if cfg.value_layout == "BLNC":
                            # BLNC: (B, L, N_h, C_h) -> flat_idx = batch_offset + (level_offset + y * W_l + x) * N_h + h_global
                            batch_offset = batch_idx * L * N_h
                            spatial_multiplier = N_h
                            base_offset = level_offset * N_h + h_global
                        else:
                            # BNLC: (B, N_h, L, C_h) -> flat_idx = batch_offset + h_global * L + (level_offset + y * W_l + x)
                            batch_offset = batch_idx * N_h * L
                            spatial_multiplier = 1
                            base_offset = h_global * L + level_offset

                        # flat_idx_00 = batch_offset + (level_offset + y0 * W_l + x0) * spatial_multiplier + base_offset
                        nisa.tensor_scalar(
                            dst=flat_idx_all[:, h_idx, level_idx, :, 0],
                            data=y0_clamped[:, h_idx, level_idx, :],
                            op0=nl.multiply,
                            operand0=int(W_l),
                            engine=nisa.engine.scalar,
                        )
                        nisa.tensor_tensor(
                            dst=flat_idx_all[:, h_idx, level_idx, :, 0],
                            data1=flat_idx_all[:, h_idx, level_idx, :, 0],
                            data2=x0_clamped[:, h_idx, level_idx, :],
                            op=nl.add,
                        )
                        nisa.tensor_scalar(
                            dst=flat_idx_all[:, h_idx, level_idx, :, 0],
                            data=flat_idx_all[:, h_idx, level_idx, :, 0],
                            op0=nl.multiply,
                            operand0=int(spatial_multiplier),
                            op1=nl.add,
                            operand1=int(batch_offset + base_offset),
                            engine=nisa.engine.scalar,
                        )

                        # flat_idx_01 = batch_offset + (level_offset + y0 * W_l + x1) * spatial_multiplier + base_offset
                        nisa.tensor_scalar(
                            dst=flat_idx_all[:, h_idx, level_idx, :, 1],
                            data=y0_clamped[:, h_idx, level_idx, :],
                            op0=nl.multiply,
                            operand0=int(W_l),
                            engine=nisa.engine.scalar,
                        )
                        nisa.tensor_tensor(
                            dst=flat_idx_all[:, h_idx, level_idx, :, 1],
                            data1=flat_idx_all[:, h_idx, level_idx, :, 1],
                            data2=x1_clamped[:, h_idx, level_idx, :],
                            op=nl.add,
                        )
                        nisa.tensor_scalar(
                            dst=flat_idx_all[:, h_idx, level_idx, :, 1],
                            data=flat_idx_all[:, h_idx, level_idx, :, 1],
                            op0=nl.multiply,
                            operand0=int(spatial_multiplier),
                            op1=nl.add,
                            operand1=int(batch_offset + base_offset),
                            engine=nisa.engine.scalar,
                        )

                        # flat_idx_10 = batch_offset + (level_offset + y1 * W_l + x0) * spatial_multiplier + base_offset
                        nisa.tensor_scalar(
                            dst=flat_idx_all[:, h_idx, level_idx, :, 2],
                            data=y1_clamped[:, h_idx, level_idx, :],
                            op0=nl.multiply,
                            operand0=int(W_l),
                            engine=nisa.engine.scalar,
                        )
                        nisa.tensor_tensor(
                            dst=flat_idx_all[:, h_idx, level_idx, :, 2],
                            data1=flat_idx_all[:, h_idx, level_idx, :, 2],
                            data2=x0_clamped[:, h_idx, level_idx, :],
                            op=nl.add,
                        )
                        nisa.tensor_scalar(
                            dst=flat_idx_all[:, h_idx, level_idx, :, 2],
                            data=flat_idx_all[:, h_idx, level_idx, :, 2],
                            op0=nl.multiply,
                            operand0=int(spatial_multiplier),
                            op1=nl.add,
                            operand1=int(batch_offset + base_offset),
                            engine=nisa.engine.scalar,
                        )

                        # flat_idx_11 = batch_offset + (level_offset + y1 * W_l + x1) * spatial_multiplier + base_offset
                        nisa.tensor_scalar(
                            dst=flat_idx_all[:, h_idx, level_idx, :, 3],
                            data=y1_clamped[:, h_idx, level_idx, :],
                            op0=nl.multiply,
                            operand0=int(W_l),
                            engine=nisa.engine.scalar,
                        )
                        nisa.tensor_tensor(
                            dst=flat_idx_all[:, h_idx, level_idx, :, 3],
                            data1=flat_idx_all[:, h_idx, level_idx, :, 3],
                            data2=x1_clamped[:, h_idx, level_idx, :],
                            op=nl.add,
                        )
                        nisa.tensor_scalar(
                            dst=flat_idx_all[:, h_idx, level_idx, :, 3],
                            data=flat_idx_all[:, h_idx, level_idx, :, 3],
                            op0=nl.multiply,
                            operand0=int(spatial_multiplier),
                            op1=nl.add,
                            operand1=int(batch_offset + base_offset),
                            engine=nisa.engine.scalar,
                        )

                # Scale all flat indices by C_h
                flat_idx_all_2d = flat_idx_all.reshape((q_actual, h_actual * N_l * N_p * 4))
                nisa.tensor_scalar(
                    dst=flat_idx_all_2d,
                    data=flat_idx_all_2d,
                    op0=nl.multiply,
                    operand0=int(C_h),
                    engine=nisa.engine.scalar,
                )

                # ====================================================================
                # Step 7: Compute combined weights (bilinear weights * attention weights)
                # ====================================================================
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 0],
                    data1=combined_weights[0:q_actual, 0],
                    data2=attn_w[0:q_actual],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 1],
                    data1=combined_weights[0:q_actual, 1],
                    data2=attn_w[0:q_actual],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 2],
                    data1=combined_weights[0:q_actual, 2],
                    data2=attn_w[0:q_actual],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 3],
                    data1=combined_weights[0:q_actual, 3],
                    data2=attn_w[0:q_actual],
                    op=nl.multiply,
                )

                # ====================================================================
                # Step 7b: Update combined weights for padding_mode zeros OOB accesses
                # ====================================================================
                if padding_mode == "zeros":
                    # Compute remaining coordinate (x0, x1) OOB masks
                    coord_oob = sbm.alloc_stack((q_actual, 4, h_actual, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=coord_oob[:, 0, :, :, :],
                        data1=x0_clamped,
                        data2=x0_unclamped,
                        op=nl.subtract,
                    )
                    nisa.tensor_tensor(
                        dst=coord_oob[:, 1, :, :, :],
                        data1=x1_unclamped,
                        data2=x1_clamped,
                        op=nl.subtract,
                    )
                    nisa.tensor_tensor(
                        dst=coord_oob[:, 2, :, :, :],
                        data1=y0_clamped,
                        data2=y0_unclamped,
                        op=nl.subtract,
                    )
                    nisa.tensor_tensor(
                        dst=coord_oob[:, 3, :, :, :],
                        data1=y1_unclamped,
                        data2=y1_clamped,
                        op=nl.subtract,
                    )

                    # Compute inverse coordinate OOB masks
                    one_minus_coord_oob = sbm.alloc_stack(
                        (q_actual, 4, h_actual, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf
                    )

                    coord_oob_2d = coord_oob.reshape((q_actual, 4 * h_actual * N_l * N_p))
                    one_minus_coord_oob_2d = one_minus_coord_oob.reshape((q_actual, 4 * h_actual * N_l * N_p))
                    nisa.tensor_scalar(
                        dst=one_minus_coord_oob_2d,
                        data=coord_oob_2d,
                        op0=nl.multiply,
                        operand0=-1,
                        op1=nl.add,
                        operand1=1,
                        engine=nisa.engine.scalar,
                    )

                    # Compute corner OOB masks
                    zeros_mask = sbm.alloc_stack((q_actual, 4, h_actual, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=zeros_mask[:, 0, :, :, :],
                        data1=one_minus_coord_oob[:, 0, :, :, :],
                        data2=one_minus_coord_oob[:, 2, :, :, :],
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )
                    nisa.tensor_tensor(
                        dst=zeros_mask[:, 1, :, :, :],
                        data1=one_minus_coord_oob[:, 1, :, :, :],
                        data2=one_minus_coord_oob[:, 2, :, :, :],
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )
                    nisa.tensor_tensor(
                        dst=zeros_mask[:, 2, :, :, :],
                        data1=one_minus_coord_oob[:, 0, :, :, :],
                        data2=one_minus_coord_oob[:, 3, :, :, :],
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )
                    nisa.tensor_tensor(
                        dst=zeros_mask[:, 3, :, :, :],
                        data1=one_minus_coord_oob[:, 1, :, :, :],
                        data2=one_minus_coord_oob[:, 3, :, :, :],
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )

                    # Update combined weights with corner OOB masks
                    combined_weights_2d = combined_weights.reshape((q_actual, 4 * h_actual * N_l * N_p))
                    zeros_mask_2d = zeros_mask.reshape((q_actual, 4 * h_actual * N_l * N_p))
                    nisa.tensor_tensor(
                        dst=combined_weights_2d,
                        data1=combined_weights_2d,
                        data2=zeros_mask_2d,
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )

                # ====================================================================
                # Step 8: Gather and accumulate
                # ====================================================================
                output_accum = []
                for head_idx in range(h_actual):
                    accum = sbm.alloc_stack((q_actual, C_h), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.memset(dst=accum[0:q_actual, :], value=0.0)
                    output_accum.append(accum)

                # Reshape value for vector_select
                value_flat_1d = value.reshape((B * L * N_h * C_h,))

                sbm.open_scope(interleave_degree=gather_interleave, name="ms_deformable_attention_gather_scope")

                for head_idx in range(h_actual):
                    for level_idx in range(N_l):
                        # Reshape to combine points and corners, then select this head and level
                        flat_idx_all_reshaped = flat_idx_all.reshape((q_actual, h_actual, N_l, N_p * 4))
                        min_C_h_P_MAX = min(C_h, P_MAX)

                        total_corners = N_p * 4
                        gathered_all = sbm.alloc_stack(
                            (min_C_h_P_MAX, num_C_h_tiles, total_corners * P_MAX), dtype=dtype, buffer=nl.sbuf, align=32
                        )

                        vector_offset_tensor_view = None
                        if q_actual < P_MAX:
                            flat_idx_padded = sbm.alloc_stack((P_MAX, total_corners), dtype=nl.uint32, buffer=nl.sbuf)
                            nisa.memset(dst=flat_idx_padded, value=0)
                            nisa.tensor_copy(
                                dst=flat_idx_padded[0:q_actual, :],
                                src=flat_idx_all_reshaped[:, head_idx, level_idx, :],
                            )
                            vector_offset_tensor_view = flat_idx_padded
                        else:
                            vector_offset_tensor_view = flat_idx_all_reshaped[:, head_idx, level_idx, :]

                        for c_tile_idx in range(num_C_h_tiles):
                            c_start = c_tile_idx * P_MAX
                            c_end = min(c_start + P_MAX, C_h)
                            c_actual = c_end - c_start

                            nisa.dma_transpose(
                                dst=gathered_all[0:c_actual, c_tile_idx, :],
                                src=value_flat_1d.ap(
                                    pattern=[[c_actual, total_corners * P_MAX], [1, c_actual]],
                                    offset=c_start,
                                    vector_offset=vector_offset_tensor_view,
                                    indirect_dim=0,
                                ),
                                axes=(1, 0),
                            )

                        # Reshape to separate points and corners
                        gathered_reshaped = gathered_all.reshape((min_C_h_P_MAX, num_C_h_tiles, N_p, 4, P_MAX))

                        # Process each point and corner
                        for point_idx in range(N_p):
                            for corner_idx in range(4):
                                gathered_corner = nl.ndarray((P_MAX, C_h), dtype=dtype, buffer=nl.psum)

                                for c_tile_idx in range(num_C_h_tiles):
                                    c_start = c_tile_idx * P_MAX
                                    c_end = min(c_start + P_MAX, C_h)
                                    c_actual = c_end - c_start

                                    nisa.nc_transpose(
                                        dst=gathered_corner[:, c_start:c_end],
                                        data=gathered_reshaped[0:c_actual, c_tile_idx, point_idx, corner_idx, :],
                                        engine=nisa.engine.tensor,
                                    )

                                if corner_idx % 2 == 0:
                                    weighted = sbm.alloc_stack((q_actual, C_h), dtype=dtype, buffer=nl.sbuf)

                                    nisa.tensor_scalar(
                                        dst=weighted,
                                        data=gathered_corner[0:q_actual, :],
                                        operand0=combined_weights[:, corner_idx, head_idx, level_idx, point_idx],
                                        op0=nl.multiply,
                                        engine=nisa.engine.scalar,
                                    )

                                    nisa.tensor_tensor(
                                        dst=output_accum[head_idx][0:q_actual, :],
                                        data1=output_accum[head_idx][0:q_actual, :],
                                        data2=weighted,
                                        op=nl.add,
                                    )
                                else:
                                    nisa.scalar_tensor_tensor(
                                        dst=output_accum[head_idx][0:q_actual, :],
                                        data=gathered_corner[0:q_actual, :],
                                        op0=nl.multiply,
                                        operand0=combined_weights[:, corner_idx, head_idx, level_idx, point_idx],
                                        op1=nl.add,
                                        operand1=output_accum[head_idx][0:q_actual, :],
                                    )

                        sbm.increment_section()

                sbm.close_scope()

                # ====================================================================
                # Step 10: Write accumulated results to HBM
                # ====================================================================
                for head_idx in range(h_actual):
                    h_global = h_start + head_idx
                    nisa.dma_copy(
                        dst=output[batch_idx, q_start:q_end, h_global * C_h : (h_global + 1) * C_h],
                        src=output_accum[head_idx][0:q_actual, :],
                    )

                # Increment section for next h_tile
                sbm.increment_section()

    sbm.close_scope()

    return output


@dataclass(frozen=True)
class MSDeformAttnConfig(nl.NKIObject):
    """Configuration for multi-scale deformable attention."""

    # Input dimensions
    B: int  # Batch size
    N_q: int  # Number of queries (global)
    N_h: int  # Number of heads
    C_h: int  # Channels per head
    N_l: int  # Number of levels
    N_p: int  # Number of sampling points per query per head per level
    L: int  # Total flattened spatial dimension (sum of H_i * W_i)

    # Spatial information
    spatial_shapes: tuple  # ((H_0, W_0), (H_1, W_1), ...)
    level_start_index: tuple  # (0, H_0*W_0, H_0*W_0 + H_1*W_1, ...)

    # Layout parameters
    value_layout: str  # "BLNC" or "BNLC"
    sampling_locations_layout: str  # "BQHLP2" or "B2QHLP"
    padding_mode: str  # "zeros" or "border"

    # Hardware constants
    P_MAX: int  # Partition dimension size (128)
    dtype: type  # Data type
    dtype_size: int  # Size of dtype in bytes

    # Sharding parameters
    q_start_global: int  # Global query start index for this shard
    local_N_q: int  # Number of queries for this shard

    # Tiling parameters
    Q_tile: int  # Queries per tile
    H_tile: int  # Number of heads per tile
    num_q_tiles: int  # Number of query tiles needed
    num_h_tiles: int  # Number of head tiles needed
    num_C_h_tiles: int  # Number of C_h tiles needed

    # Memory parameters
    h_interleave: int  # Interleave degree for h-scope
    gather_interleave: int  # Interleave degree for gather-scope
    h_scope_mem: int  # H-scope memory usage in bytes
    gather_scope_mem: int  # Gather-scope memory usage in bytes
    total_sbuf: int  # Total available SBUF in bytes


def _calculate_h_scope_memory(
    q_tile: int,
    h_tile: int,
    N_l: int,
    N_p: int,
    C_h: int,
    dtype_size: int,
    padding_mode: str,
) -> int:
    """Calculate H-scope memory usage.

    Args:
        q_tile: Number of queries in tile
        h_tile: Number of heads in tile
        N_l: Number of levels
        N_p: Number of sampling points per level
        C_h: Channels per head
        dtype_size: Size of data type in bytes
        padding_mode: "zeros" or "border"

    Returns:
        H-scope memory usage in bytes
    """
    mem = 0

    # Step 1: Load x, y, attn_w
    mem += q_tile * h_tile * N_l * N_p * 4  # x (float32)
    mem += q_tile * h_tile * N_l * N_p * 4  # y (float32)
    mem += q_tile * h_tile * N_l * N_p * dtype_size  # attn_w

    # Step 3: Bilinear coordinates
    mem += q_tile * h_tile * N_l * N_p * dtype_size  # x0_unclamped
    mem += q_tile * h_tile * N_l * N_p * dtype_size  # y0_unclamped
    mem += q_tile * h_tile * N_l * N_p * 4  # x1_unclamped (int32)
    mem += q_tile * h_tile * N_l * N_p * 4  # y1_unclamped (int32)
    mem += q_tile * h_tile * N_l * N_p * 4  # x0_clamped (uint32)
    mem += q_tile * h_tile * N_l * N_p * 4  # y0_clamped (uint32)
    mem += q_tile * h_tile * N_l * N_p * 4  # x1_clamped (uint32)
    mem += q_tile * h_tile * N_l * N_p * 4  # y1_clamped (uint32)

    # Step 4: Fractional contributions
    mem += q_tile * 2 * h_tile * N_l * N_p * dtype_size  # dx_dy
    mem += q_tile * 2 * h_tile * N_l * N_p * dtype_size  # one_minus_dx_dy

    # Step 5: Bilinear weights
    mem += q_tile * 4 * h_tile * N_l * N_p * 4  # combined_weights (float32)

    # Step 6: Flat indices
    mem += q_tile * h_tile * N_l * N_p * 4 * 4  # flat_idx_all (uint32)

    # Step 7b: Padding mode "zeros" specific
    if padding_mode == "zeros":
        mem += q_tile * 4 * h_tile * N_l * N_p * dtype_size  # coord_oob
        mem += q_tile * 4 * h_tile * N_l * N_p * 4  # one_minus_coord_oob (int32)
        mem += q_tile * 4 * h_tile * N_l * N_p * 4  # zeros_mask (int32)

    # Step 8: Output accumulators (one per head)
    mem += h_tile * q_tile * C_h * 4  # accum (float32)

    return mem


def _calculate_gather_scope_memory(
    q_tile: int,
    N_l: int,
    N_p: int,
    C_h: int,
    dtype_size: int,
    P_MAX: int,
) -> int:
    """Calculate gather-scope memory usage.

    Args:
        q_tile: Number of queries in tile
        N_l: Number of levels
        N_p: Number of sampling points per level
        C_h: Channels per head
        dtype_size: Size of data type in bytes
        P_MAX: Partition dimension size

    Returns:
        Gather-scope memory usage in bytes
    """
    mem = 0

    total_corners = N_p * 4
    num_C_h_tiles = div_ceil(C_h, P_MAX)
    min_C_h_P_MAX = min(C_h, P_MAX)

    # gathered_all allocation (once per h, l iteration)
    mem += min_C_h_P_MAX * num_C_h_tiles * total_corners * P_MAX * dtype_size

    # flat_idx_padded (only if q_tile < P_MAX, once per h, l iteration)
    if q_tile < P_MAX:
        mem += P_MAX * total_corners * 4  # uint32

    # weighted allocation (for even corners only: corners 0 and 2)
    mem += q_tile * C_h * dtype_size

    return mem


def _build_config(
    value: nl.ndarray,
    spatial_shapes: tuple,
    level_start_index: tuple,
    sampling_locations: nl.ndarray,
    attention_weights: nl.ndarray,
    value_layout: str,
    sampling_locations_layout: str,
    padding_mode: str,
    lnc: int,
    shard_id: int,
) -> MSDeformAttnConfig:
    """Build configuration from input tensors with sharding.

    Args:
        value: Value tensor in either (B, L, N_h, C_h) or (B, N_h, L, C_h) layout
        spatial_shapes: Spatial dimensions for each level
        level_start_index: Start indices for each level
        sampling_locations: Sampling coordinates in (B,N_q,N_h,N_l,N_p,2) or (B,2,N_q,N_h,N_l,N_p)
        attention_weights: Attention weights
        value_layout: "BLNC" for (B, L, N_h, C_h) or "BNLC" for (B, N_h, L, C_h)
        sampling_locations_layout: "BQHLP2" or "B2QHLP"
        padding_mode: "zeros" or "border"
        lnc: Number of NeuronCores for sharding
        shard_id: ID of this shard

    Returns:
        MSDeformAttnConfig with sharding info
    """
    # Parse input dimensions
    if value_layout == "BLNC":
        B, L, N_h, C_h = value.shape
    else:  # BNLC
        B, N_h, L, C_h = value.shape

    if sampling_locations_layout == "BQHLP2":
        _, N_q, _, N_l, N_p, _ = sampling_locations.shape
    else:  # B2QHLP
        _, _, N_q, _, N_l, N_p = sampling_locations.shape

    # Distribute queries across NeuronCores
    queries_per_nc = div_ceil(N_q, lnc)
    q_start_global = shard_id * queries_per_nc
    q_end_global = min(q_start_global + queries_per_nc, N_q)
    local_N_q = q_end_global - q_start_global

    # Hardware constants
    P_MAX = nl.tile_size.pmax
    dtype = value.dtype
    dtype_size = sizeinbytes(dtype)
    RESERVED_SBUF = 1024
    total_sbuf = nl.tile_size.total_available_sbuf_size - RESERVED_SBUF

    # Tiling parameters based on local query count
    Q_tile = min(P_MAX, local_N_q) if local_N_q > 0 else P_MAX
    num_q_tiles = div_ceil(local_N_q, Q_tile) if local_N_q > 0 else 0
    H_tile = N_h

    # Prioritize h_interleave=2, then maximize gather_interleave
    h_interleave = 2

    # Calculate h_scope memory
    h_scope_mem = _calculate_h_scope_memory(Q_tile, H_tile, N_l, N_p, C_h, dtype_size, padding_mode)

    # Find maximum gather_interleave that fits
    gather_interleave = 2
    gather_scope_mem = _calculate_gather_scope_memory(Q_tile, N_l, N_p, C_h, dtype_size, P_MAX)

    for candidate_interleave in [8, 7, 6, 5, 4, 3, 2]:
        gather_scope_available = total_sbuf // candidate_interleave
        if gather_scope_mem <= gather_scope_available:
            gather_interleave = candidate_interleave
            break

    num_h_tiles = div_ceil(N_h, H_tile)
    num_C_h_tiles = div_ceil(C_h, P_MAX)

    # Log configuration
    logger = get_logger("ms_deformable_attention")
    logger.info(
        "MSDeformAttnConfig: "
        "B=" + str(B) + ", "
        "N_q=" + str(N_q) + " (global), "
        "local_N_q=" + str(local_N_q) + ", "
        "q_start_global=" + str(q_start_global) + ", "
        "N_h=" + str(N_h) + ", "
        "C_h=" + str(C_h) + ", "
        "N_l=" + str(N_l) + ", "
        "N_p=" + str(N_p) + ", "
        "L=" + str(L) + ", "
        "value_layout=" + str(value_layout) + ", "
        "sampling_locations_layout=" + str(sampling_locations_layout) + ", "
        "padding_mode=" + str(padding_mode) + ", "
        "P_MAX=" + str(P_MAX) + ", "
        "dtype=" + str(dtype) + ", "
        "dtype_size=" + str(dtype_size) + ", "
        "Q_tile=" + str(Q_tile) + ", "
        "H_tile=" + str(H_tile) + ", "
        "num_q_tiles=" + str(num_q_tiles) + ", "
        "num_h_tiles=" + str(num_h_tiles) + ", "
        "num_C_h_tiles=" + str(num_C_h_tiles) + ", "
        "h_interleave=" + str(h_interleave) + ", "
        "gather_interleave=" + str(gather_interleave) + ", "
        "h_scope_mem=" + str(h_scope_mem) + ", "
        "gather_scope_mem=" + str(gather_scope_mem) + ", "
        "total_sbuf=" + str(total_sbuf) + "lnc=" + str(lnc)
    )

    return MSDeformAttnConfig(
        B=B,
        N_q=N_q,
        N_h=N_h,
        C_h=C_h,
        N_l=N_l,
        N_p=N_p,
        L=L,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        value_layout=value_layout,
        sampling_locations_layout=sampling_locations_layout,
        padding_mode=padding_mode,
        P_MAX=P_MAX,
        dtype=dtype,
        dtype_size=dtype_size,
        q_start_global=q_start_global,
        local_N_q=local_N_q,
        Q_tile=Q_tile,
        H_tile=H_tile,
        num_q_tiles=num_q_tiles,
        num_h_tiles=num_h_tiles,
        num_C_h_tiles=num_C_h_tiles,
        h_interleave=h_interleave,
        gather_interleave=gather_interleave,
        h_scope_mem=h_scope_mem,
        gather_scope_mem=gather_scope_mem,
        total_sbuf=total_sbuf,
    )
