# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Down projection sub-kernel for MLP TKG with LHS/RHS swap mode."""

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.interleave_copy import interleave_copy
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from ...utils.tensor_view import TensorView
from ...utils.tiled_range import TiledRange
from .mlp_parameters import MLPParameters
from .mlp_tkg_constants import (
    MLPTKGConstantsDimensionSizes,
    MLPTKGConstantsDownTileCounts,
)


def down_projection_lhs_rhs_swap(
    hidden: TensorView,
    weight: TensorView,
    output_tile: TensorView,
    weight_tiles: list[TensorView],
    bias_tile: TensorView,
    dequant_tile: TensorView,
    dims: MLPTKGConstantsDimensionSizes,
    tiles: MLPTKGConstantsDownTileCounts,
    params: MLPParameters,
    sbm: SbufManager,
    use_dge: bool = False,
):
    """Compute the down projection with dynamically selected expert weights."""
    I0, _, T = hidden.shape
    I = weight.shape[0]
    hidden_size = dims.H_per_shard
    H0 = dims.H0
    num_i_tiles = div_ceil(I, I0)
    weight_base_idx = tiles.weight_base_idx

    per_bank_t = dims._psum_fmax // T
    num_psum_banks = div_ceil(dims.H1_shard, per_bank_t)
    kernel_assert(
        num_psum_banks <= dims._psum_bmax,
        f"Required down-projection PSUM banks {num_psum_banks} exceed {dims._psum_bmax}",
    )

    result_psums = []
    for psum_idx in range(num_psum_banks):
        result_psums.append(
            nl.ndarray(
                (dims._pmax, dims._psum_fmax),
                dtype=nl.float32,
                name=f"down_psum_{sbm.get_name_prefix()}_{psum_idx}",
                buffer=nl.psum,
                address=None
                if sbm.is_auto_alloc()
                else (0, psum_idx * dims._psum_fmax * 4),
            )
        )

    for hidden_tiles in TiledRange(hidden_size, tiles.HTile):
        h1_offset = hidden_tiles.start_offset // H0
        weight_hbm_tile = weight.slice(
            dim=1,
            start=hidden_tiles.start_offset,
            end=hidden_tiles.end_offset,
        )

        for i_tile in TiledRange(I, I0):
            weight_idx = (
                weight_base_idx
                + hidden_tiles.index * num_i_tiles
                + i_tile.index
            ) % tiles.num_allocated_w_tile
            hidden_tile = hidden.slice(
                dim=0, start=0, end=i_tile.size
            ).slice(dim=1, start=i_tile.index, end=i_tile.index + 1)
            weight_tile = weight_tiles[weight_idx].slice(
                dim=0, start=0, end=i_tile.size
            ).slice(dim=1, start=0, end=hidden_tiles.size)

            nisa.dma_copy(
                dst=weight_tile.get_view(),
                src=weight_hbm_tile.slice(
                    dim=0,
                    start=i_tile.start_offset,
                    end=i_tile.end_offset,
                ).get_view(),
                dge_mode=nisa.dge_mode.swdge,
            )

            if params.use_tkg_down_proj_optimized_layout:
                for h1_tile in TiledRange(hidden_tiles.size, H0):
                    psum_idx = (h1_offset + h1_tile.index) // per_bank_t
                    psum_offset = (h1_offset + h1_tile.index) % per_bank_t
                    nisa.nc_matmul(
                        dst=result_psums[psum_idx][
                            0:H0, nl.ds(psum_offset * T, T)
                        ],
                        stationary=weight_tile.slice(
                            dim=1,
                            start=h1_tile.start_offset,
                            end=h1_tile.end_offset,
                        ).get_view(),
                        moving=hidden_tile.get_view(),
                    )
            else:
                num_h_tiles = div_ceil(hidden_tiles.size, H0)
                weight_reshaped = weight_tile.reshape_dim(
                    dim=1, shape=(H0, num_h_tiles)
                ).permute(dims=(0, 2, 1))
                for h1_tile in TiledRange(hidden_tiles.size, H0):
                    psum_idx = (h1_offset + h1_tile.index) // per_bank_t
                    psum_offset = (h1_offset + h1_tile.index) % per_bank_t
                    nisa.nc_matmul(
                        dst=result_psums[psum_idx][
                            0:H0, nl.ds(psum_offset * T, T)
                        ],
                        stationary=weight_reshaped.select(
                            dim=1, index=h1_tile.index
                        ).get_view(),
                        moving=hidden_tile.get_view(),
                    )

    output_2d = (
        output_tile.flatten_dims(start_dim=1, end_dim=2)
        if output_tile.get_dim() > 2
        else output_tile
    )
    per_bank_elements = per_bank_t * T
    for psum_tiles in TiledRange(dims.H1_shard, per_bank_t):
        num_elements = psum_tiles.size * T
        if params.quant_params.is_quant_row():
            dequant_view = (
                dequant_tile.slice(
                    dim=1,
                    start=psum_tiles.index * psum_tiles.size,
                    end=(psum_tiles.index + 1) * psum_tiles.size,
                )
                .expand_dim(dim=2)
                .broadcast(dim=2, size=T)
            )
        else:
            dequant_view = dequant_tile

        interleave_copy(
            index=psum_tiles.index,
            dst=output_2d.slice(
                dim=1,
                start=psum_tiles.index * per_bank_elements,
                end=psum_tiles.index * per_bank_elements + num_elements,
            ).get_view(),
            src=result_psums[psum_tiles.index][0:H0, 0:num_elements],
            scale=dequant_view,
            bias=None,
        )

    if bias_tile != None:
        bias_broadcast = bias_tile.expand_dim(dim=2).broadcast(dim=2, size=T)
        nisa.tensor_tensor(
            dst=output_2d.get_view(),
            data1=output_2d.get_view(),
            data2=bias_broadcast.get_view(),
            op=nl.add,
        )
