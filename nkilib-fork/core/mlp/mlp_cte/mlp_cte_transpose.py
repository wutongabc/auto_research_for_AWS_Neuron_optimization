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

"""MLP CTE transpose operations for source and intermediate tensor transposition."""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import NUM_HW_PSUM_BANKS, PSUM_BANK_SIZE, get_ceil_quotient
from ...utils.tile_info import TiledDimInfo
from ...utils.tiled_range import TiledRange
from ..mlp_parameters import MLPParameters, mlpp_has_quantized_input, mlpp_has_quantized_weights
from .mlp_cte_constants import MLPCTEConstants
from .mlp_cte_tile_info import MlpBxsIndices, MLPCTETileInfo

#
# Transpose the source tensor tile in SBUF


def transpose_source_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf_list: list[nl.ndarray],
    scale_sbuf: Optional[nl.ndarray],
    bias_sbuf: Optional[nl.ndarray],
    output_tile_sbuf_list: list[nl.ndarray],
    sbm: Optional[SbufManager] = None,
):
    if mlp_params.quant_params.is_dtype_mx():
        # If we didn't do DMA transpose, this must be H_X4_MIDDLE
        transpose_mx_source_tensor_tile(
            mlp_params,
            tile_info,
            constants,
            indices,
            source_tile_sbuf_list,
            output_tile_sbuf_list,
            sbm,
        )
        return
    apply_scale = scale_sbuf != None
    apply_bias = bias_sbuf != None

    # Alias these to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile
    hidden_dim_tile = tile_info.xpose_hidden_dim_tile
    BXS_SUBTILE_COUNT = bxs_dim_tile.subtile_dim_info.tile_count
    H_SUBTILE_SIZE = hidden_dim_tile.subtile_dim_info.tile_size

    # We will tile PSUM to make the indexing cleaner
    psum_tile_info = TiledDimInfo.build(nl.tile_size.psum_fmax, H_SUBTILE_SIZE)

    # 1-byte dtype PE transpose requires a step size of 2
    psum_step_size = 2 if mlpp_has_quantized_weights(mlp_params) else 1

    # This is the total size from the tensor that we are computing
    tensor_bxs_size = constants.get_bxs_size(mlp_params)

    if apply_scale or apply_bias:
        # For the calculation of the scale and bias indices, we are assuming that
        # the hidden subtile size == scale_sbuf.shape[0] == pmax
        if apply_scale:
            kernel_assert(
                (scale_sbuf.shape[0] == nl.tile_size.pmax) and (H_SUBTILE_SIZE == nl.tile_size.pmax),
                "Scale tile must equal the hidden dimension subtile size and they must equal PMAX",
            )
        if apply_bias:
            kernel_assert(
                (bias_sbuf.shape[0] == nl.tile_size.pmax) and (H_SUBTILE_SIZE == nl.tile_size.pmax),
                "Bias tile must equal the hidden dimension subtile size and they must equal PMAX",
            )

    res_psum_list = []
    for bank in range(constants.required_src_xpose_psum_bank_count):
        res_psum_list.append(
            nl.ndarray(
                (
                    H_SUBTILE_SIZE,
                    psum_tile_info.tile_count,
                    psum_tile_info.tile_size,
                    psum_step_size,
                ),
                dtype=constants.xpose_data_type,
                buffer=nl.psum,
                address=(0, bank * PSUM_BANK_SIZE) if sbm else None,
                name=indices.get_tensor_name("src_transpose_res_psum", f"bank{bank}"),
            )
        )

    # Loop over all the subtiles in current B x S dimension tile
    # Usig continue here to avoid deep nested code here
    for bxs_subtile_idx in range(BXS_SUBTILE_COUNT):
        # Calculate mask condition for this subtile using TiledDimInfo methods
        bxs_start = bxs_dim_tile.get_subtile_start(indices.bxs_tile_idx, bxs_subtile_idx)
        bxs_subtile_rest = tensor_bxs_size - bxs_start

        # Only process if there are valid elements in the BxS dimension
        if bxs_subtile_rest <= 0:
            continue

        # Loop over the hidden dimension tiles
        for hidden_tile_idx in range(hidden_dim_tile.tile_count):
            # Calculate hidden dimension mask condition
            hidden_tile_rest = mlp_params.hidden_size - (hidden_tile_idx * hidden_dim_tile.tile_size)
            if hidden_tile_rest > 0:
                psum_bank = (
                    bxs_subtile_idx * hidden_dim_tile.tile_count + hidden_tile_idx
                ) % constants.required_src_xpose_psum_bank_count

                _perform_hidden_transpose(
                    mlp_params,
                    tile_info,
                    constants,
                    tensor_bxs_size,
                    indices.bxs_tile_idx,
                    bxs_subtile_idx,
                    hidden_tile_idx,
                    psum_step_size,
                    source_tile_sbuf_list[bxs_subtile_idx],
                    res_psum_list[psum_bank],
                    psum_tile_info,
                )

                _apply_scale_bias_if_necessary(
                    apply_scale,
                    apply_bias,
                    tile_info,
                    tensor_bxs_size,
                    indices.bxs_tile_idx,
                    bxs_subtile_idx,
                    hidden_tile_idx,
                    hidden_tile_rest,
                    psum_step_size,
                    res_psum_list[psum_bank],
                    output_tile_sbuf_list[bxs_subtile_idx],
                    scale_sbuf,
                    bias_sbuf,
                )


#
# Transpose the intermediate tensor tile in SBUF
def transpose_intermediate_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    int_tile_sbuf_list: list[nl.ndarray],
    output_tile_sbuf_list: list[nl.ndarray],
    sbm: SbufManager,
):
    # Alias these to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile
    int_dim_tile = tile_info.xpose_intermediate_dim_tile
    BXS_SUBTILE_COUNT = bxs_dim_tile.subtile_dim_info.tile_count
    I_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size

    # We will tile PSUM to make the indexing cleaner
    psum_tile_info = TiledDimInfo.build(nl.tile_size.psum_fmax, I_SUBTILE_SIZE)

    # 1-byte dtype PE transpose requires a step size of 2
    psum_step_size = 2 if mlpp_has_quantized_weights(mlp_params) else 1

    # This is the total size from the tensor that we are computing
    tensor_bxs_size = constants.get_bxs_size(mlp_params)

    res_psum_list = []
    for bank in range(constants.required_int_xpose_psum_bank_count):
        psum_tensor = nl.ndarray(
            (
                I_SUBTILE_SIZE,
                psum_tile_info.tile_count,
                psum_tile_info.tile_size,
                psum_step_size,
            ),
            dtype=constants.xpose_data_type,
            buffer=nl.psum,
            address=(0, bank * PSUM_BANK_SIZE) if sbm else None,
            name=indices.get_tensor_name("int_transpose_res_psum", f"bank{bank}"),
        )
        res_psum_list.append(psum_tensor)

    # Loop over all the subtiles in current B x S dimension tile
    for bxs_subtile_idx in range(BXS_SUBTILE_COUNT):
        # Calculate mask condition for this subtile using TiledDimInfo methods
        bxs_start = bxs_dim_tile.get_subtile_start(indices.bxs_tile_idx, bxs_subtile_idx)
        bxs_subtile_rest = tensor_bxs_size - bxs_start

        # Only process if there are valid elements in the BxS dimension
        if bxs_subtile_rest > 0:
            # Loop over the intermediate dimension tiles
            for int_tile_idx in range(int_dim_tile.tile_count):
                # Calculate intermediate dimension mask condition
                int_tile_rest = mlp_params.intermediate_size - (int_tile_idx * int_dim_tile.tile_size)

                if int_tile_rest > 0:
                    psum_bank = (
                        bxs_subtile_idx * int_dim_tile.tile_count + int_tile_idx
                    ) % constants.required_int_xpose_psum_bank_count

                    _perform_intermediate_transpose(
                        mlp_params,
                        tile_info,
                        constants,
                        int_tile_idx,
                        int_tile_rest,
                        bxs_subtile_rest,
                        psum_step_size,
                        int_tile_sbuf_list[bxs_subtile_idx],
                        res_psum_list[psum_bank],
                        psum_tile_info,
                    )

                    _copy_intermediate_transpose_result(
                        tile_info,
                        tensor_bxs_size,
                        indices.bxs_tile_idx,
                        bxs_subtile_idx,
                        int_tile_idx,
                        int_tile_rest,
                        psum_step_size,
                        res_psum_list[psum_bank],
                        output_tile_sbuf_list[bxs_subtile_idx],
                    )


def transpose_mx_source_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf_list: list[nl.ndarray],
    output_tile_sbuf_list: list[nl.ndarray],
    sbm: Optional[SbufManager] = None,
):
    # Alias these to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile
    hidden_dim_tile = tile_info.mx_src_proj_hidden_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size  # 128
    MX_BXS_SUBTILE_SIZE = tile_info.mx_src_proj_bxs_dim_tile.subtile_dim_info.tile_size  # 256
    H_TILE_SIZE = hidden_dim_tile.tile_size  # 512
    H_SUBTILE_COUNT = hidden_dim_tile.subtile_dim_info.tile_count  # 128
    H_SUBTILE_SIZE = hidden_dim_tile.subtile_dim_info.tile_size  # 4
    BXS_BUFFER_COUNT = MX_BXS_SUBTILE_SIZE // BXS_SUBTILE_SIZE  # 2

    # 1-byte dtype PE transpose requires a step size of 2
    psum_step_size = 2 if mlpp_has_quantized_input(mlp_params) else 1

    bxs_tiles = TiledRange(constants.get_bxs_size(mlp_params), bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]

    # calculate how many elements of H will fit in PSUM at once
    # each psum bank will be [P(128_H), 2_T, 4_H, 128_T] which contains 512 elements of H
    max_h_elements_in_psum = NUM_HW_PSUM_BANKS * (PSUM_BANK_SIZE // BXS_BUFFER_COUNT // 2)
    max_h_tiles_in_psum = max_h_elements_in_psum // H_TILE_SIZE

    for h_psum_tile in TiledRange(mlp_params.hidden_size, max_h_elements_in_psum):  # 4096 in H
        # One h_psum_tile is large enough to fill PSUM. Create our PSUM buffers at this point.
        res_psum_list = []
        for bank in range(NUM_HW_PSUM_BANKS):
            res_psum_list.append(
                nl.ndarray(
                    (
                        nl.tile_size.pmax,  # 128_H
                        BXS_BUFFER_COUNT,  # 2_T
                        H_SUBTILE_SIZE,  # 4_H
                        nl.tile_size.pmax,  # 128_T
                        psum_step_size,
                    ),
                    dtype=source_tile_sbuf_list[0].dtype,
                    buffer=nl.psum,
                    address=(0, bank * PSUM_BANK_SIZE) if sbm else None,
                    name=indices.get_tensor_name("src_transpose_res_psum", f"itr{h_psum_tile.index}_bank{bank}"),
                )
            )
        for mx_bxs_subtile in TiledRange(current_bxs_tile, MX_BXS_SUBTILE_SIZE):  # 256 in BxS
            source_tile_sbuf_view = source_tile_sbuf_list[mx_bxs_subtile.index].reshape(
                (
                    nl.tile_size.pmax,  # 128_T
                    hidden_dim_tile.tile_count,  # H/512
                    BXS_BUFFER_COUNT,  # 2_T
                    H_SUBTILE_SIZE,  # 4_H
                    nl.tile_size.pmax,  # 128_H
                )
            )
            output_tile_sbuf_view = output_tile_sbuf_list[mx_bxs_subtile.index].reshape(
                (
                    nl.tile_size.pmax,  # 128_H
                    hidden_dim_tile.tile_count,  # H/512
                    BXS_BUFFER_COUNT,  # 2_T
                    BXS_SUBTILE_SIZE,  # 128_T
                    H_SUBTILE_SIZE,  # 4_H
                )
            )
            for h_tile in TiledRange(h_psum_tile, H_TILE_SIZE):  # 512 in 4096_H
                # h_tile index < 8; can be used directly as the PSUM buffer index
                bxs_subtiles = TiledRange(mx_bxs_subtile, BXS_SUBTILE_SIZE)
                for bxs_subtile in bxs_subtiles:  # 128 in 256_T
                    for h_row_tile in TiledRange(h_tile, H_SUBTILE_COUNT):  # 128 in 512_H
                        nisa.nc_transpose(
                            dst=res_psum_list[h_tile.index][
                                : h_row_tile.size,
                                bxs_subtile.index,
                                h_row_tile.index,
                                : bxs_subtile.size,
                                0,
                            ],
                            data=source_tile_sbuf_view[
                                : bxs_subtile.size,
                                h_psum_tile.index * max_h_tiles_in_psum + h_tile.index,
                                bxs_subtile.index,
                                h_row_tile.index,
                                : h_row_tile.size,
                            ],
                        )
                    # Evict half of the PSUM buffer (everything after the 2_T dim)
                    # [128_H, 2_T, 4_H, 128_T] -> [128_H, 2_T, 128_T, 4_H]
                    nisa.tensor_copy(
                        src=res_psum_list[h_tile.index].ap(
                            [
                                [
                                    BXS_BUFFER_COUNT * H_SUBTILE_SIZE * nl.tile_size.pmax * psum_step_size,
                                    h_row_tile.size,
                                ],
                                [psum_step_size, bxs_subtile.size],
                                [nl.tile_size.pmax * psum_step_size, H_SUBTILE_SIZE],
                            ],
                            offset=(bxs_subtile.index * H_SUBTILE_SIZE * nl.tile_size.pmax * psum_step_size),
                        ),
                        dst=output_tile_sbuf_view[
                            : h_row_tile.size,
                            h_psum_tile.index * max_h_tiles_in_psum + h_tile.index,
                            bxs_subtile.index,
                            : bxs_subtile.size,
                            :H_SUBTILE_SIZE,
                        ],
                    )


def _perform_hidden_transpose(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    tensor_bxs_size: int,
    bxs_tile_idx: int,
    bxs_subtile_idx: int,
    hidden_tile_idx: int,
    psum_step_size: int,
    source_tile_sbuf: nl.ndarray,
    res_psum_tensor: nl.ndarray,
    psum_tile_info,
):
    bxs_dim_tile = tile_info.bxs_dim_tile
    hidden_dim_tile = tile_info.xpose_hidden_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    H_SUBTILE_COUNT = hidden_dim_tile.subtile_dim_info.tile_count
    H_SUBTILE_SIZE = hidden_dim_tile.subtile_dim_info.tile_size

    hidden_tile_rest = mlp_params.hidden_size - (hidden_tile_idx * hidden_dim_tile.tile_size)

    for hidden_subtile_idx in range(H_SUBTILE_COUNT):
        hidden_subtile_tile_rest = hidden_tile_rest - (hidden_subtile_idx * H_SUBTILE_SIZE)
        if hidden_subtile_tile_rest > 0:
            bxs_subtile_bound = bxs_dim_tile.get_subtile_bound(bxs_tile_idx, bxs_subtile_idx)
            hidden_subtile_bound = min(hidden_subtile_tile_rest, H_SUBTILE_SIZE)

            nisa.nc_transpose(
                dst=res_psum_tensor.ap(
                    [
                        [psum_tile_info.tile_count * psum_tile_info.tile_size * psum_step_size, hidden_subtile_bound],
                        [1, 1],
                        [psum_step_size, bxs_subtile_bound],
                    ],
                    offset=hidden_subtile_idx * H_SUBTILE_SIZE * psum_step_size,
                ),
                data=source_tile_sbuf[
                    0:bxs_subtile_bound,
                    hidden_dim_tile.get_subtile_indices(hidden_tile_idx, hidden_subtile_idx, hidden_subtile_bound),
                ],
            )


def _apply_scale_bias_if_necessary(
    apply_scale: bool,
    apply_bias: bool,
    tile_info: MLPCTETileInfo,
    tensor_bxs_size: int,
    bxs_tile_idx: int,
    bxs_subtile_idx: int,
    hidden_tile_idx: int,
    hidden_tile_rest: int,
    psum_step_size: int,
    res_psum_tensor: nl.ndarray,
    output_tile_sbuf: nl.ndarray,
    scale_sbuf: Optional[nl.ndarray],
    bias_sbuf: Optional[nl.ndarray],
):
    bxs_dim_tile = tile_info.bxs_dim_tile
    hidden_dim_tile = tile_info.xpose_hidden_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    H_SUBTILE_COUNT = hidden_dim_tile.subtile_dim_info.tile_count
    H_SUBTILE_SIZE = hidden_dim_tile.subtile_dim_info.tile_size

    if apply_scale or apply_bias:
        op0 = nl.multiply if apply_scale else nl.add
        operand0 = scale_sbuf if apply_scale else bias_sbuf
        op1 = nl.add if apply_scale and apply_bias else None
        operand1 = bias_sbuf if apply_scale and apply_bias else None

        for hidden_subtile_idx in range(H_SUBTILE_COUNT):
            hidden_subtile_tile_rest = hidden_tile_rest - (hidden_subtile_idx * H_SUBTILE_SIZE)
            if hidden_subtile_tile_rest > 0:
                hidden_subtile_bound = min(hidden_subtile_tile_rest, H_SUBTILE_SIZE)
                bxs_subtile_bound = bxs_dim_tile.get_subtile_bound(bxs_tile_idx, bxs_subtile_idx)

                nisa.tensor_scalar(
                    dst=output_tile_sbuf[
                        :hidden_subtile_bound,
                        hidden_dim_tile.get_subtile_indices(hidden_tile_idx, hidden_subtile_idx, bxs_subtile_bound),
                    ],
                    data=res_psum_tensor[:hidden_subtile_bound, hidden_subtile_idx, :bxs_subtile_bound, 0],
                    op0=op0,
                    operand0=operand0[
                        : operand0.shape[0],
                        nl.ds(hidden_tile_idx * H_SUBTILE_COUNT + hidden_subtile_idx, 1),
                    ],
                    op1=op1,
                    operand1=(
                        operand1[
                            : operand1.shape[0],
                            nl.ds(hidden_tile_idx * H_SUBTILE_COUNT + hidden_subtile_idx, 1),
                        ]
                        if operand1 != None
                        else None
                    ),
                    engine=nisa.vector_engine,
                )
    else:
        hidden_subtile_bound = min(hidden_tile_rest, hidden_dim_tile.tile_size)
        res_psum_view = res_psum_tensor.reshape((nl.tile_size.pmax, nl.tile_size.psum_fmax, psum_step_size))

        nisa.tensor_copy(
            dst=output_tile_sbuf.ap(
                [
                    [output_tile_sbuf.shape[1], BXS_SUBTILE_SIZE],
                    [1, hidden_subtile_bound],
                ],
                offset=hidden_tile_idx * hidden_dim_tile.tile_size,
            ),
            src=res_psum_view[:BXS_SUBTILE_SIZE, :hidden_subtile_bound, 0],
            engine=nisa.vector_engine,
        )


def _perform_intermediate_transpose(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    int_tile_idx: int,
    int_tile_rest: int,
    bxs_subtile_rest: int,
    psum_step_size: int,
    int_tile_sbuf: nl.ndarray,
    res_psum_tensor: nl.ndarray,
    psum_tile_info,
):
    bxs_dim_tile = tile_info.bxs_dim_tile
    int_dim_tile = tile_info.xpose_intermediate_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    I_SUBTILE_COUNT = int_dim_tile.subtile_dim_info.tile_count
    I_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size

    for int_subtile_idx in range(I_SUBTILE_COUNT):
        int_subtile_rest = int_tile_rest - (int_subtile_idx * I_SUBTILE_SIZE)

        if int_subtile_rest > 0:
            bxs_subtile_bound = min(bxs_subtile_rest, BXS_SUBTILE_SIZE)
            int_subtile_bound = min(int_subtile_rest, I_SUBTILE_SIZE)
            int_tile_sbuf_view = int_tile_sbuf.reshape(
                (
                    int_tile_sbuf.shape[0],
                    int_dim_tile.tile_count,
                    I_SUBTILE_COUNT,
                    I_SUBTILE_SIZE,
                )
            )
            nisa.nc_transpose(
                dst=res_psum_tensor.ap(
                    pattern=[
                        [psum_tile_info.tile_count * psum_tile_info.tile_size * psum_step_size, int_subtile_bound],
                        [1, 1],
                        [psum_step_size, bxs_subtile_bound],
                    ],
                    offset=int_subtile_idx * I_SUBTILE_SIZE * psum_step_size,
                ),
                data=int_tile_sbuf_view[
                    :bxs_subtile_bound,
                    int_tile_idx,
                    int_subtile_idx,
                    :int_subtile_bound,
                ],
            )


def _copy_intermediate_transpose_result(
    tile_info: MLPCTETileInfo,
    tensor_bxs_size: int,
    bxs_tile_idx: int,
    bxs_subtile_idx: int,
    int_tile_idx: int,
    int_tile_rest: int,
    psum_step_size: int,
    res_psum_tensor: nl.ndarray,
    output_tile_sbuf: nl.ndarray,
):
    bxs_dim_tile = tile_info.bxs_dim_tile
    int_dim_tile = tile_info.xpose_intermediate_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    I_SUBTILE_COUNT = int_dim_tile.subtile_dim_info.tile_count
    I_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size

    actual_int_tile = min(int_tile_rest, int_dim_tile.tile_size)
    actual_int_tiles = get_ceil_quotient(actual_int_tile, I_SUBTILE_SIZE)
    bxs_subtile_bound = bxs_dim_tile.get_subtile_bound(bxs_tile_idx, bxs_subtile_idx)
    int_subtile_bound = min(int_tile_rest, I_SUBTILE_SIZE)

    res_psum_view = res_psum_tensor.reshape((BXS_SUBTILE_SIZE, I_SUBTILE_COUNT * I_SUBTILE_SIZE, psum_step_size))

    nisa.tensor_copy(
        dst=output_tile_sbuf.ap(
            [
                [
                    int_dim_tile.tile_count * I_SUBTILE_COUNT * I_SUBTILE_SIZE,
                    int_subtile_bound,
                ],
                [BXS_SUBTILE_SIZE, actual_int_tiles],
                [1, bxs_subtile_bound],
            ],
            offset=int_tile_idx * I_SUBTILE_COUNT * I_SUBTILE_SIZE,
        ),
        src=res_psum_view.ap(
            [
                [nl.tile_size.psum_fmax * psum_step_size, int_subtile_bound],
                [BXS_SUBTILE_SIZE * psum_step_size, actual_int_tiles],
                [psum_step_size, bxs_subtile_bound],
            ],
            offset=0,
        ),
        engine=nisa.vector_engine,
    )
