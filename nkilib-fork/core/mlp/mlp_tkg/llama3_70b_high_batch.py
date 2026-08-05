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
Specialized MLP TKG kernel for Llama3-70B high-batch decode.

Hardcoded for the single config:
    B*S = 256, lnc = 2 (T_shard = 128)
    hidden_size H = 8192, intermediate I = 3584
    output_dtype = bfloat16, RMSNorm + SiLU
    STATIC FP8 quant on gate/up/down weights, no projection bias, no normalization bias
    no fused_add, no skip_gate, no input_in_sbuf, no store_output_in_sbuf
    no transposed_in / transposed_out
    column-tiling on both gate/up and down projections
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ...quantization.fp8_quantize import static_quantization
from ...utils.allocator import SbufManager
from ...utils.kernel_helpers import div_ceil, get_verified_program_sharding_info, resolve_fp8_e4m3_dtype
from ...utils.logging import get_logger
from ...utils.tensor_view import TensorView
from ...utils.tiled_range import TiledRange
from ..mlp_parameters import MLPParameters

# ---------- Hardcoded constants ----------
T = 128  # tokens per NC after T-sharding (B*S / lnc = 256 / 2)
H = 8192
I = 3584
H0 = nl.tile_size.pmax  # 128
I0 = nl.tile_size.pmax  # 128
H1 = H // H0  # 64
PSUM_FMAX = nl.tile_size.psum_fmax  # 512
PSUM_BMAX = 8  # no nl symbol; hardware-fixed PSUM bank count

# Gate/up: HTile shrunk from ini=4096 to 1024 to fit 4 ring tiles in 200KB stack.
GATE_UP_HTILE = 1024
GATE_UP_NUM_W_TILES = 4
GATE_UP_H1_PER_TILE = GATE_UP_HTILE // H0  # 8
NUM_GATE_UP_PSUMS = div_ceil(I, PSUM_FMAX)  # 7  (I=3584 / 512)

# Down: HTile = 4096 (8192 halved because 16 PSUMs > 8 banks).
DOWN_HTILE = 4096
DOWN_NUM_W_TILES = 22  # measured from BIR
DOWN_NUM_ITILES = div_ceil(I, I0 * 2)  # 14
NUM_DOWN_PSUMS = DOWN_HTILE // PSUM_FMAX  # 8
NUM_DOWN_HTILES = H // DOWN_HTILE  # 2

_DGE_MODE_NONE = 3  # static DMA mode


# ============================================================================
# RMSNorm (T=128, [T, H] -> [H0, H1, T] with fused gamma multiply on PSUM drain)
# ============================================================================


def _rmsnorm_th(input_view: TensorView, gamma_hbm, output: TensorView, eps: float, sbm: SbufManager):
    """Steps: load gamma+input; x^2 + reduce -> rsqrt -> scale; transpose with fused gamma multiply."""
    input_view = input_view.flatten_dims(start_dim=0, end_dim=1)

    gamma_sb = sbm.alloc_heap(shape=(H0, H1), dtype=gamma_hbm.dtype, buffer=nl.sbuf, name="rmsnorm_th_gamma")
    input_buf = sbm.alloc_heap(
        shape=(T, H), dtype=input_view.dtype, buffer=nl.sbuf, name="rmsnorm_th_input_buf", align=32
    )
    reduced_sq = sbm.alloc_heap(shape=(T, 1), dtype=nl.float32, buffer=nl.sbuf, name="rmsnorm_th_reduced_sq")
    square_buf = sbm.alloc_heap(shape=(T, H), dtype=nl.float32, buffer=nl.sbuf, name="rmsnorm_th_square_buf", align=32)

    # gamma: [1, H] -> [H0, H1]
    gamma_view = (
        TensorView(gamma_hbm)
        .flatten_dims(start_dim=0, end_dim=1)
        .reshape_dim(dim=0, shape=[1, H0, H1])
        .select(dim=0, index=0)
    )
    nisa.dma_copy(dst=TensorView(gamma_sb).get_view(), src=gamma_view.get_view(), dge_mode=_DGE_MODE_NONE)

    # input: [T, H]
    nisa.dma_copy(dst=input_buf, src=input_view.get_view(), dge_mode=_DGE_MODE_NONE)

    # x^2 + reduce -> [T, 1]; rsqrt(mean + eps); input *= norm_factor
    nisa.activation_reduce(square_buf, op=nl.square, data=input_buf, reduce_op=nl.add, reduce_res=reduced_sq)
    nisa.activation(reduced_sq, op=nl.rsqrt, data=reduced_sq, scale=1.0 / H, bias=eps)
    nisa.tensor_scalar(input_buf, input_buf, op0=nl.multiply, operand0=reduced_sq)

    # Transpose [T, H] -> [H0, H1, T] with fused gamma multiply on PSUM drain.
    # T=128 -> padded_tile_T=128, tiles_per_psum = 512/128 = 4.
    src_view = TensorView(input_buf).reshape_dim(dim=1, shape=[H0, H1])
    for batch_start in range(0, H1, 4):
        batch_size = min(4, H1 - batch_start)
        tp_psum = nl.ndarray((H0, batch_size * T), dtype=src_view.dtype, buffer=nl.psum)
        for i in range(batch_size):
            tile_src = src_view.slice(dim=2, start=batch_start + i, end=batch_start + i + 1).squeeze_dim(dim=2)
            nisa.nc_transpose(dst=tp_psum[0:H0, i * T : i * T + T], data=tile_src.get_view())
        tp_psum_view = TensorView(tp_psum).reshape_dim(dim=1, shape=[batch_size, T])
        dst_batch = output.slice(dim=1, start=batch_start, end=batch_start + batch_size)
        gamma_batch = TensorView(gamma_sb).slice(dim=1, start=batch_start, end=batch_start + batch_size)
        nisa.tensor_tensor(
            dst_batch.get_view(),
            tp_psum_view.get_view(),
            gamma_batch.expand_dim(dim=2).broadcast(dim=2, size=T).get_view(),
            nl.multiply,
        )

    # Free per-tile buffers (square_buf, reduced_sq, input_buf, gamma_sb)
    for _ in range(4):
        sbm.pop_heap()


# ============================================================================
# Top-level kernel
# ============================================================================


def mlp_tkg_llama3_70b_high_batch(
    params: MLPParameters,
    output_tensor_hbm: nl.ndarray,
    output_stored_add_tensor_hbm: nl.ndarray,
    sbm: Optional[SbufManager] = None,
) -> list[nl.ndarray]:
    """T-shard input/output across LNC cores, then run norm + gate/up + down + store."""
    if sbm is None:
        sbm = SbufManager(0, 200 * 1024, get_logger("mlp_tkg"))
        sbm.set_name_prefix("mlp_")

    # FP8 weight tile dtype: NON_OCP -> nl.float8_e4m3 (max=240); OCP -> nl.float8_e4m3fn (max=448).
    fp8_e4m3_tile_dtype = resolve_fp8_e4m3_dtype(params.dtype_mode)

    # ----- Convert params to TensorView (mirrors convert_params_to_views) -----
    params.hidden_tensor = TensorView(params.hidden_tensor)
    params.gate_proj_weights_tensor = TensorView(params.gate_proj_weights_tensor)
    params.up_proj_weights_tensor = TensorView(params.up_proj_weights_tensor)
    params.down_proj_weights_tensor = TensorView(params.down_proj_weights_tensor)
    params.bias_params.convert_to_view()
    params.fused_add_params.convert_to_view()
    params.quant_params.convert_to_view()

    # ----- T-shard hidden + output across LNC cores -----
    _, lnc, shard_id = get_verified_program_sharding_info("mlp_tkg", (0, 1))
    T_shard = (params.batch_size * params.sequence_len) // lnc  # = T = 128

    hidden_2d = params.hidden_tensor.flatten_dims(start_dim=0, end_dim=1)
    params.hidden_tensor = hidden_2d.slice(dim=0, start=shard_id * T_shard, end=(shard_id + 1) * T_shard).expand_dim(
        dim=0
    )
    params.batch_size = 1
    params.sequence_len = T_shard
    params.shard_on_h_disabled = True

    B_in, S_in, H_in = output_tensor_hbm.shape
    output_full = TensorView(output_tensor_hbm.reshape((B_in * S_in, H_in)))
    output_tile_hbm = output_full.slice(dim=0, start=shard_id * T_shard, end=(shard_id + 1) * T_shard).expand_dim(dim=0)

    sbm.set_name_prefix(f"{sbm.get_name_prefix()}bxs_0_")
    sbm.open_scope()

    # ----- 1. RMSNorm + input load + input quantization: HBM [1, T, H] -> SBUF [H0, H1, T] -----
    input_sb_quantized = TensorView(
        sbm.alloc_heap(shape=(H0, H1 * T), dtype=fp8_e4m3_tile_dtype, buffer=nl.sbuf, name="input_sbuf_quantized")
    ).reshape_dim(dim=1, shape=[H1, T])

    input_sb = TensorView(
        sbm.alloc_heap(shape=(H0, H1 * T), dtype=params.output_dtype, buffer=nl.sbuf, name="input_sbuf")
    ).reshape_dim(dim=1, shape=[H1, T])
    _rmsnorm_th(
        input_view=params.hidden_tensor,
        gamma_hbm=params.norm_params.normalization_weights_tensor,
        output=input_sb,
        eps=params.eps,
        sbm=sbm,
    )
    input_dequant_scale = TensorView(
        sbm.alloc_stack(
            shape=(T, 1),
            dtype=params.quant_params.gate_up_in_scale.dtype,
            buffer=nl.sbuf,
            name="gate_up_in_scale_sb",
            align=4,
        )
    )
    nisa.dma_copy(
        dst=input_dequant_scale.get_view(),
        src=params.quant_params.gate_up_in_scale.slice(dim=0, start=0, end=T).get_view(),
        dge_mode=_DGE_MODE_NONE,
    )

    static_quantization(
        hidden_state=input_sb.get_view(),
        input_dequant_scale=input_dequant_scale.get_view(),
        dtype=fp8_e4m3_tile_dtype,
        sbm=sbm,
        quantized=input_sb_quantized.get_view(),
    )
    sbm.pop_heap()  # dealloc input_sb

    # ----- 2. Gate/Up projection (column-tiling, STATIC FP8 quant, no bias) -----
    gate_up_sb = TensorView(
        sbm.alloc_heap(
            shape=(I0, div_ceil(I, I0), T),
            dtype=params.hidden_tensor.dtype,
            buffer=nl.sbuf,
            name="gate_up_sbuf",
        )
    )
    sbm.open_scope()

    gate_w = params.gate_proj_weights_tensor
    up_w = params.up_proj_weights_tensor
    gate_w_scale = params.quant_params.gate_w_scale
    up_w_scale = params.quant_params.up_w_scale

    # FP32 intermediate buffers (no fused sendrecv since num_shards=1).
    gate_sb = TensorView(sbm.alloc_stack((T, I), dtype=nl.float32, name="gate_sbuf_fp32", buffer=nl.sbuf, align=4))
    up_sb = TensorView(sbm.alloc_stack((T, I), dtype=nl.float32, name="up_sbuf_fp32", buffer=nl.sbuf, align=4))

    # STATIC quant: per-token scale [T, 1].
    gate_dequant = TensorView(
        sbm.alloc_stack(shape=(T, 1), dtype=gate_w_scale.dtype, buffer=nl.sbuf, name="gate_w_scale_sb", align=4)
    )
    up_dequant = TensorView(
        sbm.alloc_stack(shape=(T, 1), dtype=up_w_scale.dtype, buffer=nl.sbuf, name="up_w_scale_sb", align=4)
    )
    nisa.dma_copy(
        dst=gate_dequant.get_view(), src=gate_w_scale.slice(dim=0, start=0, end=T).get_view(), dge_mode=_DGE_MODE_NONE
    )
    nisa.tensor_tensor(
        dst=gate_dequant.get_view(),
        data1=gate_dequant.get_view(),
        data2=input_dequant_scale.get_view(),
        op=nl.multiply,
    )
    nisa.dma_copy(
        dst=up_dequant.get_view(), src=up_w_scale.slice(dim=0, start=0, end=T).get_view(), dge_mode=_DGE_MODE_NONE
    )
    nisa.tensor_tensor(
        dst=up_dequant.get_view(),
        data1=up_dequant.get_view(),
        data2=input_dequant_scale.get_view(),
        op=nl.multiply,
    )

    # Weight ring-buffer tiles [H0, HTile/H0, I].
    weight_tiles = []
    for i in range(GATE_UP_NUM_W_TILES):
        weight_tiles.append(
            TensorView(
                sbm.alloc_stack(
                    shape=(H0, GATE_UP_H1_PER_TILE, I),
                    dtype=fp8_e4m3_tile_dtype,
                    buffer=nl.sbuf,
                    name=f"gate_up_w_tile_{i}",
                )
            )
        )

    # ----- Gate projection: PSUM banks 0..6 -----
    gate_psums = []
    for i_tile in TiledRange(I, PSUM_FMAX):
        psum_idx = i_tile.index % PSUM_BMAX
        gate_psums.append(
            nl.ndarray(
                shape=(H0, PSUM_FMAX),
                dtype=nl.float32,
                name=f"gate_{sbm.get_name_prefix()}_psum_{psum_idx}",
                buffer=nl.psum,
                address=(0, psum_idx * PSUM_FMAX * 4),
            )
        )

    for hidden_tiles in TiledRange(H, GATE_UP_HTILE):
        h1_offset = hidden_tiles.index * GATE_UP_H1_PER_TILE
        weight_idx = hidden_tiles.index % GATE_UP_NUM_W_TILES

        gate_weight_view = gate_w.reshape_dim(dim=0, shape=(H0, H1)).slice(
            dim=1, start=h1_offset, end=h1_offset + GATE_UP_H1_PER_TILE
        )
        weight_sb = weight_tiles[weight_idx].slice(dim=1, start=0, end=GATE_UP_H1_PER_TILE).slice(dim=2, start=0, end=I)
        nisa.dma_copy(dst=weight_sb.get_view(), src=gate_weight_view.get_view(), dge_mode=_DGE_MODE_NONE)

        for i_tile in TiledRange(I, PSUM_FMAX):
            for h1_local in range(0, GATE_UP_H1_PER_TILE, 2):
                nisa.nc_matmul(
                    dst=gate_psums[i_tile.index][0:T, 0 : i_tile.size],
                    stationary=input_sb_quantized.slice(
                        dim=1, start=h1_offset + h1_local, end=h1_offset + h1_local + 2
                    ).get_view(),
                    moving=weight_sb.slice(dim=1, start=h1_local, end=h1_local + 2)
                    .slice(dim=2, start=i_tile.start_offset, end=i_tile.end_offset)
                    .get_view(),
                    perf_mode=nisa.matmul_perf_mode.double_row,
                )

    # Drain gate PSUM -> SBUF; STATIC FP8 dequant.
    for i_tile in TiledRange(I, PSUM_FMAX):
        nisa.activation(
            dst=gate_sb.slice(dim=1, start=i_tile.start_offset, end=i_tile.end_offset).get_view(),
            data=gate_psums[i_tile.index][0:T, 0 : i_tile.size],
            op=nl.copy,
        )
    nisa.activation(dst=gate_sb.get_view(), data=gate_sb.get_view(), scale=gate_dequant.get_view(), op=nl.copy)

    # ----- Up projection: PSUM banks 7,0,1,2,3,4,5 (offset by NUM_GATE_UP_PSUMS=7) -----
    up_psums = []
    for i_tile in TiledRange(I, PSUM_FMAX):
        psum_idx = (i_tile.index + NUM_GATE_UP_PSUMS) % PSUM_BMAX
        up_psums.append(
            nl.ndarray(
                shape=(H0, PSUM_FMAX),
                dtype=nl.float32,
                name=f"up_{sbm.get_name_prefix()}_psum_{psum_idx}",
                buffer=nl.psum,
                address=(0, psum_idx * PSUM_FMAX * 4),
            )
        )

    for hidden_tiles in TiledRange(H, GATE_UP_HTILE):
        h1_offset = hidden_tiles.index * GATE_UP_H1_PER_TILE
        weight_idx = hidden_tiles.index % GATE_UP_NUM_W_TILES

        up_weight_view = up_w.reshape_dim(dim=0, shape=(H0, H1)).slice(
            dim=1, start=h1_offset, end=h1_offset + GATE_UP_H1_PER_TILE
        )
        weight_sb = weight_tiles[weight_idx].slice(dim=1, start=0, end=GATE_UP_H1_PER_TILE).slice(dim=2, start=0, end=I)
        nisa.dma_copy(dst=weight_sb.get_view(), src=up_weight_view.get_view(), dge_mode=_DGE_MODE_NONE)

        for i_tile in TiledRange(I, PSUM_FMAX):
            for h1_local in range(0, GATE_UP_H1_PER_TILE, 2):
                nisa.nc_matmul(
                    dst=up_psums[i_tile.index][0:T, 0 : i_tile.size],
                    stationary=input_sb_quantized.slice(
                        dim=1, start=h1_offset + h1_local, end=h1_offset + h1_local + 2
                    ).get_view(),
                    moving=weight_sb.slice(dim=1, start=h1_local, end=h1_local + 2)
                    .slice(dim=2, start=i_tile.start_offset, end=i_tile.end_offset)
                    .get_view(),
                    perf_mode=nisa.matmul_perf_mode.double_row,
                )

    # Drain up PSUM -> SBUF; STATIC FP8 dequant.
    for i_tile in TiledRange(I, PSUM_FMAX):
        nisa.activation(
            dst=up_sb.slice(dim=1, start=i_tile.start_offset, end=i_tile.end_offset).get_view(),
            data=up_psums[i_tile.index][0:T, 0 : i_tile.size],
            op=nl.copy,
        )
    nisa.activation(dst=up_sb.get_view(), data=up_sb.get_view(), scale=up_dequant.get_view(), op=nl.copy)

    # SiLU(gate) * up
    nisa.activation(dst=gate_sb.get_view(), op=nl.silu, data=gate_sb.get_view(), scale=1.0)
    nisa.tensor_tensor(dst=up_sb.get_view(), data1=gate_sb.get_view(), data2=up_sb.get_view(), op=nl.multiply)

    # Transpose [T, I] -> [I0, I1, T] via PSUM (column-tiling path).
    for i_tile in TiledRange(I, I0):
        psum_idx = i_tile.index % PSUM_BMAX
        tp_psum = nl.ndarray(
            (i_tile.size, T),
            dtype=up_sb.dtype,
            buffer=nl.psum,
            name=f"{sbm.get_name_prefix()}transpose_psum_{i_tile.index}",
            address=(0, psum_idx * PSUM_FMAX * 4),
        )
        nisa.nc_transpose(
            dst=tp_psum,
            data=up_sb.slice(dim=0, start=0, end=T)
            .slice(dim=1, start=i_tile.index * I0, end=i_tile.index * I0 + i_tile.size)
            .get_view(),
        )
        nisa.tensor_copy(
            dst=gate_up_sb.slice(dim=0, start=0, end=i_tile.size)
            .slice(dim=1, start=i_tile.index, end=i_tile.index + 1)
            .slice(dim=2, start=0, end=T)
            .get_view(),
            src=tp_psum,
        )

    sbm.close_scope()

    gate_up_sb_quantized = TensorView(
        sbm.alloc_stack(
            shape=(I0, div_ceil(I, I0), T),
            dtype=fp8_e4m3_tile_dtype,
            buffer=nl.sbuf,
            name="gate_up_sbuf_quantized",
        )
    )
    nisa.dma_copy(
        dst=input_dequant_scale.get_view(),
        src=params.quant_params.down_in_scale.slice(dim=0, start=0, end=T).get_view(),
        dge_mode=_DGE_MODE_NONE,
    )
    static_quantization(
        hidden_state=gate_up_sb.get_view(),
        input_dequant_scale=input_dequant_scale.get_view(),
        dtype=fp8_e4m3_tile_dtype,
        sbm=sbm,
        quantized=gate_up_sb_quantized.get_view(),
    )
    sbm.pop_heap()  # dealloc gate_up_sb

    sbm.pop_heap()  # dealloc input_sb_quantized

    # ----- 3. Down projection (column-tiling, STATIC FP8 quant, no bias) -----
    down_sb = TensorView(
        sbm.alloc_stack(shape=(T, H), dtype=params.hidden_tensor.dtype, buffer=nl.sbuf, name="down_sbuf")
    )
    sbm.open_scope()

    down_w = params.down_proj_weights_tensor
    down_w_scale = params.quant_params.down_w_scale

    down_dequant = TensorView(
        sbm.alloc_stack(shape=(T, 1), dtype=down_w_scale.dtype, buffer=nl.sbuf, name="down_w_scale_sb", align=4)
    )
    nisa.dma_copy(
        dst=down_dequant.get_view(), src=down_w_scale.slice(dim=0, start=0, end=T).get_view(), dge_mode=_DGE_MODE_NONE
    )
    nisa.tensor_tensor(
        dst=down_dequant.get_view(),
        data1=down_dequant.get_view(),
        data2=input_dequant_scale.get_view(),
        op=nl.multiply,
    )

    down_weight_tiles = []
    for i in range(DOWN_NUM_W_TILES):
        down_weight_tiles.append(
            TensorView(
                sbm.alloc_stack(
                    shape=(I0, 2, DOWN_HTILE),
                    dtype=fp8_e4m3_tile_dtype,
                    buffer=nl.sbuf,
                    name=f"down_w_tile_{i}",
                )
            )
        )

    for hidden_tiles in TiledRange(H, DOWN_HTILE):
        h_offset = hidden_tiles.start_offset

        result_psums = []
        for p in range(NUM_DOWN_PSUMS):
            result_psums.append(
                nl.ndarray(
                    shape=(T, PSUM_FMAX),
                    dtype=nl.float32,
                    name=f"down_psum_{sbm.get_name_prefix()}_{hidden_tiles.index}_{p}",
                    buffer=nl.psum,
                    address=(0, p * PSUM_FMAX * 4),
                )
            )

        for i_tile in TiledRange(I, I0 * 2):
            weight_idx = (hidden_tiles.index * DOWN_NUM_ITILES + i_tile.index) % DOWN_NUM_W_TILES

            hidden_slice = gate_up_sb_quantized.slice(dim=0, start=0, end=I0).slice(
                dim=1, start=i_tile.index * 2, end=i_tile.index * 2 + 2
            )
            weight_sb = (
                down_weight_tiles[weight_idx].slice(dim=0, start=0, end=I0).slice(dim=2, start=0, end=hidden_tiles.size)
            )
            weight_view_0 = down_w.slice(dim=0, start=i_tile.start_offset, end=i_tile.start_offset + I0).slice(
                dim=1, start=h_offset, end=h_offset + hidden_tiles.size
            )
            nisa.dma_copy(
                dst=weight_sb.select(dim=1, index=0).get_view(), src=weight_view_0.get_view(), dge_mode=_DGE_MODE_NONE
            )
            weight_view_1 = down_w.slice(dim=0, start=i_tile.start_offset + I0, end=i_tile.start_offset + 2 * I0).slice(
                dim=1, start=h_offset, end=h_offset + hidden_tiles.size
            )
            nisa.dma_copy(
                dst=weight_sb.select(dim=1, index=1).get_view(), src=weight_view_1.get_view(), dge_mode=_DGE_MODE_NONE
            )

            for compute_idx in range(NUM_DOWN_PSUMS):
                nisa.nc_matmul(
                    dst=result_psums[compute_idx][0:T, 0:PSUM_FMAX],
                    stationary=hidden_slice.get_view(),
                    moving=weight_sb.slice(
                        dim=2, start=compute_idx * PSUM_FMAX, end=(compute_idx + 1) * PSUM_FMAX
                    ).get_view(),
                    perf_mode=nisa.matmul_perf_mode.double_row,
                )

        # Drain PSUMs with dequant scale (column_tile_index is always 0 for factor=1 -> Scalar engine path).
        for compute_idx in range(NUM_DOWN_PSUMS):
            dst_offset = h_offset + compute_idx * PSUM_FMAX
            nisa.activation(
                dst=down_sb.slice(dim=1, start=dst_offset, end=dst_offset + PSUM_FMAX).get_view(),
                data=result_psums[compute_idx][0:T, 0:PSUM_FMAX],
                scale=down_dequant.get_view(),
                op=nl.copy,
            )

    sbm.close_scope()

    # ----- 4. Store output to HBM -----
    nisa.dma_copy(
        dst=output_tile_hbm.flatten_dims(start_dim=0, end_dim=1).get_view(),
        src=down_sb.get_view(),
    )

    sbm.close_scope()
    return [output_full.base_tensor.reshape((B_in, S_in, H_in))]
