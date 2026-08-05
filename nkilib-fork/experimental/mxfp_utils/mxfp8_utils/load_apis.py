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

"""Helper functions for loading tensor blocks from HBM to SBUF for MXFP8 kernels.

Provides three interleaving strategies for preparing data for MXFP8 quantization:

1. DGT (DMA Gather Transpose) — ``load_tile_dgt``
   Loads unswizzled [F, K] bf16 from HBM directly into interleaved SBUF layout
   via hardware gather-transpose.

2. FP32 reinterpret transpose — ``load_tile_fp32_transpose``
   Loads [F, K] bf16 from HBM, reinterprets as fp32 to halve the partition count,
   then uses 2× nc_transpose + scalar-engine tensor_copy to interleave.
   Best for data coming from HBM (weights, activations).

3. BF16 xbar transpose — ``load_tile_bf16_xbar_transpose``
   Interleaves a [P_MAX, tile_f] bf16 tile already resident in SBUF using
   4× nc_transpose + tensor_copy with unscramble AP.
   Best for on-chip intermediates that should not be spilled to HBM.
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode, oob_mode

from ....core.utils.kernel_assert import kernel_assert
from .common_dataclasses import TensorDescriptor, TileLocation
from .common_utils import get_active_sbm
from .quantize_mxfp8_utils import (
    INTERLEAVE_FACTOR,
    L_TILE_K,
    MIN_F_FOR_QUANTIZATION,
    MIN_K_FOR_QUANTIZATION,
    Q_TILE_K,
    get_fp8_dtype_x4,
)

P_MAX = 128  # 128, hardware partition dimension size
TRANSPOSE_CHUNK_SIZE = 32  # Element count per nc_transpose chunk
NUM_TRANSPOSE_CHUNKS = P_MAX // TRANSPOSE_CHUNK_SIZE  # Number of chunks in one partition
SUB_TILE = 128  # Max rows processed per sub-tile (matches P_MAX for partition alignment)
L_TILE_K_FP32 = L_TILE_K // 2  # 256, fp32 reinterpret halves the K dimension
BF16_TO_FP32_RATIO = 2  # Two bf16 elements pack into one fp32 element

"""
FP32 PE swizzle constants.

Reinterpreting bf16 as fp32 halves the partition count, so stride-2 nc_transpose
with 2 passes is sufficient to interleave all partitions.
"""
FP32_INTERLEAVE_STRIDE = 2  # nc_transpose stride in fp32 space (reads/writes every other partition)
FP32_NUM_TRANSPOSE_PASSES = 2  # Number of nc_transpose passes needed in fp32 space

"""
BF16 xbar PE swizzle constants.

Working in native bf16 requires stride-4 reads and a 2-step shuffle
(nc_transpose + tensor_copy with unscramble) to achieve the full interleave.
"""
BF16_SRC_STRIDE = 4  # nc_transpose source stride (reads every 4th partition)
BF16_NUM_TRANSPOSE_PASSES = 4  # Number of nc_transpose passes needed in bf16 space
BF16_DST_STRIDE = 8  # nc_transpose destination stride (places into every 8th slot)
BF16_DST_GROUP_SIZE = 64  # Number of elements per destination group (P_MAX / BF16_TO_FP32_RATIO)
BF16_DST_PAIR_SIZE = 2  # Adjacent element pair grouping in destination AP
BF16_DST_OFFSET_MULTIPLIER = 2  # Offset step between passes (transpose_pass_idx * 2) to fill alternating pairs
BF16_UNSCRAMBLE_STRIDE = 2  # tensor_copy unscramble: stride between interleaved groups
BF16_UNSCRAMBLE_GROUP_SIZE = 4  # tensor_copy unscramble: elements per group (INTERLEAVE_FACTOR)


def load_tile(
    load_loc: TileLocation,
    data_store_loc: Optional[TileLocation] = None,
    scale_store_loc: Optional[TileLocation] = None,
    load_scales_only: bool = False,
) -> None:
    """
    Load a tile from HBM to SBUF, dispatching by tensor format.

    Routes to the appropriate loader based on tensor metadata:
    1. MXFP8 (has scales): load data and scales via dma_copy
    2. Pre-swizzled bf16/fp16: load data via dma_copy
    3. Unswizzled bf16/fp16 (default): DMA gather transpose (DGT).

    Args:
        load_loc (TileLocation): Source tile location in HBM.
        data_store_loc (Optional[TileLocation]): Destination in SBUF for data.
        scale_store_loc (Optional[TileLocation]): Destination in SBUF for scales.
        load_scales_only (bool): If True, only load scales (skip data). Used for packed
            scale loading where scales have different HBM offsets than data.

    Returns:
        None

    Pseudocode:
        if tensor has scales (MXFP8):
            if not load_scales_only:
                dma_copy data to data_store_loc
            dma_copy scales to scale_store_loc
        elif tensor is pre-swizzled:
            dma_copy data to data_store_loc
        else:
            dma_gather_transpose data to data_store_loc
    """
    # MXFP8, preswizzled. Load data and scales separately using dma_copy
    if load_loc.tensor.is_quantized:
        kernel_assert(not load_loc.tensor.is_f_by_k, "The tensor needs to be K by F.")
        if not load_scales_only:
            data_store_loc = _load_tile_dma_copy(load_loc, data_store_loc, load_scales=False)
        scale_store_loc = _load_tile_dma_copy(load_loc, scale_store_loc, load_scales=True)
    # bf16, preswizzled. Load as is using dma_copy
    elif load_loc.tensor.is_swizzled:
        kernel_assert(not load_loc.tensor.is_f_by_k, "The tensor needs to be K by F.")
        kernel_assert(load_loc.tensor.data.dtype == nl.bfloat16, "Input tensor must be bfloat16.")
        _load_tile_dma_copy(load_loc, data_store_loc, load_scales=False)
    # bf16/fp16, unswizzled with PE swizzle: use PE transpose (indirect or direct
    # determined by vector_offset presence inside load_tile_bf16_PE_transpose).
    elif load_loc.tensor.load_with_PE_swizzle:
        load_tile_bf16_PE_transpose(load_loc, data_store_loc)
    # bf16/fp16, unswizzled with direct dma loads use DGT.
    else:
        load_tile_dgt(load_loc, data_store_loc)


def _get_dma_copy_ap(
    tensor: nl.ndarray,
    k_offset: int,
    f_offset: int,
    num_k: int,
    num_f: int,
) -> nl.ndarray:
    """
    Compute a BIR access pattern and flat offset for a dma_copy destination.

    Uses .ap() addressing so the tensor can be 2D or 3D (or any shape).
    Flattens all dimensions after the first into a single stride, then
    applies a linear offset from (k_offset, f_offset).

    Args:
        tensor (nl.ndarray): Destination SBUF tensor.
        k_offset (int): Offset in the K (first) dimension.
        f_offset (int): Offset in the flattened remaining dimensions.
        num_k (int): Number of K elements to write.
        num_f (int): Number of F elements to write.

    Returns:
        nl.ndarray: The tensor with .ap() applied.

    Pseudocode:
        stride_k = product(tensor.shape[1:])
        offset = k_offset * stride_k + f_offset
        return tensor.ap(pattern=[[stride_k, num_k], [1, num_f]], offset=offset)
    """
    stride_k = 1
    for dim in tensor.shape[1:]:
        stride_k *= dim
    offset = k_offset * stride_k + f_offset
    return tensor.ap(pattern=[[stride_k, num_k], [1, num_f]], offset=offset)


def _load_tile_dma_copy(
    load_loc: TileLocation,
    store_loc: Optional[TileLocation] = None,
    load_scales: bool = False,
) -> TileLocation:
    """
    Load a tile from HBM to SBUF using dma_copy.

    Unified loader for MXFP8 data, MXFP8 scales, and pre-swizzled bf16 tiles.
    For non-x4 MXFP8 data, uses BIR AP to reinterpret as x4 dtype during load.
    Handles boundary tiles by clamping to tensor bounds.

    Args:
        load_loc (TileLocation): Source tile location with tensor, tile dimensions, and offsets.
        store_loc (Optional[TileLocation]): Destination in SBUF. If None, allocated automatically.
        load_scales (bool): If True, load scales tensor; otherwise load data tensor.

    Returns:
        store_loc (TileLocation): The store location with loaded tensor in SBUF.

    Pseudocode:
        determine src tensor (scales if load_scales, else data)
        clamp num_k, num_f to tensor boundary
        if out of bounds: return store_loc unchanged
        allocate dst SBUF if store_loc is None
        if non-x4 MXFP8 data:
            dma_copy with BIR AP reinterpretation to x4 dtype
        else:
            dma_copy with direct slice indexing
        return store_loc
    """
    sbm = get_active_sbm()

    is_x4 = load_loc.tensor.is_x4

    kernel_assert(not load_loc.tensor.is_f_by_k, "The source tensor should be F by K.")

    if load_scales:
        src = load_loc.tensor.scales
        K, F = src.shape
        dst_dtype = src.dtype
    else:
        src = load_loc.tensor.data
        K = src.shape[0]
        # For non-x4 MXFP8, use scales' F dimension (x4-packed width)
        F = load_loc.tensor.scales.shape[1] if (load_loc.tensor.is_quantized and not is_x4) else src.shape[1]
        dst_dtype = src.dtype if (not load_loc.tensor.is_quantized or is_x4) else get_fp8_dtype_x4(str(src.dtype))

    # Clamp to boundary for partial tiles.
    # When effective_f_dim is set (stacked-expert tensors with scalar_offset),
    # clamp K to the per-expert boundary instead of the full tensor K.
    # effective_f_dim_scales will be set only for prequantized inputs,
    # in prequantized inputs shape is [K//4, F], hence the effective_f_dim
    # is used for calculating K_effective.
    if load_scales and load_loc.tensor.effective_f_dim_scales is not None:
        K_effective = load_loc.tensor.effective_f_dim_scales
    elif load_loc.tensor.effective_f_dim is not None:
        K_effective = load_loc.tensor.effective_f_dim
    else:
        K_effective = K
    num_k = min(load_loc.tile_k, K_effective - load_loc.k_offset)
    num_f = min(load_loc.tile_f, F - load_loc.f_offset)
    # Skip entirely out-of-bounds tiles (masking case where SBUF is pre-zeroed)
    if num_k <= 0 or num_f <= 0:
        return store_loc

    # Allocate destination SBUF tensor if not provided
    if store_loc == None:
        dst_sbuf = sbm.alloc_stack(shape=(num_k, num_f), dtype=dst_dtype, buffer=nl.sbuf)
        if load_scales:
            store_loc = TileLocation(
                tensor=TensorDescriptor(scales=dst_sbuf, is_f_by_k=False),
                tile_k=num_k,
                tile_f=num_f,
            )
        elif load_loc.tensor.is_quantized:
            store_loc = TileLocation(
                tensor=TensorDescriptor(data=dst_sbuf, is_x4=is_x4, is_f_by_k=False),
                tile_k=num_k,
                tile_f=num_f,
            )
        else:
            store_loc = TileLocation(
                tensor=TensorDescriptor(data=dst_sbuf, is_swizzled=True, is_f_by_k=False),
                tile_k=num_k,
                tile_f=num_f,
            )

    dst_tensor = store_loc.tensor.scales if load_scales else store_loc.tensor.data
    dst_ap = _get_dma_copy_ap(dst_tensor, store_loc.k_offset, store_loc.f_offset, num_k, num_f)

    # Determine if we need runtime scalar_offset (e.g., per-expert indexing for MoE)
    if load_scales and load_loc.tensor.scalar_offset_scales is not None:
        active_scalar_offset = load_loc.tensor.scalar_offset_scales
    else:
        active_scalar_offset = load_loc.tensor.scalar_offset
    use_scalar_offset = active_scalar_offset is not None

    # Non-x4 MXFP8 data: use BIR AP to reinterpret as x4 during load
    if load_loc.tensor.is_quantized and not is_x4 and not load_scales:
        x4_f_dim = load_loc.tensor.scales.shape[1]
        global_offset = load_loc.k_offset * x4_f_dim + load_loc.f_offset
        if use_scalar_offset:
            nisa.dma_copy(
                src=src.ap(
                    pattern=[[x4_f_dim, num_k], [1, num_f]],
                    offset=global_offset,
                    dtype=dst_dtype,
                    scalar_offset=active_scalar_offset,
                    indirect_dim=0,
                ),
                dst=dst_ap,
                dge_mode=dge_mode.hwdge,
            )
        else:
            nisa.dma_copy(
                src=src.ap(
                    pattern=[[x4_f_dim, num_k], [1, num_f]],
                    offset=global_offset,
                    dtype=dst_dtype,
                ),
                dst=dst_ap,
            )
    else:
        if use_scalar_offset:
            # Cannot use nl.ds slicing with scalar_offset; must use .ap() addressing.
            # src shape is [K, F]; stride for dim 0 (K) is F.
            src_f_dim = src.shape[1]
            static_offset = load_loc.k_offset * src_f_dim + load_loc.f_offset
            nisa.dma_copy(
                src=src.ap(
                    pattern=[[src_f_dim, num_k], [1, num_f]],
                    offset=static_offset,
                    scalar_offset=active_scalar_offset,
                    indirect_dim=0,
                ),
                dst=dst_ap,
                dge_mode=dge_mode.hwdge,
            )
        else:
            nisa.dma_copy(
                src=src[nl.ds(load_loc.k_offset, num_k), nl.ds(load_loc.f_offset, num_f)],
                dst=dst_ap,
            )

    return store_loc


def load_tile_dgt(
    load_loc: TileLocation,
    store_loc: Optional[TileLocation] = None,
) -> TileLocation:
    """
    Load an unswizzled bf16 tile from HBM to SBUF using DMA Gather Transpose.

    Flattens the source [F, K] tensor, applies vector offsets to gather the tile,
    and transposes into interleaved SBUF layout [K/4, 1, 1, tile_f*4].

    Dimensions:
        F: Source tensor first dimension (feature/free dimension)
        K: Source tensor second dimension (partition dimfension)

    Args:
        load_loc (TileLocation): Source tile location with tensor, tile dimensions, and offsets.
            If access_pattern and vector_offset are None, they are generated automatically.
        store_loc (Optional[TileLocation]): Destination in SBUF. If None, allocated automatically.

    Returns:
        store_loc (TileLocation): The store location with the loaded data tensor in SBUF.

    Pseudocode:
        validate input is F-by-K and bf16/fp16
        generate access_pattern and vector_offset if not provided
        allocate dst SBUF [tile_k/INTERLEAVE_FACTOR, 1, 1, tile_f*INTERLEAVE_FACTOR] if needed
        flatten src [F, K] -> [F*K/P_MAX, 1, 1, P_MAX]
        dma_transpose(dst, src.ap(access_pattern, vector_offset), axes=(3,1,2,0))
    """

    sbm = get_active_sbm()

    src_data = load_loc.tensor.data
    F, K = src_data.shape

    # Ensure input is F-by-K and 16-bit dtype
    kernel_assert(load_loc.tensor.is_f_by_k, "Transpose not implemented. DGT input tensor needs to be F by K.")
    kernel_assert(
        src_data.dtype in (nl.bfloat16, nl.float16),
        f"DGT input tensor must be bfloat16 or float16, got {src_data.dtype}.",
    )
    kernel_assert(
        K % MIN_K_FOR_QUANTIZATION == 0,
        f"K dimension must be divisible by {MIN_K_FOR_QUANTIZATION} for mxfp8 quantization, got {K=}",
    )
    kernel_assert(
        F % MIN_F_FOR_QUANTIZATION == 0,
        f"F dim must be divisible by {MIN_F_FOR_QUANTIZATION} for quantization, got {F=}",
    )

    # Allocate destination SBUF tensor if store_loc not provided
    if store_loc == None:
        dst_sbuf = sbm.alloc_stack(
            shape=(load_loc.tile_k // INTERLEAVE_FACTOR, 1, 1, load_loc.tile_f * INTERLEAVE_FACTOR),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
        )
        store_loc = TileLocation(
            tensor=TensorDescriptor(data=dst_sbuf, is_swizzled=True, is_f_by_k=False),
            tile_k=load_loc.tile_k,
            tile_f=load_loc.tile_f,
        )

    # Flatten source: [F, K] -> [F*K/vector_size, 1, 1, vector_size]
    vector_size = load_loc.tile_k // INTERLEAVE_FACTOR
    src_flattened = src_data.reshape((F * K // vector_size, 1, 1, vector_size))

    # Compute actual tile_f for partial tiles (boundary handling).
    # Use effective_f_dim (per-expert F) when set, otherwise full tensor F.
    F_effective = load_loc.tensor.effective_f_dim if load_loc.tensor.effective_f_dim is not None else F
    actual_tile_f = min(load_loc.tile_f, F_effective - load_loc.f_offset)

    # Slice destination for the correct tile position
    # store_loc.k_offset indexes into the tiles dimension (dim 1)
    tile_idx = store_loc.k_offset
    f_start = store_loc.f_offset * INTERLEAVE_FACTOR
    f_end = f_start + actual_tile_f * INTERLEAVE_FACTOR
    actual_physical_k = load_loc.tile_k // INTERLEAVE_FACTOR
    dst_slice = store_loc.tensor.data[nl.ds(0, actual_physical_k), nl.ds(tile_idx, 1), :, f_start:f_end]

    kernel_assert(
        load_loc.access_pattern != None and load_loc.vector_offset != None,
        "Must set load location access pattern and vector offset before ",
    )

    # Perform DMA gather transpose
    nisa.dma_transpose(
        dst=dst_slice,
        src=src_flattened.ap(
            pattern=load_loc.access_pattern,
            vector_offset=load_loc.vector_offset,
        ),
        axes=(3, 1, 2, 0),
    )

    return store_loc


def load_and_quantize_tile(
    load_loc: TileLocation,
    fp8_x4_dtype: Optional[type] = None,
    data_store_loc: Optional[TileLocation] = None,
    scale_store_loc: Optional[TileLocation] = None,
) -> TensorDescriptor:
    """
    Load a tile from HBM to SBUF and quantize to MXFP8.

    Routes to the appropriate loader based on tensor metadata via load_tile,
    then quantizes the loaded data. Returns a TensorDescriptor with quantized
    data and scales ready for nc_matmul_mx.

    Routing (handled by load_tile):
      1. MXFP8 (has scales): load data and scales via dma_copy
      2. Pre-swizzled bf16/fp16: load data via dma_copy
      3. Unswizzled bf16/fp16: load data via DMA gather transpose

    Args:
        load_loc (TileLocation): Source tile location in HBM.
        fp8_x4_dtype: Target MXFP8 quantized dtype. If None, defaults to float8_e4m3fn_x4.
        data_store_loc (Optional[TileLocation]): Destination for quantized data.
            If None, allocated as [Q_TILE_K, tile_f] fp8_x4 in SBUF.
        scale_store_loc (Optional[TileLocation]): Destination for quantization scales.
            If None, allocated as [Q_TILE_K, tile_f] uint8 in SBUF.

    Returns:
        TensorDescriptor: Quantized tensor with data (fp8_x4) and scales (uint8),
                          shape [Q_TILE_K, tile_f]. Access via .data and .scales.
    """
    sbm = get_active_sbm()

    if fp8_x4_dtype == None:
        fp8_x4_dtype = get_fp8_dtype_x4("float8_e4m3fn")

    # If tensor is already quantized, load_tile handles data+scales via dma_copy.
    if load_loc.tensor.is_quantized:
        if data_store_loc == None and scale_store_loc == None:
            store_td = TensorDescriptor(
                data=sbm.alloc_stack(shape=(load_loc.tile_k, load_loc.tile_f), dtype=fp8_x4_dtype, buffer=nl.sbuf),
                scales=sbm.alloc_stack(shape=(load_loc.tile_k, load_loc.tile_f), dtype=nl.uint8, buffer=nl.sbuf),
                is_f_by_k=False,
            )
            data_store_loc = TileLocation(tensor=store_td, tile_k=load_loc.tile_k, tile_f=load_loc.tile_f)
            scale_store_loc = TileLocation(tensor=store_td, tile_k=load_loc.tile_k, tile_f=load_loc.tile_f)
        load_tile(load_loc, data_store_loc, scale_store_loc)
        return TensorDescriptor(
            data=data_store_loc.tensor.data,
            scales=scale_store_loc.tensor.scales,
            is_quantized=True,
            is_swizzled=True,
            is_f_by_k=False,
        )

    # Unswizzled bf16: must be F-by-K for DGT
    kernel_assert(
        load_loc.tensor.is_f_by_k and not load_loc.tensor.is_swizzled,
        "load_and_quantize_tile requires unswizzled F-by-K tensor for DGT path. "
        "Pre-swizzled bf16 quantize is not yet supported.",
    )

    store_loc = load_tile_dgt(load_loc)

    # Reshape from DGT output [K/4, 1, 1, F*4] to [K/4, F*4] for quantize_mx
    src_for_quant = store_loc.tensor.data.reshape(
        (load_loc.tile_k // INTERLEAVE_FACTOR, load_loc.tile_f * INTERLEAVE_FACTOR)
    )

    if data_store_loc == None:
        data = sbm.alloc_stack(shape=(Q_TILE_K, load_loc.tile_f), dtype=fp8_x4_dtype, buffer=nl.sbuf)
    else:
        data = data_store_loc.tensor.data

    if scale_store_loc == None:
        scale = sbm.alloc_stack(shape=(Q_TILE_K, load_loc.tile_f), dtype=nl.uint8, buffer=nl.sbuf)
    else:
        scale = scale_store_loc.tensor.scales

    nisa.quantize_mx(src=src_for_quant, dst=data, dst_scale=scale)

    return TensorDescriptor(
        data=data,
        scales=scale,
        is_quantized=True,
        is_swizzled=True,
        is_f_by_k=False,
    )


def load_tile_fp32_transpose(
    load_loc: TileLocation,
    data_store_loc: Optional[TileLocation] = None,
    scale_store_loc: Optional[TileLocation] = None,
) -> None:
    """
    Load an unswizzled [F, K] bf16 tile from HBM, interleave via FP32 reinterpret
    transpose, and quantize to MXFP8 in SBUF.

    Uses the FP32 PE swizzle approach: reinterprets bf16 as fp32 to
    halve the partition count, then 2x nc_transpose with stride-2 APs interleaves
    the partitions. Suitable for data coming from HBM (weights, activations).

    For each SUB_TILE (128) rows of the tile:
      1. dma_copy [SUB_TILE, L_TILE_K] bf16 as [SUB_TILE, L_TILE_K//2] fp32
      2. 2x nc_transpose with stride-2 src/dst APs in fp32 space
      3. tensor_copy PSUM -> SBUF via scalar engine
      4. quantize_mx with bf16 reinterpret on the swizzled fp32 buffer

    Args:
        load_loc (TileLocation): Source tile location in HBM. Tensor must be
            [F, K] bf16 with is_f_by_k=True.
        data_store_loc (Optional[TileLocation]): Destination for quantized data.
            If None, allocated as [Q_TILE_K, tile_f] fp8_x4 in SBUF.
        scale_store_loc (Optional[TileLocation]): Destination for quantization scales.
            If None, allocated as [Q_TILE_K, tile_f] uint8 in SBUF.

    Returns:
        None (writes into data_store_loc and scale_store_loc)

    Pseudocode:
        validate input is F-by-K bf16, tile_k == L_TILE_K
        allocate data/scale SBUF if not provided
        for each sub-tile of SUB_TILE rows:
            dma_copy HBM bf16 as fp32 (reinterpret)
            2x nc_transpose stride-2 in fp32 -> PSUM
            tensor_copy PSUM -> SBUF (scalar engine)
            quantize_mx with bf16 reinterpret -> data, scale
    """

    sbm = get_active_sbm()

    src_data = load_loc.tensor.data
    F, K = src_data.shape

    kernel_assert(load_loc.tensor.is_f_by_k, "FP32 transpose load requires F-by-K tensor layout.")
    kernel_assert(not load_loc.tensor.is_swizzled, "FP32 transpose load requires unswizzled input.")
    kernel_assert(not load_loc.tensor.is_quantized, "FP32 transpose load requires non-quantized input.")
    kernel_assert(
        src_data.dtype in (nl.bfloat16, nl.float16),
        f"FP32 transpose load requires bfloat16 or float16 input, got {src_data.dtype}.",
    )
    kernel_assert(load_loc.tile_k == L_TILE_K, f"FP32 transpose load requires tile_k == {L_TILE_K}.")
    kernel_assert(load_loc.tile_f % SUB_TILE == 0, f"tile_f must be divisible by {SUB_TILE}.")
    kernel_assert(
        K % MIN_K_FOR_QUANTIZATION == 0,
        f"K dimension ({K}) must be divisible by {MIN_K_FOR_QUANTIZATION} for MXFP8 quantization.",
    )
    kernel_assert(
        F % MIN_F_FOR_QUANTIZATION == 0,
        f"F dimension ({F}) must be divisible by {MIN_F_FOR_QUANTIZATION} for MXFP8 quantization.",
    )

    tile_f = load_loc.tile_f
    f_offset = load_loc.f_offset
    k_offset = load_loc.k_offset
    NUM_SUB_TILES = tile_f // SUB_TILE

    fp8_x4_dtype = get_fp8_dtype_x4("float8_e4m3fn")

    # Allocate output SBUF tensors if not provided
    if data_store_loc == None:
        data_sbuf = sbm.alloc_stack(shape=(Q_TILE_K, tile_f), dtype=fp8_x4_dtype, buffer=nl.sbuf)
        data_store_loc = TileLocation(
            tensor=TensorDescriptor(data=data_sbuf, is_swizzled=True, is_f_by_k=False),
            tile_k=Q_TILE_K,
            tile_f=tile_f,
        )
    if scale_store_loc == None:
        scale_sbuf = sbm.alloc_stack(shape=(Q_TILE_K, tile_f), dtype=nl.uint8, buffer=nl.sbuf)
        scale_store_loc = TileLocation(
            tensor=TensorDescriptor(scales=scale_sbuf, is_f_by_k=False),
            tile_k=Q_TILE_K,
            tile_f=tile_f,
        )

    data_dst = data_store_loc.tensor.data
    scale_dst = scale_store_loc.tensor.scales

    K_fp32 = K // BF16_TO_FP32_RATIO
    k_fp32_offset = k_offset // BF16_TO_FP32_RATIO

    for subtile_idx in nl.affine_range(NUM_SUB_TILES):
        f_sub = f_offset + subtile_idx * SUB_TILE

        # Step 1: Load [SUB_TILE, L_TILE_K] bf16 as [SUB_TILE, L_TILE_K_FP32] fp32
        sbuf_fp32 = sbm.alloc_stack(shape=(SUB_TILE, L_TILE_K_FP32), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=sbuf_fp32,
            src=src_data.ap(
                pattern=[[K_fp32, SUB_TILE], [1, L_TILE_K_FP32]],
                offset=f_sub * K_fp32 + k_fp32_offset,
                dtype=nl.float32,
            ),
        )

        # Step 2: Stride-2 nc_transpose in fp32 space (2 passes)
        # Each pass handles alternating partitions via stride-2 AP
        psum_fp32 = nl.ndarray((SUB_TILE, L_TILE_K_FP32), dtype=nl.float32, buffer=nl.psum)
        for interleave_pass_idx in nl.affine_range(FP32_NUM_TRANSPOSE_PASSES):
            nisa.nc_transpose(
                dst=psum_fp32.ap(
                    pattern=[[L_TILE_K_FP32, SUB_TILE], [FP32_INTERLEAVE_STRIDE, SUB_TILE]],
                    offset=interleave_pass_idx,
                ),
                data=sbuf_fp32.ap(
                    pattern=[[L_TILE_K_FP32, SUB_TILE], [FP32_INTERLEAVE_STRIDE, SUB_TILE]],
                    offset=interleave_pass_idx,
                ),
            )

        # Step 3: Copy PSUM -> SBUF via scalar engine
        swizzled_fp32 = sbm.alloc_stack(shape=(SUB_TILE, L_TILE_K_FP32), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=swizzled_fp32, src=psum_fp32, engine=nisa.scalar_engine)

        # Step 4: quantize_mx with bf16 reinterpret
        f_out_start = data_store_loc.f_offset + subtile_idx * SUB_TILE
        nisa.quantize_mx(
            dst=data_dst[:, nl.ds(f_out_start, SUB_TILE)],
            src=swizzled_fp32.ap(
                pattern=[[L_TILE_K, P_MAX], [1, L_TILE_K]],
                dtype=nl.bfloat16,
            ),
            dst_scale=scale_dst[:, nl.ds(f_out_start, SUB_TILE)],
        )


def load_tile_bf16_PE_transpose(
    load_loc: TileLocation,
    data_store_loc: TileLocation,
) -> None:
    """
    Load an unswizzled [F, K] bf16 tile from HBM and interleave into the
    swizzled [K//4, F*4] layout in SBUF using BF16 PE transpose.

    This produces the same swizzled layout as DGT (load_tile_dgt) but uses
    a different mechanism: contiguous DMA loads followed by strided nc_transpose
    and tensor_copy, rather than hardware gather-transpose.

    Supports two loading modes based on load_loc.vector_offset:
      - Direct DMA (load_loc.vector_offset is None):
        Loads contiguous F rows starting at load_loc.f_offset.
      - Indirect DMA (load_loc.vector_offset is an SBUF tensor):
        Gathers non-contiguous F rows using indices from load_loc.vector_offset.
        Used for MoE-style token gathering where tokens are scattered in HBM.

    Algorithm (per SUB_TILE of 128 rows):
      1. dma_copy: Load [SUB_TILE, L_TILE_K] bf16 from HBM to SBUF.
         - Direct mode: uses f_offset-based addressing (contiguous F rows).
         - Indirect mode: uses load_loc.vector_offset + indirect_dim=0 to
           gather non-contiguous F rows by index.
      2. nc_transpose (4 passes): Read SUB_TILE contiguous partitions from SBUF,
         write into PSUM with stride-8 + 2-element pair grouping. Each pass
         offsets by 2 in the destination to fill alternating pairs.
      3. tensor_copy: Copy from PSUM to the final SBUF destination with a
         reordering access pattern that produces the [4F, 128P] layout
         required by DVE quantization and PE matmul.

    The output is a swizzled bf16 buffer — NOT quantized. Quantization (if needed)
    should be done separately via nisa.quantize_mx on the output.

    Args:
        load_loc (TileLocation): Source tile location in HBM.
            - tensor: Must be [F, K] bf16/fp16, is_f_by_k=True, not swizzled, not quantized.
            - tile_k: Must equal L_TILE_K (512).
            - tile_f: Must be divisible by SUB_TILE (128).
            - k_offset: Offset in K dimension for the tile start.
            - f_offset: Offset in F dimension (used in direct mode only).
            - vector_offset: If None, direct DMA. If set, must be an SBUF tensor
              of shape (P_MAX, NUM_SUB_TILES) where NUM_SUB_TILES = tile_f // SUB_TILE.
              Each column contains SUB_TILE (128) int32 global F-row indices for
              one sub-tile's gather operation. This layout respects SBUF's 128
              partition limit.

        data_store_loc (Optional[TileLocation]): Destination for swizzled data in SBUF.
            If None, allocated as [tile_k // INTERLEAVE_FACTOR, tile_f * INTERLEAVE_FACTOR]
            bf16 in SBUF. The TileLocation.tensor must have is_swizzled=True.

    Returns:
        None (writes swizzled bf16 data into data_store_loc)

    Examples:
        # Direct mode (contiguous rows 128..255):
        load_loc = TileLocation(
            tensor=td, tile_k=512, tile_f=128,
            k_offset=0, f_offset=128,
            vector_offset=None,  # direct DMA
        )
        load_tile_bf16_PE_transpose(load_loc)

        # Indirect mode (gather scattered tokens, tile_f=256 → 2 sub-tiles):
        # Shape (P_MAX=128, NUM_SUB_TILES=2): column 0 has first 128 row indices,
        # column 1 has next 128 row indices.
        token_indices = sbm.alloc_stack((128, 2), dtype=nl.int32)
        # ... fill token_indices with global F-row IDs ...
        load_loc = TileLocation(
            tensor=td, tile_k=512, tile_f=256,
            k_offset=0, f_offset=0,
            vector_offset=token_indices,  # indirect DMA
        )
        load_tile_bf16_PE_transpose(load_loc)
    """

    sbm = get_active_sbm()

    src_data = load_loc.tensor.data
    original_dtype = src_data.dtype

    is_f_by_k = load_loc.tensor.is_f_by_k
    # (K, F) logical extent regardless of physical layout ([F, K] vs [K, F]) — the
    # TensorDescriptor encapsulates that mapping via logical_shape.
    K, F = load_loc.tensor.logical_shape

    kernel_assert(not load_loc.tensor.is_swizzled, "BF16 transpose load requires unswizzled input.")
    kernel_assert(not load_loc.tensor.is_quantized, "BF16 transpose load requires non-quantized input.")
    kernel_assert(
        src_data.dtype in (nl.bfloat16, nl.float16),
        f"BF16 transpose load requires bfloat16 or float16 input, got {src_data.dtype}.",
    )

    kernel_assert(
        load_loc.tile_k % Q_TILE_K == 0,
        f"Currently BF16 transpose load requires tile_k % Q_TILE_K ==0, found {load_loc.tile_k % Q_TILE_K}.",
    )
    kernel_assert(load_loc.tile_f % SUB_TILE == 0, f"Currently tile_f must be divisible by {SUB_TILE}.")
    kernel_assert(
        K % MIN_K_FOR_QUANTIZATION == 0,
        f"K dimension ({K}) must be divisible by {MIN_K_FOR_QUANTIZATION} for MXFP8 quantization.",
    )
    kernel_assert(
        F % MIN_F_FOR_QUANTIZATION == 0,
        f"F dimension ({F}) must be divisible by {MIN_F_FOR_QUANTIZATION} for MXFP8 quantization.",
    )

    tile_f = load_loc.tile_f
    tile_k = load_loc.tile_k
    f_offset = load_loc.f_offset
    k_offset = load_loc.k_offset
    NUM_SUB_TILES = tile_f // SUB_TILE
    physical_tile_k = tile_k // INTERLEAVE_FACTOR

    # Determine DMA mode: indirect if vector_offset is provided, direct otherwise.
    use_indirect_dma = load_loc.vector_offset is not None

    kernel_assert(data_store_loc is not None, "data_store_loc is None")
    kernel_assert(
        len(data_store_loc.tensor.data.shape) == 4,
        f"expected 4D SBUF destination tensor, found tensor with shape {data_store_loc.tensor.data.shape}",
    )

    skip_token = False

    if load_loc.tensor.skip_dma is not None:
        skip_token = load_loc.tensor.skip_dma.skip_token

    data_dst = data_store_loc.tensor.data

    for subtile_idx in nl.affine_range(NUM_SUB_TILES):
        f_sub = f_offset + subtile_idx * SUB_TILE

        # Step 1: DMA load [SUB_TILE, L_TILE_K] bf16 from HBM to SBUF
        sbuf_data = sbm.alloc_stack(shape=(SUB_TILE, tile_k), dtype=original_dtype, buffer=nl.sbuf)

        if skip_token:
            nisa.memset(sbuf_data, value=0.0)

        if use_indirect_dma:
            # Indirect DMA token-gather is only implemented for F-by-K: the gather AP indexes
            # F-rows with K-row stride. K-by-F (K-major) gather is not supported.
            kernel_assert(is_f_by_k, "Indirect DMA (token gather) requires F-by-K layout; K-by-F is not supported.")
            # Indirect DMA: gather non-contiguous F rows using load_loc.vector_offset.
            #
            # vector_offset is shaped (P_MAX, NUM_SUB_TILES) where:
            #   - dim 0 (partitions, P_MAX=128): holds the row index values
            #   - dim 1 (free, NUM_SUB_TILES): one column per sub-tile
            # We slice column subtile_idx to get this sub-tile's 128 row indices.
            #
            # Source AP: [[K, SUB_TILE], [1, L_TILE_K]]
            #   Pair 0 [K, SUB_TILE]: stride=K (row stride), count=SUB_TILE
            #     -> indirect_dim=0: the sequential index 0,1,2,...,SUB_TILE-1
            #        is replaced by values from vector_offset. Each value is
            #        multiplied by K to compute the row's address.
            #   Pair 1 [1, L_TILE_K]: reads L_TILE_K contiguous elements per row.
            #
            # offset=k_offset: shifts within each row (selects K tile start).
            nisa.dma_copy(
                dst=sbuf_data,
                src=src_data.ap(
                    pattern=[[K, SUB_TILE], [1, tile_k]],
                    offset=k_offset,
                    vector_offset=load_loc.vector_offset[:, nl.ds(subtile_idx, 1)],
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_token else oob_mode.error,
            )
        else:
            # Direct DMA: load SUB_TILE contiguous F rows starting at f_sub
            if is_f_by_k:
                # F-by-K [F, K]: stride=K between F-rows, tile_k contiguous K-elements
                nisa.dma_copy(
                    dst=sbuf_data,
                    src=src_data.ap(
                        pattern=[[K, SUB_TILE], [1, tile_k]],
                        offset=f_sub * K + k_offset,
                        dtype=original_dtype,
                    ),
                )
            else:
                # K-by-F [K, F]: load [P_MAX, SUB_TILE] blocks from K-rows,
                # then nc_transpose to get [SUB_TILE, tile_k] (partitions=F, free=K).
                # NOTE: DGT (load_tile_dgt) is not used for K-by-F because DGT requires an
                # F-by-K source layout. Until the DMA gather-transpose API supports K-major
                # sources, K-by-F is handled here via an explicit nc_transpose, which also
                # imposes the F % 512 (full F load-tile) requirement asserted in the kernel.
                # TODO: switch to the DMA gather-transpose API to drop the explicit transpose
                # and relax the F-divisibility constraint.
                num_k_chunks = tile_k // P_MAX
                for k_chunk_idx in nl.affine_range(num_k_chunks):
                    sbuf_tmp = sbm.alloc_stack(shape=(P_MAX, SUB_TILE), dtype=original_dtype, buffer=nl.sbuf)
                    k_start = k_offset + k_chunk_idx * P_MAX
                    nisa.dma_copy(
                        dst=sbuf_tmp,
                        src=src_data[k_start : k_start + P_MAX, f_sub : f_sub + SUB_TILE],
                    )
                    # Transpose [P_MAX=128, SUB_TILE=128] -> [SUB_TILE=128, P_MAX=128]
                    # Tensor Engine nc_transpose: SBUF -> PSUM, then copy PSUM -> SBUF
                    psum_tmp = nl.ndarray((SUB_TILE, P_MAX), dtype=original_dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=psum_tmp, data=sbuf_tmp)
                    nisa.tensor_copy(
                        dst=sbuf_data[
                            nl.ds(0, SUB_TILE),
                            nl.ds(k_chunk_idx * P_MAX, P_MAX),
                        ],
                        src=psum_tmp,
                    )

        # Step 2: nc_transpose in bf16 space (4 passes)
        # PSUM has (SUB_TILE=128 partitions, tile_k free). Both src and dst use
        # SUB_TILE partitions — this is a crossbar partition shuffle, not a P↔F swap.
        # Each pass shuffles SUB_TILE partitions with tile_k/4 contiguous free elements,
        # offset by SUB_TILE each pass to cover all tile_k free elements.
        psum_interleaved = nl.ndarray(
            (physical_tile_k, SUB_TILE * INTERLEAVE_FACTOR), dtype=original_dtype, buffer=nl.psum
        )
        for interleave_pass_idx in nl.affine_range(BF16_NUM_TRANSPOSE_PASSES):
            nisa.nc_transpose(
                dst=psum_interleaved.ap(
                    pattern=[[SUB_TILE * INTERLEAVE_FACTOR, physical_tile_k], [8, SUB_TILE // 2], [1, 2]],
                    offset=interleave_pass_idx * 2,
                ),
                data=sbuf_data.ap(
                    pattern=[[tile_k, SUB_TILE], [1, physical_tile_k]],
                    offset=interleave_pass_idx * physical_tile_k,
                ),
            )

        # Step 3: tensor_copy from PSUM to final SBUF destination with reordering AP
        tile_idx = data_store_loc.k_offset
        f_start = (data_store_loc.f_offset + subtile_idx * SUB_TILE) * INTERLEAVE_FACTOR
        f_end = f_start + SUB_TILE * INTERLEAVE_FACTOR
        dst_slice = data_store_loc.tensor.data[nl.ds(0, physical_tile_k), nl.ds(tile_idx, 1), :, f_start:f_end]

        src_psum_ap = psum_interleaved.ap(
            pattern=[[SUB_TILE * INTERLEAVE_FACTOR, physical_tile_k], [8, SUB_TILE // 2], [1, 2], [2, 4]],
            dtype=original_dtype,
        )

        nisa.tensor_copy(dst=dst_slice, src=src_psum_ap)


def load_tile_bf16_xbar_transpose(
    load_loc: TileLocation,
    fp8_x4_dtype: Optional[type] = None,
) -> TensorDescriptor:
    """
    Interleave a [P_MAX, tile_f] bf16 tile already in SBUF via xbar transpose,
    then quantize to MXFP8.

    Uses the BF16 PE swizzle approach: 4x nc_transpose with stride-4
    src AP and stride-8/2-element dst AP, followed by tensor_copy PSUM -> SBUF
    with an unscramble pattern. Suitable for on-chip intermediates (e.g.,
    SiLU(gate) * up) that should not be spilled to HBM.

    Steps:
      1. 4x nc_transpose: stride-4 read from SBUF, stride-8 + 2-element
         grouping write to PSUM
      2. tensor_copy PSUM -> SBUF with unscramble AP
      3. quantize_mx from the interleaved SBUF buffer

    Args:
        load_loc (TileLocation): Source tile location in SBUF. Tensor data must be
            [P_MAX, tile_f] bf16, standalone (not a slice of a larger buffer).
        fp8_x4_dtype: Target MXFP8 quantized dtype. If None, defaults to float8_e4m3fn_x4.

    Returns:
        TensorDescriptor: Quantized tensor with data (fp8_x4) and scales (uint8),
                          shape [Q_TILE_K, tile_f // INTERLEAVE_FACTOR]. Access via .data and .scales.
    """
    sbm = get_active_sbm()
    src_sbuf = load_loc.tensor.data
    kernel_assert(src_sbuf.dtype == nl.bfloat16, "BF16 xbar transpose requires bfloat16 input.")
    kernel_assert(src_sbuf.shape[0] == P_MAX, f"Source partition dim must be {P_MAX}.")
    kernel_assert(not load_loc.tensor.is_f_by_k, "BF16 xbar transpose expects K-by-F layout (SBUF-resident).")
    kernel_assert(not load_loc.tensor.is_swizzled, "BF16 xbar transpose requires unswizzled input.")
    kernel_assert(not load_loc.tensor.is_quantized, "BF16 xbar transpose requires non-quantized input.")
    kernel_assert(
        load_loc.tile_f % INTERLEAVE_FACTOR == 0,
        f"tile_f ({load_loc.tile_f}) must be divisible by INTERLEAVE_FACTOR ({INTERLEAVE_FACTOR}).",
    )

    tile_f = load_loc.tile_f
    if fp8_x4_dtype == None:
        fp8_x4_dtype = get_fp8_dtype_x4("float8_e4m3fn")

    q_tile_f = tile_f // INTERLEAVE_FACTOR
    data = sbm.alloc_stack(shape=(Q_TILE_K, q_tile_f), dtype=fp8_x4_dtype, buffer=nl.sbuf)
    scale = sbm.alloc_stack(shape=(Q_TILE_K, q_tile_f), dtype=nl.uint8, buffer=nl.sbuf)

    # Step 1: 4-pass nc_transpose in bf16 space
    """
    Source AP: stride-4 reads every 4th partition from the [P_MAX, tile_f] input.
    Dest AP: stride-8 with 64-element groups and 2-element pairing places data
    into interleaved positions in PSUM.
    """
    psum_bf16 = nl.ndarray((P_MAX, tile_f), dtype=nl.bfloat16, buffer=nl.psum)
    for transpose_pass_idx in nl.affine_range(BF16_NUM_TRANSPOSE_PASSES):
        nisa.nc_transpose(
            dst=psum_bf16.ap(
                pattern=[[tile_f, P_MAX], [BF16_DST_STRIDE, BF16_DST_GROUP_SIZE], [1, BF16_DST_PAIR_SIZE]],
                offset=transpose_pass_idx * BF16_DST_OFFSET_MULTIPLIER,
                dtype=nl.bfloat16,
            ),
            data=src_sbuf.ap(
                pattern=[[tile_f, P_MAX], [BF16_SRC_STRIDE, P_MAX]],
                offset=transpose_pass_idx,
                dtype=nl.bfloat16,
            ),
        )

    # Step 2: Copy PSUM -> SBUF with unscramble pattern
    """
    The unscramble AP reorders the interleaved PSUM data into the final
    contiguous layout expected by quantize_mx.
    """
    swizzled_bf16 = sbm.alloc_stack(shape=(P_MAX, tile_f), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=swizzled_bf16,
        src=psum_bf16.ap(
            pattern=[
                [tile_f, P_MAX],
                [BF16_DST_STRIDE, BF16_DST_GROUP_SIZE],
                [1, BF16_DST_PAIR_SIZE],
                [BF16_UNSCRAMBLE_STRIDE, BF16_UNSCRAMBLE_GROUP_SIZE],
            ],
            dtype=nl.bfloat16,
        ),
    )

    # Step 3: Quantize from interleaved SBUF
    nisa.quantize_mx(dst=data, src=swizzled_bf16, dst_scale=scale)

    return TensorDescriptor(
        data=data,
        scales=scale,
        is_quantized=True,
        is_swizzled=True,
        is_f_by_k=False,
    )
