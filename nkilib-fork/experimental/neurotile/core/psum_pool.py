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
"""nt.psum_pool -- Grid-shaped, partial-aware PSUM allocation.

Allocates one ``nl.ndarray(buffer=nl.psum)`` per grid tile and wraps the
result in an NDSlice with a :class:`PSUMLayout`. Kernels index the pool
through the standard NDSlice surface (``psums[bank, slot].data``) rather
than a flat ``list[nl.ndarray]``.
"""

from typing import TYPE_CHECKING, Optional, Union

import nki.language as nl

from ._helpers import NUM_HW_BANKS, PSUM_BANK_SIZE
from ._validation import _validate_psum_pool_grid_args
from .factories import _make_psum_ndslice

if TYPE_CHECKING:
    from .ndslice import NDSlice


def psum_pool(
    tile_size: tuple,
    grid: Optional[tuple] = None,
    element_shape: Optional[tuple] = None,
    bank_axis: Optional[int] = None,
    bank_ids: Optional[Union[tuple, list]] = None,
    dtype=None,
) -> "NDSlice":
    """
    Allocate PSUM accumulator tiles and return them as an NDSlice.

    Args:
        tile_size (tuple[int, ...]): Per-tile (P, F) shape, rank >= 2.
            ``tile_size[0]`` is the partition (P) extent.
        element_shape (tuple[int, ...] | None): **Recommended.**
            Actual per-dim extent; the factory ceiling-divides by
            ``tile_size`` to derive the grid. When
            ``element_shape[d]`` is not a multiple of ``tile_size[d]``,
            the trailing tile on dim ``d`` is partial and ``.data``
            auto-clamps to the addressable F-width. Mutually exclusive
            with ``grid``; pass exactly one.
        grid (tuple[int, ...] | None): Tile-aligned tile-grid shape
            (extent = ``grid * tile_size``). Use only when over-
            allocating trailing partial extents is the desired
            behavior; otherwise prefer ``element_shape``.
        bank_axis (int | None): Allocation mode selector. ``None``
            with ``bank_ids=None`` -> compiler-managed; ``None`` with
            ``bank_ids`` -> every tile on its own bank; ``int`` with
            ``bank_ids`` -> slot-packed (``bank_axis`` fans across
            banks, other dims pack as slots). See API reference §11
            for the full mode table and the slot-stride rule.
        bank_ids (tuple[int, ...] | list[int] | None): Hardware bank
            indices in ``[0, 8)``. ``None`` defers placement to the
            compiler; otherwise length must match the mode (see ref).
        dtype: Element dtype. Defaults to ``nl.float32``.

    Returns:
        NDSlice: Grid + PSUMLayout pair. Index with the grid shape
        (``psums[s, i]``); ``.data`` returns the underlying
        ``nl.ndarray`` for ISA-op operands.

    Raises:
        AssertionError: ``tile_size`` invalid; neither or both of
            ``grid`` / ``element_shape`` given; ``bank_axis=int``
            without ``bank_ids``; ``bank_ids`` length / range /
            uniqueness mismatch with the selected mode; bank capacity
            exceeded (slot-packed mode).

    Example:
        # Recommended: element_shape= drives the allocation; trailing
        # partial tiles (if any) auto-clamp via PSUMLayout.
        psums = nt.psum_pool(tile_size=(128, 512), element_shape=(256, 2048))

        # Partial trailing tile (1280 = 2*512 + 256):
        psums = nt.psum_pool(
            tile_size=(128, 512), element_shape=(128, 1280),
            bank_ids=(0, 1, 2),
        )

        # Every tile on its own bank (8 tiles -> 8 banks):
        psums = nt.psum_pool(
            tile_size=(128, 512), element_shape=(256, 2048),
            bank_ids=(0, 1, 2, 3, 4, 5, 6, 7),
        )

        # Slot-packed (tile_f=128, 4 tiles per bank, stride 512):
        psums = nt.psum_pool(
            tile_size=(128, 128), element_shape=(256, 512),
            bank_axis=1, bank_ids=(0, 1, 2, 3),
        )
    """
    return _build_psum_ndslice(
        tile_size=tuple(tile_size),
        grid=tuple(grid) if grid is not None else None,
        element_shape=tuple(element_shape) if element_shape is not None else None,
        bank_axis=bank_axis,
        bank_ids=bank_ids,
        dtype=dtype,
    )


def _build_psum_ndslice(tile_size, grid, element_shape, bank_axis, bank_ids, dtype):
    """Validate, allocate one ndarray per grid tile, wrap as an NDSlice.

    Per-tile ndarrays keep each matmul accumulator independent: the
    compiler infers accumulation groups per ``nl.ndarray``, so packing
    multiple accumulators into one ndarray would tie them together and
    trip ``NCC_ISCH714``.

    Tile ``c`` is placed at
    ``address=(0, bank_id * PSUM_BANK_SIZE + slot_idx * slot_stride)``
    where ``slot_stride = max(512, 4 * tile_f)``. That stride matches
    the compiler's matmul accumulator-region quantum -- successive
    ndarrays in the same physical bank must be at least
    ``slot_stride`` F-elements apart or NCC reports ``NCC_ISCH714`` /
    ``NCC_IBIR110``. Some F-space inside each bank is intentionally
    unused so successive tiles land at distinct accumulator regions.
    """
    tile_grid_shape, num_banks, slots_per_bank, tile_f = _validate_psum_pool_grid_args(
        tile_size=tile_size,
        grid=grid,
        element_shape=element_shape,
        bank_axis=bank_axis,
        bank_ids=bank_ids,
        psum_bank_size=PSUM_BANK_SIZE,
        num_hw_banks=NUM_HW_BANKS,
    )

    _dtype = dtype if dtype is not None else nl.float32
    tile_p = tile_size[0]
    matmul_quantum = 512
    slot_stride = max(matmul_quantum, 4 * tile_f)

    tile_arrays = []
    if bank_ids is None:
        # Compiler-managed: no explicit address=, one ndarray per tile.
        for _ in range(num_banks * slots_per_bank):
            tile_arrays.append(
                nl.ndarray(
                    (tile_p, tile_f),
                    dtype=_dtype,
                    buffer=nl.psum,
                )
            )
    elif bank_axis is None:
        # Every tile on its own bank (slots_per_bank == 1).
        for tile_idx in range(num_banks):
            bank_id = bank_ids[tile_idx]
            tile_arrays.append(
                nl.ndarray(
                    (tile_p, tile_f),
                    dtype=_dtype,
                    buffer=nl.psum,
                    address=(0, bank_id * PSUM_BANK_SIZE),
                )
            )
    else:
        # Pack non-bank dims as slots within each bank.
        for bank_idx in range(num_banks):
            bank_id = bank_ids[bank_idx]
            for slot_idx in range(slots_per_bank):
                f_offset = bank_id * PSUM_BANK_SIZE + slot_idx * slot_stride
                tile_arrays.append(
                    nl.ndarray(
                        (tile_p, tile_f),
                        dtype=_dtype,
                        buffer=nl.psum,
                        address=(0, f_offset),
                    )
                )

    return _make_psum_ndslice(
        tile_arrays=tuple(tile_arrays),
        alloc_tile_size=tile_size,
        grid_shape=tile_grid_shape,
        bank_axis=bank_axis,
        slots_per_bank=slots_per_bank,
        dtype=_dtype,
        element_shape=element_shape,
    )
