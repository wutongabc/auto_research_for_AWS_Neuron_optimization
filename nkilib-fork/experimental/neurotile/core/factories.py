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
from typing import Any, Optional

import nki.language as nl

from ._helpers import (
    contiguous_strides,
    hbm_buffer_type,
    is_identity_tensor,
    physical_strides,
    product,
)
from ._validation import (
    _validate_alloc_blocks_args,
    _validate_alloc_tiles_args,
    _validate_block_size,
    _validate_block_size_rank,
    _validate_hbm_source,
    _validate_tensor_view_args,
    _validate_tiles_args,
)
from .axis import Axis, AxisLabel
from .grid import Grid
from .layout_hbm import HBMLayout
from .layout_psum import PSUMLayout
from .layout_sbuf import SBUFLayout
from .ndslice import NDSlice

# ============================================================================
# Helpers
# ============================================================================


def _pad_tile_size(size, n_batch_dims):
    """Pad tile_size with 1s for leading iteration dims.

    Example: tile_size=(128, 64) on 3D tensor -> (1, 128, 64), n_batch_dims=1.
    """
    if n_batch_dims <= 0:
        return tuple(size)
    padded = []
    for d in range(n_batch_dims):
        padded.append(1)
    for d in range(len(size)):
        padded.append(size[d])
    return tuple(padded)


def _truncate_for_skip(remaining, tile_size):
    """Truncate remaining to evenly divisible (RemainderPolicy.SKIP)."""
    result = []
    for d in range(len(remaining)):
        if d < len(tile_size) and tile_size[d] > 0:
            result.append((remaining[d] // tile_size[d]) * tile_size[d])
        else:
            result.append(remaining[d])
    return tuple(result)


def _resolve_sbuf_source(source, root):
    """Normalize an SBUF source to (raw_source, offset, dtype).

    Canonical form -- pass the NDSlice: NDSlice(SBUFLayout) carries source,
    offset, and dtype in one object. Use this form (e.g. an iteration tile
    from a tile-grid view) whenever available.

    Top-level `nl.ndarray(...)` is also accepted (offset = 0).

    Escape hatch -- raw sliced ndarray + root=<top-level>: used when a caller
    only has a slice (`t[:, a:b]`, `t[:, nl.ds(o, w)]`). `root=` supplies the
    dtype (NKI's tracer cannot resolve `.dtype` on slices); the slice's own
    `.offset` is preserved so navigation remains correct.
    """
    if isinstance(source, NDSlice):
        assert isinstance(source.layout, SBUFLayout), (
            "nt.tiles/nt.blocks(source=NDSlice, buffer_type=...): the NDSlice does not carry an SBUFLayout."
        )
        return source.layout.source, source.layout.offset, source.dtype

    if is_identity_tensor(source):
        return source, 0, source.dtype

    assert root is not None, (
        "nt.tiles/nt.blocks(source=<sliced ndarray>, buffer_type=...): "
        "source is a raw sliced view. Preferred fix: pass the NDSlice "
        "you already have (it carries source, offset, and dtype). "
        "Fallback: pass root=<top-level nl.ndarray> so dtype can be "
        "resolved against the root buffer."
    )
    assert is_identity_tensor(root), (
        "nt.tiles/nt.blocks(root=...): root must be a top-level nl.ndarray, not a view or slice."
    )
    offset = source.offset if hasattr(source, "offset") else 0
    return root, offset, root.dtype


def _resolve_hbm_root(source, root):
    """Pick the tensor that owns physical memory for the HBM layout.

    Returns `root` when passed (caller's explicit parent); otherwise
    returns `source` (which must be either a top-level tensor or a
    non-NKI test input).
    """
    return root if root is not None else source


# ============================================================================
# SBUF view construction -- single path for all SBUF Grid+Layout creation
# ============================================================================


def _make_sbuf_ndslice(sbuf_source, element_shape, tile_size, dtype, buffer_type, offset=0):
    """Create NDSlice(Grid, SBUFLayout) for an SBUF source.

    Caller supplies the addressable per-dim ``element_shape`` and the
    per-tile ``tile_size`` (uniform allocation extent). Returns the
    wrapped NDSlice.
    """
    grid, layout = SBUFLayout.build_view(
        sbuf_source,
        element_shape,
        tile_size,
        dtype,
        buffer_type,
        offset,
    )
    return NDSlice(grid, layout)


# ============================================================================
# PSUM view construction
# ============================================================================


def _make_psum_ndslice(
    tile_arrays,
    alloc_tile_size,
    grid_shape,
    bank_axis,
    slots_per_bank,
    dtype,
    element_shape=None,
):
    """Create NDSlice(Grid, PSUMLayout) for per-tile PSUM ndarrays.

    Args:
        tile_arrays: tuple of ``nl.ndarray(buffer=nl.psum)`` -- one per
            grid tile, in bank-major order (tile index =
            ``bank_idx * slots_per_bank + slot_idx``).
        alloc_tile_size: per-tile (P, F) shape.
        grid_shape: tile-grid shape (one count per dim of alloc_tile_size).
            ``grid_shape[bank_axis]`` is the number of banks; the product
            of the rest is ``slots_per_bank``.
        bank_axis: which Grid dim spreads tiles across PSUM banks.
        slots_per_bank: number of tiles co-located on each bank.
        dtype: element dtype.
        element_shape: optional per-dim addressable element extent. When
            omitted, defaults to ``grid_shape * alloc_tile_size`` (full
            tile-aligned extent).

    Returns:
        NDSlice wrapping a PSUMLayout + Grid.
    """
    if element_shape is None:
        es = []
        for d in range(len(grid_shape)):
            es.append(grid_shape[d] * alloc_tile_size[d])
        element_shape = tuple(es)
    grid = Grid.from_shape(
        element_shape=tuple(element_shape),
        tile_size=tuple(alloc_tile_size),
    )
    layout = PSUMLayout(
        tile_arrays=tile_arrays,
        offset=0,
        alloc_tile_size=tuple(alloc_tile_size),
        bank_axis=bank_axis,
        slots_per_bank=slots_per_bank,
        grid_shape=tuple(grid_shape),
        dtype=dtype,
        buffer_type=nl.psum,
    )
    return NDSlice(grid, layout)


# ============================================================================
# tiles()
# ============================================================================


def tiles(
    source: Any,
    tile_size: Optional[tuple] = None,
    access_pattern: Optional[list] = None,
    root: Optional[Any] = None,
    buffer_type: Optional[Any] = None,
    remainder: Optional[str] = None,
) -> "NDSlice":
    """
    Decompose a tensor into a logical grid of tiles. No data movement.

    Args:
        source (HBM tensor | nl.ndarray | NDSlice): The tensor to tile.
            HBM tensor: top-level or sliced (sliced requires `root=`).
            Raw SBUF ndarray: pass with `buffer_type=nl.sbuf`.
            NDSlice: re-tile / descend / promote.
        tile_size (tuple[int, ...]): Tile shape (>= 2-D). Lower-rank
            ``tile_size`` means leading source dims become batch dims
            (auto-padded with 1s). Required for raw tensor sources;
            optional for NDSlice sources (passing only `source` strips
            block axes for tile-level iteration).
        access_pattern (list[list[int, int]] | None): Source layout as
            ``[[stride_d, count_d], ...]`` -- one level per source dim.
            Raw sources only; rejected on NDSlice sources.
        root (HBM tensor | None): Parent tensor for a sliced HBM source
            (e.g. ``w_qkv[:, a:b]``). Required when source is a slice;
            without root, strides would derive from the slice's logical
            shape and multi-tile loads would read bytes from wrong offsets.
        buffer_type (nl.MemoryRegion | None): ``nl.sbuf`` for a raw SBUF
            source; ``None`` for HBM (default).
        remainder (str | None): ``"skip"`` floor-divides the grid (drops
            partial trailing tiles); ``None`` (default) keeps boundary
            tiles, surfacing them via ``view.is_remainder``.

    Returns:
        NDSlice: Tile-grid view. ``.shape`` is the tile-grid extent per
        dim, ``.tile_size`` matches `tile_size`, ``.element_shape`` is
        the source extent. For sharded views, index with a slice helper
        from :mod:`neurotile.shard_helpers`.

    Raises:
        AssertionError: Invalid arg combinations (e.g. NDSlice +
            access_pattern, NDSlice + root, NDSlice + buffer_type,
            NDSlice + remainder); raw source without `tile_size`;
            tile_size rank exceeds source rank; tile extent exceeds
            source extent on any dim; access_pattern counts exceed
            source shape; sliced HBM source without `root`.

    Example:
        # Plain tile grid on a top-level HBM tensor:
        view = nt.tiles(src, tile_size=(128, 256))

        # Sharded across cores via slice indexing on the result:
        rank = nl.program_id(0)
        n = nl.num_programs(0)
        view = nt.tiles(src, tile_size=(128, 256))[
            block_range(rank, n, 8), :
        ]

        # Strided source layout (every other row):
        view = nt.tiles(
            src,
            access_pattern=[[2 * N, M // 2], [1, N]],
            tile_size=(128, N),
        )
    """
    _validate_tiles_args(
        source=source,
        size=tile_size,
        access_pattern=access_pattern,
        buffer_type=buffer_type,
        remainder=remainder,
    )

    if isinstance(source, NDSlice):
        assert access_pattern is None, (
            "nt.tiles(source=NDSlice): access_pattern= is not supported -- "
            "the NDSlice already carries its own layout. Pass the raw "
            "tensor with access_pattern= instead."
        )
        assert buffer_type is None, (
            "nt.tiles(source=NDSlice, buffer_type=...): buffer_type= is only "
            "for raw sources. The NDSlice already carries its buffer via its "
            "layout."
        )
        assert root is None, (
            "nt.tiles(source=NDSlice, root=...): root= is only for raw "
            "sources. The NDSlice already carries its source + offset."
        )
        assert remainder is None, (
            "nt.tiles(source=NDSlice, remainder=...): remainder= applies at "
            "construct time only. Re-tile inherits the source view's "
            "remaining region."
        )
        return _retile_ndslice(source, tile_size)

    return _build_tiled_ndslice(
        source=source,
        size=tile_size,
        access_pattern=access_pattern,
        root=root,
        buffer_type=buffer_type,
        remainder=remainder,
    )


def _retile_ndslice(source, tile_size):
    """Handle ``nt.tiles(source=NDSlice)`` -- descend or re-tile.

    Both paths strip block axes first (re-tiling a block view descends
    to tile granularity) and then reset the cursor to the first
    non-batch dim so iteration starts on the resulting tile grid.

    - ``tile_size is None``  -> descend: drop block axes, keep the
      view's existing tile structure.
    - ``tile_size`` given    -> retile: drop block axes, rebuild the
      tile grid at the new ``tile_size`` via ``Grid.tile()``. Outer
      shard / broadcast axes are preserved.

    The Layout is asked to ``retile`` only when ``tile_size`` is given.
    HBMLayout returns self (HBM strides are source-element units,
    independent of tile grain); SBUFLayout recomputes its strides for
    the new tile grain.
    """
    if tile_size is None and source.grid.tile_size is None:
        return source

    base_grid = source.grid.strip_block()

    if tile_size is None:
        new_grid = base_grid.with_cursor(base_grid.n_batch_dims)
        return NDSlice(new_grid, source.layout)

    new_tile_size = tuple(tile_size)
    source.grid.validate_retile_to(new_tile_size)
    new_grid = base_grid.tile(new_tile_size).with_cursor(base_grid.n_batch_dims)
    new_layout = source.layout.retile(new_tile_size, new_grid.remaining)
    return NDSlice(new_grid, new_layout)


def _split_access_pattern(access_pattern):
    """Split AP into (strides, element_shape) tuples.

    access_pattern is `[[stride_0, count_0], ...]` -- stride_d is the element
    stride along dim d; count_d is the element extent along dim d.
    """
    strides = []
    element_shape = []
    for level in access_pattern:
        strides.append(level[0])
        element_shape.append(level[1])
    return tuple(strides), tuple(element_shape)


def _resolve_element_shape(source, access_pattern):
    """Pick the canonical element_shape.

    When `access_pattern` is given, its counts supply the element extents;
    otherwise fall back to `source.shape`.
    """
    if access_pattern is not None:
        _, ap_shape = _split_access_pattern(access_pattern)
        return ap_shape
    return tuple(source.shape)


def _build_tiled_ndslice(
    source,
    size,
    access_pattern,
    root,
    buffer_type,
    remainder,
):
    """Single-pipeline NDSlice builder for raw-tensor sources.

    All composition axes flow through the same steps:
      1. Normalize source -> (storage tensor, offset, dtype, is_sbuf).
      2. Resolve element_shape (from AP counts / source.shape).
      3. Resolve strides (from AP if given, else physical layout).
      4. Apply remainder policy to compute `remaining`.
      5. Build Grid + Layout.
      6. Drop unit batch dims, return NDSlice.
    """
    is_sbuf = buffer_type is not None

    # Step 1: normalize source -> (storage tensor, offset, dtype).
    if is_sbuf:
        storage, offset, dtype = _resolve_sbuf_source(source, root)
    else:
        _validate_hbm_source(source, root)
        storage = _resolve_hbm_root(source, root)
        offset = source.offset if hasattr(source, "offset") else 0
        dtype = source.dtype

    # Step 2: element_shape. AP counts override source.shape.
    element_shape = _resolve_element_shape(source, access_pattern)

    # Step 3: strides. AP overrides physical layout.
    # Strides come from the source (not storage/root) because a leading-index
    # slice like weights[i] reduces rank -- the slice's get_pattern() reports
    # correct rank-matched strides, while the root's strides have extra dims.
    if access_pattern is not None:
        strides, _ = _split_access_pattern(access_pattern)
    elif is_sbuf:
        strides = None  # SBUFLayout computes strides from tile grid; see below.
    else:
        strides = physical_strides(source, source.shape)

    # Step 4: tile size + remainder policy.
    tile_size = tuple(size)
    ndim = len(element_shape)
    n_batch_dims = ndim - len(tile_size)
    padded_tile_size = _pad_tile_size(tile_size, n_batch_dims)

    effective_element_shape = element_shape
    if remainder == "skip":
        effective_element_shape = _truncate_for_skip(element_shape, padded_tile_size)

    # Step 5: Grid + Layout.
    grid = Grid.from_shape(
        element_shape=effective_element_shape,
        tile_size=padded_tile_size,
        n_batch_dims=n_batch_dims,
    )

    if is_sbuf:
        view = _make_sbuf_ndslice(
            storage,
            element_shape,
            padded_tile_size,
            dtype,
            buffer_type,
            offset=offset,
        )
        layout = view.layout
        grid = view.grid if remainder != "skip" else grid
    else:
        layout = HBMLayout(
            source=storage,
            offset=offset,
            strides=strides,
            dtype=dtype,
            buffer_type=hbm_buffer_type(),
        )

    return NDSlice(grid, layout)


# ============================================================================
# blocks()
# ============================================================================


def blocks(
    source: Any,
    block_size: tuple,
    tile_size: Optional[tuple] = None,
    access_pattern: Optional[list] = None,
    root: Optional[Any] = None,
    buffer_type: Optional[Any] = None,
) -> "NDSlice":
    """
    Group tiles into blocks for coalesced DMA. No data movement.

    Args:
        source (HBM tensor | nl.ndarray | NDSlice): The tensor to block-tile.
            HBM tensor: top-level or sliced (sliced requires `root=`).
            Raw SBUF ndarray: pass with `buffer_type=nl.sbuf`.
            NDSlice: promote an existing tile view to a block view, or
            re-tile and promote in one step.
        block_size (tuple[int, ...]): Tiles per block, rank >= 2. Required.
            Must not exceed the source rank.
        tile_size (tuple[int, ...] | None): Tile shape (>= 2-D). Required
            for raw tensor sources; optional for NDSlice sources where it
            is inherited from the view (re-tile only happens when supplied).
        access_pattern (list[list[int, int]] | None): Source layout as
            ``[[stride_d, count_d], ...]`` -- one level per source dim.
            Raw sources only; rejected on NDSlice sources.
        root (HBM tensor | None): Parent tensor for a sliced HBM source.
            Required when source is a slice; see :func:`tiles` for details.
        buffer_type (nl.MemoryRegion | None): ``nl.sbuf`` for a raw SBUF
            source; ``None`` for HBM (default).

    Returns:
        NDSlice: Block-grid view. ``.shape`` is the block-grid extent per
        dim, ``.block_size`` is `block_size`, ``.tile_size`` is the
        underlying tile shape. Index with a slice helper from
        :mod:`neurotile.shard_helpers` for sharded views.

    Raises:
        AssertionError: Missing or invalid block_size; block_size rank
            exceeds source rank; raw source without tile_size; NDSlice
            source combined with raw-source-only kwargs (root,
            access_pattern, buffer_type); sliced HBM source without
            `root`.

    Example:
        # Raw source: tile then promote to blocks of 2x2 tiles each.
        view = nt.blocks(src, tile_size=(128, 512), block_size=(2, 2))

        # Promote an existing tile view into block granularity:
        tiles_view = nt.tiles(src, tile_size=(128, 512))
        block_view = nt.blocks(tiles_view, block_size=(2, 2))

        # Re-tile and promote in one step:
        block_view = nt.blocks(
            tiles_view, tile_size=(64, 256), block_size=(2, 2),
        )
    """
    assert block_size is not None, "nt.blocks(...): block_size= is required. For a tile-level view, use nt.tiles()."
    _validate_block_size(block_size)
    _validate_block_size_rank(block_size, source)

    # Raw sources require tile_size= (NDSlice inherits its tile_size). Check
    # here so the error names blocks(), not the forwarded tiles() call.
    if not isinstance(source, NDSlice):
        assert tile_size is not None, (
            "nt.blocks(<raw source>, ...): tile_size= is required for raw "
            "tensor sources. For an existing NDSlice, tile_size= is "
            "optional (inherited from the view)."
        )

    # Validate tile_size / AP / buffer_type with the same rules as
    # nt.tiles(). tiles() is re-called for the raw-source path below, but
    # that dispatcher checks only its own args -- so validate explicitly
    # here to cover the NDSlice-source case too.
    _validate_tiles_args(
        source=source,
        size=tile_size,
        access_pattern=access_pattern,
        buffer_type=buffer_type,
        remainder=None,
    )

    # --- NDSlice source: promote / re-tile + promote ---
    if isinstance(source, NDSlice):
        assert root is None, (
            "nt.blocks(NDSlice, ...): root= is only for raw sources. The NDSlice already knows its root via its layout."
        )
        assert access_pattern is None, "nt.blocks(NDSlice, ...): access_pattern= is only for raw sources."
        assert buffer_type is None, (
            "nt.blocks(NDSlice, ...): buffer_type= is only for raw sources. "
            "The NDSlice already knows its buffer via its layout."
        )

        view = _retile_ndslice(source, tile_size)
        return _prepend_block_level(view, block_size)

    # --- Raw source: build tile-level view -> prepend block axis ---
    tile_view = tiles(
        source,
        tile_size=tile_size,
        access_pattern=access_pattern,
        root=root,
        buffer_type=buffer_type,
    )
    return _prepend_block_level(tile_view, block_size)


def _prepend_block_level(view, block_size):
    """Prepend a block axis above the outermost tile axis on each non-batch dim.

    Delegates to Grid.with_block, which inserts a block axis with
    step = block_size * tile_step on each dim with bs >= 1, preserving
    any outer shard / broadcast axes.
    """
    grid = view.grid
    padded_block_size = _pad_tile_size(block_size, grid.n_batch_dims)
    # Strip leading batch padding (with_block reads tail offset).
    block_for_dims = padded_block_size[grid.n_batch_dims :]
    new_grid = grid.with_block(block_for_dims)
    return NDSlice(new_grid, view.layout)


# ============================================================================
# tensor_view()
# ============================================================================


def tensor_view(
    source: nl.ndarray,
    access_pattern: Optional[list[list[int]]] = None,
    buffer_type: Optional[nl.MemoryRegion] = None,
) -> NDSlice:
    """
    Create an untiled NDSlice over a raw tensor. No data movement.

    Use as the starting point for reshape / permute / flatten chains
    before tiling. To re-tile, pass the result to nt.tiles().

    Args:
        source: HBM tensor or raw SBUF ndarray (with buffer_type=nl.sbuf).
            NDSlice sources are rejected -- chain transforms on the view
            directly.
        access_pattern: Source layout ``[[stride, count], ...]``.
            Defaults to contiguous.
        buffer_type: nl.sbuf for a raw SBUF source; nl.shared_hbm /
            nl.private_hbm or None for HBM. nl.psum is rejected.

    Returns:
        Element-level NDSlice (one element per ``shape`` slot).

    Raises:
        AssertionError on misuse: NDSlice source, AP rank or count
            mismatch, invalid buffer_type.

    Example:
        view = nt.tensor_view(src).reshape((B*S, H)).flatten_dims(0, 1)
        gamma_bc = nt.tensor_view(gamma_sb.data, buffer_type=nl.sbuf)
    """
    _validate_tensor_view_args(source, access_pattern, buffer_type)

    element_shape = _resolve_element_shape(source, access_pattern)
    if access_pattern is not None:
        strides, _ = _split_access_pattern(access_pattern)
    else:
        strides = contiguous_strides(element_shape)
    ndim = len(element_shape)

    if buffer_type is not None and buffer_type == nl.sbuf:
        # SBUF: element-level view with tile_size = element_shape.
        ts = []
        for _d in range(ndim):
            ts.append(1)
        return _make_sbuf_ndslice(source, element_shape, tuple(ts), source.dtype, buffer_type)

    # HBM (default, or buffer_type is an HBM enum). buffer_type=nl.sbuf on a
    # non-SBUF tensor is rejected upstream by _validate_buffer_type.
    grid = Grid.from_shape(element_shape, None)
    layout = HBMLayout(
        source=source,
        offset=0,
        strides=strides,
        dtype=source.dtype,
        buffer_type=hbm_buffer_type(),
    )
    return NDSlice(grid, layout)


# ============================================================================
# alloc -- stubs
# ============================================================================


def alloc_tiles(
    tile_size: tuple[int, ...],
    grid: Optional[tuple[int, ...]] = None,
    buffer_type: Optional[nl.MemoryRegion] = None,
    dtype=None,
    element_shape: Optional[tuple[int, ...]] = None,
) -> NDSlice:
    """
    Allocate a tiled buffer and return an NDSlice over it.

    Pass either ``grid`` (tile-aligned allocation: extent = ``grid * tile_size``)
    or ``element_shape`` (exact extent; last tile per dim is a remainder when
    not divisible). Omit both for a single-tile allocation.

    Args:
        tile_size: Per-tile shape (>= 2-D). Dim 0 maps to SBUF P axis.
        grid: Tile-grid shape (one count per tile_size dim). Mutually
            exclusive with ``element_shape``.
        buffer_type: nl.sbuf for SBUF; nl.shared_hbm / nl.private_hbm for
            HBM. nl.psum is rejected -- use nt.psum_pool() for PSUM.
        dtype: Element dtype (e.g. nl.bfloat16). Required.
        element_shape: Logical element extent per dim. Mutually exclusive
            with ``grid``. Tile grid is inferred via ceiling division.

    Returns:
        NDSlice over the freshly allocated buffer.

    Raises:
        AssertionError on misuse: missing buffer_type / dtype, both grid
            and element_shape passed, rank or value mismatches.

    Example:
        # Tile-aligned SBUF accumulator
        acc = nt.alloc_tiles(tile_size=(128, 512), grid=(M, N),
                             buffer_type=nl.sbuf, dtype=nl.bfloat16)

        # Exact-extent HBM output (with remainder tiles on last col)
        out = nt.alloc_tiles(tile_size=(128, 512),
                             element_shape=(128, 1792),
                             buffer_type=nl.shared_hbm, dtype=nl.bfloat16)
    """
    _validate_alloc_tiles_args(tile_size, grid, element_shape, buffer_type, dtype)

    tile_size = tuple(tile_size)
    tile_p = tile_size[0]
    tile_f = product(tile_size, start=1)

    # Derive grid from element_shape if not provided
    if element_shape is not None:
        element_shape = tuple(element_shape)
        if grid is None:
            tile_shape = []
            for d in range(len(tile_size)):
                if d < len(element_shape):
                    tile_shape.append((element_shape[d] + tile_size[d] - 1) // tile_size[d])
                else:
                    tile_shape.append(1)
            tile_shape = tuple(tile_shape)
        else:
            tile_shape = tuple(grid)
    else:
        # Tile grid: (1, 1, ...) for single tile, or user-specified
        if grid is None:
            tile_shape = []
            for d in range(len(tile_size)):
                tile_shape.append(1)
            tile_shape = tuple(tile_shape)
        else:
            tile_shape = tuple(grid)

    total_tiles = 1
    for g in tile_shape:
        total_tiles = total_tiles * g

    # Allocate buffer
    if buffer_type == hbm_buffer_type():
        # HBM allocation
        if element_shape is None:
            element_shape = []
            for d in range(len(tile_size)):
                if d < len(tile_shape):
                    element_shape.append(tile_shape[d] * tile_size[d])
                else:
                    element_shape.append(tile_size[d])
            element_shape = tuple(element_shape)

        hbm = nl.ndarray(element_shape, dtype=dtype, buffer=buffer_type)
        strides = contiguous_strides(element_shape)
        alloc_grid = Grid.from_shape(element_shape, tile_size)
        layout = HBMLayout(hbm, 0, strides, dtype, buffer_type)
        return NDSlice(alloc_grid, layout)
    else:
        # SBUF / PSUM allocation -- flat 2D (P-tiles fold into F-columns)
        if element_shape is not None:
            p_tiles = (element_shape[0] + tile_p - 1) // tile_p
            actual_f = product(element_shape, start=1) * p_tiles
            sbuf = nl.ndarray((tile_p, actual_f), dtype=dtype, buffer=buffer_type)
            view_element_shape = element_shape
        else:
            sbuf = nl.ndarray((tile_p, total_tiles * tile_f), dtype=dtype, buffer=buffer_type)
            view_element_shape = []
            for d in range(len(tile_size)):
                if d < len(tile_shape):
                    view_element_shape.append(tile_shape[d] * tile_size[d])
                else:
                    view_element_shape.append(tile_size[d])
            view_element_shape = tuple(view_element_shape)
        grid_obj, layout = SBUFLayout.build_view(
            sbuf,
            view_element_shape,
            tile_size,
            dtype,
            buffer_type,
        )

        # grid=None (single tile, no grid): collapse the tile axis into a
        # single elem axis whose count is the full per-dim element extent.
        # The walk is unchanged (one tile == full extent); the descent shape
        # changes from "tile-level (count=1)" to "element-level (count=ext)"
        # so that downstream `.ap()` addresses the whole tile.
        # grid=(M,N) (explicit grid, even 1x1): keep tile level for navigation.
        if grid is None and element_shape is None:
            _elem_axes = []
            for _d in range(grid_obj.ndim):
                _ext = grid_obj.element_shape[_d]
                _elem_axes.append(Axis(count=_ext, step=1, dim=_d, label=AxisLabel.ELEM))
            grid_obj = Grid(
                element_shape=grid_obj.element_shape,
                axes=tuple(_elem_axes),
                cursor=grid_obj.cursor,
                n_batch_dims=grid_obj.n_batch_dims,
                tiled=grid_obj.tiled,
            )

        return NDSlice(grid_obj, layout)


def alloc_blocks(
    tile_size: tuple[int, ...],
    block_size: tuple[int, ...],
    grid: Optional[tuple[int, ...]] = None,
    buffer_type: Optional[nl.MemoryRegion] = None,
    dtype=None,
    element_shape: Optional[tuple[int, ...]] = None,
) -> NDSlice:
    """
    Allocate a block-structured buffer and return an NDSlice over it.

    Like ``alloc_tiles`` but the returned NDSlice carries a block level
    above the tile grid. Pass either ``grid`` (block-aligned: tile grid =
    ``grid * block_size``) or ``element_shape`` (exact extent; ceiling-divides
    into the tile grid). Omit both for a single-block allocation.

    Args:
        tile_size: Per-tile shape (>= 2-D). Dim 0 maps to SBUF P axis.
        block_size: Tiles per block (one count per ``tile_size`` dim).
        grid: Block-grid shape. Mutually exclusive with ``element_shape``.
        buffer_type: nl.sbuf for SBUF; nl.shared_hbm / nl.private_hbm for
            HBM. nl.psum is rejected -- use nt.psum_pool() for PSUM.
        dtype: Element dtype (e.g. nl.bfloat16). Required.
        element_shape: Logical element extent per dim. Mutually exclusive
            with ``grid``. Tile grid ceiling-divides; block grid is then
            derived.

    Returns:
        NDSlice over the freshly allocated buffer with a block-level Grid.

    Raises:
        AssertionError on misuse: missing buffer_type / dtype, both grid
            and element_shape passed, rank or value mismatches.

    Example:
        # Block-aligned SBUF accumulator: 4x2 block grid, 2x2 tiles per block
        out = nt.alloc_blocks(
            tile_size=(128, 512),
            block_size=(2, 2),
            grid=(4, 2),
            buffer_type=nl.sbuf, dtype=nl.bfloat16,
        )

        # Exact-extent SBUF block buffer with remainder tiles
        gated = nt.alloc_blocks(
            tile_size=(128, 512),
            block_size=(1, 4),
            element_shape=(128, 1792),
            buffer_type=nl.sbuf, dtype=nl.bfloat16,
        )
    """
    _validate_alloc_blocks_args(
        tile_size,
        block_size,
        grid,
        element_shape,
        buffer_type,
        dtype,
    )

    # Allocate as a flat tile grid; alloc_tiles handles both grid= and
    # element_shape= paths. Then promote with _prepend_block_level (the same
    # helper used by nt.blocks(NDSlice, block_size=)) so block layering is
    # consistent across all block-producing factories.
    if element_shape is not None:
        # Forward exact extent; alloc_tiles ceil-divides into the tile grid.
        flat = alloc_tiles(
            tile_size,
            element_shape=element_shape,
            buffer_type=buffer_type,
            dtype=dtype,
        )
    elif grid is not None:
        # Block grid -> flat tile grid by element-wise multiply.
        tile_grid = []
        for d in range(len(block_size)):
            tile_grid.append(grid[d] * block_size[d])
        flat = alloc_tiles(
            tile_size,
            grid=tuple(tile_grid),
            buffer_type=buffer_type,
            dtype=dtype,
        )
    else:
        # Single block: tile grid = block_size.
        flat = alloc_tiles(
            tile_size,
            grid=tuple(block_size),
            buffer_type=buffer_type,
            dtype=dtype,
        )

    return _prepend_block_level(flat, tuple(block_size))


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "tiles",
    "blocks",
    "tensor_view",
    "alloc_tiles",
    "alloc_blocks",
    "NDSlice",
    "Grid",
    "HBMLayout",
    "SBUFLayout",
]
