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
"""Input validation for the public factories.

Houses the `_validate_*` helpers used by `tiles()`, `blocks()`,
`alloc_tiles()`, `alloc_blocks()`, and `tensor_view()`. Keeping them
in one place keeps `factories.py` focused on dispatch + builder
pipelines.

Each helper raises `AssertionError` with a named, user-facing message
on misuse.
"""

import nki.language as nl

from ._helpers import is_identity_tensor
from .ndslice import NDSlice

# Valid string values for remainder=. Exposed so callers / tests / docstrings
# can reference one authoritative list.
_VALID_REMAINDER = (None, "skip")


# ============================================================================
# Shape arg helpers (shared by tiles / blocks / alloc_tiles / alloc_blocks)
# ============================================================================


def _validate_shape_arg(value, label, caller="nt.tiles/nt.blocks"):
    """Shared check: tuple/list of positive ints, rank >= 2.

    Used for `tile_size=` (nt.tiles / nt.blocks / nt.alloc_tiles /
    nt.alloc_blocks) and `block_size=` (nt.blocks / nt.alloc_blocks).
    Minimum rank is 2 because NeuronCore tiles have a P (partition)
    and F (free) axis; a 1-D shape has no meaningful partition layout.

    Args:
        value: The tuple/list to validate.
        label: User-visible kwarg name (e.g. "tile_size", "block_size").
        caller: Function name prefix in error messages (e.g.
            "nt.tiles/nt.blocks", "nt.alloc_tiles", "nt.alloc_blocks").
    """
    assert isinstance(value, (tuple, list)), caller + ": " + label + "= must be a tuple or list of ints."
    assert len(value) >= 2, (
        caller
        + ": "
        + label
        + "= must have at least 2 dims (P, F), got "
        + str(len(value))
        + " dim(s): "
        + str(tuple(value))
    )
    for d in range(len(value)):
        s = value[d]
        assert isinstance(s, int) and not isinstance(s, bool), caller + ": " + label + "[" + str(d) + "] must be int."
        assert s > 0, caller + ": " + label + "[" + str(d) + "] must be > 0, got " + str(s)


def _validate_tile_size(size, caller="nt.tiles/nt.blocks"):
    """tile_size must be a tuple/list of positive ints, rank >= 2."""
    _validate_shape_arg(size, "tile_size", caller=caller)


def _validate_block_size(block_size, caller="nt.tiles/nt.blocks"):
    """block_size must be a tuple/list of positive ints, rank >= 2."""
    _validate_shape_arg(block_size, "block_size", caller=caller)


def _validate_block_size_rank(block_size, source):
    """block_size rank must not exceed source rank.

    The block level is prepended onto each source dim's stack. Extra block
    dims beyond source rank have nowhere to land and would be silently
    dropped, producing a grid with fewer block dims than requested.
    """
    if isinstance(source, NDSlice):
        source_rank = source.ndim
    else:
        source_rank = len(source.shape)
    assert len(block_size) <= source_rank, (
        "nt.blocks(): block_size has "
        + str(len(block_size))
        + " dims but source has "
        + str(source_rank)
        + " dims; block_size rank must not exceed source rank."
    )


def _validate_remainder(remainder):
    """remainder must be None or one of the documented string options."""
    assert remainder in _VALID_REMAINDER, (
        "nt.tiles/nt.blocks: remainder="
        + str(remainder)
        + " is not valid. Must be one of "
        + str(_VALID_REMAINDER)
        + "."
    )


def _validate_buffer_type(buffer_type):
    """buffer_type must be None, nl.sbuf, nl.private_hbm, or nl.shared_hbm.

    Raw tensor sources: nl.sbuf for SBUF-backed ndarrays; nl.shared_hbm /
    nl.private_hbm or None for HBM tensors. nl.psum is rejected -- PSUM
    buffers are produced via nt.psum_pool(), not via tile factories.
    """
    if buffer_type is None:
        return
    assert isinstance(buffer_type, nl.MemoryRegion), (
        "nt.tiles/nt.blocks: buffer_type= must be None or an nl.MemoryRegion "
        "(nl.sbuf, nl.shared_hbm, nl.private_hbm). Got " + str(buffer_type) + "."
    )
    assert buffer_type != nl.psum, (
        "nt.tiles/nt.blocks: buffer_type=nl.psum is not supported. Use "
        "nl.sbuf for raw SBUF sources, nl.shared_hbm / nl.private_hbm for "
        "HBM, or leave unset. PSUM buffers are produced via nt.psum_pool()."
    )


# ============================================================================
# Access pattern + tile/source compatibility
# ============================================================================


def _validate_access_pattern(access_pattern):
    """access_pattern must be a list of [stride, count] pairs.

    The AP defines the logical view's rank, shape, and strides; it is
    not constrained to match the source's rank. The AP's rank is the
    rank of the resulting logical view (each level becomes one dim with
    count=count, stride=stride). The source is only required to have
    enough bytes to hold the AP's largest addressed offset; that's
    checked by ``_validate_ap_addresses_fit_source``.
    """
    if access_pattern is None:
        return
    assert isinstance(access_pattern, (list, tuple)), (
        "nt.tiles/nt.blocks: access_pattern= must be a list of [stride, count] pairs."
    )
    assert len(access_pattern) >= 1, "nt.tiles/nt.blocks: access_pattern= must have at least one [stride, count] level."
    for d in range(len(access_pattern)):
        level = access_pattern[d]
        assert isinstance(level, (list, tuple)) and len(level) == 2, (
            "nt.tiles/nt.blocks: access_pattern[" + str(d) + "] must be [stride, count]; got " + str(level)
        )
        stride = level[0]
        count = level[1]
        assert isinstance(stride, int) and not isinstance(stride, bool), (
            "nt.tiles/nt.blocks: access_pattern[" + str(d) + "][0] (stride) must be int."
        )
        assert stride > 0, (
            "nt.tiles/nt.blocks: access_pattern[" + str(d) + "][0] (stride) must be > 0, got " + str(stride)
        )
        assert isinstance(count, int) and not isinstance(count, bool), (
            "nt.tiles/nt.blocks: access_pattern[" + str(d) + "][1] (count) must be int."
        )
        assert count > 0, "nt.tiles/nt.blocks: access_pattern[" + str(d) + "][1] (count) must be > 0, got " + str(count)


def _validate_tile_size_fits_source(size, source_shape):
    """Per-dim tile size <= source element extent on that dim.

    ``tile_size[d]`` operates on ``source_shape[n_batch + d]``
    (position-matched after auto-padding leading batch dims). If you want
    the tile's P-axis to partition an *inner* source dim (transpose-on-
    load), use ``.load(transpose=True)`` on a contiguous tile instead.
    """
    n_batch = len(source_shape) - len(size)
    if n_batch < 0:
        return  # handled by the earlier rank check caller
    for d in range(len(size)):
        src_d = source_shape[n_batch + d]
        assert size[d] <= src_d, (
            "nt.tiles/nt.blocks: tile_size["
            + str(d)
            + "]="
            + str(size[d])
            + " exceeds source extent "
            + str(src_d)
            + " on dim "
            + str(n_batch + d)
            + " of source shape "
            + str(source_shape)
            + ". tile_size[d] operates on source dim (n_batch + d); use "
            ".load(transpose=True) if you intended to map the tile's "
            "P-axis to an inner source dim."
        )


def _ap_element_shape(access_pattern):
    """Per-level counts as a tuple -- the AP's view shape."""
    counts = []
    for level in access_pattern:
        counts.append(level[1])
    return tuple(counts)


def _validate_ap_addresses_fit_source(access_pattern, source_shape):
    """AP must not address bytes beyond the source's flat extent.

    The AP can have any rank (independent of source rank). What matters
    is that the largest element offset the AP can address fits within
    the source: the source must hold every element the view claims to
    expose. The largest offset is ``sum((count_i - 1) * stride_i)`` --
    the position of the last element in the unfolded walk.
    """
    if access_pattern is None:
        return
    max_offset = 0
    for d in range(len(access_pattern)):
        stride = access_pattern[d][0]
        count = access_pattern[d][1]
        max_offset = max_offset + (count - 1) * stride
    source_extent = 1
    for d in range(len(source_shape)):
        source_extent = source_extent * source_shape[d]
    assert max_offset < source_extent, (
        "nt.tiles/nt.blocks: access_pattern addresses element offset "
        + str(max_offset)
        + " (largest) but source has only "
        + str(source_extent)
        + " elements (shape "
        + str(source_shape)
        + "). The AP claims to expose elements past the source's end."
    )


def _validate_dim_tuple(value, label, expected_rank, caller="nt.alloc_tiles"):
    """value must be a tuple/list of positive ints with `expected_rank` dims.

    Args:
        caller: Function name prefix used in error messages.
    """
    assert isinstance(value, (tuple, list)), caller + ": " + label + "= must be a tuple or list of ints."
    assert len(value) == expected_rank, (
        caller
        + ": "
        + label
        + " has "
        + str(len(value))
        + " dims but tile_size has "
        + str(expected_rank)
        + " dims; ranks must match."
    )
    for d in range(len(value)):
        s = value[d]
        assert isinstance(s, int) and not isinstance(s, bool), caller + ": " + label + "[" + str(d) + "] must be int."
        assert s > 0, caller + ": " + label + "[" + str(d) + "] must be > 0, got " + str(s)


# ============================================================================
# Per-factory entry points (compose the helpers above)
# ============================================================================


def _validate_alloc_tiles_args(
    tile_size,
    grid,
    element_shape,
    buffer_type,
    dtype,
    caller="nt.alloc_tiles",
):
    """Input validation for nt.alloc_tiles() and (via caller=) nt.alloc_blocks().

    Asserts:
      - tile_size is a positive-int tuple/list, rank >= 2.
      - buffer_type is one of None / nl.sbuf / nl.shared_hbm / nl.private_hbm
        (nl.psum rejected -- use nt.psum_pool).
      - buffer_type is required (allocation must know which memory region).
      - dtype is required.
      - At most one of grid, element_shape is given (they express the same
        intent; passing both forces ambiguous precedence).
      - grid / element_shape ranks match tile_size and entries are positive.

    Args:
        caller: Function name prefix used in error messages (lets
            nt.alloc_blocks errors say "nt.alloc_blocks: ..." instead of
            the default "nt.alloc_tiles: ...").
    """
    _validate_tile_size(tile_size, caller=caller)
    assert buffer_type is not None, (
        caller + "(...): buffer_type= is required (nl.sbuf, nl.shared_hbm, or nl.private_hbm)."
    )
    _validate_buffer_type(buffer_type)
    assert dtype is not None, caller + "(...): dtype= is required."
    assert grid is None or element_shape is None, (
        caller + "(...): pass either grid= or element_shape=, not both. "
        "Both express the allocation extent; specifying both is ambiguous. "
        "Use grid= for tile-aligned allocations and element_shape= when the "
        "actual extent is not a multiple of tile_size."
    )
    if grid is not None:
        _validate_dim_tuple(grid, "grid", len(tile_size), caller=caller)
    if element_shape is not None:
        _validate_dim_tuple(element_shape, "element_shape", len(tile_size), caller=caller)


def _validate_alloc_blocks_args(
    tile_size,
    block_size,
    grid,
    element_shape,
    buffer_type,
    dtype,
):
    """Input validation for nt.alloc_blocks().

    Reuses _validate_alloc_tiles_args for the shared contract and adds
    block_size validation: positive ints, rank >= 2, rank matches tile_size.
    """
    _validate_alloc_tiles_args(
        tile_size,
        grid,
        element_shape,
        buffer_type,
        dtype,
        caller="nt.alloc_blocks",
    )
    _validate_block_size(block_size, caller="nt.alloc_blocks")
    assert len(block_size) == len(tile_size), (
        "nt.alloc_blocks(): block_size has "
        + str(len(block_size))
        + " dims but tile_size has "
        + str(len(tile_size))
        + " dims; ranks must match."
    )


def _validate_tiles_args(
    source,
    size,
    access_pattern,
    buffer_type,
    remainder,
):
    """Central gate for all nt.tiles()/nt.blocks() public input validation.

    Fires on wrong types, out-of-range values, rank mismatches, and unknown
    enum strings.
    """
    _validate_remainder(remainder)
    _validate_buffer_type(buffer_type)

    if isinstance(source, NDSlice):
        # Re-tile or descend. size is optional; AP is rejected (handled at
        # the tiles() dispatch layer).
        if size is not None:
            _validate_tile_size(size)
        return

    # Raw tensor: tile_size is required.
    assert size is not None, "nt.tiles(): tile_size= is required for raw tensor sources."
    _validate_tile_size(size)

    source_shape = tuple(source.shape)
    _validate_access_pattern(access_pattern)
    _validate_ap_addresses_fit_source(access_pattern, source_shape)

    # The view's logical shape comes from the AP if given, else from
    # source.shape directly. tile_size rank / extent checks are against
    # the logical shape -- the AP is the source-of-truth for what the
    # view exposes.
    view_shape = _ap_element_shape(access_pattern) if access_pattern is not None else source_shape

    assert len(size) <= len(view_shape), (
        "nt.tiles(): tile_size has "
        + str(len(size))
        + " dims but the view has "
        + str(len(view_shape))
        + " dims (from "
        + ("access_pattern=" if access_pattern is not None else "source.shape")
        + "); tile_size must not exceed the view's rank."
    )

    # Tile extent per dim must not exceed the view's element extent on that dim.
    _validate_tile_size_fits_source(tuple(size), view_shape)

    # Higher-rank AP (more levels than source.ndim) currently requires
    # tile_size to match the view exactly -- one tile covers the whole
    # view, no tile-grid iteration. Per-tile DMA descriptors for a
    # multi-level walk need offset adjustments that the AP-emit path
    # does not yet derive correctly; until that lands, restrict to the
    # single-tile case.
    # TODO: support multi-tile iteration over higher-rank-AP views by
    # making the AP emitter aware of per-tile offsets in the multi-level
    # walk; today the static pattern is built from the parent's full
    # walk and a child tile's load would re-issue the parent's DMA.
    if access_pattern is not None and len(access_pattern) > len(source_shape):
        n_batch = len(view_shape) - len(size)
        padded_size = []
        for d in range(n_batch):
            padded_size.append(1)
        for d in range(len(size)):
            padded_size.append(size[d])
        assert tuple(padded_size) == tuple(view_shape), (
            "nt.tiles(): higher-rank access_pattern (more levels than "
            "source.ndim) currently requires tile_size to match the view "
            "shape exactly (single-tile load, no tile-grid iteration). "
            "Got tile_size="
            + str(tuple(size))
            + " (padded to "
            + str(tuple(padded_size))
            + ") but the view shape from access_pattern is "
            + str(tuple(view_shape))
            + ". TODO: support multi-tile iteration over higher-rank-AP views."
        )


def _validate_hbm_source(source, root):
    """Require root= when an HBM source is a sliced view of another tensor.

    Sliced views (e.g. `w_qkv[:, a:b]`) carry a narrowed logical shape
    but physical strides of the parent. Without root=, the factory
    would derive strides from the slice's logical shape and silently
    emit wrong APs -- multi-tile loads would read bytes from wrong
    offsets. Passing root= fixes the coordinate space to the parent:
    strides and offsets are correct, and `indirect_dim` consistently
    indexes into root's dims.

    Non-NKI sources (numpy, MockTensor in tests) skip this check since
    they lack `_pattern`.
    """
    if not hasattr(source, "_pattern"):
        return
    if is_identity_tensor(source):
        return
    assert root is not None, (
        "nt.tiles/nt.blocks(source=<sliced view>): the HBM source is a "
        "slice of another tensor (e.g. `w_qkv[:, a:b]`). Pass "
        "root=<parent tensor> so strides and offsets are derived against "
        "the parent's physical memory. Without root=, strides would come "
        "from the slice's logical shape and multi-tile loads would read "
        "bytes from wrong offsets."
    )
    assert is_identity_tensor(root), (
        "nt.tiles/nt.blocks(root=...): root must be a top-level tensor, not a view or slice."
    )


def _validate_tensor_view_args(source, access_pattern, buffer_type):
    """Input validation for nt.tensor_view().

    tensor_view is a constructor over a raw tensor; chain transforms
    (.reshape, .permute, .flatten_dims, ...) on the result. NDSlice
    sources are rejected -- the existing view already has a layout to
    chain against.
    """
    assert not isinstance(source, NDSlice), (
        "nt.tensor_view(source=NDSlice): tensor_view is a constructor over "
        "a raw tensor. To chain transforms on an existing NDSlice, call "
        ".reshape() / .permute() / .flatten_dims() on it directly."
    )
    _validate_buffer_type(buffer_type)
    source_shape = tuple(source.shape)
    _validate_access_pattern(access_pattern)
    _validate_ap_addresses_fit_source(access_pattern, source_shape)


# ============================================================================
# psum_pool() validation
# ============================================================================


def _validate_psum_pool_grid_args(
    tile_size,
    grid,
    element_shape,
    bank_axis,
    bank_ids,
    psum_bank_size=2048,
    num_hw_banks=8,
):
    """Input validation for nt.psum_pool.

    Allocation modes accepted:

    - ``bank_axis=None``, ``bank_ids=None``: compiler-managed (no
      ``address=`` placement; the caller skips that path).
    - ``bank_axis=None``, ``bank_ids=...``: every grid tile on its own
      bank. ``len(bank_ids) == product(grid)``; ``slots_per_bank == 1``.
    - ``bank_axis=int``, ``bank_ids=...``: pack the non-bank dims as
      slots within each bank. ``len(bank_ids) == grid[bank_axis]``;
      slot stride must satisfy compiler accumulator-region rule.
    - ``bank_axis=int``, ``bank_ids=None``: rejected.

    Returns:
        ``(tile_grid_shape, num_banks, slots_per_bank, tile_f)`` --
        ``num_banks`` is ``product(grid)`` in the all-fanout mode and
        ``grid[bank_axis]`` otherwise; ``slots_per_bank`` is 1 in the
        compiler-managed and all-fanout modes.
    """
    _validate_shape_arg(tile_size, "tile_size", caller="nt.psum_pool")

    assert grid is not None or element_shape is not None, (
        "nt.psum_pool: pass either grid= (tile-aligned allocation) or "
        "element_shape= (actual extent; the factory ceiling-divides by "
        "tile_size to derive the grid)."
    )
    assert grid is None or element_shape is None, (
        "nt.psum_pool: pass either grid= or element_shape=, not both. "
        "Both express the allocation extent; specifying both is ambiguous. "
        "Use grid= for tile-aligned allocations and element_shape= when "
        "the actual extent is not a multiple of tile_size."
    )

    if grid is not None:
        _validate_dim_tuple(grid, "grid", len(tile_size), caller="nt.psum_pool")
        tile_grid_shape = tuple(grid)
    else:
        _validate_dim_tuple(
            element_shape,
            "element_shape",
            len(tile_size),
            caller="nt.psum_pool",
        )
        es = tuple(element_shape)
        tg = []
        for d in range(len(tile_size)):
            tg.append((es[d] + tile_size[d] - 1) // tile_size[d])
        tile_grid_shape = tuple(tg)

    tile_count = 1
    for d in range(len(tile_grid_shape)):
        tile_count = tile_count * tile_grid_shape[d]

    tile_f = 1
    for d in range(1, len(tile_size)):
        tile_f = tile_f * tile_size[d]

    # Reject the ``bank_axis=int, bank_ids=None`` combination.
    if bank_axis is not None and bank_ids is None:
        assert False, (
            "nt.psum_pool: bank_axis= without bank_ids= is not supported. "
            "Either pass bank_ids= to pin placement, or leave both as None "
            "to let the compiler manage allocation."
        )

    # Compiler-managed: no explicit placement. Each tile is its own
    # ndarray (no ``address=``); compiler picks banks.
    if bank_axis is None and bank_ids is None:
        return tile_grid_shape, tile_count, 1, tile_f

    # bank_ids is non-None from here on.
    assert isinstance(bank_ids, (tuple, list)), (
        "nt.psum_pool: bank_ids= must be a tuple or list of ints in [0, " + str(num_hw_banks) + ")."
    )
    for i in range(len(bank_ids)):
        b = bank_ids[i]
        assert isinstance(b, int) and not isinstance(b, bool), (
            "nt.psum_pool: bank_ids[" + str(i) + "] must be int, got " + str(b)
        )
        assert 0 <= b < num_hw_banks, (
            "nt.psum_pool: bank_ids["
            + str(i)
            + "]="
            + str(b)
            + " is out of range; PSUM has "
            + str(num_hw_banks)
            + " banks (0.."
            + str(num_hw_banks - 1)
            + ")."
        )

    if bank_axis is None:
        # Every tile on its own bank.
        assert len(bank_ids) == tile_count, (
            "nt.psum_pool: bank_axis=None expects len(bank_ids)= "
            + str(tile_count)
            + " (one bank per grid tile), got "
            + str(len(bank_ids))
            + "."
        )
        assert tile_count <= num_hw_banks, (
            "nt.psum_pool: "
            + str(tile_count)
            + " grid tiles exceed the "
            + str(num_hw_banks)
            + " available PSUM banks. Reduce grid "
            "size or pack tiles via an explicit bank_axis= int."
        )
        # Duplicate bank IDs in ``bank_axis=None`` mode would alias multiple
        # tiles onto the same physical bank; the all-fanout contract says
        # every tile gets its own bank.
        for i in range(len(bank_ids)):
            b = bank_ids[i]
            for j in range(i):
                assert bank_ids[j] != b, (
                    "nt.psum_pool: bank_axis=None requires unique bank_ids "
                    "(every grid tile on its own bank). bank_ids["
                    + str(j)
                    + "] and bank_ids["
                    + str(i)
                    + "] both = "
                    + str(b)
                    + "."
                )
        return tile_grid_shape, tile_count, 1, tile_f

    # bank_axis = int: fan one dim across banks; pack the rest as slots.
    assert isinstance(bank_axis, int) and not isinstance(bank_axis, bool), (
        "nt.psum_pool: bank_axis= must be int or None, got " + str(bank_axis)
    )
    assert 0 <= bank_axis < len(tile_size), (
        "nt.psum_pool: bank_axis="
        + str(bank_axis)
        + " is out of range for tile_size with "
        + str(len(tile_size))
        + " dim(s)."
    )

    num_banks = tile_grid_shape[bank_axis]
    assert num_banks <= num_hw_banks, (
        "nt.psum_pool: tile_grid_shape[bank_axis]="
        + str(num_banks)
        + " exceeds the "
        + str(num_hw_banks)
        + " available PSUM banks. Reduce the count along bank_axis or "
        "pick a different bank_axis."
    )
    assert len(bank_ids) == num_banks, (
        "nt.psum_pool: len(bank_ids)="
        + str(len(bank_ids))
        + " must match tile_grid_shape[bank_axis]="
        + str(num_banks)
        + "."
    )

    slots_per_bank = 1
    for d in range(len(tile_grid_shape)):
        if d != bank_axis:
            slots_per_bank = slots_per_bank * tile_grid_shape[d]

    # Accumulation-region constraint (verified via raw-NKI probes,
    # see psum_pool._build_psum_ndslice docstring): slots in one bank
    # must sit at least ``max(512, 4 * tile_f)`` F-elements apart.
    matmul_quantum = 512
    slot_stride = max(matmul_quantum, 4 * tile_f)
    used_per_bank = slots_per_bank * slot_stride
    assert used_per_bank <= psum_bank_size, (
        "nt.psum_pool: "
        + str(slots_per_bank)
        + " slots per bank x slot_stride="
        + str(slot_stride)
        + " (= max("
        + str(matmul_quantum)
        + ", 4 * tile_f="
        + str(tile_f)
        + ")) = "
        + str(used_per_bank)
        + " elements, exceeding the "
        + str(psum_bank_size)
        + "-element PSUM bank capacity. "
        "Reduce slots_per_bank (smaller grid on non-bank dims), shrink "
        "tile_size on F, or pass bank_axis=None to spread every tile "
        "across its own bank."
    )

    return tile_grid_shape, num_banks, slots_per_bank, tile_f
