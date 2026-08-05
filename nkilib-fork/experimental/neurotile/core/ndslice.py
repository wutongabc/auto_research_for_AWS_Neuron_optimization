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
from typing import Optional

import nki.language as nl

from ._helpers import replace_at, sbuf_buffer_type, validate_index_key
from .axis import Axis, AxisLabel, IndirectKind
from .grid import Grid
from .layout_hbm import HBMLayout
from .layout_sbuf import SBUFLayout
from .transforms import (
    compute_broadcast,
    compute_expand_dim,
    compute_flatten_dims,
    compute_fold,
    compute_permute,
    compute_reshape,
    compute_reshape_dim,
    compute_squeeze_dim,
)

# ============================================================================
# NDSlice
# ============================================================================


def _shift_dims_after_drop(dims, dropped_dims):
    """Re-number ``dims`` after ``dropped_dims`` were removed from the grid.

    A dim id ``D`` becomes ``D - count(d in dropped_dims where d < D)``;
    if ``D`` itself was dropped, it's removed from the result.
    """
    result = []
    for d in dims:
        if d in dropped_dims:
            continue
        shift = 0
        for dd in dropped_dims:
            if dd < d:
                shift = shift + 1
        result.append(d - shift)
    return result


def _is_post_consume_leaf(grid, dim):
    """True when ``dim`` has exactly one axis left and that axis is a
    sub-tile leaf left over after a parent index consumed the iteration
    level above it.

    Triggered states:
      - PARTITION: only the partition dim of a tiled view carries this
        label, and only after the TILE / BLOCK above it was consumed.
      - ELEM with ``walked < element_shape[dim]``: the elem-leaf was
        clamped after a prior index (post-consume sub-tile span).

    NOT triggered for untouched batch / untiled ELEM (``walked ==
    element_shape[dim]``) or broadcast (``step == 0``) -- those are
    still indexable by partial keys.
    """
    axes = grid.axes_for(dim)
    if len(axes) != 1:
        return False
    only = axes[0]
    if only.label == AxisLabel.PARTITION:
        return True
    if only.label not in (AxisLabel.ELEM, AxisLabel.BROADCAST):
        return False
    if only.step == 0:
        return False
    return only.count * only.step < grid.element_shape[dim]


def _is_partition_leaf_only(grid, dim):
    """True when ``dim`` has exactly one axis and it's the partition
    leaf. Int indexing on this state narrows the partition to count=1
    (sub-tile P-row) rather than fully consuming the dim -- SBUF
    allocations must keep a partition-row axis (2-D physical memory).
    """
    axes = grid.axes_for(dim)
    return len(axes) == 1 and axes[0].label == AxisLabel.PARTITION


class NDSlice(nl.NKIObject):
    """
    The user-facing tile view over a tensor. Pairs a Grid with a Layout.

    ``NDSlice`` is what every public factory (``nt.tiles``, ``nt.blocks``,
    ``nt.alloc_tiles``, ``nt.tensor_view``) returns and what every kernel
    indexes, iterates, and DMAs. It carries two collaborators:

    - **Grid** (``self.grid``) -- the logical iteration schedule: what
      tiles exist, in what order, at what level. See :class:`Grid`.
    - **Layout** (``self.layout``) -- the physical addressing: source
      tensor, byte offset, strides, optional indirect runtime offset.
      :class:`HBMLayout` for HBM-backed views, :class:`SBUFLayout` for
      SBUF views (e.g. the result of ``.load()``).

    All indexing (``view[i, j]``, ``view[a:b, :]``, ``view[runtime_k]``),
    iteration (``view.tolist()``, ``for x in view``), and DMA
    (``.load()``, ``.store()``, ``.stream()``) goes through here.
    Operations are immutable -- each returns a new ``NDSlice`` rather
    than mutating in place.

    Attributes:
        shape (tuple[int, ...]): Per-dim item count from the iteration
            cursor onward (what users see on the view).
        element_shape (tuple[int, ...]): Per-dim remaining-element extent
            (narrows as dims are indexed).
        tile_size (tuple[int, ...] | None): Per-dim tile granularity, or
            ``None`` for untiled views.
        block_size (tuple[int, ...] | None): Per-dim block granularity,
            or ``None`` when no block axis.
        ndim (int): Number of source dims.
        is_remainder (bool): True when this view's outermost iteration
            on any dim straddles or sits past a partial-tile boundary
            (combined Grid + Layout-offset check; either alone misses
            the trailing-tile case).
        dtype: Element dtype.
        source: Root tensor (HBM) or SBUF buffer the view addresses.
        offset (int): Compile-time element offset into ``source``.
        strides (tuple[int, ...]): Per-dim element strides.
        buffer_type: ``nl.shared_hbm`` / ``nl.private_hbm`` / ``nl.sbuf`` / ``nl.psum``.
        data: For SBUF views, the underlying ``nl.ndarray``; ``None``
            for HBM (use ``.load()`` to materialize).
    """

    def __init__(self, grid, layout, dma_override=None, load_dst=None, load_pattern_override=None, load_out_shape=None):
        self.grid = grid
        self.layout = layout

        self.shape = grid.shape
        self.element_shape = grid.remaining
        self.tile_size = grid.tile_size
        self.tile_shape = grid.tile_shape if grid.is_tiled() else None
        self.block_size = grid.block_size
        self.block_shape = grid.block_shape
        self.is_tiled = grid.is_tiled()
        self.is_blocked = grid.is_blocked()
        self.ndim = grid.ndim
        # Layout owns the offset, so the partial-trailing-tile check is
        # Layout's responsibility (it ORs Grid's flag with offset+axis arithmetic).
        self.is_remainder = layout.is_remainder(grid)

        # Forwarded from Layout
        self.dtype = layout.dtype
        self.source = layout.source
        self.offset = layout.offset
        self.strides = layout.strides
        self.buffer_type = layout.buffer_type

        self.data = layout.get_data(grid)

        # DMA override for non-standard load/store paths (immutable after construction).
        # ("partition_fold", recipe) -- multi-DMA for P-dim folds
        # ("ap_override", pattern) -- custom HBM AP for free-dim folds
        self._dma_override = dma_override

        # Stream defaults -- populated by BlockStream.tolist() to route DMAs
        # into rotating buffer slots without the caller having to thread
        # dst=/pattern_override=/out_shape= through every load() call. Each
        # is used only when the corresponding kwarg on load() is None.
        self._load_dst = load_dst
        self._load_pattern_override = load_pattern_override
        self._load_out_shape = load_out_shape

    # ================================================================
    # Indexing
    # ================================================================

    def _is_sbuf_element_level(self):
        """True if SBUF view at element level -- enables sub_index path.

        Checks: Layout is SBUFLayout AND all dims at element level (step==1)
        AND remaining fits in a single tile (no tile navigation needed).
        """
        if isinstance(self.layout, SBUFLayout):
            grid = self.grid
            for d in range(grid.ndim):
                if not grid.is_element(d):
                    return False
            if grid.tile_size is not None:
                for d in range(grid.ndim):
                    if grid.remaining[d] > grid.tile_size[d]:
                        return False
            return True
        return False

    def __getitem__(self, key):
        """Index the view -- dispatches based on view type and level.

        SBUF at element level: sub_index path (direct array slicing).
        Otherwise: Grid + Layout coordination (descend/narrow + advance).
        """
        if self._is_sbuf_element_level():
            return self._sub_index(key)

        if isinstance(key, tuple):
            return self._index_multi(key)
        return self._index_multi((key,))

    def _sub_index(self, key):
        """SBUF sub-tile indexing -- bypasses Grid entirely."""
        new_layout, result_shape = self.layout.sub_index(key, self.element_shape)

        result_shape = tuple(result_shape)
        # Element-level grid: one elem axis per dim, count = result_shape[d].
        minimal_grid = Grid.from_shape(
            element_shape=result_shape,
            tile_size=result_shape,
        )

        return NDSlice(minimal_grid, new_layout)

    def _index_multi(self, keys):
        """Process multi-dimensional index keys left-to-right.

        Keys map to source dims 0..len(keys)-1 (numpy convention).
        Partial indexing (``len(keys) < ndim``) leaves trailing dims at
        their current level.

        Per-key dispatch:
          - int            : grid.consume + layout.advance(stride, k)
          - slice [a:b]    : grid.narrow + layout.advance(stride, a)
          - slice [a:b:s]  : grid.split (peer-walk) + grid.consume +
                             grid.narrow + layout.advance(stride, a)
          - SBUF vector    : grid.consume + layout.set_indirect(VECTOR, ...)
          - runtime scalar : grid.consume + layout.set_indirect(SCALAR, ...)

        Sub-tile guard: when an int key lands on a dim whose only
        remaining axis is a sub-tile leaf (PART / consumed ELEM) AND
        the cursor has already advanced past this dim, the key is
        deflected to the cursor's dim. This handles ``row[j]`` after
        ``row = v[i, :].load()`` -- the user's single key targets the
        surviving iteration dim, not the consumed PARTITION leaf.

        After all keys, the cursor advances past every consumed dim and
        any dim whose surviving outer is no longer iteration-level (e.g.
        a slice that landed exactly on a tile-leaf). If the cursor would
        run off the end and any other dim still has an iteration-level
        outer, it wraps to expose the next-level grid (e.g. a consumed
        block reveals its interior tile grid).

        cleanup() drops fully-consumed dims and auto-pops trivial tile
        levels after batch collapse; ``drop_dim`` adjusts the cursor for
        dropped dims so no remapping is needed here.
        """
        grid = self.grid
        layout = self.layout

        assert len(keys) <= grid.ndim, (
            f"view[...]: too many keys ({len(keys)}) for {grid.ndim}-D "
            "view. Drop trailing keys or index through a higher-rank view."
        )

        # The new cursor must land past every consumed key (consume
        # removes the iter-level axis on its dim, so the dim is no
        # longer at the same iteration level).
        consumed_dims = []

        for pos in range(len(keys)):
            dim = pos
            k = keys[pos]

            # Partial-key deflect: when a single int key lands on a
            # non-batch dim the cursor has already advanced past,
            # redirect to the cursor's dim. The cursor identifies the
            # next iteration level the user wants to step through; dims
            # before it (and past the batch zone) are "inside" a
            # parent's iteration step (sub-tile leaves or per-block
            # tile grids inside a consumed block) and should not be
            # re-indexed implicitly. Batch dims (``dim < n_batch_dims``)
            # are always user-addressable -- ``v[batch_idx]`` on a
            # batched view consumes the slab regardless of cursor.
            if (
                len(keys) == 1
                and isinstance(k, int)
                and grid.n_batch_dims <= dim < grid.cursor
                and grid.cursor < grid.ndim
            ):
                dim = grid.cursor

            validate_index_key(
                k,
                dim,
                grid.current_count(dim),
                grid=grid,
                context="view[...]",
            )

            advance_step = layout.advance_step(grid, dim)

            if isinstance(k, int):
                normalized_k = k if k >= 0 else grid.current_count(dim) + k
                # Sub-tile P-narrow: an int on a dim whose only remaining
                # axis is the partition leaf shrinks PART to count=1
                # rather than dropping it. SBUF allocation must stay 2-D
                # (partition row axis required), so we keep the axis with
                # count=1 instead of consuming it. The layout still
                # advances by k partition-rows.
                if _is_partition_leaf_only(grid, dim):
                    grid = grid.narrow(dim, 1)
                    layout = layout.advance(dim, normalized_k, advance_step)
                else:
                    grid = grid.consume(dim)
                    layout = layout.advance(dim, normalized_k, advance_step)
                    grid = grid.truncate_to_source(dim, layout.dim_offset_elements(dim, grid.element_shape))
                consumed_dims.append(dim)

            elif isinstance(k, slice):
                start = k.start if k.start is not None else 0
                stop = k.stop if k.stop is not None else grid.current_count(dim)
                slice_step = k.step if k.step is not None else 1

                if slice_step != 1:
                    # Stepped slice: factor a peer-walk axis (split + pop peer).
                    # peer is outer with step = self.step (one item per peer slot);
                    # owned is inner with count = self.count // step,
                    # step = self.step * step (skips past peers).
                    grid = grid.split_peers(
                        dim,
                        num_peers=slice_step,
                        peer_label=AxisLabel.SHARD,
                        owned_label=AxisLabel.TILE,
                    )
                    grid = grid.consume(dim)  # drop peer axis -- this rank's slot
                    # ceil((stop - start) / step) -- the count of owned items.
                    count = (stop - start + slice_step - 1) // slice_step
                else:
                    count = stop - start

                grid = grid.narrow(dim, count)
                layout = layout.advance(dim, start, advance_step)
                grid = grid.truncate_to_source(dim, layout.dim_offset_elements(dim, grid.element_shape))

            else:
                # Runtime scalar OR vector tensor. Vector inputs may arrive
                # as a loaded NDSlice (over SBUF) OR a raw nl.ndarray with
                # shape[0] > 1 -- both route to vector_offset.
                if isinstance(k, NDSlice) and isinstance(k.layout, SBUFLayout):
                    vector_count = k.element_shape[0]
                    is_vector = vector_count > 1
                    index_data = k.layout.source
                elif hasattr(k, "shape") and len(k.shape) >= 1:
                    vector_count = k.shape[0]
                    is_vector = vector_count > 1
                    index_data = k
                else:
                    vector_count = 1
                    is_vector = False
                    index_data = k

                if is_vector:
                    # Vector gather: pop the outer axis on dim, then narrow
                    # the next axis (the partition/leaf walk) to the vector
                    # count -- this rank visits exactly ``vector_count``
                    # gathered positions on this dim.
                    grid = grid.consume(dim)
                    if grid.outer_axis(dim) is not None:
                        grid = grid.narrow(dim, vector_count)
                    layout = layout.set_indirect(IndirectKind.VECTOR, index_data, dim)
                else:
                    grid = grid.consume(dim)
                    layout = layout.advance(dim, index_data, advance_step)
                consumed_dims.append(dim)

        grid, dropped_dims = grid.cleanup()
        if hasattr(layout, "drop_dims"):
            layout = layout.drop_dims(dropped_dims)
        consumed_dims = _shift_dims_after_drop(consumed_dims, dropped_dims)

        grid = grid.with_cursor_past_consumed(consumed_dims)
        return NDSlice(grid, layout)

    # ================================================================
    # Iteration
    # ================================================================

    def tolist(self, dim: Optional[int] = None) -> list:
        """
        Materialize sub-views as a Python list along ``dim``.

        Args:
            dim: None (default) follows the grid's current cursor
                (outermost active iteration axis); batch dims are
                auto-dropped on consumption. An int iterates along that
                dim explicitly, cursor unchanged.

        Returns:
            list[NDSlice] of one sub-view per tile step.

        Raises:
            AssertionError when ``dim`` is not None / int, or out of range.

        Example:
            for tile in tiles.tolist():               # cursor-driven
                ...
            for col in tiles.tolist(dim=1):           # column-first
                for tile in col.tolist():
                    ...
        """
        if dim is None:
            return self._tolist_cursor()
        assert isinstance(dim, int) and not isinstance(dim, bool), (
            "NDSlice.tolist: dim= must be int or None, got " + str(dim)
        )
        assert 0 <= dim < self.ndim, (
            "NDSlice.tolist: dim=" + str(dim) + " is out of range for ndim=" + str(self.ndim) + "."
        )
        return self._tolist_along_dim(dim)

    def _tolist_cursor(self):
        """Iterate the next active dim -- batch dims first, then cursor.

        Unconsumed batch dims always come first (each iteration drops the
        dim entirely so a child sees one less batch dim). Once batch dims
        are exhausted, the iteration follows the grid cursor through the
        tile dims, advancing the cursor on each child so the next
        ``tolist()`` walks the next tile dim.
        """
        if self.ndim == 0:
            return []

        # Walk the leftmost still-present batch dim first.
        if self.grid.n_batch_dims > 0:
            return self._iterate_dim(dim=0, drop=True)

        cursor_dim = self.grid.cursor
        if cursor_dim >= self.grid.ndim:
            return [self]
        return self._iterate_dim(dim=cursor_dim, drop=False, advance_cursor=True)

    def _iterate_dim(self, dim, drop=False, advance_cursor=False):
        """Materialize one child per outer-axis step on ``dim``.

        Args:
            dim: dim to iterate.
            drop: drop ``dim`` from the child Grid + Layout (batch consumption).
            advance_cursor: bump child cursor to ``dim + 1`` (tile dim walk).
        """
        step = self.layout.advance_step(self.grid, dim)
        count = self.grid.current_count(dim)
        result = []
        for i in range(count):
            child_grid = self.grid.narrow(dim, 1)
            child_layout = self.layout.advance(dim, i, step)
            if drop:
                child_grid = child_grid.drop_dim(dim)
                child_layout = child_layout.drop_dim(dim)
            elif advance_cursor:
                child_grid = child_grid.with_cursor(dim + 1)
            result.append(NDSlice(child_grid, child_layout))
        return result

    def _tolist_along_dim(self, dim):
        """Iterate along an explicit dim -- cursor untouched, no batch-drop."""
        return self._iterate_dim(dim)

    def _enumerate(self, start=0, mode=None):
        """Return list of (index, sub_view) pairs -- always local indices."""
        items = self.tolist()
        result = []
        for i in range(len(items)):
            result.append((start + i, items[i]))
        return result

    def __iter__(self):
        return iter(self.tolist())

    def __len__(self):
        total = 1
        for s in self.shape:
            total = total * s
        return total

    # ================================================================
    # Dimensional iteration
    # ================================================================

    def stream(
        self,
        dim: Optional[int] = None,
        buffer_count: int = 2,
        dtype=None,
        pattern_override: Optional[list] = None,
        out_shape: Optional[tuple[int, ...]] = None,
    ):
        """
        Create a rotating-buffer stream along ``dim``.

        Pre-allocates ``buffer_count`` SBUF buffers sized for one step along
        ``dim``. Each ``stream.load(k)`` DMAs into slot ``k % buffer_count``,
        overlapping the next DMA with the current compute. Use for flowing
        operands (RHS in matmul); for stationary operands hoist via
        ``view.load()`` once.

        Args:
            dim: Iteration axis. Defaults to the view's cursor -- the
                first dim that still has an iteration-level outer axis.
                On a fresh view that's dim 0; on a child of
                ``parent[i, :]`` it's dim 1. Pass an explicit int to
                stream a non-default axis.
            buffer_count: Rotating-buffer count; 2 for double-buffer,
                3 for triple-buffer pipelines. Default 2.
            dtype: Override dtype for the rotating buffers. Defaults to
                source dtype.
            pattern_override: Custom HBM AP per step (advanced; replaces
                the auto-generated pattern). Requires ``out_shape``.
            out_shape: SBUF allocation shape per step. Required when
                ``pattern_override`` is set.

        Returns:
            BlockStream wrapping the rotating SBUF pool.

        Raises:
            AssertionError: View is SBUF-backed (stream is HBM-only --
                rotating buffers are SBUF, not the source); ``dim`` is
                not an int or out of range; ``buffer_count`` is not a
                positive int; ``pattern_override`` is set without
                ``out_shape``; the view still has unconsumed batch
                dims (raised by BlockStream.__init__).

        Example:
            stream = tiles[:, 0].stream(buffer_count=2)
            for k in nl.affine_range(K):
                tile = stream.load(k)
                # compute with tile.data overlaps next DMA
        """
        assert isinstance(self.layout, HBMLayout), (
            "NDSlice.stream: only valid on HBM views (the rotating "
            "buffers live in SBUF; the view is the HBM source). For an "
            "already-loaded SBUF view, iterate it with .tolist() or "
            "indexing instead."
        )
        assert self.grid.n_batch_dims == 0, (
            "NDSlice.stream: requires all batch dims to be consumed "
            "first (got " + str(self.grid.n_batch_dims) + " unconsumed). "
            "Index or iterate the batch dims before calling .stream()."
        )
        if dim is None:
            dim = self.grid.cursor
            assert dim < self.ndim, (
                "NDSlice.stream: cursor (=" + str(dim) + ") is past the "
                "view's ndim=" + str(self.ndim) + ". The view has no "
                "iteration dim left; either pass dim= explicitly or "
                "stream a parent view that still has iteration axes."
            )
        else:
            assert isinstance(dim, int) and not isinstance(dim, bool), "NDSlice.stream: dim= must be int, got " + str(
                dim
            )
            assert 0 <= dim < self.ndim, (
                "NDSlice.stream: dim=" + str(dim) + " is out of range for ndim=" + str(self.ndim) + "."
            )
        assert isinstance(buffer_count, int) and not isinstance(buffer_count, bool), (
            "NDSlice.stream: buffer_count= must be int, got " + str(buffer_count)
        )
        assert buffer_count >= 1, "NDSlice.stream: buffer_count= must be >= 1, got " + str(buffer_count)
        if pattern_override is not None:
            assert out_shape is not None, (
                "NDSlice.stream: pattern_override= requires out_shape="
                "(<P>, <F>) so the SBUF rotating buffers can be allocated."
            )
        return BlockStream(
            view=self,
            dim=dim,
            buffer_count=buffer_count,
            dtype=dtype,
            pattern_override=pattern_override,
            out_shape=out_shape,
        )

    # ================================================================
    # DMA
    # ================================================================

    def load(
        self,
        dtype=None,
        transpose: bool = False,
        dge_mode=None,
        oob_mode=None,
        oob_value: Optional[float] = None,
        dst: Optional[nl.ndarray] = None,
        pattern_override: Optional[list] = None,
        out_shape: Optional[tuple[int, ...]] = None,
    ) -> "NDSlice":
        """
        DMA the view's region from HBM to SBUF and return the SBUF NDSlice.

        Single ``nisa.dma_copy`` per call (the single-DMA contract); the only
        exception is partition-dim ``.fold()``, which emits K coalesced DMAs.

        Args:
            dtype: Cast on DMA. Defaults to source dtype.
            transpose: DMA-transpose path. F<=128 uses a direct transpose;
                F%128==0 uses the tiled path; other shapes are not supported
                (no fallback).
            dge_mode: ``nisa.dge_mode.*`` -- DMA generation mode hint.
            oob_mode: ``nisa.oob_mode.skip`` to suppress out-of-bounds DMA
                faults at boundaries; pair with ``oob_value`` to pre-fill.
            oob_value: Pre-fill SBUF with this value before DMA. Requires
                ``oob_mode=nisa.oob_mode.skip``.
            dst: Pre-allocated SBUF ndarray to load into. The library
                allocates a fresh buffer when omitted.
            pattern_override: Custom HBM access pattern (advanced; replaces
                the auto-generated AP). Requires ``out_shape``.
            out_shape: SBUF allocation shape. Required when
                ``pattern_override`` is set.

        Returns:
            NDSlice over the SBUF buffer.

        Raises:
            AssertionError when called on an SBUF view, when batch dims are
            unconsumed, when ``pattern_override`` is given without
            ``out_shape``, or when ``oob_value`` is given without
            ``oob_mode``.

        Example:
            tile = tiles[i, j].load()                       # auto-allocate
            tile = tiles[i, j].load(dst=preallocated_sbuf)  # caller-provided
            col  = tiles[:, m].load()                       # coalesced col load

        Notes:
            ``.fold()`` recipes attached upstream are routed automatically
            here -- partition fold issues K coalesced DMAs, free-dim fold
            applies the recorded AP override.
        """
        assert self.grid.tile_size is not None
        assert isinstance(self.layout, HBMLayout), "load() is only valid on HBM views"
        assert self.grid.n_batch_dims == 0, (
            "load() requires all batch dims to be consumed first (got "
            + str(self.grid.n_batch_dims)
            + " unconsumed batch dim(s), element_shape="
            + str(self.grid.remaining)
            + "). Index or iterate the batch dims before calling .load()."
        )
        if oob_value is not None:
            assert oob_mode is not None, (
                "NDSlice.load: oob_value= requires oob_mode= "
                "(typically nisa.oob_mode.skip) -- without an oob_mode, the "
                "value is never written."
            )

        # Partition fold: multi-DMA path
        if self._dma_override is not None and self._dma_override[0] == "partition_fold":
            load_dtype = dtype if dtype is not None else self.dtype
            sbuf_grid, sbuf_layout = HBMLayout.load_partition_fold(
                self.layout,
                self.grid,
                self._dma_override[1],
                load_dtype,
                dge_mode,
            )
            return NDSlice(sbuf_grid, sbuf_layout)

        # AP override (free-dim fold): use stored pattern for HBM side
        if self._dma_override is not None and self._dma_override[0] == "ap_override":
            if pattern_override is None:
                pattern_override = self._dma_override[1]

        # Stream defaults: fall through to BlockStream-provided rotating buffer,
        # pattern, and out_shape when the caller didn't pass them. Preserves
        # the load semantics BlockStream iteration relies on.
        if dst is None and self._load_dst is not None:
            effective_dtype = dtype if dtype is not None else self.dtype
            if not self.is_remainder and effective_dtype == self._load_dst.dtype:
                dst = self._load_dst
        if pattern_override is None and self._load_pattern_override is not None:
            pattern_override = self._load_pattern_override
        if out_shape is None and self._load_out_shape is not None:
            out_shape = self._load_out_shape

        # After internal defaults resolve, pattern_override needs out_shape (or dst)
        # so the SBUF destination shape is known.
        if pattern_override is not None and dst is None:
            assert out_shape is not None, (
                "NDSlice.load: pattern_override= requires out_shape= or dst= so the SBUF destination shape is known."
            )

        # Standard path: layout handles DMA + SBUF wrapping
        sbuf_grid, sbuf_layout = self.layout.load(
            self.grid,
            dtype=dtype,
            dst=dst,
            oob_mode=oob_mode,
            oob_value=oob_value,
            out_shape=out_shape,
            transpose=transpose,
            dge_mode=dge_mode,
            pattern_override=pattern_override,
        )
        return NDSlice(sbuf_grid, sbuf_layout)

    def store(
        self,
        data,
        dtype=None,
        dge_mode=None,
        oob_mode=None,
        pattern_override: Optional[list] = None,
    ) -> None:
        """
        DMA SBUF data into the view's HBM region.

        Single ``nisa.dma_copy`` per call (the single-DMA contract); the only
        exception is partition-dim ``.fold()``, which emits K coalesced DMAs.

        Args:
            data: SBUF source -- either a raw ``nl.ndarray`` or an access-
                pattern view from ``.ap()`` on a loaded NDSlice. Pass an
                NDSlice's ``.ap()`` view, not the NDSlice itself.
            dtype: Cast on DMA. Defaults to ``data``'s dtype.
            dge_mode: ``nisa.dge_mode.*`` -- DMA generation mode hint.
            oob_mode: ``nisa.oob_mode.skip`` to suppress out-of-bounds DMA
                faults at boundaries.
            pattern_override: Custom HBM access pattern (advanced).

        Raises:
            AssertionError when called on an SBUF view, when ``data`` is an
            NDSlice (must call ``.ap()`` first), or when batch dims are
            unconsumed.

        Example:
            dst_tiles[i, j].store(tile.ap())          # store loaded tile
            dst_tiles[i, j].store(sbuf_result)        # store raw nl.ndarray

        Notes:
            ``.fold()`` recipes attached upstream are routed automatically.
        """
        assert self.grid.tile_size is not None
        assert isinstance(self.layout, HBMLayout), "store() is only valid on HBM views"
        assert not isinstance(data, NDSlice), "store() expects ndarray or .ap() view, not NDSlice. Use data.ap()"
        assert self.grid.n_batch_dims == 0, (
            "store() requires all batch dims to be consumed first (got "
            + str(self.grid.n_batch_dims)
            + " unconsumed batch dim(s), element_shape="
            + str(self.grid.remaining)
            + "). Index or iterate the batch dims before calling .store()."
        )
        if pattern_override is not None:
            assert isinstance(pattern_override, (list, tuple)), (
                "store() pattern_override= must be a list/tuple of [stride, count] pairs."
            )
            for level in pattern_override:
                assert isinstance(level, (list, tuple)) and len(level) == 2, (
                    "store() pattern_override= levels must each be [stride, count] pairs."
                )

        # Partition fold store: K separate DMAs (reverse of load)
        if self._dma_override is not None and self._dma_override[0] == "partition_fold":
            HBMLayout.store_partition_fold(
                self.source,
                self.offset,
                self._dma_override[1],
                self.element_shape,
                data,
                dge_mode,
            )
            return

        # AP override (free-dim fold): use stored pattern for HBM side
        if self._dma_override is not None and self._dma_override[0] == "ap_override":
            if pattern_override is None:
                pattern_override = self._dma_override[1]

        self.layout.store(data, self.grid, oob_mode=oob_mode, dge_mode=dge_mode, pattern_override=pattern_override)

    def ap(self):
        """
        Build the NKI access-pattern view describing this NDSlice's region.

        Use anywhere an NKI op accepts an access-pattern view -- ``nisa.dma_copy``
        for manual DMA, or any ``nisa.*`` compute / activation op for SBUF
        views when finer control is needed than ``.data`` provides. For HBM
        views the AP reflects source + offset + strides + indirect parameters;
        for SBUF views it reflects the flat F-offset layout.

        Takes no arguments -- the region is fully determined by the NDSlice.
        For a custom AP pattern over the same region, pass
        ``pattern_override=`` to ``.load()`` / ``.store()`` instead.

        Returns:
            An NKI access-pattern view (the same object kind that
            ``.load()`` / ``.store()`` build internally).

        Example:
            # Manual HBM -> SBUF DMA
            nisa.dma_copy(dst=sbuf, src=hbm_tiles[i, j].ap())

            # Manual SBUF -> HBM DMA
            nisa.dma_copy(dst=dst_tiles[i, j].ap(), src=sbuf_buf.ap())

            # SBUF view fed directly to a compute op
            nisa.tensor_tensor(out, gamma_x, inv_rms_bc.ap(), nl.multiply)
        """
        return self.layout.ap(self.grid)

    # ================================================================
    # Transforms -- delegate to standalone functions + Layout
    # ================================================================

    def _transform_strides(self):
        """Strides for transform computation."""
        return self.layout.transform_strides(self.element_shape)

    def _apply_transform(self, new_element_shape, new_strides):
        """Transformed view: single tile covering the new shape.

        Builds a new Grid with tile_size = new_element_shape (single tile,
        directly loadable). Delegates layout construction to
        Layout.apply_transform: HBM creates new layout with updated strides,
        SBUF returns self (pure reinterpretation -- sub_index uses
        Grid.remaining for F-offset computation).

        SBUF: element-level Grid for immediate sub_index access.
        HBM: tile-level Grid for standard load path.
        """
        tile_size = tuple(new_element_shape)
        new_layout = self.layout.apply_transform(new_strides)

        # SBUF: element-level grid with one ELEM axis per dim sized to the new
        # element_shape -- NDSlice.element_shape reads grid.remaining (count*step
        # of outer axis), so axes must carry the actual counts.
        if isinstance(self.layout, SBUFLayout):
            _elem_axes = []
            for _d in range(len(new_element_shape)):
                _elem_axes.append(Axis(count=new_element_shape[_d], step=1, dim=_d, label=AxisLabel.ELEM))
            new_grid = Grid.from_shape(
                element_shape=new_element_shape,
                tile_size=tile_size,
                axes=tuple(_elem_axes),
            )
        else:
            new_grid = Grid.from_shape(element_shape=new_element_shape, tile_size=tile_size)

        return NDSlice(new_grid, new_layout)

    def _validate_dim(self, dim, label="dim"):
        """Shared dim validator for transform methods."""
        assert isinstance(dim, int) and not isinstance(dim, bool), "NDSlice." + label + " must be int, got " + str(dim)
        assert 0 <= dim < self.ndim, (
            "NDSlice." + label + "=" + str(dim) + " is out of range for ndim=" + str(self.ndim) + "."
        )

    def reshape_dim(self, dim: int, shape: tuple) -> "NDSlice":
        """
        Split one dimension into sub-dimensions (metadata only; no DMA).

        Args:
            dim: Dim to split (in ``[0, ndim)``).
            shape: Sub-dim sizes; product must equal ``element_shape[dim]``.

        Returns:
            New NDSlice with the dim replaced by len(shape) new dims.

        Example:
            view = nt.tensor_view(src).reshape_dim(1, (8, 128))   # split F
        """
        self._validate_dim(dim, "reshape_dim: dim")
        result = compute_reshape_dim(self.element_shape, self._transform_strides(), dim, shape)
        return self._apply_transform(result[0], result[1])

    def reshape(self, new_shape: tuple) -> "NDSlice":
        """
        Full reshape to ``new_shape`` (metadata only; no DMA).

        Requires contiguous layout (or a layout where the new shape factors
        through the existing strides). Use ``reshape_dim`` / ``flatten_dims``
        for surgical reshapes that don't touch every axis.

        Args:
            new_shape: New element shape; must have same total element count.

        Returns:
            New NDSlice over the same memory with the new shape.

        Example:
            view = nt.tensor_view(src).reshape((B * S, H))
        """
        result = compute_reshape(self.element_shape, self._transform_strides(), new_shape)
        return self._apply_transform(result[0], result[1])

    def permute(self, dims: tuple) -> "NDSlice":
        """
        Reorder dimensions (metadata only; no DMA).

        Args:
            dims: Permutation of ``range(ndim)``.

        Returns:
            New NDSlice with dims reordered.

        Raises:
            AssertionError on SBUF views attempting to move the partition
            dim (dim 0); P is hardware-fixed and cannot be permuted.

        Example:
            view = nt.tensor_view(w).reshape((H1, P)).permute((1, 0))
        """
        if isinstance(self.layout, SBUFLayout):
            assert dims[0] == 0, (
                "SBUF views cannot permute the partition dimension (dim 0). "
                "Got permute(" + str(tuple(dims)) + ") on SBUF NDSlice."
            )
        result = compute_permute(self.element_shape, self._transform_strides(), dims)
        return self._apply_transform(result[0], result[1])

    def flatten_dims(self, start_dim: int, end_dim: int) -> "NDSlice":
        """
        Merge dimensions ``[start_dim..end_dim]`` into one (metadata only).

        Args:
            start_dim: First dim to merge (``0 <= start_dim < ndim``).
            end_dim: Last dim to merge (``start_dim <= end_dim < ndim``).

        Returns:
            New NDSlice with the merged dim's size = product of merged sizes.

        Example:
            flat = nt.tensor_view(x).flatten_dims(0, 1)   # (B, S, H) -> (B*S, H)
        """
        self._validate_dim(start_dim, "flatten_dims: start_dim")
        self._validate_dim(end_dim, "flatten_dims: end_dim")
        assert start_dim <= end_dim, (
            "NDSlice.flatten_dims: start_dim=" + str(start_dim) + " must be <= end_dim=" + str(end_dim) + "."
        )
        result = compute_flatten_dims(self.element_shape, self._transform_strides(), start_dim, end_dim)
        return self._apply_transform(result[0], result[1])

    def squeeze_dim(self, dim: int) -> "NDSlice":
        """
        Remove a size-1 dimension (metadata only; no DMA).

        Args:
            dim: Dim to remove; must have ``element_shape[dim] == 1``.

        Returns:
            New NDSlice with the size-1 dim removed.

        Raises:
            AssertionError when the named dim is not size 1.
        """
        self._validate_dim(dim, "squeeze_dim: dim")
        result = compute_squeeze_dim(self.element_shape, self._transform_strides(), dim)
        return self._apply_transform(result[0], result[1])

    def expand_dim(self, dim: int) -> "NDSlice":
        """
        Insert a size-1 dimension at position ``dim`` (metadata only).

        Stride 0 -- no memory cost; the new dim is repeated logically.
        Pair with ``broadcast(dim, size)`` to expose the dim with a real
        size for downstream consumers.

        Args:
            dim: Insert position in ``[0, ndim]`` (note: end position
                allowed for trailing insertion).

        Returns:
            New NDSlice with one extra size-1 dim.

        Example:
            bc = inv_rms.expand_dim(2).broadcast(2, S)
        """
        result = compute_expand_dim(self.element_shape, self._transform_strides(), dim)
        return self._apply_transform(result[0], result[1])

    def broadcast(self, dim: int, size: int) -> "NDSlice":
        """
        Broadcast a size-1 dim to ``size`` (stride-0; no memory cost).

        Args:
            dim: Dim with current size 1.
            size: Target broadcast size (positive int).

        Returns:
            New NDSlice with the dim's size expanded; stride remains 0.

        Raises:
            AssertionError when the dim's current size is not 1.

        Example:
            gamma_bc = gamma.expand_dim(1).broadcast(1, BxS)
        """
        self._validate_dim(dim, "broadcast: dim")
        assert isinstance(size, int) and not isinstance(size, bool) and size > 0, (
            "NDSlice.broadcast: size= must be a positive int, got " + str(size)
        )
        result = compute_broadcast(self.element_shape, self._transform_strides(), dim, size)
        return self._apply_transform(result[0], result[1])

    def slice(self, dim: int, start: int, end: int) -> "NDSlice":
        """
        Narrow dim to elements ``[start, end)`` -- element-level slicing.

        Unlike ``__getitem__`` which operates at the current Grid level
        (tile/block), ``slice()`` always operates at element level. Works
        on both HBM and SBUF.

        Args:
            dim: Dim to narrow (``0 <= dim < ndim``).
            start: Start element index (inclusive).
            end: End element index (exclusive). Must satisfy
                ``start < end <= element_shape[dim]``.

        Returns:
            New NDSlice over the narrowed region.
        """
        self._validate_dim(dim, "slice: dim")
        assert isinstance(start, int) and not isinstance(start, bool), "NDSlice.slice: start must be int, got " + str(
            start
        )
        assert isinstance(end, int) and not isinstance(end, bool), "NDSlice.slice: end must be int, got " + str(end)
        assert 0 <= start < end <= self.element_shape[dim], (
            "NDSlice.slice: invalid range ["
            + str(start)
            + ", "
            + str(end)
            + ") for dim="
            + str(dim)
            + " (element_shape[dim]="
            + str(self.element_shape[dim])
            + ")."
        )
        count = end - start

        # SBUF: delegate to sub_index for all dims (P and F)
        if isinstance(self.layout, SBUFLayout):
            key = []
            for d in range(self.ndim):
                if d == dim:
                    key.append(slice(start, end))
                else:
                    key.append(slice(0, self.element_shape[d]))
            return self._sub_index(tuple(key))

        # HBM: advance offset + narrow shape
        new_element_shape = replace_at(self.element_shape, dim, count)
        new_tile_size = replace_at(self.element_shape, dim, count)

        new_grid = Grid.from_shape(new_element_shape, tile_size=new_tile_size)

        # Advance layout offset by start elements along dim (step=1 for element level)
        new_layout = self.layout.advance(dim, start, 1)

        return NDSlice(new_grid, new_layout)

    def fold(self, src_dim: int, into_dim: int, position: str = "outer") -> "NDSlice":
        """
        Fold ``src_dim`` into ``into_dim`` (metadata + DMA recipe).

        Combines two dims into one for compact iteration. Two flavors:

          - **Free-dim fold** (``src_dim > 0`` and ``into_dim > 0``):
            stores a pre-fold AP pattern on the returned NDSlice; the
            DMA on the next ``.load()`` / ``.store()`` issues a single
            coalesced transfer using the recorded pattern.
          - **Partition fold** (``src_dim == 0`` or ``into_dim == 0``):
            attaches a multi-DMA recipe -- one DMA per slice along the
            partition dim, coalesced over the free axis.

        Chained folds propagate the AP base through the chain.

        Args:
            src_dim: Dim being folded away.
            into_dim: Dim that absorbs ``src_dim``.
            position: ``"outer"`` (default) places ``src_dim`` outside
                ``into_dim`` in the merged stride; ``"inner"`` places it
                inside.

        Returns:
            New NDSlice with ndim - 1 and a fold recipe attached.
        """
        self._validate_dim(src_dim, "fold: src_dim")
        self._validate_dim(into_dim, "fold: into_dim")
        assert src_dim != into_dim, "NDSlice.fold: src_dim and into_dim must differ, got " + str(src_dim)
        assert position in ("outer", "inner"), "NDSlice.fold: position= must be 'outer' or 'inner', got " + str(
            position
        )
        strides = self._transform_strides()
        result = compute_fold(self.element_shape, strides, src_dim, into_dim, position)
        view = self._apply_transform(result[0], result[1])

        # Determine AP base: use existing override if present (chained fold),
        # otherwise build from current strides + element_shape.
        if self._dma_override is not None and self._dma_override[0] == "ap_override":
            ap_base = self._dma_override[1]
        else:
            ap_base = []
            for d in range(len(self.element_shape)):
                ap_base.append([strides[d], self.element_shape[d]])

        is_partition_fold = src_dim == 0 or into_dim == 0

        if is_partition_fold:
            K = self.element_shape[src_dim]
            P_per_slice = self.element_shape[into_dim]
            fold_stride = strides[src_dim]
            base_pattern = []
            for i in range(len(ap_base)):
                if i != src_dim:
                    base_pattern.append(ap_base[i])
            override = ("partition_fold", (K, P_per_slice, fold_stride, base_pattern))
            # Partition fold's multi-DMA path resolves shape internally;
            # no out_shape needed.
            return NDSlice(view.grid, view.layout, dma_override=override)

        # Free-dim fold: pattern_override on .load() needs the post-fold
        # element shape so the SBUF destination can be sized.
        override = ("ap_override", ap_base)
        return NDSlice(
            view.grid,
            view.layout,
            dma_override=override,
            load_out_shape=tuple(view.element_shape),
        )

    def split(self, dim: int, n: int) -> "NDSlice":
        """
        Split a dim into ``n`` equal chunks (shorthand for ``reshape_dim``).

        Args:
            dim: Dim to split.
            n: Chunk count; must evenly divide ``element_shape[dim]``.

        Returns:
            New NDSlice with the dim replaced by two dims:
            ``(n, element_shape[dim] // n)``.

        Raises:
            AssertionError when ``n`` does not divide ``element_shape[dim]``.

        Example:
            view.split(1, 4)   # (B, S) -> (B, 4, S/4)
        """
        self._validate_dim(dim, "split: dim")
        assert isinstance(n, int) and not isinstance(n, bool) and n > 0, (
            "NDSlice.split: n must be a positive int, got " + str(n)
        )
        assert self.element_shape[dim] % n == 0, (
            "NDSlice.split: element_shape["
            + str(dim)
            + "]="
            + str(self.element_shape[dim])
            + " is not divisible by n="
            + str(n)
            + "."
        )
        chunk_size = self.element_shape[dim] // n
        return self.reshape_dim(dim, (n, chunk_size))

    def _drop_unit_batch_dims(self):
        """Drop batch dims narrowed to a single tile (e.g., rank dim after shard).

        Any batch dim whose remaining fits in a single tile is consumed and
        dropped -- matches the source-tensor convention that rank dims
        collapse once the rank shard is applied.
        """
        grid = self.grid
        layout = self.layout
        drop_dims = []
        for d in range(grid.n_batch_dims):
            tile_step = grid.tile_size[d] if grid.tile_size is not None else 1
            if grid.remaining[d] <= tile_step:
                drop_dims.append(d)

        for d_idx in range(len(drop_dims) - 1, -1, -1):
            d = drop_dims[d_idx]
            # drop_dim removes all axes on `d` and shrinks element_shape;
            # an explicit consume() before it would be redundant.
            grid = grid.drop_dim(d)
            layout = layout.drop_dim(d)

        return NDSlice(grid, layout)

    # ================================================================
    # Remainder
    # ================================================================

    def whole_tiles(self) -> list:
        """
        Return the non-remainder sub-views from ``tolist()``.

        Splits a remainder-bearing view into its clean and boundary halves;
        callers process each with the right policy (e.g. plain ``.load()``
        for whole tiles, ``.load(oob_mode=skip)`` for remainder tiles).

        Returns:
            list[NDSlice] -- only sub-views with ``.is_remainder == False``.

        Example:
            for tile in tiles.whole_tiles():
                tile.load()              # full-extent DMA
            for rem in tiles.remainder_tiles():
                rem.load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
        """
        items = self.tolist()
        result = []
        for item in items:
            if not item.is_remainder:
                result.append(item)
        return result

    def remainder_tiles(self) -> list:
        """
        Return the remainder sub-views from ``tolist()``.

        Pair with ``whole_tiles()`` to split iteration on a view whose
        outermost extent is not divisible by the tile granularity.

        Returns:
            list[NDSlice] -- only sub-views with ``.is_remainder == True``.

        Example: see ``whole_tiles``.
        """
        items = self.tolist()
        result = []
        for item in items:
            if item.is_remainder:
                result.append(item)
        return result

    # ================================================================
    # Representation
    # ================================================================

    def __repr__(self):
        return (
            "NDSlice(shape="
            + str(self.shape)
            + ", element_shape="
            + str(self.element_shape)
            + ", offset="
            + str(self.offset)
            + ", buffer_type="
            + str(self.buffer_type)
            + ")"
        )


# ============================================================================
# BlockStream
# ============================================================================


class BlockStream(nl.NKIObject):
    """
    Rotating-buffer DMA pipeline along one dimension of an HBM view.

    Constructed via :meth:`NDSlice.stream`. Pre-allocates ``buffer_count``
    SBUF buffers sized for **one step** along the streaming dim; each
    ``stream.load(k)`` DMAs into slot ``k % buffer_count``, overlapping
    the next DMA with the current compute. Use this for flowing
    operands (e.g. the RHS of a streamed matmul); for stationary
    operands hoist with a one-shot ``view.load()`` instead.

    Two access shapes on the same rotating pool:

    - **Iteration** -- ``for sv in stream: sv.load()`` (or
      ``stream.tolist()``). tolist() returns NDSlice children with
      ``load_dst`` pre-filled to the rotating buffer at slot
      ``k % buffer_count``, so ``sv.load()`` routes into the right slot
      automatically.
    - **Random access** -- ``stream.load(k)``, ``stream[k]``,
      ``stream.store(k)`` for prolog / steady / epilog manual
      pipelining.

    Asserts at construction that the source view's batch dims are all
    consumed (``n_batch_dims == 0``); the rotating slots address one
    streaming step at a time, not a multi-batch slab.
    """

    def __init__(self, view, dim, buffer_count=2, dtype=None, pattern_override=None, out_shape=None):
        self._view = view
        self._dim = dim
        self._step = view.layout.advance_step(view.grid, dim)
        self.count = view.grid.current_count(dim)
        self._buffer_count = buffer_count
        self._dtype = dtype if dtype is not None else view.dtype
        self._pattern_override = pattern_override
        self._out_shape = out_shape
        assert view.grid.n_batch_dims == 0, (
            "stream() requires all batch dims to be consumed first (got "
            + str(view.grid.n_batch_dims)
            + " unconsumed batch dim(s), element_shape="
            + str(view.grid.remaining)
            + "). Index or iterate the batch dims before calling .stream()."
        )
        self._buffers = self._allocate_buffers()

    def _child(self, i):
        """Build the NDSlice view at stream position i (no buffer binding).

        A stream step consumes the streamed-dim's iteration level (same
        semantics as a single-int index on the dim) and places the
        cursor past the consumed dim. The loaded view then exposes the
        next-level grid (e.g. one block's interior tile grid after a
        per-block stream step).
        """
        child_grid = self._consumed_grid_for_step()
        child_layout = self._view.layout.advance(self._dim, i, self._step)
        return NDSlice(child_grid, child_layout)

    def _bound_child(self, i):
        """Build the NDSlice at stream position i with stream-default kwargs
        pre-bound so sv.load() routes into the rotating buffer."""
        child_grid = self._consumed_grid_for_step()
        child_layout = self._view.layout.advance(self._dim, i, self._step)
        return NDSlice(
            child_grid,
            child_layout,
            load_dst=self._buffers[i % self._buffer_count],
            load_pattern_override=self._pattern_override,
            load_out_shape=self._out_shape,
        )

    def _consumed_grid_for_step(self):
        """Grid representing one stream step: consume the streamed dim's
        outer axis and place the cursor at the next iterable position."""
        return self._view.grid.consume(self._dim).with_cursor_past_consumed((self._dim,))

    def _allocate_buffers(self):
        """Allocate buffer_count SBUF ndarrays sized for one stream step.

        Buffer size matches what load() would allocate for one step's
        worth of data: tile_p rows x (F-extent * P-tiles) columns.
        """
        if self._out_shape is not None:
            buffers = []
            for i in range(self._buffer_count):
                sbuf = nl.ndarray(tuple(self._out_shape), dtype=self._dtype, buffer=nl.sbuf)
                buffers.append(sbuf)
            return buffers

        # Ask HBM layout what SBUF shape it would allocate for one step.
        child = self._child(0)
        sbuf_shape = child.layout.sbuf_shape_for(child.grid)

        buffers = []
        for i in range(self._buffer_count):
            sbuf = nl.ndarray(sbuf_shape, dtype=self._dtype, buffer=nl.sbuf)
            buffers.append(sbuf)
        return buffers

    # ================================================================
    # Iteration
    # ================================================================

    def tolist(self) -> list:
        """
        Materialize stream positions as a list of NDSlice children.

        Each child has its rotating-buffer slot pre-bound: calling
        ``.load()`` on child ``k`` DMAs into slot ``k % buffer_count``.
        Use to consume the stream end-to-end without explicit index math.

        Returns:
            list[NDSlice] -- one child per stream step, with load_dst
            pre-filled to the corresponding rotating slot.

        Example:
            stream = tiles[:, 0].stream(buffer_count=2)
            for sv in stream.tolist():
                tile = sv.load()      # rotates buffers automatically
        """
        result = []
        for i in range(self.count):
            result.append(self._bound_child(i))
        return result

    def _enumerate(self, start=0, mode=None):
        """Return (index, NDSlice) pairs."""
        items = self.tolist()
        result = []
        for i in range(len(items)):
            result.append((start + i, items[i]))
        return result

    def __iter__(self):
        return iter(self.tolist())

    def __len__(self):
        return self.count

    # ================================================================
    # Direct access by index -- random stream position (prolog/epilog)
    # ================================================================

    def load(
        self,
        index: int,
        dtype=None,
        dge_mode=None,
        oob_mode=None,
        oob_value: Optional[float] = None,
    ) -> "NDSlice":
        """
        DMA stream position ``index`` into rotating slot ``index % buffer_count``.

        The slot's NDSlice is returned; ``.data`` is the underlying SBUF
        ``nl.ndarray``. With ``buffer_count=2``, iteration k overlaps the
        DMA into slot ``(k+1) % 2`` with the compute on slot ``k % 2``.

        Args:
            index: Stream position (0-indexed).
            dtype: Cast on DMA. Defaults to stream dtype. Passing a
                different dtype bypasses the rotating slot (allocates a
                fresh SBUF buffer for this call).
            dge_mode / oob_mode / oob_value: Forwarded to ``NDSlice.load``.

        Returns:
            NDSlice over the rotating slot (or a fresh SBUF if dtype mismatch).

        Example:
            for k in nl.affine_range(K):
                tile = stream.load(k)
                nisa.nc_matmul(acc, lhs.data, tile.data)
        """
        child = self._child(index)
        dst = self._buffers[index % self._buffer_count]
        if dtype is not None and dtype != self._dtype:
            dst = None
        return child.load(
            dst=dst,
            dtype=dtype,
            dge_mode=dge_mode,
            oob_mode=oob_mode,
            oob_value=oob_value,
            pattern_override=self._pattern_override,
            out_shape=self._out_shape,
        )

    def __getitem__(self, index: int) -> "NDSlice":
        """
        Return the NDSlice wrapping rotating slot ``index % buffer_count``.

        No DMA -- the slot is the pre-allocated buffer. Use for output
        streaming (write into the slot, then ``stream.store(index)``) or
        when reading a slot loaded by an earlier call.

        Args:
            index: Stream position (0-indexed).

        Returns:
            NDSlice over the rotating slot's SBUF buffer.

        Example:
            # Write into rotating slot, then DMA back to HBM.
            nisa.tensor_copy(out_stream[k].data, ...)
            out_stream.store(k)
        """
        sbuf = self._buffers[index % self._buffer_count]
        tile_size = tuple(sbuf.shape)
        tile_shape = []
        for d in range(len(tile_size)):
            tile_shape.append(1)
        tile_shape = tuple(tile_shape)
        grid, layout = SBUFLayout.build_view(sbuf, tile_size, tile_shape, self._dtype, sbuf_buffer_type())
        return NDSlice(grid, layout)

    def store(self, index: int, oob_mode=None, dge_mode=None) -> None:
        """
        DMA the rotating slot at ``index % buffer_count`` back to HBM.

        Mirror of ``load(index)``. Use for output streaming: fill the
        slot via ``stream[k]`` (or ``get_tile(k)``), then call
        ``stream.store(k)`` to write to the corresponding HBM position.

        Args:
            index: Stream position (0-indexed).
            oob_mode: ``nisa.oob_mode.skip`` to suppress out-of-bounds
                DMA faults at boundaries.
            dge_mode: ``nisa.dge_mode.*`` -- DMA generation mode hint.

        Example:
            for k in nl.affine_range(K):
                nisa.tensor_copy(out_stream[k].data, psum_acc)
                out_stream.store(k)
        """
        sbuf = self._buffers[index % self._buffer_count]
        child = self._child(index)
        child.store(
            SBUFLayout.contiguous_ap(sbuf),
            oob_mode=oob_mode,
            dge_mode=dge_mode,
            pattern_override=self._pattern_override,
        )
