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
import nki.isa as nisa
import nki.language as nl

from ._helpers import (
    MAX_SBUF_PARTITION_ROWS,
    ceiling_div,
    remove_at,
    sbuf_buffer_type,
)
from .ap_emitter import APEmitter
from .axis import IndirectKind, IndirectOffset
from .grid import Grid
from .layout_sbuf import SBUFLayout


class HBMLayout(nl.NKIObject):
    """
    Physical addressing for an HBM-backed view.

    Holds the byte-level information :class:`Grid` doesn't carry: source
    tensor, compile-time offset, per-dim strides, and an optional
    :class:`~neurotile.core.axis.IndirectOffset` for runtime indexing.
    Pairs with a Grid inside an :class:`~neurotile.core.ndslice.NDSlice`
    to back ``.load()`` / ``.store()`` and to build access patterns via
    ``.ap()``.

    Immutable: ``advance`` / ``set_indirect`` / ``drop_dim`` /
    ``apply_transform`` all return fresh ``HBMLayout`` instances rather
    than mutating in place. The Grid composes those operations; HBMLayout
    never inspects axes directly except to emit the AP.

    Attributes:
        source: Root HBM tensor used by ``.ap()`` calls.
        root_source: Parent tensor for stride anchoring on sliced views;
            equals ``source`` when not set explicitly.
        offset (int): Compile-time element offset into ``source``.
        strides (tuple[int, ...]): Per-dim element strides; multiplied
            by ``axis.step`` at AP emission to get physical byte stride.
        dtype: Element dtype.
        buffer_type: ``nl.shared_hbm`` / ``nl.private_hbm`` (or a
            sentinel in test mode).
        indirect (IndirectOffset | None): Runtime scalar / vector offset
            for indirect indexing (``view[k]`` with runtime ``k``); the
            AP emitter routes it through ``scalar_offset=`` /
            ``vector_offset=`` instead of folding into ``offset``.
    """

    def __init__(
        self,
        source,
        offset,
        strides,
        dtype,
        buffer_type=None,
        indirect=None,
        root_source=None,
    ):
        self.source = source
        self.root_source = root_source if root_source is not None else source
        self.offset = offset
        self.strides = strides
        self.dtype = dtype
        self.buffer_type = buffer_type
        self.indirect = indirect

    # ================================================================
    # Position primitives
    # ================================================================

    def advance_step(self, grid, dim):
        """Per-item advance step on `dim` in this layout's native unit.

        HBM strides are at source-element granularity, so the native step
        is simply the Grid's per-item element step.
        """
        return grid.current_step(dim)

    def advance(self, dim, k, step):
        """Advance offset along `dim` by `k` items at `step` source-units per item.

        For compile-time int `k`, folds k * step * strides[dim] into self.offset.
        For runtime k, stashes k * step (source-element units, unscaled by
        stride) in self.indirect -- the AP emitter multiplies by stride at
        the time it issues `.ap(scalar_offset=...)`.
        """
        if isinstance(k, int):
            new_offset = self.offset + k * step * self.strides[dim]
            return HBMLayout(
                self.source,
                new_offset,
                self.strides,
                self.dtype,
                self.buffer_type,
                self.indirect,
                self.root_source,
            )
        # Runtime k -> indirect (in source-element units).
        scaled = k if step == 1 else k * step
        if self.indirect is not None and self.indirect.kind == IndirectKind.SCALAR and self.indirect.dim == dim:
            scaled = self.indirect.value + scaled
        new_indirect = IndirectOffset(kind=IndirectKind.SCALAR, value=scaled, dim=dim)
        return HBMLayout(
            self.source,
            self.offset,
            self.strides,
            self.dtype,
            self.buffer_type,
            new_indirect,
            self.root_source,
        )

    def set_indirect(self, kind, value, dim):
        """Replace self.indirect with a new tagged offset."""
        return HBMLayout(
            self.source,
            self.offset,
            self.strides,
            self.dtype,
            self.buffer_type,
            IndirectOffset(kind=kind, value=value, dim=dim),
            self.root_source,
        )

    def drop_dim(self, dim):
        """Remove dim from strides. indirect.dim stays unchanged (source-relative)."""
        return HBMLayout(
            self.source,
            self.offset,
            remove_at(self.strides, dim),
            self.dtype,
            self.buffer_type,
            self.indirect,
            self.root_source,
        )

    def drop_dims(self, dims):
        """Drop multiple dims in reverse index order."""
        result = self
        for d_idx in range(len(dims) - 1, -1, -1):
            result = result.drop_dim(dims[d_idx])
        return result

    # ================================================================
    # Strides / transforms
    # ================================================================

    def transform_strides(self, element_shape):
        """Strides for transform computation."""
        return self.strides

    def apply_transform(self, new_strides):
        """Return layout for a transformed view (reshape/permute/broadcast)."""
        return HBMLayout(
            self.source,
            self.offset,
            new_strides,
            self.dtype,
            self.buffer_type,
            self.indirect,
            self.root_source,
        )

    def retile(self, new_tile_size, new_remaining):
        """Return a fresh layout with the same fields.

        HBM strides are source-element granularity, so re-tiling does
        not change the layout's contents. We still return a new
        instance to keep ``Layout.retile`` aligned with every other
        layout transform (each returns a fresh value, never aliases
        ``self``), so callers can treat layouts as immutable values.
        """
        return HBMLayout(
            self.source,
            self.offset,
            self.strides,
            self.dtype,
            self.buffer_type,
            self.indirect,
            self.root_source,
        )

    # ================================================================
    # Data access
    # ================================================================

    def tile_data(self):
        """HBM source tensor for direct NKI indexing."""
        return self.source

    def get_data(self, grid):
        """HBM views have no tile-local data; returns None."""
        return None

    def ap(self, grid):
        """Build N-D HBM access pattern from grid.axes + self.strides.

        The merge-clamp uses the per-dim *addressable* extent (source
        extent minus this layout's prior offset) so partial trailing
        tiles produce APs that don't walk past the source.
        """
        addressable = []
        for d in range(grid.ndim):
            offset_d = self.dim_offset_elements(d, grid.element_shape)
            addressable.append(grid.element_shape[d] - offset_d)
        levels = APEmitter.emit(grid.axes, self.strides, tuple(addressable))
        return HBMLayout._apply_ap(self.source, self.offset, levels, self.indirect)

    # ================================================================
    # Load / store
    # ================================================================

    def load(
        self,
        grid,
        dtype=None,
        dst=None,
        oob_mode=None,
        oob_value=None,
        out_shape=None,
        transpose=False,
        dge_mode=None,
        pattern_override=None,
    ):
        """Load from HBM to SBUF. Returns (Grid, SBUFLayout)."""
        load_dtype = dtype if dtype is not None else self.dtype

        if transpose:
            return self._load_transpose(grid, load_dtype, dst=dst)

        remaining = grid.remaining
        owned_extents = grid.owned_extents()
        effective_tile_size, tile_shape = HBMLayout.compute_effective_tiles(
            remaining,
            grid.tile_size,
            owned_extents=owned_extents,
        )
        sbuf_shape, p_tiles = HBMLayout.compute_sbuf_alloc(
            remaining,
            grid.tile_size,
            owned_extents=owned_extents,
        )
        tile_p = effective_tile_size[0]

        assert p_tiles == 1 or remaining[0] % tile_p == 0, (
            "Multi-tile load with P-remainder: remaining[0]="
            + str(remaining[0])
            + " not divisible by tile_p="
            + str(tile_p)
        )

        sbuf = self._resolve_sbuf(dst, out_shape, sbuf_shape, load_dtype)
        if oob_value is not None:
            nisa.memset(sbuf, oob_value)

        self._issue_load_dma(
            grid,
            sbuf,
            effective_tile_size,
            tile_shape,
            tile_p,
            p_tiles,
            pattern_override,
            dst,
            out_shape,
            oob_mode,
            dge_mode,
        )

        return HBMLayout._wrap_sbuf_load(
            sbuf,
            grid,
            self,
            effective_tile_size,
            tile_shape,
            load_dtype,
            out_shape=out_shape,
        )

    def store(self, data, grid, oob_mode=None, dge_mode=None, pattern_override=None):
        """Build HBM AP, issue DMA from SBUF to HBM."""
        if pattern_override is not None:
            hbm_ap = HBMLayout._apply_ap(self.source, self.offset, pattern_override, self.indirect)
        else:
            hbm_ap = self.ap(grid)
        HBMLayout._dma_copy(dst=hbm_ap, src=data, oob_mode=oob_mode, dge_mode=dge_mode)

    def sbuf_shape_for(self, grid):
        """Compute SBUF allocation shape for a load from this region."""
        return HBMLayout.compute_sbuf_alloc(
            grid.remaining,
            grid.tile_size,
            owned_extents=grid.owned_extents(),
        )[0]

    # ================================================================
    # Remainder
    # ================================================================

    def dim_offset_elements(self, dim, element_shape):
        """Total elements consumed on `dim` from the source origin.

        Folds prior slice-starts and int advances into a single per-dim
        position in element units. The ``element_shape`` arg is the
        per-dim source extent used as the modular period to strip
        outer-dim contributions from the flat offset; for a contiguous
        row-major layout (strides ``(F, 1)``), dividing by
        ``strides[d]`` peels off this-dim-and-outer contributions, then
        ``mod element_shape[d]`` discards the outer contribution.
        """
        if self.strides is None:
            return 0
        if dim >= len(self.strides):
            return 0
        stride = self.strides[dim]
        if stride == 0:
            return 0
        period = element_shape[dim] if dim < len(element_shape) else 1
        if period <= 0:
            return 0
        return (self.offset // stride) % period

    def is_remainder(self, grid):
        """True when this view sits on a partial trailing tile.

        Sources: Grid's own bit (multi-tile parent / truncated leaf),
        the offset crossing element_shape on any dim, or an indirect
        offset (runtime; treated as worst case).
        """
        if self.indirect is not None or grid.is_remainder:
            return True
        for d in range(grid.ndim):
            outer = grid.outer_axis(d)
            if outer is None or outer.step == 0:
                continue
            dim_offset = self.dim_offset_elements(d, grid.element_shape)
            if dim_offset + outer.count * outer.step > grid.element_shape[d]:
                return True
        return False

    # ================================================================
    # Static helpers (tile/SBUF allocation math, DMA emission)
    # ================================================================

    @staticmethod
    def compute_effective_tiles(remaining, tile_size, owned_extents=None):
        """Compute effective tile size and tile shape from region and tile config.

        `owned_extents` (when provided) supplies per-dim owned-element counts
        so tile_shape reflects owned tiles (skipping shard gaps).
        """
        effective_tile_size = []
        tile_shape = []
        for d in range(len(tile_size)):
            if d < len(remaining):
                ets = min(tile_size[d], remaining[d])
            else:
                ets = tile_size[d]
            effective_tile_size.append(ets)
            if ets <= 0:
                tile_shape.append(1)
                continue
            if owned_extents is not None and d < len(owned_extents):
                tile_shape.append(ceiling_div(owned_extents[d], ets))
            elif d < len(remaining):
                tile_shape.append(ceiling_div(remaining[d], ets))
            else:
                tile_shape.append(1)
        return (tuple(effective_tile_size), tuple(tile_shape))

    @staticmethod
    def compute_sbuf_alloc(remaining, tile_size, owned_extents=None):
        """Compute SBUF allocation shape and owned P-tile count."""
        eff_ts, _ = HBMLayout.compute_effective_tiles(
            remaining,
            tile_size,
            owned_extents=owned_extents,
        )
        tile_p = eff_ts[0]
        p_span = owned_extents[0] if owned_extents is not None else remaining[0]
        p_tiles = p_span // tile_p
        if p_tiles > 1:
            f_span = (p_span,) + tuple(remaining[1:])
            total_f = SBUFLayout.f_extent(f_span, tile_p)
            return (tile_p, total_f), p_tiles
        return tuple(remaining), p_tiles

    @staticmethod
    def load_partition_fold(layout, grid, recipe, dtype, dge_mode):
        """Load with K separate DMAs for partition fold. Returns (Grid, SBUFLayout)."""
        sbuf_buffer = HBMLayout._partition_fold_load_dma(
            layout.source,
            layout.offset,
            recipe,
            grid.remaining,
            dtype,
            dge_mode,
        )
        return SBUFLayout.build_view(
            sbuf_buffer,
            grid.tile_size,
            grid.tile_size,
            dtype,
            sbuf_buffer_type(),
        )

    @staticmethod
    def store_partition_fold(source, offset, fold_recipe, element_shape, data, dge_mode):
        """Store with K separate DMAs for partition fold."""
        K, P_per_slice, fold_stride, base_pattern = fold_recipe
        f_total = 1
        for d in range(1, len(element_shape)):
            f_total = f_total * element_shape[d]
        f_per_slice = f_total
        for k in range(K):
            dst_offset = offset + k * fold_stride
            hbm_ap = source.ap(pattern=base_pattern, offset=dst_offset)
            p_start = k * P_per_slice
            sbuf_offset = p_start * f_total
            sbuf_ap = data.ap(
                pattern=[[f_total, P_per_slice], [1, f_per_slice]],
                offset=sbuf_offset,
            )
            HBMLayout._dma_copy(dst=hbm_ap, src=sbuf_ap, dge_mode=dge_mode)

    # ================================================================
    # Repr
    # ================================================================

    def __repr__(self):
        parts = "HBMLayout(offset=" + str(self.offset)
        parts = parts + ", strides=" + str(self.strides)
        if self.indirect is not None:
            parts = parts + ", indirect=" + str(self.indirect)
        return parts + ")"

    # ================================================================
    # Private instance helpers
    # ================================================================

    def _load_transpose(self, grid, dtype, dst=None):
        """DMA transpose path. Returns (Grid, SBUFLayout).

        DMA transpose requires a 2D (P, F) view. If the grid has trivial-1
        dims (from batch consume, block-iter, etc.), pick the two
        non-trivial dims as P and F. Iterate in ascending dim order so
        the result is naturally sorted.
        """
        p_dim_idx = -1
        f_dim_idx = -1
        for d in range(grid.ndim):
            if grid.remaining[d] > 1:
                if p_dim_idx < 0:
                    p_dim_idx = d
                elif f_dim_idx < 0:
                    f_dim_idx = d
        if p_dim_idx < 0 or f_dim_idx < 0:
            for d in range(grid.ndim):
                if d == p_dim_idx or d == f_dim_idx:
                    continue
                if p_dim_idx < 0:
                    p_dim_idx = d
                elif f_dim_idx < 0:
                    f_dim_idx = d

        remaining_2d = (grid.remaining[p_dim_idx], grid.remaining[f_dim_idx])
        strides_2d = (self.strides[p_dim_idx], self.strides[f_dim_idx])
        sbuf, transposed_ts, ts = HBMLayout._dma_transpose(
            self.source,
            self.offset,
            strides_2d,
            remaining_2d,
            dtype,
            self.indirect,
            dst=dst,
        )
        element_shape = []
        for d in range(len(ts)):
            element_shape.append(ts[d] * transposed_ts[d])
        return SBUFLayout.build_view(
            sbuf,
            tuple(element_shape),
            transposed_ts,
            dtype,
            sbuf_buffer_type(),
        )

    def _issue_load_dma(
        self,
        grid,
        sbuf,
        effective_tile_size,
        tile_shape,
        tile_p,
        p_tiles,
        pattern_override,
        dst,
        out_shape,
        oob_mode,
        dge_mode,
    ):
        """Build HBM + SBUF APs and issue DMA copy.

        Both APs walk the *addressable* extent (post-offset), not the
        SBUF buffer's allocation. The buffer can be larger (uniform
        rotating-pool slot, or pre-allocated dst) -- the trailing
        partial region is left uninitialized; the partial-aware Grid
        on the returned view ensures downstream consumers slice to
        the actual extent.
        """
        if pattern_override is not None:
            hbm_ap = HBMLayout._apply_ap(self.source, self.offset, pattern_override, self.indirect)
        else:
            hbm_ap = self.ap(grid)

        if out_shape is not None:
            # Custom load: SBUF buffer is exactly out_shape; walk it as a
            # single contiguous tile rather than re-using the HBM grid's
            # tile structure.
            sbuf_remaining = tuple(out_shape)
            sbuf_tile_p = out_shape[0]
            sbuf_p_tiles = 1
            ets = None
            ts = None
        else:
            addressable = []
            for d in range(grid.ndim):
                offset_d = self.dim_offset_elements(d, grid.element_shape)
                owned = grid._owned_extent(d)
                addressable.append(min(owned, grid.element_shape[d] - offset_d))
            sbuf_remaining = tuple(addressable)
            sbuf_tile_p = tile_p
            sbuf_p_tiles = p_tiles
            use_matched = pattern_override is None and dst is None
            ets = effective_tile_size if use_matched else None
            ts = tile_shape if use_matched else None
        sbuf_ap = SBUFLayout._build_ap(sbuf, sbuf_remaining, sbuf_tile_p, sbuf_p_tiles, ets, ts)
        HBMLayout._dma_copy(dst=sbuf_ap, src=hbm_ap, oob_mode=oob_mode, dge_mode=dge_mode)

    def _resolve_sbuf(self, dst, out_shape, default_shape, dtype):
        """Resolve SBUF buffer: use dst, allocate from out_shape, or default."""
        if dst is not None:
            return dst
        if out_shape is not None:
            return nl.ndarray(tuple(out_shape), dtype=dtype, buffer=nl.sbuf)
        return nl.ndarray(default_shape, dtype=dtype, buffer=nl.sbuf)

    # ================================================================
    # Private static helpers (DMA emission, transpose paths, fold)
    # ================================================================

    @staticmethod
    def _apply_ap(source, offset, pattern, indirect):
        """Build AP with optional indirect dispatch. Single source of truth."""
        if indirect is None:
            return source.ap(pattern=pattern, offset=offset)
        if indirect.kind == IndirectKind.SCALAR:
            return source.ap(
                pattern=pattern,
                offset=offset,
                scalar_offset=indirect.value,
                indirect_dim=indirect.dim,
            )
        return source.ap(
            pattern=pattern,
            offset=offset,
            vector_offset=indirect.value,
            indirect_dim=indirect.dim,
        )

    @staticmethod
    def _dma_copy(dst, src, oob_mode=None, dge_mode=None):
        """Issue nisa.dma_copy with optional oob_mode and dge_mode."""
        if oob_mode is not None and dge_mode is not None:
            nisa.dma_copy(dst, src, oob_mode=oob_mode, dge_mode=dge_mode)
        elif oob_mode is not None:
            nisa.dma_copy(dst, src, oob_mode=oob_mode)
        elif dge_mode is not None:
            nisa.dma_copy(dst, src, dge_mode=dge_mode)
        else:
            nisa.dma_copy(dst=dst, src=src)

    @staticmethod
    def _dma_transpose(source, offset, strides, remaining, dtype, indirect, dst=None):
        """Load with nisa.dma_transpose. Returns (sbuf, tile_size, tile_shape)."""
        assert len(remaining) == 2
        p_dim = remaining[0]
        f_dim = remaining[1]
        row_stride = strides[0]

        if f_dim <= MAX_SBUF_PARTITION_ROWS:
            return HBMLayout._dma_transpose_direct(
                source,
                offset,
                row_stride,
                p_dim,
                f_dim,
                dtype,
                indirect,
                dst=dst,
            )
        assert f_dim % MAX_SBUF_PARTITION_ROWS == 0
        return HBMLayout._dma_transpose_tiled(
            source,
            offset,
            row_stride,
            p_dim,
            f_dim,
            dtype,
            indirect,
            dst=dst,
        )

    @staticmethod
    def _issue_dma_transpose(source, offset, hbm_pattern, sbuf, sbuf_pattern, indirect):
        """Build APs and issue nisa.dma_transpose."""
        hbm_ap = HBMLayout._apply_ap(source, offset, hbm_pattern, indirect)
        sbuf_ap = sbuf.ap(pattern=sbuf_pattern, offset=0)
        nisa.dma_transpose(dst=sbuf_ap, src=hbm_ap)

    @staticmethod
    def _dma_transpose_direct(source, offset, row_stride, p_dim, f_dim, dtype, indirect, dst=None):
        """Direct DMA transpose for F <= 128."""
        transposed_shape = (f_dim, p_dim)
        if dst is not None:
            assert tuple(dst.shape) == transposed_shape
            sbuf = dst
        else:
            sbuf = nl.ndarray(transposed_shape, dtype=dtype, buffer=nl.sbuf)
        hbm_pattern = [[row_stride, p_dim], [row_stride, 1], [row_stride, 1], [1, f_dim]]
        sbuf_pattern = [[p_dim, f_dim], [p_dim, 1], [p_dim, 1], [1, p_dim]]
        HBMLayout._issue_dma_transpose(source, offset, hbm_pattern, sbuf, sbuf_pattern, indirect)
        return (sbuf, transposed_shape, (1, 1))

    @staticmethod
    def _dma_transpose_tiled(source, offset, row_stride, p_dim, f_dim, dtype, indirect, dst=None):
        """Tiled DMA transpose for F > MAX_SBUF_PARTITION_ROWS."""
        num_chunks = f_dim // MAX_SBUF_PARTITION_ROWS
        transposed_shape = (MAX_SBUF_PARTITION_ROWS, p_dim * num_chunks)
        if dst is not None:
            assert tuple(dst.shape) == transposed_shape
            sbuf = dst
        else:
            sbuf = nl.ndarray(transposed_shape, dtype=dtype, buffer=nl.sbuf)
        hbm_pattern = [
            [row_stride, p_dim],
            [1, 1],
            [MAX_SBUF_PARTITION_ROWS, num_chunks],
            [1, MAX_SBUF_PARTITION_ROWS],
        ]
        sbuf_pattern = [
            [p_dim * num_chunks, MAX_SBUF_PARTITION_ROWS],
            [1, 1],
            [p_dim, num_chunks],
            [1, p_dim],
        ]
        HBMLayout._issue_dma_transpose(source, offset, hbm_pattern, sbuf, sbuf_pattern, indirect)
        return (sbuf, transposed_shape, (1, 1))

    @staticmethod
    def _partition_fold_load_dma(source, offset, fold_recipe, element_shape, dtype, dge_mode):
        """Load with K separate DMAs for partition fold. Returns SBUF ndarray."""
        K, P_per_slice, fold_stride, base_pattern = fold_recipe
        total_P = K * P_per_slice
        f_total = 1
        for d in range(1, len(element_shape)):
            f_total = f_total * element_shape[d]
        f_per_slice = f_total
        sbuf = nl.ndarray((total_P, f_total), dtype=dtype, buffer=nl.sbuf)
        for k in range(K):
            src_offset = offset + k * fold_stride
            hbm_ap = source.ap(pattern=base_pattern, offset=src_offset)
            p_start = k * P_per_slice
            sbuf_offset = p_start * f_total
            sbuf_ap = sbuf.ap(
                pattern=[[f_total, P_per_slice], [1, f_per_slice]],
                offset=sbuf_offset,
            )
            HBMLayout._dma_copy(dst=sbuf_ap, src=hbm_ap, dge_mode=dge_mode)
        return sbuf

    @staticmethod
    def _wrap_sbuf_load(
        sbuf,
        hbm_grid,
        hbm_layout,
        effective_tile_size,
        tile_shape,
        dtype,
        out_shape=None,
    ):
        """Wrap raw SBUF buffer as (Grid, SBUFLayout) after a load.

        Types the SBUF view at the HBM source's *addressable* per-dim
        extent: the Grid carries the partial-trailing-tile extent
        (clamped via the layout offset) so downstream ``[k, i].data``
        reports per-tile partial widths correctly. The SBUF buffer's
        physical allocation may be larger -- AP / DMA emission walks at
        the uniform allocation stride, while view typing uses the
        addressable extent. Single rule, no conditional fallback.

        With ``out_shape`` (load() with custom ``pattern_override`` /
        ``out_shape``), the loaded SBUF tile has ``out_shape`` extent --
        independent of the HBM source tile_size -- so the view is typed
        at ``out_shape`` as a single tile.
        """
        if out_shape is not None:
            sbuf_grid, sbuf_layout = SBUFLayout.build_view(
                sbuf,
                tuple(out_shape),
                tuple(out_shape),
                dtype,
                sbuf_buffer_type(),
                block_size=hbm_grid.block_size,
            )
            return sbuf_grid, sbuf_layout

        # Per-dim addressable extent: min(grid's owned walk, source-extent
        # remaining after the HBM layout's prior offset).
        addressable = []
        for d in range(hbm_grid.ndim):
            owned = hbm_grid._owned_extent(d)
            offset_d = hbm_layout.dim_offset_elements(d, hbm_grid.element_shape)
            remaining = hbm_grid.element_shape[d] - offset_d
            addressable.append(min(owned, remaining))

        sbuf_grid, sbuf_layout = SBUFLayout.build_view(
            sbuf,
            tuple(addressable),
            tuple(effective_tile_size),
            dtype,
            sbuf_buffer_type(),
            block_size=hbm_grid.block_size,
        )

        # The loaded SBUF view mirrors the HBM iteration state: same per-dim
        # axis depth, same cursor. Truncate the freshly-built SBUF grid to
        # match HBM's per-dim depth (drop outer TILE/BLOCK wrappers HBM has
        # already consumed) and copy HBM's cursor so .shape reports the
        # right iteration extent.
        new_axes = []
        for d in range(sbuf_grid.ndim):
            sbuf_axes_d = sbuf_grid.axes_for(d)
            if d < hbm_grid.ndim:
                hbm_count = len(hbm_grid.axes_for(d))
            else:
                hbm_count = len(sbuf_axes_d)
            keep = min(len(sbuf_axes_d), hbm_count)
            kept = sbuf_axes_d[len(sbuf_axes_d) - keep :]
            for ax in kept:
                new_axes.append(ax)

        sbuf_grid = Grid(
            element_shape=sbuf_grid.element_shape,
            axes=tuple(new_axes),
            cursor=hbm_grid.cursor,
            n_batch_dims=sbuf_grid.n_batch_dims,
            tiled=sbuf_grid.tiled,
        )

        return sbuf_grid, sbuf_layout
