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
from typing import Any, Optional, Union

import nki.language as nl


def _validate_shard_args(num_shards, total, caller):
    """Shared positive-int check for shard helper arg dims.

    `num_shards` and `total` must be compile-time positive ints.
    `rank` is checked at the call site (it can be a runtime expression).
    """
    assert isinstance(num_shards, int) and not isinstance(num_shards, bool), (
        caller + ": num_shards must be an int, got " + str(num_shards)
    )
    assert num_shards >= 1, caller + ": num_shards must be >= 1, got " + str(num_shards)
    assert isinstance(total, int) and not isinstance(total, bool), caller + ": total must be an int, got " + str(total)
    assert total >= 1, caller + ": total must be >= 1, got " + str(total)


def block_range(rank: Union[int, Any], num_shards: int, total: int) -> slice:
    """
    Build a contiguous-block shard range for one rank.

    Each rank owns ``total // num_shards`` consecutive units along the
    shard dim. Pair the returned slice with NDSlice subscripting to
    produce a sharded view.

    Args:
        rank (int | runtime scalar): This core's rank index. Compile-time
            ``int`` or a runtime expression such as ``nl.program_id(0)``.
        num_shards (int): Total core count. Must divide `total` evenly;
            use :func:`uneven_block_range` when divisibility is not
            guaranteed.
        total (int): Total unit count along the shard dim. The unit is
            whatever the slice indexes -- tiles for ``nt.tiles(...)``,
            blocks for ``nt.blocks(...)``.

    Returns:
        slice: ``slice(start, start + owned)`` covering this rank's range.
        When `rank` is a runtime scalar, ``start`` is a runtime expression
        and NDSlice routes it through the layout's indirect-offset path.

    Raises:
        AssertionError: ``total`` is not divisible by ``num_shards``.

    Example:
        # Shard a tile-grid across the SPMD launch grid:
        rank = nl.program_id(0)
        n = nl.num_programs(0)
        n_tiles = nt.ceiling_div(M, 128)
        view = nt.tiles(src, tile_size=(128, 512))[
            block_range(rank, n, n_tiles), :
        ]
    """
    _validate_shard_args(num_shards, total, "block_range")
    assert total % num_shards == 0, (
        "block_range requires total ("
        + str(total)
        + ") divisible by num_shards ("
        + str(num_shards)
        + "). Use uneven_block_range for remainder distribution."
    )
    owned = total // num_shards
    # Skip `rank * 1` -- Beta 3 parser rejects runtime_scalar * 1.
    if owned == 1:
        start = rank
    else:
        start = rank * owned
    return slice(start, start + owned)


def uneven_block_range(rank: Union[int, Any], num_shards: int, total: int) -> slice:
    """
    Build a contiguous-block shard range with the remainder spread to early ranks.

    When ``total`` doesn't divide evenly by ``num_shards``, ranks
    ``[0, total % num_shards)`` each get one extra unit so the entire
    `total` range is covered. The uneven case requires a compile-time
    ``int`` rank because the per-rank owned count varies.

    Args:
        rank (int | runtime scalar): This core's rank index. Must be a
            compile-time ``int`` when ``total`` does not divide evenly.
            A runtime scalar is accepted only when divisibility holds, in
            which case this delegates to :func:`block_range`.
        num_shards (int): Total core count.
        total (int): Total unit count along the shard dim.

    Returns:
        slice: ``slice(start, start + owned)`` for this rank's range.

    Raises:
        AssertionError: Runtime ``rank`` combined with non-divisible
            ``total`` (per-rank branching is not expressible at trace
            time). Use :func:`block_range` or pad ``total`` to a multiple
            of ``num_shards``.

    Example:
        # 5 cores, 12 work units -- ranks 0, 1 own 3 each; ranks 2, 3, 4 own 2.
        for rank in range(5):
            r = uneven_block_range(rank=rank, num_shards=5, total=12)
            owned_count = r.stop - r.start  # 3, 3, 2, 2, 2
    """
    _validate_shard_args(num_shards, total, "uneven_block_range")
    base_units = total // num_shards
    remainder = total % num_shards

    if not isinstance(rank, int):
        # Runtime rank: must divide evenly (per-rank branch impossible).
        assert remainder == 0, (
            "uneven_block_range with runtime rank requires total ("
            + str(total)
            + ") divisible by num_shards ("
            + str(num_shards)
            + "). Use block_range or adjust dimensions."
        )
        return block_range(rank, num_shards, total)

    # Compile-time rank: ranks [0, remainder) get base+1, rest get base.
    if rank < remainder:
        owned = base_units + 1
        extra_before = rank
    else:
        owned = base_units
        extra_before = remainder
    start = rank * base_units + extra_before
    return slice(start, start + owned)


def interleaved_range(rank: Union[int, Any], num_shards: int, total: int) -> slice:
    """
    Build a round-robin shard range for one rank.

    Each rank owns every ``num_shards``-th unit along the shard dim
    (rank 0 owns 0, num_shards, 2*num_shards, ...; rank 1 owns 1,
    num_shards+1, ...). The returned slice has ``step == num_shards``.
    NDSlice's stepped-slice handling factors a peer-walk axis from this,
    leaving an inner stride-``num_shards`` walk for the rank's own units.

    Args:
        rank (int | runtime scalar): This core's rank index.
        num_shards (int): Total core count. Must divide ``total`` evenly.
        total (int): Total unit count along the shard dim.

    Returns:
        slice: ``slice(rank, total, num_shards)`` -- a stepped range
        addressing only this rank's units.

    Raises:
        AssertionError: ``total`` is not divisible by ``num_shards``.

    Example:
        rank = nl.program_id(0)
        n = nl.num_programs(0)
        view = nt.tiles(src, tile_size=(128, 256))[
            interleaved_range(rank, n, n_tiles), :
        ]
    """
    _validate_shard_args(num_shards, total, "interleaved_range")
    assert total % num_shards == 0, (
        "interleaved_range requires total (" + str(total) + ") divisible by num_shards (" + str(num_shards) + ")."
    )
    return slice(rank, total, num_shards)


def get_shard_info(
    tensor_shape: tuple,
    tile_size: tuple,
    shard_dim: int = 0,
    num_shards: Optional[int] = None,
    shard_id: Optional[Any] = None,
) -> dict:
    """
    Compute a partition summary for a sharded tile grid.

    Diagnostic helper -- not used by the runtime DMA path. Returns a
    dictionary describing how a sharded view would partition the tile
    grid, useful for trace-time printing or asserting expected
    structure.

    Args:
        tensor_shape (tuple[int, ...]): Element shape of the source tensor.
        tile_size (tuple[int, ...]): Tile shape used by ``nt.tiles()`` /
            ``nt.blocks()`` -- determines the tile grid extent.
        shard_dim (int): Dim along which sharding splits tiles. Defaults
            to 0.
        num_shards (int | None): Total core count. Defaults to
            ``nl.num_programs(0)``.
        shard_id (int | runtime scalar | None): This core's rank.
            Defaults to ``nl.program_id(0)``.

    Returns:
        dict: Keys ``total_tiles``, ``tiles_per_shard``, ``shard_id``,
        ``num_shards``, ``shard_dim``.

    Raises:
        AssertionError: ``tensor_shape`` and ``tile_size`` are not
            same-length tuples; ``shard_dim`` is not an int in
            ``[0, len(tensor_shape))``; ``tensor_shape[shard_dim]`` is
            not divisible by ``tile_size[shard_dim]``.

    Example:
        info = get_shard_info(
            tensor_shape=(1024, 4096), tile_size=(128, 512),
            shard_dim=0, num_shards=2, shard_id=0,
        )
        # info["total_tiles"] == 8, info["tiles_per_shard"] == 4
    """
    assert isinstance(tensor_shape, (tuple, list)), "get_shard_info: tensor_shape must be a tuple/list, got " + str(
        tensor_shape
    )
    assert isinstance(tile_size, (tuple, list)), "get_shard_info: tile_size must be a tuple/list, got " + str(tile_size)
    assert len(tensor_shape) == len(tile_size), (
        "get_shard_info: tensor_shape and tile_size must have the same "
        "rank; got tensor_shape rank " + str(len(tensor_shape)) + " vs tile_size rank " + str(len(tile_size)) + "."
    )
    assert isinstance(shard_dim, int) and not isinstance(shard_dim, bool), (
        "get_shard_info: shard_dim must be an int, got " + str(shard_dim)
    )
    assert 0 <= shard_dim < len(tensor_shape), (
        "get_shard_info: shard_dim="
        + str(shard_dim)
        + " is out of range for tensor_shape rank "
        + str(len(tensor_shape))
        + "."
    )

    if num_shards is None:
        num_shards = nl.num_programs(0)
    if shard_id is None:
        shard_id = nl.program_id(0)

    assert tensor_shape[shard_dim] % tile_size[shard_dim] == 0, (
        "get_shard_info: tensor_shape["
        + str(shard_dim)
        + "]="
        + str(tensor_shape[shard_dim])
        + " not divisible by tile_size["
        + str(shard_dim)
        + "]="
        + str(tile_size[shard_dim])
    )
    total_tiles = tensor_shape[shard_dim] // tile_size[shard_dim]
    tiles_per_shard = total_tiles // num_shards

    return {
        "total_tiles": total_tiles,
        "tiles_per_shard": tiles_per_shard,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "shard_dim": shard_dim,
    }


__all__ = [
    "block_range",
    "uneven_block_range",
    "interleaved_range",
    "get_shard_info",
]
