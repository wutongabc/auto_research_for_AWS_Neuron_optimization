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
"""nkilib.experimental.neurotile - Tile Iterator Library for NKI.

User-facing API: see neurotile_api_reference.md.

Canonical consumer style:
    from nkilib_src.nkilib.experimental import neurotile as nt
    nt.tiles(...)
    nt.blocks(...)
    nt.alloc_tiles(...)
    # etc.
"""

from .core._helpers import ceiling_div, largest_divisor
from .core.factories import alloc_blocks, alloc_tiles, blocks, tensor_view, tiles
from .core.psum_pool import psum_pool
from .core.shard_helpers import (
    block_range,
    get_shard_info,
    interleaved_range,
    uneven_block_range,
)

__all__ = [
    "alloc_blocks",
    "alloc_tiles",
    "block_range",
    "blocks",
    "ceiling_div",
    "get_shard_info",
    "interleaved_range",
    "largest_divisor",
    "psum_pool",
    "tensor_view",
    "tiles",
    "uneven_block_range",
]
