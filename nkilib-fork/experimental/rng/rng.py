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

"""RNG kernels for GPSIMD engine state management and random number generation on Trainium2."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import assert_shape

NUM_LANES = 128
NUM_RNG_SEEDS = 6
DTYPE_SIZE_INT32 = 4


@nki.jit
def get_rng_state_gpsimd(tensor_state: nl.ndarray):
    """
    Retrieve the current RNG state from the GPSIMD engine.

    Reads all 128 lanes of RNG state from the GPSIMD engine into SBUF,
    then copies only lane 0's seeds to a new output HBM tensor.

    Input shape range is constant [1, NUM_RNG_SEEDS]

    Dimensions:
        L: Number of GPSIMD lanes (128)
        S: Number of RNG seeds per lane (6)

    Args:
        tensor_state (nl.ndarray): [1, NUM_RNG_SEEDS], dtype uint32, HBM tensor
            used only for shape/dtype reference.

    Returns:
        output (nl.ndarray): [1, NUM_RNG_SEEDS], dtype uint32, HBM tensor
            containing the 6 RNG seeds from lane 0.

    Pseudocode:
        state = ndarray(shape=(128, 6), dtype=uint32)  # SBUF buffer
        state = gpsimd_engine.get_rng_state()           # Read all lanes
        output = state[0, 0:6]                          # Copy lane 0 to HBM
    """
    assert_shape(tensor_state, (1, NUM_RNG_SEEDS), "tensor_state")
    output = nl.ndarray(shape=(1, NUM_RNG_SEEDS), dtype=nl.uint32, buffer=nl.shared_hbm)
    state = nl.ndarray(shape=(NUM_LANES, NUM_RNG_SEEDS), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.rand_get_state(dst=state, engine=nisa.gpsimd_engine)
    nisa.dma_copy(dst=output, src=state[0:1, 0:NUM_RNG_SEEDS])
    return output


@nki.jit
def set_rng_state_gpsimd(tensor_state: nl.ndarray):
    """
    Set the RNG state for the GPSIMD engine by broadcasting seeds to all lanes.

    Loads 6 seeds from HBM, broadcasts them to all 128 GPSIMD lanes,
    and writes the state to the engine.

    Input shape range is constant [1, NUM_RNG_SEEDS]

    Dimensions:
        L: Number of GPSIMD lanes (128)
        S: Number of RNG seeds per lane (6)

    Args:
        tensor_state (nl.ndarray): [1, NUM_RNG_SEEDS], dtype uint32, HBM tensor
            containing the 6 seeds to broadcast.

    Returns:
        output (nl.ndarray): [1, NUM_RNG_SEEDS], dtype uint32, HBM tensor
            echoing back the seeds that were set.

    Pseudocode:
        seed = dma_load(tensor_state)                    # Load seeds from HBM
        state = broadcast(seed, shape=(128, 6))          # Broadcast to all lanes
        gpsimd_engine.set_rng_state(state)               # Write state to engine
    """
    assert_shape(tensor_state, (1, NUM_RNG_SEEDS), "tensor_state")
    output = nl.ndarray(shape=(1, NUM_RNG_SEEDS), dtype=nl.uint32, buffer=nl.shared_hbm)
    seed = nl.ndarray(shape=(1, NUM_RNG_SEEDS), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=seed, src=tensor_state)
    state = nl.broadcast_to(seed, (NUM_LANES, NUM_RNG_SEEDS))
    nisa.rand_set_state(src_seeds=state, engine=nisa.gpsimd_engine)
    nisa.dma_copy(dst=output, src=seed)
    return output.view(nl.int32)


@nki.jit
def generate_random(output: nl.ndarray, n_elements: int):
    """
    Generate random int32 values, tiling to fit SBUF.

    Generates n_elements random int32 values using the GPSIMD RNG engine
    and writes them to a new output HBM tensor. Uses sequential_range because
    rand carries implicit RNG state across iterations (loop-carried dependency).

    Output shape range should be [1, n_elements] where n_elements is unbounded

    Dimensions:
        N: Number of random elements to generate (n_elements)
        F: Tile size on free dimension, determined by available SBUF

    Args:
        output (nl.ndarray): [1, n_elements], dtype int32, HBM tensor
            to be filled with random values.
        n_elements (int): Number of random int32 values to generate.

    Returns:
        output (nl.ndarray): [1, n_elements], dtype int32, HBM tensor
            filled with random values.

    Notes:
        - Uses sequential_range (not affine_range) due to loop-carried RNG state dependency
        - Remainder tile is handled separately after full tiles

    Pseudocode:
        tile_free_size = total_sbuf_size // sizeof(int32)
        n_full_tiles = n_elements // tile_free_size
        remainder = n_elements % tile_free_size
        for tile_idx in range(n_full_tiles):
            random_buffer = rng(shape=(128, tile_free_size))
            output[0, tile_idx * tile_free_size : (tile_idx+1) * tile_free_size] = random_buffer[0]
        if remainder > 0:
            random_buffer = rng(shape=(128, remainder))
            output[0, n_full_tiles * tile_free_size : ...] = random_buffer[0]
    """

    # Compute tile size from SBUF capacity
    tile_free_size = nl.tile_size.total_available_sbuf_size // DTYPE_SIZE_INT32
    n_full_tiles, remainder = divmod(n_elements, tile_free_size)

    for tile_idx in nl.sequential_range(n_full_tiles):
        offset = tile_idx * tile_free_size
        random_buffer = nl.ndarray([NUM_LANES, tile_free_size], dtype=nl.int32, buffer=nl.sbuf)
        nisa.rng(dst=random_buffer, engine=nisa.engine.gpsimd)
        nisa.dma_copy(dst=output[0:1, offset : offset + tile_free_size], src=random_buffer[0:1, 0:tile_free_size])

    if remainder > 0:
        offset = n_full_tiles * tile_free_size
        random_buffer = nl.ndarray([NUM_LANES, remainder], dtype=nl.int32, buffer=nl.sbuf)
        nisa.rng(dst=random_buffer, engine=nisa.engine.gpsimd)
        nisa.dma_copy(dst=output[0:1, offset : offset + remainder], src=random_buffer[0:1, 0:remainder])
    return output
