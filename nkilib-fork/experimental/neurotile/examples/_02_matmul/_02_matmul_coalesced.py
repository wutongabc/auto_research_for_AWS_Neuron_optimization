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
Blocked matmul with coalesced block load/store via nt.blocks(): each
block-load DMAs BLOCK_K x BLOCK_M (or BLOCK_K x BLOCK_N) tiles in a
single coalesced transfer; per-tile matmul accumulates into an
nt.alloc_tiles output block; one coalesced store per output block.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from nkilib_src.nkilib.experimental import neurotile as nt


@nki.jit
def matmul_coalesced(
    lhsT,
    rhs,
    TILES_IN_BLOCK_M=4,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=4,
):
    """Blocked matmul: C = lhsT.T @ rhs with coalesced block load/store."""
    # lhsT: [K, M],  rhs: [K, N]  ->  C: [M, N]
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    K, M = lhsT.shape
    _, N = rhs.shape
    C = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    lhsT_blocks = nt.blocks(
        lhsT,
        tile_size=(TILE_K, TILE_M),
        block_size=(TILES_IN_BLOCK_K, TILES_IN_BLOCK_M),
    )
    rhs_blocks = nt.blocks(
        rhs,
        tile_size=(TILE_K, TILE_N),
        block_size=(TILES_IN_BLOCK_K, TILES_IN_BLOCK_N),
    )
    out_blocks = nt.blocks(
        C,
        tile_size=(TILE_M, TILE_N),
        block_size=(M // TILE_M, TILES_IN_BLOCK_N),
    )

    for nb in range(rhs_blocks.shape[1]):
        # Pre-allocate one output block of partials.
        acc = nt.alloc_tiles(
            tile_size=(TILE_M, TILE_N),
            grid=(M // TILE_M, TILES_IN_BLOCK_N),
            buffer_type=nl.sbuf,
            dtype=lhsT.dtype,
        )
        nisa.memset(acc.data, 0.0)

        for kb in nl.sequential_range(rhs_blocks.shape[0]):
            rhs_block = rhs_blocks[kb, nb].load()  # block: [BLOCK_K*128, BLOCK_N*512]

            for mb in range(lhsT_blocks.shape[1]):
                lhsT_block = lhsT_blocks[kb, mb].load()  # block: [BLOCK_K*128, BLOCK_M*128]

                for bm in range(TILES_IN_BLOCK_M):
                    for bn in range(TILES_IN_BLOCK_N):
                        psum = nl.ndarray(
                            (TILE_M, TILE_N),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        for bk in range(TILES_IN_BLOCK_K):
                            nisa.nc_matmul(
                                dst=psum,
                                stationary=lhsT_block[bk, bm].data,
                                moving=rhs_block[bk, bn].data,
                            )
                        acc_tile = acc[mb * TILES_IN_BLOCK_M + bm, bn].data
                        nisa.tensor_tensor(
                            dst=acc_tile,
                            data1=acc_tile,
                            data2=psum,
                            op=nl.add,
                        )

        # Single coalesced store back to HBM.
        out_blocks[0, nb].store(acc.ap())

    return C


# ============================================================================
# Helpers
# ============================================================================


def to_device(t):
    import torch_xla.core.xla_model as xm

    return t.to(xm.xla_device())


def to_cpu(t):
    return t.cpu() if isinstance(t, torch.Tensor) else t


# ============================================================================
# Tests
# ============================================================================


def test_matmul_coalesced():
    """Single block per dimension."""
    torch.manual_seed(42)
    M, K, N = 512, 512, 1024
    lhs = torch.rand(M, K, dtype=torch.bfloat16)
    rhs = torch.rand(K, N, dtype=torch.bfloat16)
    result = matmul_coalesced(to_device(lhs.T.contiguous()), to_device(rhs))
    expected = lhs @ rhs
    assert torch.allclose(to_cpu(result).to(torch.bfloat16), expected, rtol=1e-2, atol=1e-2)
    print("matmul_coalesced: PASSED")


def test_matmul_coalesced_multi_block():
    """Multiple blocks per dimension."""
    torch.manual_seed(42)
    M, K, N = 512, 512, 1024
    lhs = torch.rand(M, K, dtype=torch.bfloat16)
    rhs = torch.rand(K, N, dtype=torch.bfloat16)
    result = matmul_coalesced(
        to_device(lhs.T.contiguous()),
        to_device(rhs),
        TILES_IN_BLOCK_M=2,
        TILES_IN_BLOCK_N=1,
        TILES_IN_BLOCK_K=2,
    )
    expected = lhs @ rhs
    assert torch.allclose(to_cpu(result).to(torch.bfloat16), expected, rtol=1e-2, atol=1e-2)
    print("matmul_coalesced_multi_block: PASSED")


def main():
    test_matmul_coalesced()
    test_matmul_coalesced_multi_block()


if __name__ == "__main__":
    main()
