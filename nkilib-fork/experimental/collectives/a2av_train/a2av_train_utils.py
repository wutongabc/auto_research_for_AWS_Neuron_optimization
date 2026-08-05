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
"""Shared helpers for the a2av_train dispatch / combine kernels.

This module exposes module-private helpers used by ``permute_a2av`` and
``unpermute_a2av`` to build the inputs ``ncc.all_to_all_v`` expects:

- :func:`_exclusive_cumsum_u32` computes the per-destination *scatter
  displacement* table (exclusive cumulative sum in tokens) and materialises
  it in both SBUF and HBM so the packer loop can look it up via indirect DMA.
- :func:`_write_a2av_v_metadata` writes the ``(4, EP)`` uint32 metadata
  tensor consumed by the collective (counts, displacements, then two zero
  rows populated by the runtime when ``recv_counts_known=False`` and
  ``has_rdispls=False``).
- :func:`_validate_a2av_indices_counts` asserts the shared shape/dtype
  contract on ``send_indices`` and ``counts`` (plus ``T >= 1``, ``EP >= 2``).
  Called at entry from both ``permute_a2av`` and ``unpermute_a2av``.

All three helpers are plain Python functions (not decorated with
``@nki.jit``) so they inline into the caller's tracing context.
"""

import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert


def _exclusive_cumsum_u32(
    counts_hbm: nl.ndarray,
    EP: int,
) -> tuple[nl.ndarray, nl.ndarray, nl.ndarray]:
    """Exclusive cumulative sum of ``counts_hbm`` in tokens.

    Produces the scatter-displacement table
    ``[0, counts[0], counts[0] + counts[1], ..., sum(counts[:EP - 1])]``,
    which is the per-destination row offset into the packed send buffer
    that ``ncc.all_to_all_v`` expects.

    Args:
        counts_hbm (nl.ndarray): [1, EP] int32/uint32 HBM tensor.
            Per-destination counts in tokens.
        EP (int): collective world size (static).

    Returns:
        counts_sb (nl.ndarray): [1, EP] uint32 SBUF. Raw counts (in tokens)
            loaded from ``counts_hbm``; returned so callers can reuse
            without a second DMA.
        sdispls_sb (nl.ndarray): [1, EP] uint32 SBUF. Exclusive cumulative
            sum (scatter displacements) in tokens.
        sdispls_hbm (nl.ndarray): [1, EP] uint32 shared_hbm. Same data as
            ``sdispls_sb``, materialised in HBM for indirect-DMA lookups
            from the packer loops.

    Notes:
        ``tensor_tensor_scan`` requires float32 operands, so counts are
        cast through float32 and back to uint32. This is safe for the
        count ranges used in practice (sums up to hundreds of thousands
        of tokens fit comfortably in fp32 mantissa).
    """
    counts_sb = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=counts_sb, src=counts_hbm)

    counts_f = nl.ndarray((1, EP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(counts_f, counts_sb)

    sdispls_f = nl.ndarray((1, EP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(sdispls_f, 0.0)
    init_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(init_sb, 0.0)
    # ones_sb acts as a multiplicative identity for the scan:
    #   acc = acc * 1 + counts[i]   → exclusive cumulative sum.
    ones_sb = nl.ndarray((1, EP - 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_sb, 1.0)
    nisa.tensor_tensor_scan(
        initial=init_sb,
        data0=ones_sb,
        op0=nl.multiply,
        data1=counts_f[0, : EP - 1],
        op1=nl.add,
        dst=sdispls_f[0, 1:],
    )

    sdispls_sb = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(sdispls_sb, sdispls_f)
    sdispls_hbm = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.shared_hbm, name="a2av_v_sdispls_hbm")
    nisa.dma_copy(dst=sdispls_hbm, src=sdispls_sb)

    return counts_sb, sdispls_sb, sdispls_hbm


def _write_a2av_v_metadata(
    metadata_hbm: nl.ndarray,
    counts_sb: nl.ndarray,
    sdispls_sb: nl.ndarray,
    H: int,
    EP: int,
) -> None:
    """Write the four rows of the ``(4, EP)`` uint32 metadata tensor that
    ``ncc.all_to_all_v`` consumes.

        row 0: counts   * H    (send-side counts, in elements)
        row 1: sdispls  * H    (packed send displacements, in elements)
        row 2: zeros           (runtime fills recv_counts when
                                recv_counts_known=False)
        row 3: zeros           (runtime fills recv_displs when
                                has_rdispls=False)

    Args:
        metadata_hbm (nl.ndarray): [4, EP] uint32 shared_hbm tensor,
            pre-allocated by the caller so the caller controls the
            tensor ``name``.
        counts_sb (nl.ndarray): [1, EP] uint32 SBUF. Counts in tokens.
        sdispls_sb (nl.ndarray): [1, EP] uint32 SBUF. Exclusive cumulative
            sum of ``counts_sb`` in tokens.
        H (int): hidden size (elements per token).
        EP (int): collective world size.
    """
    counts_el = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(data=counts_sb, op0=nl.multiply, operand0=H, dst=counts_el)

    sdispls_el = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(data=sdispls_sb, op0=nl.multiply, operand0=H, dst=sdispls_el)

    zeros_sb = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.memset(zeros_sb, 0)

    nisa.dma_copy(dst=metadata_hbm[0:1, :], src=counts_el)
    nisa.dma_copy(dst=metadata_hbm[1:2, :], src=sdispls_el)
    nisa.dma_copy(dst=metadata_hbm[2:3, :], src=zeros_sb)
    nisa.dma_copy(dst=metadata_hbm[3:4, :], src=zeros_sb)


def _validate_a2av_indices_counts(
    send_indices: nl.ndarray,
    counts: nl.ndarray,
    T: int,
    EP: int,
) -> None:
    """Assert the shape / dtype contract shared by ``permute_a2av`` and
    ``unpermute_a2av``.

    Checks:
      - ``send_indices`` is ``[T, EP]`` int32.
      - ``counts`` is ``[1, EP]`` int32/uint32.
      - ``EP`` and ``T`` are positive (``EP >= 2`` because the scan helper
        needs at least two slots).

    Args:
        send_indices (nl.ndarray): [T, EP] int32 HBM tensor. MoE routing
            indices shared across dispatch and combine.
        counts (nl.ndarray): [1, EP] int32/uint32 HBM tensor. ``send_counts``
            for dispatch, ``recv_counts`` for combine.
        T (int): number of local tokens (compile-time).
        EP (int): collective world size (compile-time).
    """
    kernel_assert(
        T >= 1,
        f"T must be >= 1, got T={T}",
    )
    kernel_assert(
        EP >= 2,
        f"EP must be >= 2 (exclusive-cumsum scan requires at least two slots), got EP={EP}",
    )
    kernel_assert(
        tuple(send_indices.shape) == (T, EP),
        f"send_indices must be (T={T}, EP={EP}), got {tuple(send_indices.shape)}",
    )
    kernel_assert(
        send_indices.dtype == nl.int32,
        f"send_indices must be int32, got {send_indices.dtype}",
    )
    kernel_assert(
        tuple(counts.shape) == (1, EP),
        f"counts must be (1, EP={EP}), got {tuple(counts.shape)}",
    )
    kernel_assert(
        counts.dtype in (nl.int32, nl.uint32),
        f"counts must be int32/uint32, got {counts.dtype}",
    )
