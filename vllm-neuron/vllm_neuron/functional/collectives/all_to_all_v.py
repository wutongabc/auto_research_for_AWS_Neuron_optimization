# SPDX-License-Identifier: Apache-2.0
import os
from typing import Optional, Tuple

import torch

import nki
import nki.isa as nisa
import nki.collectives as ncc
import nki.language as nl

from vllm.distributed.parallel_state import GroupCoordinator
from vllm_neuron import envs
from vllm_neuron.nki.nki_hop import wrap_nki


def all_to_all_v(
    input: torch.Tensor,
    output: torch.Tensor,
    group: GroupCoordinator,
    metadata: torch.Tensor,
    recv_counts_known: bool = False,
    has_rdispls: bool = False,
    priority: Optional[int] = None,
    cc_use_intermediate_io: bool = False,
) -> torch.Tensor:
    """Perform a variable-length all-to-all on the given replica group and input/output tensors.

    Unlike all_to_all which splits and concatenates along a collective_dim, all_to_all_v treats tensors as flat buffers of elements.
    Counts and displacements in the metadata tensor are in elements (row-major order), not slices along a particular dimension.

    This API is a thin wrapper on top of the nki.collectives.all_to_all_v instruction, which executes the collective.

    Args:
        input: Input tensor to redistribute.
        output: Output tensor to store result.
        group: GroupCoordinator for ranks that the collective will be executed across.
        metadata: Metadata tensor of shape (4, world_size), dtype uint32.
            Row 0: send counts, Row 1: send displacements,
            Row 2: recv counts (can be empty), Row 3: recv displacements (can be empty).
        recv_counts_known: Whether the collective should populate row 2 of metadata with recv_counts.
        has_rdispls: Not yet supported. Whether row 3 of metadata contains real recv_displs.
        priority: Not yet supported. DMA quality-of-service priority level 0-3 where lower is higher (Trn3+ only).
        cc_use_intermediate_io: Whether all_to_all_v should utilize intermediate buffers for collective I/O,
            which comes with a small performance penalty. Necessary in some cases, since NEFF I/O cannot also be collective I/O.

    Returns:
        output: Output tensor populated with data from the collective operation.
        metadata: Original metadata tensor, with recv_counts (row 2) optionally populated by the collective.

    Example:
        >>> # world_size=8, each rank sends 4 elements to every other rank
        >>> # input shape: (32,), output shape: (32,), metadata shape: (4, 8)
        >>> metadata = torch.zeros(4, 8, dtype=torch.uint32)
        >>> metadata[0] = torch.tensor([4, 4, 4, 4, 4, 4, 4, 4])  # send counts
        >>> metadata[1] = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28])  # send displs
        >>> output, metadata = all_to_all_v(input, output, group, metadata)
        >>> # output[0:4] = data from rank 0, output[4:8] = data from rank 1, ...
    """

    # Convert from GroupCoordinator -> tuple[int]
    group = tuple(group.ranks)

    _validate_all_to_all_v(group, metadata, priority)

    wrapped = wrap_nki(_all_to_all_v_nki)

    return wrapped[2](
        input=input,
        output=output,
        group=group,
        metadata=metadata,
        recv_counts_known=recv_counts_known,
        has_rdispls=has_rdispls,
        priority=priority,
        cc_use_intermediate_io=cc_use_intermediate_io,
    )


def _validate_all_to_all_v(group, metadata, priority):
    """
    Check if the collective inputs are valid.

    Constraints:
        - NeuronSwitch must be used (VLLM_NEURON_SWITCH_CC=1)
        - Not using CPU mode (NKI simulation does not support collectives yet)
        - Priority must be None (not yet supported)
        - Metadata must have uint32 dtype
        - Metadata must have 4 rows
        - Metadata must have the same number of columns as group.size
        - Group size must be at least 8
    """

    assert not envs.VLLM_NEURON_CPU_MODE, (
        f"all_to_all_v collective is not supported on CPU mode, got {envs.VLLM_NEURON_CPU_MODE=}"
    )
    assert priority is None, (
        f"all_to_all_v collective does not yet support priority != None, but got {priority=}"
    )
    assert metadata.dtype in (torch.uint32, "uint32"), (
        f"Expected metadata.dtype == torch.uint32, but got {metadata.dtype=}"
    )
    assert len(metadata.shape) == 2, f"Expected 2D metadata, but got {metadata.shape=}"
    assert metadata.shape[0] == 4, (
        f"Expected dim0 of metadata of size 4, but got {metadata.shape=}"
    )
    assert metadata.shape[1] == len(group), (
        f"Expected dim1 of metadata equal to process group size, but got {metadata.shape=}, {len(group)=}"
    )
    is_inter_node_trn2 = "NEURON_RT_ROOT_COMM_ID" in os.environ
    min_group_size = 2 if is_inter_node_trn2 else 8
    assert len(group) >= min_group_size, (
        f"all_to_all_v collective requires at least {min_group_size} ranks in collective group, but got {len(group)=}"
    )


@nki.jit
def _all_to_all_v_nki(
    input: nl.ndarray,
    output: nl.ndarray,
    group: Tuple[int],
    metadata: nl.ndarray,
    recv_counts_known: bool = False,
    has_rdispls: bool = False,
    priority: Optional[int] = None,
    cc_use_intermediate_io: bool = False,
) -> nl.ndarray:
    """Thin wrapper of nki.collectives.all_to_all_v, which executes a variable-length all-to-all collective."""
    # Workaround: XLA may deliver metadata as int32 instead of uint32 due to
    # XLA's lack of native unsigned int support. The ncc.all_to_all_v collective
    # requires uint32 metadata. Bitcast unconditionally — the output must remain
    # uint32 to satisfy HLO operand_output_aliases.
    if metadata.dtype != nl.uint32:
        metadata = metadata.view(nl.uint32)

    # Convert from tuple[int] to list[list[int]]
    replica_group = ncc.ReplicaGroup([list(group)])

    # Legalize 1D input tensors to 2D
    original_input_shape = input.shape
    original_output_shape = output.shape
    new_input_shape = input.shape if len(input.shape) > 1 else (input.shape[0], 1)
    new_output_shape = output.shape if len(output.shape) > 1 else (output.shape[0], 1)
    input = input.reshape(new_input_shape)
    output = output.reshape(new_output_shape)

    # NEFF I/O cannot be collective I/O; when kernel I/O is NEFF I/O, copy to intermediate buffers before calling collective
    if cc_use_intermediate_io:
        cc_input = nl.ndarray(input.shape, input.dtype, buffer=nl.shared_hbm)
        cc_output = nl.ndarray(output.shape, output.dtype, buffer=nl.shared_hbm)
        cc_metadata = nl.ndarray(metadata.shape, metadata.dtype, buffer=nl.shared_hbm)

        nisa.dma_copy(cc_input, input)
        nisa.dma_copy(cc_output, output)
        nisa.dma_copy(cc_metadata, metadata)
    else:
        cc_input = input
        cc_output = output
        cc_metadata = metadata

    # Call collective
    ncc.all_to_all_v(
        srcs=[cc_input],
        dsts=[cc_output],
        replica_group=replica_group,
        metadata_tensor=cc_metadata,
        recv_counts_known=recv_counts_known,
        has_rdispls=has_rdispls,
        priority=priority,
    )

    # When kernel I/O is NEFF I/O, copy back from intermediate buffers after calling collective
    if cc_use_intermediate_io:
        nisa.dma_copy(output, cc_output)
        nisa.dma_copy(metadata, cc_metadata)

    # Reset I/O to original shapes
    input = input.reshape(original_input_shape)
    output = output.reshape(original_output_shape)

    return output, metadata
