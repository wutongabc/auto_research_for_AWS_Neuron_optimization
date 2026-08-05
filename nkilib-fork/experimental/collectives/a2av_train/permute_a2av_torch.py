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
"""Torch reference for permute_a2av kernel."""

import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from ..collectives_torch import _to_numpy, _to_torch, get_pg, get_rank


def permute_a2av_torch_ref(
    hidden_states: np.ndarray,
    send_indices: np.ndarray,
    send_counts: np.ndarray,
    replica_group: ReplicaGroup,
) -> dict:
    """Torch ref: gather by send_indices, pack, all-to-all-v, return recv_data + metadata."""
    T, H = hidden_states.shape
    EP = send_indices.shape[1]
    dtype = hidden_states.dtype
    rank = get_rank()
    pg = get_pg(replica_group)

    # Step 1: Gather tokens into packed send buffer per destination.
    send_chunks = []
    send_sizes = []
    for ep_dst in range(EP):
        count = int(send_counts[0, ep_dst])
        send_sizes.append(count * H)
        if count > 0:
            idx = send_indices[:count, ep_dst].astype(np.intp)
            send_chunks.append(_to_torch(hidden_states[idx, :]).reshape(-1))
        else:
            send_chunks.append(torch.zeros(0, dtype=_to_torch(hidden_states[:1]).reshape(-1).dtype))

    send_flat = torch.cat(send_chunks)

    # Step 2: Exchange sizes to know recv_sizes.
    send_sizes_t = torch.tensor(send_sizes, dtype=torch.long)
    recv_sizes_t = torch.zeros(EP, dtype=torch.long)
    dist.all_to_all_single(recv_sizes_t, send_sizes_t, group=pg)
    recv_sizes = [int(recv_sizes_t[s]) for s in range(EP)]

    # Step 3: Variable-length all-to-all exchange.
    recv_total = sum(recv_sizes)
    recv_flat = torch.zeros(recv_total, dtype=send_flat.dtype)
    dist.all_to_all_single(recv_flat, send_flat, output_split_sizes=recv_sizes, input_split_sizes=send_sizes, group=pg)

    # Step 4: Place received data into [EP*T, H] layout.
    recv_data = np.zeros((EP * T, H), dtype=dtype)
    offset = 0
    for src in range(EP):
        n_rows = recv_sizes[src] // H
        if n_rows > 0:
            recv_data[src * T : src * T + n_rows, :] = _to_numpy(
                recv_flat[offset : offset + recv_sizes[src]].reshape(n_rows, H), dtype
            )
        offset += recv_sizes[src]

    # Step 5: Build metadata [4, EP] uint32.
    metadata = np.zeros((4, EP), dtype=np.uint32)
    metadata[0, :] = send_counts[0, :].astype(np.uint32) * H
    sdispls = np.concatenate([[0], np.cumsum(send_counts[0, :-1])]).astype(np.uint32)
    metadata[1, :] = sdispls * H
    metadata[2, :] = np.array([recv_sizes[s] for s in range(EP)], dtype=np.uint32)

    return {"recv_data": recv_data, "metadata": metadata}
