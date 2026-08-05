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
"""Torch reference for unpermute_a2av kernel."""

import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from ..collectives_torch import _to_numpy, _to_torch, get_pg, get_rank


def unpermute_a2av_torch_ref(
    output: np.ndarray,
    send_indices: np.ndarray,
    recv_counts: np.ndarray,
    replica_group: ReplicaGroup,
) -> dict:
    """Torch ref: repack output, all-to-all-v, scatter-add to original positions."""
    T = send_indices.shape[0]
    EP = send_indices.shape[1]
    _, H = output.shape
    dtype = output.dtype
    rank = get_rank()
    pg = get_pg(replica_group)

    # Step 1: Pack output rows into send buffer using recv_counts as send sizes.
    send_chunks = []
    send_sizes = []
    for src in range(EP):
        count = int(recv_counts[0, src])
        n_elems = count * H
        send_sizes.append(n_elems)
        if count > 0:
            send_chunks.append(_to_torch(output[src * T : src * T + count, :]).reshape(-1))
        else:
            send_chunks.append(torch.zeros(0, dtype=_to_torch(output[:1]).reshape(-1).dtype))

    send_flat = torch.cat(send_chunks)

    # Step 2: Exchange sizes to know recv_sizes.
    send_sizes_t = torch.tensor(send_sizes, dtype=torch.long)
    recv_sizes_t = torch.zeros(EP, dtype=torch.long)
    dist.all_to_all_single(recv_sizes_t, send_sizes_t, group=pg)
    recv_sizes = [int(recv_sizes_t[d]) for d in range(EP)]

    # Step 3: Variable-length all-to-all exchange.
    recv_total = sum(recv_sizes)
    recv_flat = torch.zeros(recv_total, dtype=send_flat.dtype)
    dist.all_to_all_single(recv_flat, send_flat, output_split_sizes=recv_sizes, input_split_sizes=send_sizes, group=pg)

    # Step 4: Scatter-add received rows to original token positions.
    result = np.zeros((T, H), dtype=dtype)
    offset = 0
    for d in range(EP):
        n_rows = recv_sizes[d] // H
        if n_rows > 0:
            chunk = _to_numpy(recv_flat[offset : offset + recv_sizes[d]].reshape(n_rows, H), dtype)
            idx = send_indices[:n_rows, d].astype(np.intp)
            for r in range(n_rows):
                if 0 <= idx[r] < T:
                    result[idx[r], :] += chunk[r, :]
        offset += recv_sizes[d]

    return {"result": result}
