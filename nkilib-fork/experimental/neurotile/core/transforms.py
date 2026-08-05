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
from ._helpers import contiguous_strides, insert_at

# ============================================================================
# Transforms -- all return (new_element_shape, new_strides)
# ============================================================================


def compute_reshape_dim(element_shape, strides, dim, new_sub_shape):
    """Split one dimension into multiple sub-dimensions.

    element_shape[dim] is replaced by new_sub_shape. Strides are recomputed
    by subdividing strides[dim] from innermost outward.
    """
    base_stride = strides[dim]

    # Sub-strides from innermost outward, then reverse to outermost-first
    sub_strides = []
    running_stride = base_stride
    for i in range(len(new_sub_shape) - 1, -1, -1):
        sub_strides.append(running_stride)
        running_stride = running_stride * new_sub_shape[i]
    sub_strides.reverse()

    # Build output: replace dim with sub-dims, keep others unchanged
    new_shape = []
    new_strides = []
    for d in range(len(element_shape)):
        if d == dim:
            for si in range(len(new_sub_shape)):
                new_shape.append(new_sub_shape[si])
                new_strides.append(sub_strides[si])
        else:
            new_shape.append(element_shape[d])
            new_strides.append(strides[d])

    return (tuple(new_shape), tuple(new_strides))


def compute_reshape(element_shape, strides, new_shape):
    """Full reshape (requires contiguous layout).

    Computes new contiguous strides, preserving the base stride (strides[-1])
    for non-unit innermost dimensions.
    """
    new_strides = contiguous_strides(new_shape)

    base = strides[-1] if len(strides) > 0 else 1
    if base != 1:
        scaled = []
        for s in new_strides:
            scaled.append(s * base)
        new_strides = tuple(scaled)

    return (tuple(new_shape), new_strides)


def compute_permute(element_shape, strides, order):
    """Reorder dimensions.

    order[i] = which old dim goes to position i.
    """
    new_shape = []
    new_strides = []
    for i in range(len(order)):
        new_shape.append(element_shape[order[i]])
        new_strides.append(strides[order[i]])
    return (tuple(new_shape), tuple(new_strides))


def compute_flatten_dims(element_shape, strides, start, end):
    """Merge dimensions [start..end] (inclusive) into one.

    The merged dimension gets the innermost stride (strides[end]).
    """
    merged_size = 1
    for d in range(start, end + 1):
        merged_size = merged_size * element_shape[d]

    new_shape = []
    new_strides = []
    for d in range(len(element_shape)):
        if d == start:
            new_shape.append(merged_size)
            new_strides.append(strides[end])
        elif d > start and d <= end:
            pass  # merged into start
        else:
            new_shape.append(element_shape[d])
            new_strides.append(strides[d])

    return (tuple(new_shape), tuple(new_strides))


def compute_squeeze_dim(element_shape, strides, dim):
    """Remove a size-1 dimension."""
    assert element_shape[dim] == 1
    new_shape = []
    new_strides = []
    for d in range(len(element_shape)):
        if d != dim:
            new_shape.append(element_shape[d])
            new_strides.append(strides[d])
    return (tuple(new_shape), tuple(new_strides))


def compute_expand_dim(element_shape, strides, dim):
    """Insert a size-1 dimension with stride 0 (broadcastable)."""
    return (
        insert_at(element_shape, dim, 1),
        insert_at(strides, dim, 0),
    )


def compute_broadcast(element_shape, strides, dim, size):
    """Broadcast a size-1 dimension to given size (stride becomes 0)."""
    assert element_shape[dim] == 1
    new_shape = []
    new_strides = []
    for d in range(len(element_shape)):
        if d == dim:
            new_shape.append(size)
            new_strides.append(0)
        else:
            new_shape.append(element_shape[d])
            new_strides.append(strides[d])
    return (tuple(new_shape), tuple(new_strides))


def compute_fold(element_shape, strides, src_dim, into_dim, position="outer"):
    """Fold src_dim into into_dim (removes src_dim).

    If position="outer": into_dim size = src * into, stride = into's stride.
    If position="inner": into_dim size = into * src, stride = src's stride.

    The stride is kept from the into_dim (outer) or src_dim (inner) to
    preserve the HBM AP pattern for store operations. For load, the SBUF
    AP is built contiguously regardless.
    """
    assert src_dim != into_dim
    if position == "outer":
        new_size = element_shape[src_dim] * element_shape[into_dim]
        new_stride = strides[into_dim]
    else:
        new_size = element_shape[into_dim] * element_shape[src_dim]
        new_stride = strides[src_dim]

    new_shape = []
    new_strides = []
    for d in range(len(element_shape)):
        if d == src_dim:
            pass  # folded away
        elif d == into_dim:
            new_shape.append(new_size)
            new_strides.append(new_stride)
        else:
            new_shape.append(element_shape[d])
            new_strides.append(strides[d])
    return (tuple(new_shape), tuple(new_strides))
