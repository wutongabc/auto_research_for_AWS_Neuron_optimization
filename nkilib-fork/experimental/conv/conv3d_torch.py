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

"""PyTorch reference implementation for 3D convolution kernel testing."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.utils.common_types import ActFnType


def conv3d_torch_ref(
    x_in: torch.Tensor,
    filters: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    activation_fn: Optional[ActFnType] = None,
    lnc_shard: bool = False,
) -> dict[str, torch.Tensor]:
    """
    PyTorch reference implementation of 3D convolution kernel.

    Args:
        x_in (torch.Tensor): Input tensor of shape [B, C_in, D, H, W]
        filters (torch.Tensor): Filter weights of shape [K_d, K_h, K_w, C_in, C_out]
        bias (Optional[torch.Tensor]): Optional bias tensor of shape [C_out]
        stride (tuple[int, int, int]): Stride for convolution. Default (1, 1, 1).
        padding (tuple[int, int, int, int, int, int]):
            Flattened padding tuple (pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right).
            Default (0, 0, 0, 0, 0, 0).
        dilation (tuple[int, int, int]): Dilation factor for dilated convolution. Default (1, 1, 1).
        activation_fn (Optional[ActFnType]): Optional activation function type. Default None.
        lnc_shard (bool): Whether LNC sharding is enabled (unused in reference). Default False.

    Returns:
        dict[str, torch.Tensor]: Dictionary with key "out" containing output tensor of shape
            [B, C_out, D_out, H_out, W_out]
    """
    K_d, K_h, K_w, C_in, C_out = filters.shape
    pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding

    # Handle asymmetric padding by manually padding the input
    if pad_d_left != pad_d_right or pad_h_top != pad_h_bottom or pad_w_left != pad_w_right:
        x_in = F.pad(
            x_in, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_d_left, pad_d_right), mode="constant", value=0
        )
        conv_padding = (0, 0, 0)
    else:
        conv_padding = (pad_d_left, pad_h_top, pad_w_left)

    # Create nn.Conv3d module
    conv = nn.Conv3d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=(K_d, K_h, K_w),
        stride=stride,
        padding=conv_padding,
        dilation=dilation,
        bias=(bias is not None),
    )

    # Set weights: transpose from [K_d, K_h, K_w, C_in, C_out] to [C_out, C_in, K_d, K_h, K_w]
    with torch.no_grad():
        conv.weight.copy_(filters.permute(4, 3, 0, 1, 2))
        if bias is not None:
            conv.bias.copy_(bias)

    # Perform convolution
    output = conv(x_in)

    # Apply activation function if specified
    if activation_fn is not None:
        if activation_fn == ActFnType.SiLU:
            output = F.silu(output)
        elif activation_fn == ActFnType.GELU:
            output = F.gelu(output)
        elif activation_fn == ActFnType.GELU_Tanh_Approx:
            output = F.gelu(output, approximate="tanh")
        elif activation_fn == ActFnType.Swish:
            output = F.silu(output)
        elif activation_fn == ActFnType.ReLU:
            output = F.relu(output)
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

    return {"out": output.detach()}
