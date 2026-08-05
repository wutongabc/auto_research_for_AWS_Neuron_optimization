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

"""PyTorch reference implementation for rmsnorm_mx_quantize_tkg kernel.

Implements fused RMSNorm + MX quantization matching the hardware kernel output
layout. The quantized outputs (out_quant, out_scale) are returned as numpy arrays
with x4/uint8 dtypes since torch_ref_wrapper cannot convert these types.

Note: x4 packed numpy arrays are the only numpy outputs — they are numpy because
torch has no equivalent dtype. All other outputs are torch tensors.
"""

from typing import Optional

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import torch

from ..utils.mx_torch_common import quantize_mx_golden

_QMX_OUTPUT_DTYPES = [nl.float8_e4m3fn_x4, nl.float8_e5m2_x4]
_MX_UNPACKED_PACKED_MAP = {
    nl.float8_e4m3fn: dt.float8_e4m3fn_x4,
    nl.float8_e5m2: dt.float8_e5m2_x4,
}


def rmsnorm_mx_quantize_tkg_torch_ref(
    inp: torch.Tensor,
    gamma: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    hidden_actual: Optional[int] = None,
    output_quant_in_sbuf: bool = True,
    output_quant_packed: bool = False,
    _out_quant_dtype=nl.float8_e4m3fn_x4,
) -> dict:
    """PyTorch reference for fused RMSNorm + MX quantization TKG kernel.

    Matches the signature and output layout of rmsnorm_mx_quantize_tkg_wrapper.

    Args:
        inp (torch.Tensor): [B, S, H], Input tensor.
        gamma (torch.Tensor): [1, H], RMSNorm weight.
        residual (Optional[torch.Tensor]): [B, S, H], Optional residual tensor.
        hidden_actual (Optional[int]): Actual hidden dimension for padded inputs.
        output_quant_in_sbuf (bool): Unused, for signature compatibility with kernel.
        output_quant_packed (bool):
            When True, out_quant contains transposed and concatenated quantized tensor and scale, with shape [BxS, H * 5/4] and out_scale is not returned.
                In this case, the quantized tensor uses an unpacked FP8 dtype, and out_scale is bitcast to the same FP8 dtype as the quantized tensor.
            When False, out_quant and out_scale have layout [H0, num_H512_tiles, BxS].
        _out_quant_dtype: output quantize dtype (float8_e4m3fn_x4, float8_e5m2_x4, float8_e4m3fn, or float8_e5m2)

    Returns:
        dict with keys:
            out: torch.Tensor [H0, BxS, H1] — normalized output
            out_quant: numpy x4 dtype [H0, num_H512_tiles, BxS] or numpy fp8 [BxS, H * 5/4] — MX quantized
            out_scale: numpy uint8 [H0, num_H512_tiles, BxS] (only when output_quant_packed=False)
            out_residual: torch.Tensor [BxS, H] — (only when residual is provided)
    """
    B, S, H = inp.shape
    BxS = B * S
    H0, H1 = 128, H // 128
    _q_width = 4
    num_H512_tiles = H1 // _q_width

    if residual is not None:
        hidden = inp + residual
        out_residual = hidden.reshape(BxS, H)
    else:
        hidden = inp
        out_residual = None

    """
    RMSNorm in f32 for numerical accuracy.

    Hardware computes (input*gamma) in f16 then *rsqrt in f16, but matching this exactly
    in torch is not feasible (tensor engine rounding differs from IEEE 754). The f32 golden
    is close enough for `out` validation (within 1% rtol). For `out_quant`, small f16
    precision differences get amplified by MX block exponent quantization, so `out_quant`
    uses a skip validator in the test (correctness follows from `out` + `out_scale` passing
    plus quantize_mx_golden matching nisa.quantize_mx exactly).
    """
    hidden_f32 = hidden.float()
    gamma_f32 = gamma.float()

    if hidden_actual != None:
        sum_squares = torch.sum(hidden_f32**2, dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(sum_squares / hidden_actual + 1e-6)
    else:
        inv_rms = torch.rsqrt(torch.mean(hidden_f32**2, dim=-1, keepdim=True) + 1e-6)

    norm_f32 = hidden_f32 * inv_rms * gamma_f32
    result_reshaped = norm_f32.to(torch.float16).reshape(BxS, H1, H0).permute(2, 0, 1)

    """
    MX quantization via quantize_mx_golden (handles scale→hardware quadrant layout).
    Use f16→f32 values: f16 matches what hardware feeds to nisa.quantize_mx,
    f32 avoids numpy dtype issues in the golden math.
    """
    result_f32 = result_reshaped.float()
    qmx_input_2D = result_f32.reshape(H0, BxS, _q_width, num_H512_tiles).permute(0, 3, 1, 2).reshape(H0, -1)

    qmx_dtype = (
        _out_quant_dtype if _out_quant_dtype in _QMX_OUTPUT_DTYPES else _MX_UNPACKED_PACKED_MAP[_out_quant_dtype]
    )
    out_data_dummy = np.empty((H0, qmx_input_2D.shape[1] // _q_width), dtype=qmx_dtype)
    out_scale_dummy = np.empty((H0, qmx_input_2D.shape[1] // _q_width), dtype=np.uint8)
    mx_result = quantize_mx_golden(qmx_input_2D.numpy(), out_data_dummy, out_scale_dummy)

    mx_data = mx_result["out_data_hbm"].reshape(H0, num_H512_tiles, BxS)
    out_scale = mx_result["out_scale_hbm"].reshape(H0, num_H512_tiles, BxS)

    # Optionally pack output as [BxS, H * 5/4] in fp8
    if output_quant_packed:
        # [128_H, H/512, BxS]fp8x4 -> [BxS, H/512, 128_H]fp8x4 -> [BxS, H/512, 128_H * 4_H]fp8 -> [BxS, H/512 * 128_H * 4_H]fp8 = [BxS, H]fp8
        mx_data = np.ascontiguousarray(np.transpose(mx_data, (2, 1, 0))).view(_out_quant_dtype).reshape(BxS, H)

        # [128_H, H/512, BxS]u8 -> [BxS, H/512, 128_H]u8 -> [BxS, H/512, 128_H]fp8 -> [BxS, H/512 * 128_H]fp8 = [Bxs, H/4]fp8
        out_scale = np.ascontiguousarray(np.transpose(out_scale, (2, 1, 0))).view(_out_quant_dtype).reshape(BxS, H // 4)

        # [BxS, H/512 * 128_H * 4_H + H/512 * 128_H]fp8 = [BxS, H * 5/4]fp8
        mx_data = np.concatenate((mx_data, out_scale), axis=1)

    result_dict = {
        "out": result_reshaped,
        "out_quant": mx_data,
    }
    if not output_quant_packed:
        result_dict["out_scale"] = out_scale
    if out_residual is not None:
        result_dict["out_residual"] = out_residual.to(inp.dtype)
    return result_dict
