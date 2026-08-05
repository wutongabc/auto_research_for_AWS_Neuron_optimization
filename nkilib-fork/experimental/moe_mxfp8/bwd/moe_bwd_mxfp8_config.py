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

"""Configuration classes for MXFP8 MoE backward pass kernel."""

from dataclasses import dataclass

import nki
import nki.language as nl
from nki.dtype import float8_e4m3fn_x4

from ...matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from ...moe.bwd.moe_bwd_parameters import ActFnType, AffinityOption, ClampLimits, ShardOption, SkipMode


@dataclass
class MXFP8MOEBwdConfig(nl.NKIObject):
    """Configuration for the MXFP8 MoE backward pass kernel.

    Groups all tuning knobs and configuration options into a single dataclass.

    Args:
        compute_dtype (nki.dtype): Compute data type for intermediate results (default: nl.bfloat16).
        fp8_x4_dtype (type): MXFP8 packed data type (default: float8_e4m3fn_x4).
        activation_type (ActFnType): Activation function type (default: SiLU).
        shard_option (ShardOption): LNC2 sharding strategy (default: SHARD_ON_FREE).
        affinity_option (AffinityOption): Affinity scaling dimension (default: AFFINITY_ON_H).
        phase1_config (MatmulMxfp8KernelConfig): Matmul config for Phase 1 (gate_up_proj_output_grad + SwiGLU bwd + ea grad).
        phase2_config (MatmulMxfp8KernelConfig): Matmul config for Phase 2 (hidden_states_grad).
        phase3_config (MatmulMxfp8KernelConfig): Matmul config for Phase 3 (gate_up_weight_grad).
        phase4_config (MatmulMxfp8KernelConfig): Matmul config for Phase 4 (down_weight_grad).
        is_tensor_update_accumulating (bool): Whether to accumulate into existing gradients.
        skip_grad_initialization (bool): Whether to skip zeroing gradient outputs.
        clamp_limits (ClampLimits): Optional gradient clamping limits.
        skip_dma (SkipMode): OOB handling mode for DMA operations.
        bias (bool): Whether to compute bias gradients.
    """

    # Compute settings
    compute_dtype: nki.dtype = nl.bfloat16
    fp8_x4_dtype: type = float8_e4m3fn_x4
    activation_type: ActFnType = ActFnType.SiLU

    # Sharding & affinity
    shard_option: ShardOption = ShardOption.SHARD_ON_FREE
    affinity_option: AffinityOption = AffinityOption.AFFINITY_ON_H

    # Per-phase matmul configs. When None, defaults are constructed with
    # TILES_IN_BLOCK_*=1 and the remaining fields (M, K, N, bd, tile shapes)
    # left for the kernel entry point to fill from tensor shapes.
    phase1_config: MatmulMxfp8KernelConfig = None
    phase2_config: MatmulMxfp8KernelConfig = None
    phase3_config: MatmulMxfp8KernelConfig = None
    phase4_config: MatmulMxfp8KernelConfig = None

    # Accumulation
    is_tensor_update_accumulating: bool = True
    skip_grad_initialization: bool = False

    # Gradient clamping
    clamp_limits: ClampLimits = None

    # Skip DMA for OOB
    skip_dma: SkipMode = None

    # Bias
    bias: bool = False

    def __post_init__(self):
        """Initialize default matmul configs and optional fields."""
        if self.phase1_config == None:
            self.phase1_config = MatmulMxfp8KernelConfig(
                M=0,
                K=0,
                N=0,
                TILES_IN_BLOCK_M=1,
                TILES_IN_BLOCK_N=1,
                TILES_IN_BLOCK_K=1,
            )
        if self.phase2_config == None:
            self.phase2_config = MatmulMxfp8KernelConfig(
                M=0,
                K=0,
                N=0,
                TILES_IN_BLOCK_M=1,
                TILES_IN_BLOCK_N=1,
                TILES_IN_BLOCK_K=1,
            )
        if self.phase3_config == None:
            self.phase3_config = MatmulMxfp8KernelConfig(
                M=0,
                K=0,
                N=0,
                TILES_IN_BLOCK_M=1,
                TILES_IN_BLOCK_N=1,
                TILES_IN_BLOCK_K=1,
            )
        if self.phase4_config == None:
            self.phase4_config = MatmulMxfp8KernelConfig(
                M=0,
                K=0,
                N=0,
                TILES_IN_BLOCK_M=1,
                TILES_IN_BLOCK_N=1,
                TILES_IN_BLOCK_K=1,
            )
        if self.clamp_limits == None:
            self.clamp_limits = ClampLimits()
