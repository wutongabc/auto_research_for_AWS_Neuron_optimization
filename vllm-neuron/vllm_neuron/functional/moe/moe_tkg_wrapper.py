# SPDX-License-Identifier: Apache-2.0
import nki
import nki.language as nl
from typing import Optional

from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    MoEAllToAllVStrategy,
)

from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
from nkilib.core.utils.tensor_view import TensorView


# TODO: this wrapper needs to be upstreamed into NKL.
@nki.jit
def moe_tkg_wrapper(
    hidden_input: nl.ndarray,
    expert_gate_up_weights: nl.ndarray,
    expert_down_weights: nl.ndarray,
    expert_affinities: nl.ndarray,
    expert_index: nl.ndarray,
    is_all_expert: bool,
    rank_id: Optional[nl.ndarray] = None,
    expert_gate_up_bias: Optional[nl.ndarray] = None,
    expert_down_bias: Optional[nl.ndarray] = None,
    expert_gate_up_weights_scale: Optional[nl.ndarray] = None,
    expert_down_weights_scale: Optional[nl.ndarray] = None,
    hidden_input_scale: Optional[nl.ndarray] = None,
    expert_gate_up_input_scale: Optional[nl.ndarray] = None,
    expert_down_input_scale: Optional[nl.ndarray] = None,
    mask_unselected_experts: bool = False,
    expert_affinities_eager: Optional[nl.ndarray] = None,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.NO_SCALE,
    activation_fn: ActFnType = ActFnType.SiLU,
    output_dtype: nki.dtype = None,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    output_in_sbuf: bool = False,
    is_all_expert_dynamic: bool = False,
    block_size: int = None,
    input_dequant_scale: Optional[nl.ndarray] = None,
    all_to_all_v_strategy: MoEAllToAllVStrategy = MoEAllToAllVStrategy.DISABLED,
):
    """
    Thin wrapper around moe_block_tkg that handles uint16 -> float4_e2m1fn_x4 and uint32 -> float8_e4m3fn_x4 dtype reinterpretation.
    """

    # Handle uint16 -> float4_e2m1fn_x4 conversion
    if expert_gate_up_weights.dtype == nl.uint16:
        expert_gate_up_weights = TensorView(expert_gate_up_weights).reinterpret_cast(
            nl.float4_e2m1fn_x4
        )
    if expert_down_weights.dtype == nl.uint16:
        expert_down_weights = TensorView(expert_down_weights).reinterpret_cast(
            nl.float4_e2m1fn_x4
        )

    # Handle uint32 -> float8_e4m3fn_x4 conversion
    if expert_gate_up_weights.dtype == nl.uint32:
        expert_gate_up_weights = TensorView(expert_gate_up_weights).reinterpret_cast(
            nl.float8_e4m3fn_x4
        )
    if expert_down_weights.dtype == nl.uint32:
        expert_down_weights = TensorView(expert_down_weights).reinterpret_cast(
            nl.float8_e4m3fn_x4
        )

    return moe_tkg(
        hidden_input=hidden_input,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        is_all_expert=is_all_expert,
        rank_id=rank_id,
        expert_gate_up_bias=expert_gate_up_bias,
        expert_down_bias=expert_down_bias,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        hidden_input_scale=hidden_input_scale,
        expert_gate_up_input_scale=expert_gate_up_input_scale,
        expert_down_input_scale=expert_down_input_scale,
        mask_unselected_experts=mask_unselected_experts,
        expert_affinities_eager=expert_affinities_eager,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        activation_fn=activation_fn,
        output_dtype=output_dtype,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        output_in_sbuf=output_in_sbuf,
        is_all_expert_dynamic=is_all_expert_dynamic,
        block_size=block_size,
        input_dequant_scale=input_dequant_scale,
        all_to_all_v_strategy=all_to_all_v_strategy,
    )
