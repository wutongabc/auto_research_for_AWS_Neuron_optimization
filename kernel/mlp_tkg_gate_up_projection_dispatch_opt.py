"""Dispatch selective-expert LHS/RHS projection to the SBUF-scale reuse path."""

from .mlp_tkg_gate_up_projection import (
    _post_process_gate_up,
    run_gate_up_projection_non_lhs_rhs_swap,
)
from .mlp_tkg_gate_up_projection_lhs_rhs_swap_opt import (
    run_gate_up_projection_lhs_rhs_swap,
)
from ...utils.kernel_assert import kernel_assert


def process_gate_up_projection(
    hidden,
    output,
    params,
    dims,
    sbm,
    T_offset=0,
    share_memory_scope=False,
    use_dge=False,
):
    if params.use_tkg_gate_up_proj_column_tiling:
        kernel_assert(
            not share_memory_scope,
            "share_memory_scope is not supported with column tiling",
        )
        (
            tiles,
            gate_sb_view,
            up_sb_view,
            gate_up_recv,
            use_fused_gate_up_sendrecv,
            gate_up_sb_fp32,
        ) = run_gate_up_projection_non_lhs_rhs_swap(
            hidden=hidden,
            output=output,
            params=params,
            dims=dims,
            sbm=sbm,
        )
    else:
        tiles, gate_sb_view, up_sb_view, gate_up_recv = (
            run_gate_up_projection_lhs_rhs_swap(
                hidden=hidden,
                output=output,
                params=params,
                dims=dims,
                sbm=sbm,
                T_offset=T_offset,
                share_memory_scope=share_memory_scope,
                use_dge=use_dge,
            )
        )
        use_fused_gate_up_sendrecv = False
        gate_up_sb_fp32 = None

    return _post_process_gate_up(
        output=output,
        params=params,
        dims=dims,
        sbm=sbm,
        gate_sb_view=gate_sb_view,
        up_sb_view=up_sb_view,
        gate_up_recv=gate_up_recv,
        use_fused_gate_up_sendrecv=use_fused_gate_up_sendrecv,
        gate_up_sb_fp32=gate_up_sb_fp32,
        tiles=tiles,
    )
