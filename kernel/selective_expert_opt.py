# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selective-expert TKG variant that fuses first-expert scaling into init."""

_BASE_PATCH = (
    "/dev3/zigeng/bc/BrowseComp-Plus/patches/nkilib_tkg_swdge/"
    "selective_expert_impl.py"
)

with open(_BASE_PATCH, "r", encoding="utf-8") as _source_file:
    _source = _source_file.read()

_old = '''            if params.expert_params.expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
                # Apply affinity and accumulate to SB
                nisa.tensor_scalar(
                    dst=down_sb,
                    data=down_sb,
                    op0=nl.multiply,
                    operand0=expert_affinity_sb[:, expert_k_idx],
                )
            if expert_k_idx == 0:
                nisa.tensor_copy(dst=output_temp[0 : dims.H0, 0 : dims.H1_shard, local_token_idx], src=down_sb)
            else:
                nisa.tensor_tensor(
                    dst=output_temp[0 : dims.H0, 0 : dims.H1_shard, local_token_idx],
                    data1=output_temp[0 : dims.H0, 0 : dims.H1_shard, local_token_idx],
                    data2=down_sb,
                    op=nl.add,
                )
'''

_new = '''            output_accum = output_temp[0 : dims.H0, 0 : dims.H1_shard, local_token_idx]
            if params.expert_params.expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
                if expert_k_idx == 0:
                    nisa.tensor_scalar(
                        dst=output_accum,
                        data=down_sb,
                        op0=nl.multiply,
                        operand0=expert_affinity_sb[:, expert_k_idx],
                    )
                else:
                    nisa.tensor_scalar(
                        dst=down_sb,
                        data=down_sb,
                        op0=nl.multiply,
                        operand0=expert_affinity_sb[:, expert_k_idx],
                    )
                    nisa.tensor_tensor(
                        dst=output_accum,
                        data1=output_accum,
                        data2=down_sb,
                        op=nl.add,
                    )
            elif expert_k_idx == 0:
                nisa.tensor_copy(dst=output_accum, src=down_sb)
            else:
                nisa.tensor_tensor(
                    dst=output_accum,
                    data1=output_accum,
                    data2=down_sb,
                    op=nl.add,
                )
'''

if _old not in _source:
    raise RuntimeError("selective-expert fusion anchor not found")
_source = _source.replace(_old, _new, 1)
exec(compile(_source, _BASE_PATCH, "exec"), globals(), globals())
