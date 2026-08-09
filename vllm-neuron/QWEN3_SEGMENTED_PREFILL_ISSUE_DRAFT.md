# Issue draft: Add native Qwen3 and Qwen3MoE support

## Suggested title

Add native BF16 Qwen3/Qwen3MoE support, including segmented prefill and prefix caching

## Motivation

vLLM Neuron currently has production-oriented Llama and GPT-OSS paths but no
registered text implementation for `Qwen3ForCausalLM` or
`Qwen3MoeForCausalLM`. This prevents Qwen3 dense and sparse checkpoints from
running without a downstream model port, and long-context serving additionally
requires the backend's segmented-prefill path.

## Proposed scope

- Register `Qwen3ForCausalLM` and `Qwen3MoeForCausalLM`.
- Implement BF16 Qwen3 attention with Q/K RMSNorm, GQA, RoPE, paged KV cache,
  full prefill, segmented prefill, and fused decode.
- Implement dense SwiGLU and Qwen3MoE routing/expert paths for prefill/decode.
- Support TP and the existing EP path.
- Read both legacy `rope_theta` and current Transformers `rope_parameters`.
- Compute RoPE phase in FP32 before casting cos/sin to BF16.
- Reuse the existing segmented-attention kernel and scheduler/APC machinery.

## Why this is not a new attention kernel

GPT-OSS controls with one and two KV heads per rank and `head_dim=128` match
their Hugging Face references. Tiny Qwen controls covering MHA, GQA, Q/K norm,
and the production Qwen head layout also match. The Qwen-specific failures were
caused by a missing 1M RoPE base and BF16 position-phase aliasing, not by the
AWS segmented-attention kernel.

## Acceptance criteria

- Qwen3-0.6B TP=2 compiles and warms prefill/decode on Trn2.
- A prompt longer than the 512-token bucket runs through multiple prefill
  segments and matches HF BF16 under greedy decoding.
- Repeated long prompts use prefix caching without changing outputs.
- A native tiny Qwen3MoE fixture compiles segmented prefill/decode and matches
  HF BF16 under teacher forcing.
- CPU unit tests cover modern/legacy RoPE config parsing and FP32 phase math.

## Validation completed

- Dense Qwen3: 1509-token prompt, 512-token segment, three segments, 16/16 HF
  token match; repeated request also 16/16, with 46.7% APC hit rate.
- Dense boundary tests: 508, 521, and 807 tokens matched 16/16.
- Tiny Qwen3MoE: 807-token prompt, 32 experts/top-8, 8/8 teacher-forced token
  match; repeated-prefix test reached 90.9% APC hit rate.

## Known limitations / follow-ups

- Scope is Qwen3/Qwen3MoE text, not Qwen2/Qwen2.5 or multimodal Qwen.
- BF16 and `head_dim=128` only; no FP8 KV cache, DCP, sliding window, sinks, or
  speculative decoding in this model implementation.
- Segmented prefill currently inherits batch-size-1/no-mixed-prefill-decode
  backend restrictions.
- A 989-token adversarial prompt changes only the first greedy token versus
  full prefill; HF's top-two margin is 0.25 BF16 logits and teacher forcing
  matches the remaining 15/15 steps. Keep this in model accuracy qualification.
- TP=2 on-device-sampler `logprobs` returns five top logprobs per generated
  step; first-step top-5 token IDs match HF BF16 exactly.
