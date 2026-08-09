# Pull request draft: Native Qwen3/Qwen3MoE support

## Suggested title

feat(model): add native BF16 Qwen3 and Qwen3MoE support

## Summary

This change adds a native Neuron implementation shared by the Hugging Face
`Qwen3ForCausalLM` and `Qwen3MoeForCausalLM` architectures. It covers dense and
sparse models, TP/EP weight loading, full and segmented prefill, paged KV cache,
prefix caching, fused decode, and on-device sampling.

The implementation uses the existing vLLM Neuron segmented-attention API. No
Qwen-specific fork of the NKI attention kernel is introduced.

## Main changes

- Add and register the Qwen3/Qwen3MoE model module.
- Implement fused QKV projection with Q/K RMSNorm and Qwen RoPE.
- Add dense SwiGLU and Qwen3MoE router/expert execution for prefill/decode.
- Add Qwen expert weight loaders and TP/EP sharding.
- Write prefill K/V into the canonical paged cache and dispatch to segmented
  attention when `kv_segment_size` is present.
- Support both `rope_theta` and Transformers' modern `rope_parameters` shape.
- Keep position/frequency phase computation in FP32; cast only cos/sin outputs.
- Add config/RoPE unit regressions and update model usage/limitation docs.

## Root cause found during long-context validation

Official Qwen3 configs expose a 1,000,000 RoPE base through
`rope_parameters`. Falling back to 10,000 corrupts long-context attention.
Separately, casting positions to BF16 before multiplying by inverse frequencies
aliases integer positions above 256. Reading the modern config and retaining
FP32 phase math fixes both full and segmented prefill.

## Hardware validation

Environment: Trn2.3xlarge, vLLM Neuron 0.21.0, BF16.

- `Qwen/Qwen3-0.6B`, TP=2, max length 2048, segment 512:
  - 1509-token prompt executed three prefill segments.
  - Neuron greedy output matched HF CPU BF16 16/16 tokens.
  - Repeated request matched 16/16; APC hit rate was 46.7%.
- Boundary prompts at 508, 521, and 807 tokens matched 16/16.
- 989-token stress prompt: teacher forcing matched 15/16; only the first token
  swapped to HF's second-ranked candidate at a 0.25-logit margin. Full prefill
  matched 16/16. This is documented as a BF16 kernel-ordering accuracy edge.
- Tiny Qwen3MoE, TP=2, 2 layers, 32 experts, top-8, segment 512:
  - Compiled and warmed segmented prefill and fused decode.
  - 807-token teacher-forced comparison matched HF BF16 8/8.
  - Repeated-prefix run reached 90.9% APC hit rate.
- Unit tests: 11 passed for RoPE/config validation, padding-slot sanitization,
  TP sampling/logits gathering, and async ODS logprobs handling.

## Compatibility and limitations

- Qwen3/Qwen3MoE text only; Qwen2/Qwen2.5 and Qwen multimodal are out of scope.
- BF16 and decode `head_dim=128` only.
- No FP8 KV cache, context-parallel segmented attention, sliding window,
  attention sinks, or Eagle3/speculative decoding in this implementation.
- Segmented prefill is currently batch size 1 and cannot mix prefill/decode.
- Bias-enabled and non-QK-normalized variants are rejected as unsupported.
- Async on-device-sampler `logprobs` synchronizes gathered logits only when
  requested; token-only requests retain the existing async fast path.

## Review focus

- Paged-cache write and segmented-attention layout/aliasing.
- Qwen RoPE compatibility across supported Transformers versions.
- Qwen3MoE EP collectives and checkpoint sharding.
- Whether the 989-token BF16 ranking edge meets the project's accuracy bar.
- Placement of model-level fail-fast validation for unsupported configs.
