# Qwen3 / Qwen3MoE support

This repository contains a native BF16 Neuron implementation shared by the
Hugging Face architectures `Qwen3ForCausalLM` and
`Qwen3MoeForCausalLM`.

## Implemented paths

- Qwen3 attention with Q/K RMSNorm, GQA, standard RoPE, sequence-parallel
  prefill, paged KV cache, full flash-attention prefill, segmented prefill, and
  fused decode attention.
- Automatic prefix caching with segmented prefill. K/V are written to the
  canonical paged cache before the segmented attention kernel reads them.
- Both legacy `rope_theta` and modern Transformers `rope_parameters` configs.
  RoPE phases are computed in FP32 before cos/sin are cast to BF16.
- Dense SwiGLU MLP with tensor-parallel intermediate sharding.
- Qwen3MoE router semantics for both `norm_topk_prob=true` and `false`.
- Fused per-expert gate/up loading, expert down loading, blockwise CTE prefill,
  and fused TKG decode.
- Tensor parallelism and expert parallelism. With `--enable-expert-parallel`,
  experts are partitioned across EP ranks and outputs are reduced across the
  TP world.
- Dense-only layers and sparse cadence through `mlp_only_layers` and
  `decoder_sparse_step`.
- Both architecture names are registered without `--hf-overrides`.

The active implementation is
`vllm_neuron/model/qwen3/model_bf16.py`; Qwen attention and MoE math do not
delegate to the GPT-OSS model.

## Hardware validation

Validation was performed on a trn2.3xlarge (four NeuronCores):

- `Qwen/Qwen3-0.6B`, TP=2: compiled prefill and decode graphs, completed
  hardware warmup, and matched Hugging Face CPU BF16 for short and segmented
  prompts. A 1509-token prompt with a 512-token segment bucket exercised three
  prefill segments and matched all 16 generated tokens; repeating the request
  also matched all 16 tokens with a 46.7% prefix-cache hit rate.
- Bucket-boundary coverage at 508, 521, 807, 989, and 1509 prompt tokens. The
  508/521/807 cases matched 16/16 tokens. At 989 tokens, segmented prefill chose
  HF's second-ranked first token (the HF top-two margin was 0.25 BF16 logits),
  while teacher forcing matched the remaining 15/15 steps. Full prefill matched
  16/16 for the same prompt. This is a numerical-ordering edge, not context
  truncation; it should remain in accuracy qualification.
- A native tiny Qwen3MoE checkpoint, TP=2: Neuron greedy output exactly matched
  Hugging Face CPU BF16 for four generated tokens. A 2-layer, 32-expert, top-8
  fixture also compiled and warmed segmented prefill/decode; an 807-token
  teacher-forced comparison matched 8/8 tokens.
- `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`, TP=4 + EP=4: loaded the complete
  48-layer checkpoint, compiled four rank-specific prefill and four decode
  graphs, warmed every core, and generated coherent completion and chat output.
- A one-layer random fixture retaining Tongyi dimensions (H=2048, I_moe=768,
  128 experts, top-8, 32 Q heads, 4 KV heads), TP=4 + EP=4: under identical
  autoregressive prefixes, the four Neuron-selected tokens ranked [2, 2, 1, 1]
  in the CPU BF16 distributions. The first two differed from CPU top-1 by only
  0.15625 and 0.03125 logits; the last two matched top-1. This checks the
  production-width EP path without overstating random-model argmax stability.

Artificial 1024-token vocabularies are poor sampler fixtures at TP=4 because
`vocab_per_rank=256` coincides with the default `max_top_k=256`. Use a normal
Qwen vocabulary or validate raw/model distributions separately.

## Model-specific notes

Some local Tongyi-DeepResearch copies contain weights and config but no
tokenizer files. Pass a compatible Qwen3 tokenizer explicitly with
`--tokenizer`; this is checkpoint packaging, not a model limitation.

For MoE on four cores, use:

```text
--tensor-parallel-size 4 --enable-expert-parallel
```

This yields EP=4 and 32 of Tongyi's 128 experts per rank.

## Current scope and limitations

- Registered text architectures are `Qwen3ForCausalLM` and
  `Qwen3MoeForCausalLM`. Qwen2/Qwen2.5 and multimodal Qwen models are separate
  implementations and are not covered by this module.
- The implementation is BF16-only. FP8 weights/KV cache, sliding-window
  attention, attention sinks, and context-parallel segmented attention are not
  implemented here.
- The fused decode path currently requires `head_dim=128`. Bias-enabled or
  non-QK-normalized Qwen variants are not implemented and are rejected during
  model construction.
- Segmented prefill inherits the backend restriction of prefill batch size 1;
  mixed prefill/decode batches are not supported.
- Eagle3/speculative decoding and Qwen-specific auxiliary hidden states are not
  implemented.
- On-device-sampler OpenAI `logprobs` is validated with TP=2: two generated
  steps returned five top logprobs each, and the first-step top-5 token IDs
  matched HF BF16 exactly.
- Production qualification still needs representative official Qwen3MoE model
  accuracy, performance, memory, concurrency, and long-running soak coverage.
