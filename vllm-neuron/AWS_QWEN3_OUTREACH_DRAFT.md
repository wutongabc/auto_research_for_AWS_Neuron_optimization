# AWS team outreach drafts

## Slack draft

Hi Neuron team — I have a native BF16 Qwen3/Qwen3MoE port working on the
vLLM Neuron 0.21.0 branch and would appreciate review, especially around the
segmented-prefill cache contract and MoE EP collectives.

The port registers `Qwen3ForCausalLM` and `Qwen3MoeForCausalLM`, implements
Q/K-normalized GQA, dense SwiGLU, Qwen3MoE prefill/decode, TP/EP loading, paged
KV cache, segmented prefill, and APC. It reuses the existing segmented NKI
kernel rather than adding a Qwen-specific attention kernel.

Two Qwen-specific long-context issues were found and fixed: current
Transformers stores the official 1M RoPE base in `rope_parameters`, and RoPE
phase must remain FP32 because BF16 aliases integer positions above 256.

Trn2.3xlarge results:

- Qwen3-0.6B TP2, 512-token segment: a 1509-token prompt (three segments)
  matched HF BF16 16/16 tokens; the repeated request also matched and APC was
  active.
- Tiny Qwen3MoE TP2 (32 experts, top-8): 807-token segmented prefill compiled,
  warmed, and matched HF 8/8 teacher-forced tokens.
- GPT-OSS and tiny-Qwen controls show the existing segmented kernel handles
  multi-KV-head GQA and `head_dim=128` correctly.

One accuracy edge remains worth your view: at 989 tokens, one adversarial dense
prompt swaps only the first greedy token to HF's second-ranked candidate; the
HF top-two margin is 0.25 BF16 logits, and teacher forcing then matches 15/15.
Full prefill matches 16/16. Would you consider this within the expected
segmented-vs-flash BF16 tolerance, or is there a preferred raw-logit/R-ratio
qualification workflow for this backend?

I also fixed the async on-device-sampler OpenAI `logprobs` path by returning
gathered logits from Qwen and synchronizing those logits only for requests that
ask for logprobs. In a TP=2 test, both generated steps returned top-5 results,
and the first-step top-5 token IDs matched HF BF16 exactly.

## Email draft

Subject: Review request: native Qwen3/Qwen3MoE support with segmented prefill on vLLM Neuron

Hi AWS Neuron team,

I have completed a first native BF16 implementation for Qwen3 and Qwen3MoE on
the vLLM Neuron 0.21.0 codebase and would like to coordinate an upstream review.

The change includes Qwen3/Qwen3MoE architecture registration, Q/K-normalized
GQA, dense and sparse MLP paths, TP/EP checkpoint sharding, full and segmented
prefill, paged KV cache, fused decode, and automatic prefix caching. The
attention path uses the existing segmented-attention NKI implementation.

On Trn2.3xlarge, Qwen3-0.6B at TP=2 matched Hugging Face CPU BF16 for a
1509-token prompt across three 512-token prefill segments (16/16 generated
tokens), including a repeated APC request. A tiny 2-layer Qwen3MoE model with
32 experts/top-8 also compiled and matched 8/8 teacher-forced tokens for an
807-token segmented prompt.

The main correctness fixes were model-specific: handling the modern
`rope_parameters` location for Qwen's 1M RoPE base, and computing RoPE phase in
FP32 before casting cos/sin to BF16.

I would particularly value feedback on:

1. The canonical paged-cache write followed by segmented-attention read.
2. TP/EP semantics for Qwen3MoE and the preferred large-model qualification.
3. The expected accuracy threshold for a documented 0.25-logit BF16 top-1
   boundary difference at one 989-token prompt.
4. Whether synchronizing gathered logits only for async requests that ask for
   `logprobs` matches the preferred backend design.

I can share the branch, patch, full command lines, compiled graph hashes, and
the tiny reproducible checkpoints. Thank you for any guidance on the preferred
upstream test matrix and review owners.

Best,
[Name]
