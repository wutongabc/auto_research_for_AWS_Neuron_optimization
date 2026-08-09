# Qwen3 / Qwen3MoE (Text) Model Recipe

<!-- meta: description: Model recipe for deploying Qwen3 and Qwen3MoE text models
with vLLM on Neuron, including supported checkpoints, feature support, accuracy
results, FP8 KV cache, EAGLE3 speculative decoding, disaggregated inference, and
hardware validation for dense and MoE variants on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3, Qwen3MoE, Qwen3ForCausalLM,
Qwen3MoeForCausalLM, dense, MoE, BF16, FP8, KV cache, EAGLE3, speculative
decoding, disaggregated inference, model recipe, model card, LLM serving, Trn2,
Trainium -->
<!-- meta: date_updated: 2026-07-27 -->
<!-- Content type: model-card -->

## Introduction

[Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)
is a family of dense and Mixture-of-Experts (MoE) language models developed by
the Qwen team. Qwen3 models support multilingual text generation, reasoning, and
tool use. The MoE variants (e.g. Tongyi-DeepResearch-30B-A3B) use top-k expert
routing for efficient inference at scale.

Qwen3 and Qwen3MoE are supported for inference serving with
[vLLM](https://github.com/vllm-project/vllm) using the Neuron SDK on AWS
Trainium2 (`trn2`) hardware.

**Compatible model checkpoints:**

| Model | HuggingFace | Type | Hardware |
|-------|-------------|------|----------|
| Qwen3-0.6B | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Dense | Trn2 |
| Qwen3-4B | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Dense | Trn2 |
| Qwen3-8B | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Dense | Trn2 |
| Qwen3-32B | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Dense | Trn2 |
| Tongyi-DeepResearch-30B-A3B | [Alibaba-NLP/Tongyi-DeepResearch-30B-A3B](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B) | MoE (128 experts, top-8) | Trn2 |

> Both `Qwen3ForCausalLM` and `Qwen3MoeForCausalLM` architectures are
> registered natively — no `--hf-overrides` required. All models run in BF16
> with optional FP8 KV cache quantization.

## Features

Per-model feature availability for Qwen3/Qwen3MoE. See the
[features guide](../guides/features-guide.md) for configuration details and the
cross-model feature compatibility matrix.

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Text | ✅ |
| **Quantization** | BF16 weights | ✅ |
| | FP8 KV cache | ✅ |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Expert parallelism (EP) | ✅ (MoE) |
| | Pipeline parallelism (PP) | ❌ |
| | Context parallelism (CP) | ❌ |
| **Performance** | Continuous batching | ✅ |
| | Segmented prefill | ✅ |
| | Prefix caching (APC) | ✅ |
| | On-device sampling (greedy, top-k, top-p) | ✅ |
| | Speculative decoding (EAGLE3) | ✅ |
| | Disaggregated inference (1P1D) | ✅ |
| **Serving** | OpenAI-compatible logprobs | ✅ |
| **Compilation** | torch.compile (XLA backend) | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Qwen3/Qwen3MoE
- ❌ Not supported: may be considered for future releases

## Deploy on a single instance

The simplest deployment: one server serving the whole model with tensor
parallelism. Use it for functional validation, development, and light workloads.

### Dense models (Qwen3-0.6B through Qwen3-32B)

```bash
vllm serve Qwen/Qwen3-32B \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --enable-prefix-caching \
    --additional-config '{
        "neuron_config": {
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4]
        }
    }'
```

### MoE models (Tongyi-DeepResearch-30B-A3B)

```bash
vllm serve Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --enable-prefix-caching \
    --additional-config '{
        "neuron_config": {
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4]
        }
    }'
```

This yields EP=4 (32 of 128 experts per rank). The `--enable-expert-parallel`
flag is required for MoE models.

### Validate the server

```bash
curl -i http://localhost:8000/health

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-32B",
        "prompt": "The capital of France is ",
        "max_tokens": 16
    }'
```

## FP8 KV cache

FP8 KV cache reduces KV cache memory by 50% (BF16 → FP8), enabling larger batch
sizes or longer context windows within the same HBM budget. The implementation
uses static quantization scales loaded from the checkpoint.

### Enable FP8 KV cache

Add `--kv-cache-dtype fp8` to your serve command:

```bash
vllm serve Qwen/Qwen3-32B \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --max-model-len 65536 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    --additional-config '{
        "neuron_config": {
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [8]
        }
    }'
```

### How it works

- K and V values are quantized to `float8_e4m3fn` before writing to the paged
  cache, using per-layer `k_scale` / `v_scale` factors loaded from the
  checkpoint.
- On decode, the scales are fused into the attention megakernel's
  `softmax_scale` and `W_out` parameters, avoiding a separate dequantization
  step.
- When the hardware supports it, a **packed FP8** layout pairs two FP8 values
  in one BF16 slot for higher memory bandwidth utilization. A per-bucket
  viability check automatically falls back to unpacked layout when the batch
  geometry does not support packing.

### Requirements

- The model checkpoint must include `k_scale` and `v_scale` tensors (most
  Qwen3 checkpoints include these; they are generated during calibration).
- If the checkpoint does not include scale tensors, FP8 KV cache is still
  activated but uses a default scale of 1.0 (no quantization benefit).

## Speculative decoding (EAGLE3)

EAGLE3 speculative decoding uses a lightweight draft model to speculatively
generate multiple tokens per step, with the target model verifying them in a
single forward pass. This reduces per-token latency by amortizing the
memory-bound KV cache read across multiple tokens.

### Enable EAGLE3

```bash
vllm serve Qwen/Qwen3-32B \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --speculative-config '{
        "method": "eagle3",
        "model": "path/to/eagle3-draft-model",
        "num_speculative_tokens": 5
    }' \
    --additional-config '{
        "neuron_config": {
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4]
        }
    }'
```

### How it works

- The target model (Qwen3) collects auxiliary hidden states from three layers
  during forward: an early layer (layer 2), a middle layer, and a late layer
  (N-3). These are concatenated and passed to the draft model.
- The draft model (a small Llama-based model, `Eagle3LlamaForCausalLM`) uses
  these auxiliary states to predict the next K tokens.
- The target model verifies all K tokens in one batched forward pass using
  rejection sampling. Accepted tokens are emitted immediately; rejected tokens
  trigger a resample from the target distribution.
- The `@async_speculative_decoding` decorator handles the draft-verify loop
  asynchronously, overlapping draft computation with target verification.

### Architecture details

| Component | Role |
|-----------|------|
| `SupportsEagle3` mixin | Declares EAGLE3 capability to the framework |
| `get_eagle3_aux_hidden_state_layers()` | Returns default layers: `(2, num_layers // 2, num_layers - 3)` |
| `set_aux_hidden_state_layers(layers)` | Allows the framework to override layer selection |
| `rejection_sampler(metadata, sampled_tokens)` | Verifies draft tokens against target distribution |

## Deploy with disaggregated inference (1P1D)

Disaggregated inference separates prompt processing (prefill) from token
generation (decode) onto different NeuronCore groups. This eliminates
head-of-line blocking where long prompts delay decode requests, and lets you
optimize each phase independently.

For the full mechanics of DI on Neuron — the NIXL KV connector, the proxy
router, and how to scale from 1P1D to xPyD — see the
[disaggregated inference tutorial](../tutorials/tutorial-di-1p1d-xpyd.md).

### Single-node 1P1D (trn2.48xlarge)

This recipe deploys prefill and decode on the same `trn2.48xlarge` instance,
with KV cache transferred over loopback via NIXL / LIBFABRIC.

| Role | NeuronCores | Parallelism | Port |
|------|-------------|-------------|------|
| Prefill (`kv_producer`) | 0–15 | TP4 | 8100 |
| Decode (`kv_consumer`) | 16–31 | TP8 DP2 | 8200 |
| Proxy | — | — | 8000 |

**Prerequisites:**

```bash
pip install nixl
export VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0
export VLLM_NEURON_COMPILATION_TIMEOUT=2400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_ENGINE_READY_TIMEOUT_S=1800
```

**Launch prefill:**

```bash
export NEURON_VISIBLE_DEVICES="0-15"
export VLLM_NIXL_SIDE_CHANNEL_PORT=5559

vllm serve Qwen/Qwen3-32B \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 1 \
    --no-disable-hybrid-kv-cache-manager \
    --port 8100 \
    --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_producer", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}' \
    --additional-config '{
        "neuron_config": {
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [1]
        }
    }'
```

**Launch decode:**

```bash
export NEURON_VISIBLE_DEVICES="16-31"
export VLLM_NIXL_SIDE_CHANNEL_PORT=5659

vllm serve Qwen/Qwen3-32B \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --max-logprobs 0 \
    --no-disable-hybrid-kv-cache-manager \
    --port 8200 \
    --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_consumer", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}' \
    --additional-config '{
        "neuron_config": {
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4]
        }
    }'
```

**Launch proxy (after both servers report startup complete):**

```bash
python3 examples/vllm_neuron/vllm/disaggregated_inference/toy_proxy_server.py \
    --port 8000 \
    --host 0.0.0.0 \
    --prefiller-host 127.0.0.1 --prefiller-port 8100 \
    --decoder-host 127.0.0.1 --decoder-port 8200
```

**Validate:**

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-32B",
        "prompt": "Count the number 1, 2, 3",
        "max_tokens": 200
    }'
```

See `examples/vllm_neuron/models/qwen3/32b/bf16/single_node_DI/` for
ready-to-run shell scripts.

### Qwen3-specific DI notes

- Qwen3 is a dense model (except for Tongyi MoE variants), so there is no
  expert parallelism to configure on prefill/decode independently.
- The `no-disable-hybrid-kv-cache-manager` flag is required for DI; it enables
  the KV cache manager that supports block-level RDMA transfer.
- FP8 KV cache can be combined with DI (`--kv-cache-dtype fp8`), which halves
  the NIXL transfer volume between prefill and decode.

## Accuracy Validation

Accuracy measured on Trn2 hardware with BF16 weights.

**Dense (Qwen3-0.6B, TP=2, segment size 512):**

| Test | Result |
|------|--------|
| 1509-token prompt (3 segments), greedy decode | 16/16 tokens match HF BF16 |
| Repeated request (APC active, 46.7% hit) | 16/16 tokens match |
| Boundary prompts (508, 521, 807 tokens) | 16/16 tokens match |
| 989-token adversarial prompt, segmented | 15/16 (first token swaps to HF rank-2 at 0.25-logit margin) |
| 989-token prompt, full prefill | 16/16 tokens match |

**MoE (tiny Qwen3MoE, TP=2, 32 experts, top-8, segment 512):**

| Test | Result |
|------|--------|
| 807-token segmented prefill, teacher forcing | 8/8 tokens match HF BF16 |
| Repeated-prefix run (APC active, 90.9% hit) | Match |

**MoE (Tongyi-DeepResearch-30B-A3B, TP=4, EP=4):**

| Test | Result |
|------|--------|
| Full 48-layer checkpoint load + compile | ✅ |
| Coherent completion and chat output | ✅ |

## Limitations

- Text-only. Qwen2/Qwen2.5 and multimodal Qwen are separate implementations.
- BF16 weights only (`head_dim=128`). MXFP4 weight quantization is not yet
  supported.
- Segmented prefill inherits batch-size-1 restriction (no mixed prefill/decode).
- Sliding-window attention, attention sinks, and bias-enabled variants are not
  supported and are rejected during model construction.
- Some MoE checkpoints (e.g. local Tongyi copies) may lack tokenizer files —
  pass a compatible Qwen3 tokenizer with `--tokenizer`.

## Tutorials

- [Tutorial: Configure disaggregated inference](../tutorials/tutorial-di-1p1d-xpyd.md)
  — General 1P1D and xPyD setup with NIXL on Neuron.
- [Example scripts: Qwen3-32B single-node DI](../../examples/vllm_neuron/models/qwen3/32b/bf16/single_node_DI/)
  — Ready-to-run prefill, decode, proxy, and client scripts.
