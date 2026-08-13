# Multi-Model Optimization Port — Benchmark Results

Date: 2026-08-13
Platform: trn2.48xlarge (64 NeuronCores, 24GB HBM/core)

## Summary

Ported Tongyi-30B-A3B prefill optimizations (Local-Q + Context Parallel + Local-MoE/MLP)
to three additional models. Results measured as avg prefill tok/s on the last 50% of turns.

| Model | Config | Baseline (tok/s) | Optimized (tok/s) | Speedup |
|-------|--------|-----------------|-------------------|---------|
| GPT-OSS-20B | TP=8, 16K ctx | 433 | 24,732 | **57.1x** |
| GPT-OSS-20B | TP=8, 128K ctx | COMPILE FAIL | 13,395 | baseline can't compile |
| GPT-OSS-120B | TP=16, 16K ctx | HBM OOM | 12,977 | baseline OOMs |
| GPT-OSS-120B | TP=16, 128K ctx | HBM OOM | 13,433 | baseline OOMs |
| Qwen3-VL-32B | TP=16, 32K ctx | 9,840 | 27,425 | **2.8x** |
| Qwen3-VL-32B | TP=8, 128K ctx | 4,439 | 10,602 | **2.4x** |

All optimized runs: 100% correctness (top-1 logit match).

## Key Findings

1. **GPT-OSS-20B (57x speedup)**: The MoE model benefits massively from Local-Q + CP + Local-MoE.
   Baseline runs at 433 tok/s because it performs full all_gather before every attention/MoE block.
   Our optimization eliminates the bulk of inter-rank communication during prefill.

2. **GPT-OSS-120B (baseline impossible)**: The unoptimized code cannot fit in 24GB HBM per core
   at TP=16. Our Local-Q/Local-MoE pattern reduces peak memory by processing only local tokens
   (T/world_size) instead of the full sequence through QKV and MoE, enabling the model to run.

3. **Qwen3-VL-32B (2.4-2.8x)**: Dense model (no MoE), so gains come purely from Local-Q + CP
   + Local-MLP. The 2.8x on the official config (TP=16) is higher than 2.4x on TP=8 because
   CP reduces more communication at higher TP degrees.

## Optimizations Applied

- **Local-Q**: Skip hidden all_gather in attention; do QKV on local tokens (T/world_size),
  all_gather only the small KV tensors, all_reduce O-proj output
- **Context Parallel (CP)**: Split k_prior (cached KV) across TP ranks, dual flash_attention
  calls with online softmax reduction
- **Local-MoE** (GPT-OSS): Skip all_gather before MoE, process local tokens with tp_degree=1,
  all_reduce output
- **Local-MLP** (Qwen3-VL): Same pattern for dense MLP — skip all_gather, process local tokens,
  all_reduce output

## Why Baseline Fails on OSS-120B

The unoptimized attention pattern does `all_gather(hidden_states)` before computing Q/K/V,
which means each core temporarily holds the full sequence (T tokens * hidden_dim). For OSS-120B
at TP=16, this creates a tensor that exceeds the 24GB HBM limit (28.62GB required).

Our Local-Q pattern never materializes the full-sequence hidden states on any single core.

## Reproduction

```bash
# Build docker image
docker tag public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04 browsecomp-neuron:0.21.0

# Run a single benchmark (example: OSS-20B optimized, official config)
export HF_TOKEN=<your-token>
bash docker/run_bench.bash test-oss20b "0,1,2,3,4,5,6,7" /dev3/zigeng/bc/opt 8100
docker exec -d test-oss20b bash -c "PORT=8100 bash run/serve_oss20b_official.bash > /tmp/server.log 2>&1"
# Wait for "Server ready" in logs, then:
docker exec test-oss20b python benchmark/prefill_bench.py --config benchmark/config_oss20b_official.json --save-baseline
```
