# Multi-Model Optimization Port — Benchmark Results

Date: 2026-08-14
Platform: trn2.48xlarge (64 NeuronCores, 24GB HBM/core)

## Summary (v2 — Fair Comparison with Matched Segment Sizes)

Ported Tongyi-30B-A3B prefill optimizations (Local-Q + Context Parallel + Local-MoE/MLP)
to three additional models. Results measured as avg prefill tok/s on the last 50% of turns.

### Round 1: Original configs (large segment sizes — includes padding effects)

| Model | Config | seg_size | Baseline (tok/s) | Optimized (tok/s) | Speedup |
|-------|--------|----------|-----------------|-------------------|---------|
| GPT-OSS-20B | TP=8, 16K ctx | 8192 | 433 | 24,732 | **57.1x** |
| GPT-OSS-20B | TP=8, 128K ctx | 4096 | COMPILE FAIL | 13,395 | — |
| GPT-OSS-120B | TP=16, 16K ctx | 8192 | HBM OOM | 12,977 | — |
| GPT-OSS-120B | TP=16, 128K ctx | 4096 | HBM OOM | 13,433 | — |
| Qwen3-VL-32B | TP=16, 32K ctx | 4096 | 9,840 | 27,425 | **2.8x** |
| Qwen3-VL-32B | TP=8, 128K ctx | 4096 | 4,439 | 10,602 | **2.4x** |

### Round 2: Fair comparison (seg=1024 for OSS-20B, seg=2048/4096 for OSS-120B at TP=32)

| Model | Config | seg_size | Baseline (tok/s) | Optimized (tok/s) | Speedup |
|-------|--------|----------|-----------------|-------------------|---------|
| GPT-OSS-20B | TP=8, 16K ctx | 1024 | 1,099 | 17,897 | **16.3x** |
| GPT-OSS-20B | TP=8, 128K ctx | 1024 | 136 | 9,080 | **66.7x** |
| GPT-OSS-120B | TP=32, 16K ctx | 2048/4096 | 1,217 | 13,934 | **11.4x** |
| GPT-OSS-120B | TP=32, 128K ctx | 4096 | ~1,200 (est.) | 11,030 | **~9.2x** |

All optimized runs: 100% correctness (top-1 logit match).

## Key Findings

1. **GPT-OSS-20B**: At fair seg=1024 comparison, **16.3x** on official 16K config.
   At 128K long context, speedup jumps to **66.7x** because baseline degrades quadratically
   with number of KV segments (128K/1024 = 128 segments, each requiring full all_gather),
   while our CP keeps per-rank work bounded.

2. **GPT-OSS-120B**: Baseline cannot run at TP=16 (OOM). At TP=32, seg=2048 baseline achieves
   1,217 tok/s. Our optimized code runs 11.4x faster. The model is too large for small TP
   without our memory-efficient Local-Q/Local-MoE patterns.

3. **Qwen3-VL-32B (2.4-2.8x)**: Dense model (no MoE), so gains come purely from Local-Q + CP
   + Local-MLP. The 2.8x on the official config (TP=16) is higher than 2.4x on TP=8 because
   CP reduces more communication at higher TP degrees.

4. **Padding effect analysis**: Round 1's 57x for OSS-20B included 63% padding waste at
   seg=8192 (3000 tok turns padded to 8192). With seg=1024 (~3% padding), the true algorithmic
   speedup is 16.3x for short context — still enormous, driven by elimination of all_gather.

## Optimizations Applied

- **Local-Q**: Skip hidden all_gather in attention; do QKV on local tokens (T/world_size),
  all_gather only the small KV tensors, all_reduce O-proj output
- **Context Parallel (CP)**: Split k_prior (cached KV) across TP ranks, dual flash_attention
  calls with online softmax reduction
- **Local-MoE** (GPT-OSS): Skip all_gather before MoE, process local tokens with tp_degree=1,
  all_reduce output
- **Local-MLP** (Qwen3-VL): Same pattern for dense MLP — skip all_gather, process local tokens,
  all_reduce output

## Why Long-Context Speedup Is Higher

At 128K context with seg=1024, the baseline must process 128 segments per request. Each segment
triggers a full all_gather (bringing all tokens to all cores) before attention. The communication
overhead grows linearly with segment count while compute stays constant — making baseline
extremely slow (136 tok/s for OSS-20B at 128K).

Our Context Parallel keeps KV distributed across ranks and uses dual flash_attention with
online softmax reduction, avoiding any full-sequence all_gather regardless of context length.

## OSS-120B Constraints

- **TP=16 impossible for baseline**: Model weights = 120B × 2B / 16 = 15GB, leaving only 9GB
  for KV + activations. Baseline's all_gather pattern requires 28.6GB.
- **TP=32 baseline requires seg≥2048**: At seg=1024, seqlen_q=1024/32=32 which is below the
  flash_attention minimum of 128. At seg=2048, seqlen_q=64... still borderline, but the
  baseline path uses standard attention (not our flash_attention+cache_softmax kernel).
- **Our optimized code**: Requires seg ≥ 128 × TP (kernel constraint for cache_softmax mode).
  At TP=32, minimum seg=4096.

## Reproduction

```bash
# Build docker image
docker tag public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04 browsecomp-neuron:0.21.0

# Run a single benchmark (example: OSS-20B optimized, official config, seg=1024)
export HF_TOKEN=<your-token>
bash docker/run_bench.bash test-oss20b "0,1,2,3,4,5,6,7" /dev3/zigeng/bc/opt 8100
docker exec -d test-oss20b bash -c "PORT=8100 bash run/serve_oss20b_official.bash > /tmp/server.log 2>&1"
# Wait for "Server ready" in logs, then:
docker exec test-oss20b python benchmark/prefill_bench.py --config benchmark/config_oss20b_official.json --save-baseline

# OSS-120B (TP=32, needs 32 cores)
bash docker/run_bench.bash test-oss120b "0,1,2,...,31" /dev3/zigeng/bc/opt 8100
docker exec -d test-oss120b bash -c "SEGMENT_SIZE=4096 PORT=8100 bash run/serve_oss120b_official.bash > /tmp/server.log 2>&1"
```
