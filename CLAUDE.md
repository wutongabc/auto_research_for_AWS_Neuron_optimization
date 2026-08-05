# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonomous optimization loop for Tongyi-30B-A3B (Qwen3MoE) prefill throughput on AWS Trainium 2. An AI agent iterates on serving configuration, vLLM model code, and NKI kernels to maximize prefill tok/s while maintaining >99% top-1 logit correctness.

The full agent specification is in `program.md` — read it before starting any optimization work.

## Running Experiments

```bash
# Start the container (daemon mode)
bash docker/run.bash -d

# Launch model and run benchmark (fast mode: TP=4, 16K context)
docker exec neuron-prefill bash -c "cd /dev3/zigeng/bc/opt && bash run/serve_fast.bash > /dev/null 2>&1 & sleep 30 && python benchmark/prefill_bench.py --config benchmark/config_fast.json > run.log 2>&1"

# Extract results
grep "^avg_prefill_tok_per_s:\|^correctness_pct:\|^compile_time_s:" run.log

# Full model validation (TP=8, 128K context) — end of optimization only
docker exec neuron-prefill bash -c "cd /dev3/zigeng/bc/opt && bash run/serve_full.bash > /dev/null 2>&1 & sleep 120 && python benchmark/prefill_bench.py --config benchmark/config_full.json > run_full.log 2>&1"
```

## Architecture

**Model**: Qwen3MoE — 30B total params, 3B active/token, 128 experts, top-8 routing, 48 layers.

**Hardware**: trn2 with 8 NeuronCores. Fast model uses cores 0-3 (TP=4), full model uses all 8 (TP=8).

**Container**: `browsecomp-neuron:0.21.0` — runs privileged with host networking. The vLLM fork at `/dev3/zigeng/bc/vllm-neuron/` is mounted read-only at `/opt/vllm-neuron`. Kernel and vLLM patches from this repo are bind-mounted to override container packages.

**Key paths inside container**:
- nkilib kernels: `/opt/conda/lib/python3.13/site-packages/nkilib/core/moe/moe_tkg/`
- vLLM-neuron: `/opt/vllm-neuron/vllm_neuron/`

## Editable Files (by phase)

| Phase | Budget | Editable |
|-------|--------|----------|
| 1: Params | 2h | `run/config.env`, `run/serve_fast.bash` |
| 2: Model | 4h | `vLLM-neuron/__init__.py` (monkey-patches Qwen3 MoE model code) |
| 3: Kernel | 6h | `kernel/moe_tkg.py`, `kernel/selective_expert_opt.py` (NKI kernels) |

## Read-Only (never modify)

- `benchmark/` — judge program, configs, baseline logits
- `docker/run.bash` — container configuration
- `program.md` — agent specification

## Keep/Discard Logic

After each experiment:
- **Keep** if tok/s improved AND correctness >= 99.0%
- **Discard** (revert commit) if correctness < 99.0% or tok/s did not improve
- Log every experiment to `results.tsv` regardless of outcome

## Key Metric

`avg_prefill_tok_per_s` — average across last 50% of turns (long-context steady-state). This is the single number that determines keep/discard.

## Tech Stack

- Python 3.13 (in container), Bash
- vLLM-Neuron (custom fork) for model serving
- NKI (Neuron Kernel Interface) for custom kernels — uses `nki.jit`, `nki.language`, `nki.isa`
- Docker image with Neuron SDK, served via OpenAI-compatible API on port 8100
