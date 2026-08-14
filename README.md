# Neuron Prefill Optimization

Autonomous optimization loop for LLM prefill throughput on AWS Trainium 2.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — an AI agent iterates on model serving configuration, vLLM model code, and NKI kernels to maximize prefill tok/s.

## Results

![Speedup Comparison](assets/speedup_comparison.png)

| Model | Type | Official Config | 128K Context | Correctness |
|-------|------|----------------|--------------|-------------|
| Tongyi-30B-A3B | MoE | **17.6x** | **24.0x** | 100% |
| GPT-OSS-20B | MoE | **16.3x** | **66.7x** | 100% |
| GPT-OSS-120B | MoE | **11.4x** | **~9.2x** | 100% |
| Qwen3-VL-32B | Dense | **2.8x** | **2.4x** | 100% |

MoE models benefit most from Local-Q + Context Parallel + Local-MoE (eliminating all_gather).
Dense models gain 2-3x from Local-Q + CP + Local-MLP alone.

> Detailed per-round results, analysis, and constraints → [optimization_report_en.md](optimization_report_en.md) | [中文版](optimization_report_cn.md)

### Tongyi-30B-A3B Optimization Timeline

![Optimization Timeline](assets/optimization_timeline.png)

| Round | Focus | Medium (32K) | Full (128K) |
|-------|-------|--------------|-------------|
| 1 | Param tuning (segment size, KV dtype, block size) | 712 → 845 (+19%) | 258 → 365 (+41%) |
| 2 | Model code (GQA broadcast, BF16 attn) + NKI flash_attention | 845 → 4,269 (+405%) | 365 → 1,533 (+320%) |
| 3 | Context Parallel + Local-Q (skip hidden all_gather) | 4,269 → 12,503 (+193%) | 1,533 → 6,200 (+304%) |

## Core Optimizations

- **Local-Q**: Skip hidden all_gather; QKV on local tokens only, all_gather small KV, all_reduce output
- **Context Parallel (CP)**: Split cached KV across ranks, dual flash_attention with online softmax reduction
- **Local-MoE/MLP**: Process MoE/MLP on local tokens (tp_degree=1), all_reduce output

## Structure

```
opt/
├── program.md              # AI agent instructions
├── benchmark/              # Judge program, configs, results
│   └── all_results/        # Detailed reports and CSVs
├── run/                    # Serving scripts and parameters
├── vllm-neuron/            # Optimized vLLM-neuron fork
├── nkilib-fork/            # Optimized NKI kernels
└── docker/                 # Container configuration
```

## Quick Start

Modify `program.md` to your liking. Then open your AI-assistant (Codex, Claude Code, Opencode) and let it read `program.md` and start optimizing.

```bash
bash docker/run.bash -d
docker exec neuron-prefill bash -c "cd /dev3/zigeng/bc/opt && bash run/serve_medium.bash"
```

## Insights

- AI agents excel at parameter tuning, calling existing modules, and writing small patches. Writing 4000-line flash attention from scratch in a single session is beyond current capability.
- Long-context optimization (attention/CP) required manual steering — the agent initially fixated on MoE for 12 hours before being redirected.
- A clean, constrained directory structure (explicit editable vs read-only) is critical for productive auto-research.

