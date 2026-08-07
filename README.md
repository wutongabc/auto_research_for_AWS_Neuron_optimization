# Neuron Prefill Optimization

Autonomous optimization loop for Tongyi-30B-A3B (Qwen3MoE) prefill throughput on AWS Trainium 2.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — an AI agent iterates on model serving configuration, vLLM model code, and NKI kernels to maximize prefill tok/s.

## Structure

```
opt/
├── program.md          # AI agent instructions (the "research org code")
├── benchmark/          # READ-ONLY judge program and configs
├── run/                # AI-editable: serving scripts and parameters
├── kernel/             # AI-editable: NKI kernel patches
├── vLLM-neuron/        # AI-editable: vLLM-neuron patches
├── docker/             # Container configuration
└── local-models/       # Compiled artifacts (gitignored)
```

## Phases

| Phase | Budget | Target | Compile? |
|-------|--------|--------|----------|
| 1: Params | 2h | run/config.env, run/serve_fast.bash | Rarely |
| 2: Model | 4h | vLLM-neuron | Yes |
| 3: Kernel | 6h | kernel | Yes |

## Quick Start

Modify the `program.md` instructions to your liking. Then open your AI-assistant like Codex, Claude Code or Opencode and let it read `program.md` and start optimizing.

## Prgress and failure that have been made are written in `optimization_report_en.md` and `optimization_report_cn,md`

## Insights and Experiences

- In long time running experiences, AI-assistant might struggle in one corner case. For example, in the first 12 hours optimization, it focused on MoE optimization and hardly care about other part. In the second 12 hours optimizaiton, I manauly let AI-assistant focus on long context optimization and it optimized the attention part.

- Auto-research is capable with modifying parameters, call for exisiting code modules and write small patch of code. It is hard for auto-research to write a whole flash-attention part of 4000 lines in a single 12 hours auto-research. In my first 12 hours run, it tried 6 hours to modify NKI, failed for about 50 experiments and got 0.6% progress.

- It is important for auto-research to create a clean and toy directory and specific what it can modify and what it cannot, other than a heavy deveploment directory.
