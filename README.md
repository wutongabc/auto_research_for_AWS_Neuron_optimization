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
| 2: Model | 4h | vLLM-neuron/ patches | Yes |
| 3: Kernel | 6h | kernel/ patches | Yes |

## Quick Start

```bash
bash docker/run.bash -d           # Start container
docker exec -it neuron-prefill bash
bash run/serve_fast.bash &        # Launch model
python benchmark/prefill_bench.py --config benchmark/config_fast.json
```
