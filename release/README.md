# vllm-neuron-opt

Optimized vLLM-Neuron inference on AWS Trainium 2.

This image extends the official `pytorch-inference-vllm-neuronx` with optimized vLLM-Neuron model code and NKI kernels for improved prefill throughput.

## Requirements

- AWS Trainium 2 instance (trn2)
- Docker
- Model weights (HuggingFace format)

## Quick Start

```bash
# 1. Build the optimized image
bash build.sh

# 2. Start a container (interactive shell)
bash run.sh

# 3. Inside the container, download and serve a model
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /model
vllm serve /model --tensor-parallel-size 4 --max-model-len 32768 --port 8100
```

Or mount local weights:

```bash
EXTRA_ARGS="-v /path/to/model:/model" bash run.sh
```

## Configuration

### Container Launch

Use env files to select which NeuronCores are visible to the container:

```bash
# 4 NeuronCores (for TP=4 models)
ENV_FILE=configs/tp4.env MODEL_PATH=/path/to/model bash run.sh

# 8 NeuronCores (for TP=8 models)
ENV_FILE=configs/tp8.env MODEL_PATH=/path/to/model bash run.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV_FILE` | | Path to .env config file (sets NEURON_VISIBLE_DEVICES) |
| `COMPILE_CACHE` | | Path to persist Neuron compile cache |
| `EXTRA_ARGS` | | Additional docker run arguments (e.g., `-v /path:/mount`) |
| `CONTAINER_NAME` | `vllm-neuron-opt` | Docker container name |
| `IMAGE_NAME` | `zigeng/vllm-neuron-opt` | Docker image name |
| `IMAGE_TAG` | `0.21.0.1.0.0-1.0.0` | Docker image tag |

### Inside the Container

Start vllm serve with model-specific parameters:

```bash
# Qwen3-30B-A3B (MoE, TP=4)
vllm serve /model --tensor-parallel-size 4 --max-model-len 32768 --port 8100

# Llama 3 70B (TP=8, longer context)
vllm serve /model --tensor-parallel-size 8 --max-model-len 131072 --port 8100
```

### Compile Cache

First compilation on Neuron hardware takes time. Persist the cache to skip recompilation on subsequent runs:

```bash
COMPILE_CACHE=/path/to/cache MODEL_PATH=/path/to/model bash run.sh
```

## Supported Models

Models supported by vLLM-Neuron, including:
- Qwen3MoE (e.g., Qwen/Qwen3-30B-A3B)
- Llama 3
- Gemma 4

## Base Image

Built on: `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04`
