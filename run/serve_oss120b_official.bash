#!/usr/bin/env bash
set -euo pipefail

# Serve GPT-OSS-120B with official-like params (TP=32, 16K context, BF16)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="openai/gpt-oss-120b"
TP_SIZE=32
MAX_MODEL_LEN=16384
MAX_NUM_BATCHED_TOKENS=${SEGMENT_SIZE:-4096}
MAX_NUM_SEQS=1
PORT=${PORT:-8100}

export NEURON_CC_FLAGS="--optlevel 1 --model-type transformer --enable-fast-loading-neuron-binaries"
export NEURON_COMPILE_NUM_WORKERS=${NEURON_COMPILE_WORKERS:-16}
export VLLM_NEURON_COMPILATION_TIMEOUT=2400

COMPILE_START=$(date +%s)

SERVE_ARGS=(
    --model "$MODEL_PATH"
    --tensor-parallel-size "$TP_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-num-seqs "$MAX_NUM_SEQS"
    --block-size 64
    --port "$PORT"
    --host "0.0.0.0"
    --dtype bfloat16
    --optimization-level 1
    --enable-chunked-prefill
    --additional-config "{\"neuron_config\": {\"kv_segment_size_buckets\": [${MAX_NUM_BATCHED_TOKENS}], \"num_batched_tokens_buckets\": [${MAX_NUM_BATCHED_TOKENS}], \"num_seqs_buckets\": [1]}}"
)

echo "=== Serving GPT-OSS-120B (Official: TP=$TP_SIZE, ctx=$MAX_MODEL_LEN, seg=$MAX_NUM_BATCHED_TOKENS) ==="
echo "  Port: $PORT"

mkdir -p "$REPO_ROOT/.compile-artifacts"
cd "$REPO_ROOT/.compile-artifacts"
python -m vllm.entrypoints.openai.api_server "${SERVE_ARGS[@]}" &
VLLM_PID=$!

echo "Waiting for server to be ready..."
for i in $(seq 1 480); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        COMPILE_END=$(date +%s)
        echo "Server ready! (compile+load time: $((COMPILE_END - COMPILE_START))s)"
        break
    fi
    sleep 5
done

wait $VLLM_PID
