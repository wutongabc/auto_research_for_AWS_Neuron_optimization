#!/usr/bin/env bash
set -euo pipefail

# Serve gpt-oss-20B in LONG mode (TP=8, 128K context)
# Extended context matching Tongyi full benchmark for cross-model comparison.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="openai/gpt-oss-20b"
TP_SIZE=8
MAX_MODEL_LEN=131072
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=1
PORT=8100
HOST="0.0.0.0"

export NEURON_CC_FLAGS="--optlevel 3"
export NEURON_COMPILE_NUM_WORKERS=8
export VLLM_NEURON_COMPILATION_TIMEOUT=2400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION="0.15"

COMPILE_START=$(date +%s)

SERVE_ARGS=(
    --model "$MODEL_PATH"
    --tensor-parallel-size "$TP_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --port "$PORT"
    --host "$HOST"
    --enable-chunked-prefill
    --enable-prefix-caching
    --no-disable-hybrid-kv-cache-manager
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [8192], "num_seqs_buckets": [1]}}'
)

echo "=== Serving gpt-oss-20B (LONG: TP=$TP_SIZE, ctx=$MAX_MODEL_LEN) ==="
echo "  Model: $MODEL_PATH"
echo "  Port: $PORT"
echo "  Batched tokens: $MAX_NUM_BATCHED_TOKENS"
echo ""

mkdir -p "$REPO_ROOT/.compile-artifacts"
cd "$REPO_ROOT/.compile-artifacts"
python -m vllm.entrypoints.openai.api_server "${SERVE_ARGS[@]}" &
VLLM_PID=$!

echo "Waiting for server to be ready..."
for i in $(seq 1 480); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        COMPILE_END=$(date +%s)
        export NEURON_COMPILE_TIME_S=$((COMPILE_END - COMPILE_START))
        echo "Server ready! (compile+load time: ${NEURON_COMPILE_TIME_S}s)"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        wait "$VLLM_PID"
        exit $?
    fi
    sleep 5
done

wait $VLLM_PID
