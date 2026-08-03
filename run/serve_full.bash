#!/usr/bin/env bash
set -euo pipefail

# Serve Tongyi-30B-A3B in FULL mode (TP=8, 128K context)
# Used only for final validation at the end of Phase 3.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Load config
source "$SCRIPT_DIR/config.env"

# Full model overrides
TP_SIZE=8
MAX_MODEL_LEN=131072
CONTEXT_LENGTH_BUCKETS="4096,8192,16384,32768,65536,131072"
DECODE_CONTEXT_LENGTH_BUCKETS="131072"

# Export Neuron env vars
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS}"
export NEURON_COMPILE_NUM_WORKERS="${NEURON_COMPILE_WORKERS}"
export VLLM_NEURON_FUSED_MOE_NKI="${VLLM_NEURON_FUSED_MOE_NKI}"
export VLLM_NEURON_FP8_EXPERT_WEIGHTS="${VLLM_NEURON_FP8_EXPERT_WEIGHTS}"
export VLLM_NEURON_FP8_ONLY="${VLLM_NEURON_FP8_ONLY}"

# Track compile time
COMPILE_START=$(date +%s)

SERVE_ARGS=(
    --model "$MODEL_PATH"
    --tensor-parallel-size "$TP_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --block-size "$BLOCK_SIZE"
    --port "$PORT"
    --host "$HOST"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --override-neuron-config "{\"context_length_buckets\": [${CONTEXT_LENGTH_BUCKETS}], \"decode_context_length_buckets\": [${DECODE_CONTEXT_LENGTH_BUCKETS}]}"
)

if [[ "${ENABLE_CHUNKED_PREFILL:-0}" == "1" ]]; then
    SERVE_ARGS+=(--enable-chunked-prefill --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

if [[ -n "${SCHEDULING_POLICY:-}" ]]; then
    SERVE_ARGS+=(--scheduling-policy "$SCHEDULING_POLICY")
fi

echo "=== Serving Tongyi-30B-A3B (FULL: TP=$TP_SIZE, ctx=$MAX_MODEL_LEN) ==="
echo "  Port: $PORT"
echo "  Buckets: $CONTEXT_LENGTH_BUCKETS"
echo "  KV dtype: $KV_CACHE_DTYPE"
echo "  FP8 experts: $VLLM_NEURON_FP8_EXPERT_WEIGHTS"
echo ""

python -m vllm_neuron.entrypoints.openai.api_server "${SERVE_ARGS[@]}" &
VLLM_PID=$!

echo "Waiting for server to be ready..."
for i in $(seq 1 240); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        COMPILE_END=$(date +%s)
        export NEURON_COMPILE_TIME_S=$((COMPILE_END - COMPILE_START))
        echo "Server ready! (compile+load time: ${NEURON_COMPILE_TIME_S}s)"
        break
    fi
    sleep 5
done

wait $VLLM_PID
