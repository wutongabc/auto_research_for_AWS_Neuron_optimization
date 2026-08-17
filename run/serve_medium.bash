#!/usr/bin/env bash
set -euo pipefail

# Serve Tongyi-30B-A3B in MEDIUM mode (TP=4, 32K context)
# Shows long-context attention behavior without full-model compile cost.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Load config
source "$SCRIPT_DIR/config.env"

# Medium model overrides
TP_SIZE=4
MAX_MODEL_LEN=30208
export NEURON_VISIBLE_DEVICES="8,9,10,11"

# Force fresh compilation (bypass stale pre-compiled cache)
export VLLM_CACHE_ROOT="$REPO_ROOT/.cache-medium-1l-bs128"

# Layer reduction for faster prefill
export VLLM_NEURON_NUM_LAYERS=1

# Reduce expert routing (top-4 instead of top-8)
export VLLM_NEURON_NUM_EXPERTS_PER_TOK=4

# Export Neuron env vars
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS}"
export NEURON_COMPILE_NUM_WORKERS="${NEURON_COMPILE_WORKERS}"
export VLLM_NEURON_FUSED_MOE_NKI="${VLLM_NEURON_FUSED_MOE_NKI}"
export VLLM_NEURON_FP8_EXPERT_WEIGHTS="${VLLM_NEURON_FP8_EXPERT_WEIGHTS}"
export VLLM_NEURON_FP8_ONLY="${VLLM_NEURON_FP8_ONLY}"
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION="${KV_BUDGET_CAP}"

# Track compile time
COMPILE_START=$(date +%s)

# Build vLLM serve command
SERVE_ARGS=(
    --model "$MODEL_PATH"
    --tensor-parallel-size "$TP_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --block-size "$BLOCK_SIZE"
    --port "$PORT"
    --host "$HOST"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --optimization-level 1
    --max-logprobs "${MAX_LOGPROBS:-5}"
    --additional-config "{\"neuron_config\": {\"num_batched_tokens_buckets\": [${CONTEXT_LENGTH_BUCKETS}], \"decode_context_length_buckets\": [${DECODE_CONTEXT_LENGTH_BUCKETS}]}}"
)

# Optional: chunked prefill
if [[ "${ENABLE_CHUNKED_PREFILL:-0}" == "1" ]]; then
    SERVE_ARGS+=(--enable-chunked-prefill)
fi

# Optional: scheduling policy
if [[ -n "${SCHEDULING_POLICY:-}" ]]; then
    SERVE_ARGS+=(--scheduling-policy "$SCHEDULING_POLICY")
fi

echo "=== Serving Tongyi-30B-A3B (MEDIUM: TP=$TP_SIZE, ctx=$MAX_MODEL_LEN) ==="
echo "  Port: $PORT"
echo "  Buckets: $CONTEXT_LENGTH_BUCKETS"
echo "  KV dtype: $KV_CACHE_DTYPE"
echo "  FP8 experts: $VLLM_NEURON_FP8_EXPERT_WEIGHTS"
echo ""

# Launch vLLM (cd to .compile-artifacts so neuronx-cc metrics stay out of repo root)
mkdir -p "$REPO_ROOT/.compile-artifacts"
cd "$REPO_ROOT/.compile-artifacts"
python -m vllm.entrypoints.openai.api_server "${SERVE_ARGS[@]}" &
VLLM_PID=$!

# Wait for server ready (check health endpoint)
echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        COMPILE_END=$(date +%s)
        export NEURON_COMPILE_TIME_S=$((COMPILE_END - COMPILE_START))
        echo "Server ready! (compile+load time: ${NEURON_COMPILE_TIME_S}s)"
        break
    fi
    sleep 5
done

# Keep running
wait $VLLM_PID
