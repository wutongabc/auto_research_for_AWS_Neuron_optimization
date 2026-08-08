#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Run gpt-oss-20B LONG (64K) Baseline Benchmarks across 3 Time-Points
#
# Config: TP=8, 64K context, 21×3000 tokens
#
# Time-point 1: Before Round 1 (unoptimized code)
#   - vllm-neuron fork: 94ffb83 (stock)
#   - Params: batched=1024, block=64, O3 (same params, only CODE differs)
#
# Time-point 2: After Round 1 (stock code, same as TP1)
#   - vllm-neuron fork: 94ffb83 (stock)
#   - Params: batched=1024, block=64, O3
#
# Time-point 3: After Round 2 (optimized code)
#   - vllm-neuron fork: 353e1ab (HEAD)
#   - Params: batched=1024, block=64, O3
#
# Usage:
#   bash run/run_gptoss_long_baselines.bash          # run all 3
#   bash run/run_gptoss_long_baselines.bash 1        # run only time-point 1
#   bash run/run_gptoss_long_baselines.bash 2        # run only time-point 2
#   bash run/run_gptoss_long_baselines.bash 3        # run only time-point 3
###############################################################################

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VLLM_FORK="/dev3/zigeng/bc/opt/vllm-neuron"
RESULTS_DIR="$REPO_ROOT/benchmark/baseline_results"
CONTAINER="neuron-prefill"

MODEL_PATH="openai/gpt-oss-20b"
TP_SIZE=8
MAX_MODEL_LEN=65536
PORT=8100

# Commits
VLLM_STOCK="94ffb83"
VLLM_OPTIMIZED="353e1ab"

mkdir -p "$RESULTS_DIR" "$REPO_ROOT/logs"

RUN_TP="${1:-all}"

log() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================================"
    echo ""
}

kill_server() {
    docker exec "$CONTAINER" bash -c "pkill -9 -f 'vllm.entrypoints' 2>/dev/null; pkill -9 -f 'VLLM::Worker' 2>/dev/null; exit 0" 2>/dev/null || true
    sleep 8
}

wait_for_server() {
    local timeout="${1:-7200}"
    local elapsed=0
    echo "Waiting for server to be ready (timeout: ${timeout}s)..."
    while [[ $elapsed -lt $timeout ]]; do
        if docker exec "$CONTAINER" curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "Server ready after ${elapsed}s"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        if (( elapsed % 120 == 0 )); then
            echo "  ...${elapsed}s elapsed"
        fi
    done
    echo "ERROR: Server not ready after ${timeout}s"
    return 1
}

run_benchmark() {
    local config="$1"
    local output="$2"

    docker exec "$CONTAINER" bash -c \
        "cd /dev3/zigeng/bc/opt && python benchmark/prefill_bench.py --config $config" \
        | tee "$output"
}

setup_vllm_fork() {
    local commit="$1"
    log "Setting vllm-neuron fork to $commit"
    git -C "$VLLM_FORK" checkout -f "$commit" --quiet
    echo "  vllm-neuron now at: $(git -C "$VLLM_FORK" rev-parse --short HEAD)"

    # Patch: add gpt_oss_mxfp4 to supported_quantization so the model loads on trn2
    local platform_file="$VLLM_FORK/vllm_neuron/vllm/platform.py"
    if ! grep -q 'gpt_oss_mxfp4' "$platform_file" 2>/dev/null; then
        sed -i 's/"modelopt",/"modelopt",\n        "gpt_oss_mxfp4",/' "$platform_file"
        echo "  Patched platform.py: added gpt_oss_mxfp4 to supported_quantization"
    fi
}

run_timepoint() {
    local tp="$1"
    local tp_label=""
    local batched_tokens=""
    local block_size=""
    local optlevel=""

    case "$tp" in
        1) tp_label="before_r1"; batched_tokens=1024; block_size=64; optlevel=2 ;;
        2) tp_label="after_r1"; batched_tokens=1024; block_size=64; optlevel=2 ;;
        3) tp_label="after_r2"; batched_tokens=1024; block_size=64; optlevel=2 ;;
    esac

    log "TIME-POINT $tp ($tp_label): Setting up environment"

    # Setup vllm-neuron fork
    case "$tp" in
        1|2) setup_vllm_fork "$VLLM_STOCK" ;;
        3)   setup_vllm_fork "$VLLM_OPTIMIZED" ;;
    esac

    # Kill any existing server
    kill_server

    # Ensure container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
        log "Starting container..."
        bash "$REPO_ROOT/docker/run.bash" -d
        sleep 10
    fi

    # Write benchmark config for this timepoint
    local bench_config="benchmark/config_gptoss_long_tp${tp}.json"
    cat > "$REPO_ROOT/$bench_config" << EOF
{
    "model_name": "gpt-oss-20b-long-${tp_label}",
    "base_url": "http://localhost:${PORT}",
    "tokenizer": "openai/gpt-oss-20b",
    "tokens_per_turn": 3000,
    "num_turns": 21,
    "num_cores": 8,
    "peak_tflops_per_core": 380.0,
    "active_params_b": 3.8,
    "baseline_logits_save_path": "benchmark/baseline_logits_gptoss_long.json",
    "results_output": "benchmark/baseline_results/gptoss_long_${tp_label}.json"
}
EOF

    # Start server
    local server_log="$REPO_ROOT/logs/server_gptoss_long_${tp_label}.log"
    log "Starting gpt-oss-20B server (LONG: tp=$tp, $tp_label)"
    echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN, batched_tokens=$batched_tokens, block_size=$block_size, optlevel=$optlevel"

    docker exec -d "$CONTAINER" bash -c "
        export NEURON_CC_FLAGS='--optlevel $optlevel'
        export NEURON_COMPILE_NUM_WORKERS=8
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
        export VLLM_NEURON_COMPILATION_TIMEOUT=7200
        export VLLM_NEURON_BARRIER_TIMEOUT=7200
        export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION='0.15'
        mkdir -p /dev3/zigeng/bc/opt/.compile-artifacts
        cd /dev3/zigeng/bc/opt/.compile-artifacts
        python -m vllm.entrypoints.openai.api_server \
            --model $MODEL_PATH \
            --tensor-parallel-size $TP_SIZE \
            --max-model-len $MAX_MODEL_LEN \
            --max-num-seqs 1 \
            --max-num-batched-tokens $batched_tokens \
            --block-size $block_size \
            --port $PORT \
            --host 0.0.0.0 \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --no-disable-hybrid-kv-cache-manager \
            --additional-config '{\"neuron_config\": {\"num_batched_tokens_buckets\": [$batched_tokens], \"num_seqs_buckets\": [1]}}' \
            > $server_log 2>&1
    "

    # Wait for server (128K compilation is slow)
    if ! wait_for_server 3600; then
        echo "FAILED: Server did not start for tp=$tp ($tp_label)"
        echo "Last 20 lines of server log:"
        tail -20 "$server_log" 2>/dev/null || true
        kill_server
        return 1
    fi

    # Run benchmark
    local result_log="$RESULTS_DIR/gptoss_long_${tp_label}.log"
    run_benchmark "$bench_config" "$result_log"

    # Print results
    echo ""
    echo "--- gpt-oss-20B LONG Results (tp=$tp, $tp_label) ---"
    grep "^avg_prefill_tok_per_s:\|^correctness_pct:\|^compile_time_s:" "$result_log" 2>/dev/null || echo "(no results found)"
    echo ""

    kill_server
}

restore_repo() {
    log "Restoring vllm-neuron to HEAD"
    git -C "$VLLM_FORK" checkout "$VLLM_OPTIMIZED" --quiet 2>/dev/null || true
    echo "  vllm-neuron restored to $VLLM_OPTIMIZED"
}

trap restore_repo EXIT

# Free neuron cores: stop competing containers
docker stop browsecomp-neuron 2>/dev/null || true

# Main execution
log "GPT-OSS-20B LONG (64K) BASELINE BENCHMARK — Starting $(date '+%Y-%m-%d %H:%M:%S')"
echo "Model: $MODEL_PATH"
echo "Config: TP=$TP_SIZE, ctx=$MAX_MODEL_LEN, 21 turns x 3000 tokens"
echo "Results: $RESULTS_DIR/gptoss_long_*.log"
echo ""

case "$RUN_TP" in
    1)   run_timepoint 1 ;;
    2)   run_timepoint 2 ;;
    3)   run_timepoint 3 ;;
    all)
        run_timepoint 1
        run_timepoint 2
        run_timepoint 3
        ;;
    *)
        echo "Usage: $0 [1|2|3|all]"
        exit 1
        ;;
esac

# Print summary
log "SUMMARY"
echo ""
printf "%-20s %-20s\n" "Time-point" "Long (TP=8, 128K)"
printf "%-20s %-20s\n" "----------" "-----------------"
for tp_label in before_r1 after_r1 after_r2; do
    tps=$(grep "^avg_prefill_tok_per_s:" "$RESULTS_DIR/gptoss_long_${tp_label}.log" 2>/dev/null | awk '{print $2}') || true
    [[ -z "$tps" ]] && tps="N/A"
    printf "%-20s %-20s\n" "$tp_label" "$tps"
done
echo ""
log "DONE — $(date '+%Y-%m-%d %H:%M:%S')"
