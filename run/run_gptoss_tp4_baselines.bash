#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Run gpt-oss-20B Baseline Benchmarks — All 3 Time-Points at TP=4
#
# Uses NeuronDevice 0 (cores 0-3) only. Does NOT affect cores 8-63
# or the browsecomp-neuron container.
#
# Time-point 1: Before Round 1 (unoptimized)
#   - vllm-neuron fork: 94ffb83 (stock)
#   - Params: MAX_NUM_BATCHED_TOKENS=4096, BLOCK_SIZE=32, optlevel 2
#
# Time-point 2: After Round 1 (param-optimized)
#   - vllm-neuron fork: 94ffb83 (stock)
#   - Params: MAX_NUM_BATCHED_TOKENS=512, BLOCK_SIZE=64, optlevel 3
#
# Time-point 3: After Round 2 (fully optimized)
#   - vllm-neuron fork: 353e1ab (HEAD, model-level optimizations)
#   - Params: MAX_NUM_BATCHED_TOKENS=1024, BLOCK_SIZE=64, optlevel 3
#
# Usage:
#   bash run/run_gptoss_tp4_baselines.bash
###############################################################################

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VLLM_FORK="/dev3/zigeng/bc/opt/vllm-neuron"
RESULTS_DIR="$REPO_ROOT/benchmark/baseline_results"
CONTAINER="neuron-prefill"

MODEL_PATH="openai/gpt-oss-20b"
TP_SIZE=4
MAX_MODEL_LEN=16384
PORT=8200

# Commits
VLLM_STOCK="94ffb83"
VLLM_OPTIMIZED="353e1ab"

mkdir -p "$RESULTS_DIR" "$REPO_ROOT/logs"

log() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================================"
    echo ""
}

kill_server() {
    docker exec "$CONTAINER" bash -c "pkill -f 'vllm.entrypoints' 2>/dev/null; exit 0" 2>/dev/null || true
    sleep 5
}

wait_for_server() {
    local timeout="${1:-3600}"
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

setup_vllm_fork() {
    local commit="$1"
    log "Setting vllm-neuron fork to $commit"
    git -C "$VLLM_FORK" checkout "$commit" --quiet
    echo "  vllm-neuron now at: $(git -C "$VLLM_FORK" rev-parse --short HEAD)"

    local platform_file="$VLLM_FORK/vllm_neuron/vllm/platform.py"
    if ! grep -q 'gpt_oss_mxfp4' "$platform_file" 2>/dev/null; then
        sed -i 's/"modelopt",/"modelopt",\n        "gpt_oss_mxfp4",/' "$platform_file"
        echo "  Patched platform.py: added gpt_oss_mxfp4 to supported_quantization"
    fi
}

ensure_container() {
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
        echo "Container $CONTAINER already running"
        return 0
    fi

    # Remove stale container if exists
    docker rm -f "$CONTAINER" 2>/dev/null || true

    log "Starting container $CONTAINER (device 0 only, cores 0-3)"

    local ZIGENG_ROOT="/dev3/zigeng"
    local BC_ROOT="$ZIGENG_ROOT/bc"
    local BROWSECOMP_ROOT="$BC_ROOT/BrowseComp-Plus"
    local HF_CACHE_DIR="$BROWSECOMP_ROOT/local_models/huggingface"
    local NKILIB_FORK="$REPO_ROOT/nkilib-fork"

    docker run -d \
        --name "$CONTAINER" \
        --network host \
        --privileged \
        $(for d in /dev/neuron*; do [[ -c "$d" ]] && echo "--device=$d:$d"; done) \
        -v "$ZIGENG_ROOT:/dev3/zigeng" \
        -v "$REPO_ROOT:/dev3/zigeng/bc/opt" \
        -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
        -v "$VLLM_FORK:/opt/vllm-neuron" \
        -v "$NKILIB_FORK:/opt/conda/lib/python3.13/site-packages/nkilib" \
        -w /dev3/zigeng/bc/opt \
        -e "HF_HOME=/root/.cache/huggingface" \
        -e "TRANSFORMERS_CACHE=/root/.cache/huggingface" \
        -e "PYTHONPATH=/opt/vllm-neuron" \
        -e "PYTHONDONTWRITEBYTECODE=1" \
        -e "NEURON_VISIBLE_DEVICES=0,1,2,3" \
        -e "FIRST_CORE=0" \
        -e "NEURON_SKIP_EFA_AFFINITY=1" \
        ${HF_TOKEN:+-e "HF_TOKEN=$HF_TOKEN"} \
        browsecomp-neuron:0.21.0 /bin/bash -c "tail -f /dev/null"

    sleep 5
    echo "  Container started, NEURON_VISIBLE_DEVICES=0 (cores 0-3)"
}

run_timepoint() {
    local tp="$1"
    local tp_label=""
    local batched_tokens=""
    local block_size=""
    local optlevel=""

    case "$tp" in
        1) tp_label="before_r1"; batched_tokens=1024; block_size=32; optlevel=2 ;;
        2) tp_label="after_r1"; batched_tokens=512; block_size=64; optlevel=3 ;;
        3) tp_label="after_r2"; batched_tokens=1024; block_size=64; optlevel=3 ;;
    esac

    log "TIME-POINT $tp ($tp_label): TP=4, batched=$batched_tokens, block=$block_size, opt=$optlevel"

    # Setup vllm-neuron fork
    case "$tp" in
        1|2) setup_vllm_fork "$VLLM_STOCK" ;;
        3)   setup_vllm_fork "$VLLM_OPTIMIZED" ;;
    esac

    kill_server

    # Write benchmark config
    local bench_config="benchmark/config_gptoss_tp4_${tp_label}.json"
    cat > "$REPO_ROOT/$bench_config" << EOF
{
    "model_name": "gpt-oss-20b-tp4-${tp_label}",
    "base_url": "http://localhost:${PORT}",
    "tokenizer": "openai/gpt-oss-20b",
    "tokens_per_turn": 3000,
    "num_turns": 10,
    "num_cores": 4,
    "peak_tflops_per_core": 380.0,
    "active_params_b": 3.8,
    "baseline_logits_save_path": "benchmark/baseline_logits_gptoss_short.json",
    "results_output": "benchmark/baseline_results/gptoss_tp4_${tp_label}.json"
}
EOF

    # Start server
    local server_log="$REPO_ROOT/logs/server_gptoss_tp4_${tp_label}.log"
    log "Starting gpt-oss-20B server (TP=4, $tp_label)"

    docker exec -d "$CONTAINER" bash -c "
        export NEURON_CC_FLAGS='--optlevel $optlevel'
        export NEURON_COMPILE_NUM_WORKERS=8
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
        export NEURON_VISIBLE_DEVICES=0,1,2,3
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

    if ! wait_for_server 3600; then
        echo "FAILED: Server did not start for tp=$tp ($tp_label)"
        echo "Last 20 lines of server log:"
        tail -20 "$server_log" 2>/dev/null || true
        kill_server
        return 1
    fi

    # Run benchmark
    local result_log="$RESULTS_DIR/gptoss_tp4_${tp_label}.log"
    docker exec "$CONTAINER" bash -c \
        "cd /dev3/zigeng/bc/opt && python benchmark/prefill_bench.py --config $bench_config" \
        | tee "$result_log"

    echo ""
    echo "--- gpt-oss-20B Results (TP=4, $tp_label) ---"
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

# Main
log "GPT-OSS-20B TP=4 BASELINE BENCHMARK (cores 0-3 only)"
echo "Model: $MODEL_PATH"
echo "Config: TP=4, ctx=$MAX_MODEL_LEN, 10 turns x 3000 tokens"
echo "Port: $PORT (won't conflict with other services)"
echo "Results: $RESULTS_DIR/gptoss_tp4_*.log"
echo ""

ensure_container

run_timepoint 1
run_timepoint 2
run_timepoint 3

# Summary
log "SUMMARY"
echo ""
printf "%-20s %-20s\n" "Time-point" "TP=4, 16K ctx"
printf "%-20s %-20s\n" "----------" "-------------"
for tp_label in before_r1 after_r1 after_r2; do
    tps=$(grep "^avg_prefill_tok_per_s:" "$RESULTS_DIR/gptoss_tp4_${tp_label}.log" 2>/dev/null | awk '{print $2}') || true
    [[ -z "$tps" ]] && tps="N/A"
    printf "%-20s %-20s\n" "$tp_label" "$tps"
done
echo ""
log "DONE"
