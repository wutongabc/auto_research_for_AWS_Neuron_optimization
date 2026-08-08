#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Run All 6 Baseline Benchmarks
#
# Matrix: 3 time-points × 2 configs (medium TP=4, full TP=8)
#
# Time-point 1: Before Round 1 (unoptimized)
#   - vllm-neuron fork: 94ffb83 (stock)
#   - nkilib-fork: original (8d61d09 state, _MAX_SEQLEN=131072)
#   - config.env: from commit a616588
#   - serve_medium: TP=4, MAX_MODEL_LEN=32768
#   - serve_full: TP=8, MAX_MODEL_LEN=131072
#
# Time-point 2: After Round 1 / Before Round 2 (param-optimized)
#   - vllm-neuron fork: 94ffb83 (stock — same as TP1)
#   - nkilib-fork: original (same as TP1)
#   - config.env: from commit 4b55f0b
#   - serve_medium: TP=4, MAX_MODEL_LEN=32768
#   - serve_full: TP=8, MAX_MODEL_LEN=131072
#
# Time-point 3: After Round 2 (fully optimized, current HEAD)
#   - vllm-neuron fork: 353e1ab (HEAD)
#   - nkilib-fork: current (HEAD, _MAX_SEQLEN=132096)
#   - config.env: current HEAD
#   - serve_medium: TP=4, MAX_MODEL_LEN=30208
#   - serve_full: TP=8, MAX_MODEL_LEN=131072
#
# Usage:
#   bash run/run_all_baselines.bash          # run all 6
#   bash run/run_all_baselines.bash 1        # run only time-point 1 (benchmarks #1 and #2)
#   bash run/run_all_baselines.bash 2        # run only time-point 2
#   bash run/run_all_baselines.bash 3        # run only time-point 3
###############################################################################

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VLLM_FORK="/dev3/zigeng/bc/opt/vllm-neuron"
NKILIB_FORK="$REPO_ROOT/nkilib-fork"
RESULTS_DIR="$REPO_ROOT/benchmark/baseline_results"
CONTAINER="neuron-prefill"

# Commits
VLLM_BEFORE_R1="94ffb83"
VLLM_BEFORE_R2="94ffb83"
VLLM_AFTER_R2="353e1ab"

NKILIB_ORIGINAL_SEQLEN="131072"
NKILIB_FINAL_SEQLEN="132096"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Determine which time-points to run
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
    docker exec "$CONTAINER" bash -c "pkill -f 'vllm.entrypoints' 2>/dev/null; exit 0" 2>/dev/null || true
    sleep 5
}

wait_for_server() {
    local timeout="${1:-600}"
    local elapsed=0
    echo "Waiting for server to be ready (timeout: ${timeout}s)..."
    while [[ $elapsed -lt $timeout ]]; do
        if docker exec "$CONTAINER" curl -s "http://localhost:8100/health" > /dev/null 2>&1; then
            echo "Server ready after ${elapsed}s"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        echo "  ...${elapsed}s elapsed"
    done
    echo "ERROR: Server not ready after ${timeout}s"
    return 1
}

run_benchmark() {
    local config="$1"
    local output="$2"
    local save_baseline="${3:-false}"

    local bench_args="--config $config"
    if [[ "$save_baseline" == "true" ]]; then
        bench_args="$bench_args --save-baseline"
    fi

    docker exec "$CONTAINER" bash -c \
        "cd /dev3/zigeng/bc/opt && python benchmark/prefill_bench.py $bench_args" \
        | tee "$output"
}

setup_vllm_fork() {
    local commit="$1"
    log "Setting vllm-neuron fork to $commit"
    git -C "$VLLM_FORK" checkout "$commit" --quiet
    echo "  vllm-neuron now at: $(git -C "$VLLM_FORK" rev-parse --short HEAD)"
}

setup_nkilib_seqlen() {
    local seqlen="$1"
    log "Setting nkilib-fork _MAX_SEQLEN to $seqlen"
    sed -i "s/_MAX_SEQLEN = [0-9]*/_MAX_SEQLEN = $seqlen/" \
        "$NKILIB_FORK/core/attention/attention_cte.py"
    grep "_MAX_SEQLEN" "$NKILIB_FORK/core/attention/attention_cte.py" | head -1
}

write_config_env() {
    local timepoint="$1"
    log "Writing config.env for time-point $timepoint"

    case "$timepoint" in
        1)
            # From commit a616588: unoptimized
            cat > "$REPO_ROOT/run/config.env" << 'ENVEOF'
# Neuron Prefill Optimization — Environment Configuration
# BENCHMARK: Time-point 1 (Before Round 1, unoptimized)

# === Model ===
MODEL_PATH="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"

# === Server ===
PORT=8100
HOST="0.0.0.0"

# === Neuron Compilation ===
NEURON_CC_FLAGS="--optlevel 2"
NEURON_COMPILE_WORKERS=8

# === vLLM Serving Parameters ===
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=1
BLOCK_SIZE=32
KV_CACHE_DTYPE="fp8"

# === Bucket Configuration ===
CONTEXT_LENGTH_BUCKETS="4096"
DECODE_CONTEXT_LENGTH_BUCKETS="8192"

# === MoE Configuration ===
VLLM_NEURON_FUSED_MOE_NKI=1
VLLM_NEURON_FP8_EXPERT_WEIGHTS=0
VLLM_NEURON_FP8_ONLY=0

# === Scheduling ===
SCHEDULING_POLICY="fcfs"
ENABLE_CHUNKED_PREFILL=1

# === Memory ===
KV_BUDGET_CAP=0.05
ENVEOF
            ;;
        2)
            # From commit 4b55f0b: param-optimized
            cat > "$REPO_ROOT/run/config.env" << 'ENVEOF'
# Neuron Prefill Optimization — Environment Configuration
# BENCHMARK: Time-point 2 (After Round 1, param-optimized)

# === Model ===
MODEL_PATH="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"

# === Server ===
PORT=8100
HOST="0.0.0.0"

# === Neuron Compilation ===
NEURON_CC_FLAGS="--optlevel 3"
NEURON_COMPILE_WORKERS=8

# === vLLM Serving Parameters ===
MAX_NUM_BATCHED_TOKENS=512
MAX_NUM_SEQS=1
BLOCK_SIZE=64
KV_CACHE_DTYPE="auto"

# === Bucket Configuration ===
CONTEXT_LENGTH_BUCKETS="512"
DECODE_CONTEXT_LENGTH_BUCKETS="8192"

# === MoE Configuration ===
VLLM_NEURON_FUSED_MOE_NKI=1
VLLM_NEURON_FP8_EXPERT_WEIGHTS=0
VLLM_NEURON_FP8_ONLY=0

# === Scheduling ===
SCHEDULING_POLICY="fcfs"
ENABLE_CHUNKED_PREFILL=1

# === Memory ===
KV_BUDGET_CAP=0.05
ENVEOF
            ;;
        3)
            # Current HEAD: fully optimized
            cat > "$REPO_ROOT/run/config.env" << 'ENVEOF'
# Neuron Prefill Optimization — Environment Configuration
# BENCHMARK: Time-point 3 (After Round 2, fully optimized)

# === Model ===
MODEL_PATH="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"

# === Server ===
PORT=8100
HOST="0.0.0.0"

# === Neuron Compilation ===
NEURON_CC_FLAGS="--optlevel 3"
NEURON_COMPILE_WORKERS=8

# === vLLM Serving Parameters ===
MAX_NUM_BATCHED_TOKENS=1024
MAX_NUM_SEQS=1
BLOCK_SIZE=64
KV_CACHE_DTYPE="auto"

# === Bucket Configuration ===
CONTEXT_LENGTH_BUCKETS="1024"
DECODE_CONTEXT_LENGTH_BUCKETS="8192"

# === MoE Configuration ===
VLLM_NEURON_FUSED_MOE_NKI=1
VLLM_NEURON_FP8_EXPERT_WEIGHTS=0
VLLM_NEURON_FP8_ONLY=0

# === Scheduling ===
SCHEDULING_POLICY="fcfs"
ENABLE_CHUNKED_PREFILL=1

# === Memory ===
KV_BUDGET_CAP=0.05
ENVEOF
            ;;
    esac
    echo "  config.env written"
}

set_medium_model_len() {
    local model_len="$1"
    sed -i "s/^MAX_MODEL_LEN=.*/MAX_MODEL_LEN=$model_len/" "$REPO_ROOT/run/serve_medium.bash"
    echo "  serve_medium.bash MAX_MODEL_LEN=$model_len"
}

run_timepoint() {
    local tp="$1"
    local tp_label=""
    local medium_model_len=""

    case "$tp" in
        1) tp_label="before_r1"; medium_model_len=32768 ;;
        2) tp_label="after_r1"; medium_model_len=32768 ;;
        3) tp_label="after_r2"; medium_model_len=30208 ;;
    esac

    log "TIME-POINT $tp ($tp_label): Setting up environment"

    # Setup vllm-neuron fork
    case "$tp" in
        1|2) setup_vllm_fork "$VLLM_BEFORE_R1" ;;
        3)   setup_vllm_fork "$VLLM_AFTER_R2" ;;
    esac

    # Setup nkilib-fork
    case "$tp" in
        1|2) setup_nkilib_seqlen "$NKILIB_ORIGINAL_SEQLEN" ;;
        3)   setup_nkilib_seqlen "$NKILIB_FINAL_SEQLEN" ;;
    esac

    # Write config.env
    write_config_env "$tp"

    # Set medium model len
    set_medium_model_len "$medium_model_len"

    # Ensure container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
        log "Starting container..."
        bash "$REPO_ROOT/docker/run.bash" -d
        sleep 10
    fi

    # --- MEDIUM BENCHMARK (TP=4) ---
    log "TIME-POINT $tp ($tp_label): Running MEDIUM benchmark (TP=4, ctx=${medium_model_len})"
    kill_server

    local medium_log="$RESULTS_DIR/medium_${tp_label}.log"
    local medium_results="benchmark/baseline_results/medium_${tp_label}.json"
    local save_baseline="false"

    # First medium run (tp=1) saves baseline logits
    if [[ "$tp" == "1" ]] && [[ ! -f "$REPO_ROOT/benchmark/baseline_logits_medium.json" ]]; then
        save_baseline="true"
    fi

    # Update benchmark config to use our results output path
    local medium_config="/tmp/config_medium_tp${tp}.json"
    cat > "$medium_config" << EOF
{
    "model_name": "tongyi-30b-medium-tp${tp}",
    "base_url": "http://localhost:8100",
    "tokenizer": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "tokens_per_turn": 3000,
    "num_turns": 10,
    "num_cores": 4,
    "peak_tflops_per_core": 380.0,
    "active_params_b": 3.0,
    "baseline_logits_path": "benchmark/baseline_logits_medium.json",
    "baseline_logits_save_path": "benchmark/baseline_logits_medium.json",
    "results_output": "$medium_results"
}
EOF

    if [[ "$save_baseline" == "true" ]]; then
        # Remove baseline_logits_path so it triggers save
        cat > "$medium_config" << EOF
{
    "model_name": "tongyi-30b-medium-tp${tp}",
    "base_url": "http://localhost:8100",
    "tokenizer": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "tokens_per_turn": 3000,
    "num_turns": 10,
    "num_cores": 4,
    "peak_tflops_per_core": 380.0,
    "active_params_b": 3.0,
    "baseline_logits_save_path": "benchmark/baseline_logits_medium.json",
    "results_output": "$medium_results"
}
EOF
    fi

    # Copy config into container-accessible path
    cp "$medium_config" "$REPO_ROOT/benchmark/config_medium_bench.json"

    # Start medium server
    docker exec -d "$CONTAINER" bash -c \
        "cd /dev3/zigeng/bc/opt && bash run/serve_medium.bash > logs/server_baseline_medium_${tp_label}.log 2>&1"

    if ! wait_for_server 900; then
        echo "FAILED: Medium server did not start for tp=$tp" | tee -a "$medium_log"
        kill_server
    else
        # Run benchmark
        run_benchmark "benchmark/config_medium_bench.json" "$medium_log" "$save_baseline"
        kill_server
    fi

    echo ""
    echo "--- Medium Results (tp=$tp, $tp_label) ---"
    grep "^avg_prefill_tok_per_s:\|^correctness_pct:\|^compile_time_s:" "$medium_log" 2>/dev/null || true
    echo ""

    # --- FULL BENCHMARK (TP=8) ---
    log "TIME-POINT $tp ($tp_label): Running FULL benchmark (TP=8, ctx=131072)"
    kill_server

    local full_log="$RESULTS_DIR/full_${tp_label}.log"
    local full_results="benchmark/baseline_results/full_${tp_label}.json"
    save_baseline="false"

    # First full run (tp=1) saves baseline logits
    if [[ "$tp" == "1" ]] && [[ ! -f "$REPO_ROOT/benchmark/baseline_logits_full.json" ]]; then
        save_baseline="true"
    fi

    local full_config="/tmp/config_full_tp${tp}.json"
    cat > "$full_config" << EOF
{
    "model_name": "tongyi-30b-full-tp${tp}",
    "base_url": "http://localhost:8100",
    "tokenizer": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "tokens_per_turn": 3000,
    "num_turns": 42,
    "num_cores": 8,
    "peak_tflops_per_core": 380.0,
    "active_params_b": 3.0,
    "baseline_logits_path": "benchmark/baseline_logits_full.json",
    "baseline_logits_save_path": "benchmark/baseline_logits_full.json",
    "results_output": "$full_results"
}
EOF

    if [[ "$save_baseline" == "true" ]]; then
        cat > "$full_config" << EOF
{
    "model_name": "tongyi-30b-full-tp${tp}",
    "base_url": "http://localhost:8100",
    "tokenizer": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "tokens_per_turn": 3000,
    "num_turns": 42,
    "num_cores": 8,
    "peak_tflops_per_core": 380.0,
    "active_params_b": 3.0,
    "baseline_logits_save_path": "benchmark/baseline_logits_full.json",
    "results_output": "$full_results"
}
EOF
    fi

    cp "$full_config" "$REPO_ROOT/benchmark/config_full_bench.json"

    # Start full server (no NEURON_VISIBLE_DEVICES restriction — uses all 8 cores)
    docker exec -d "$CONTAINER" bash -c \
        "cd /dev3/zigeng/bc/opt && bash run/serve_full.bash > logs/server_baseline_full_${tp_label}.log 2>&1"

    if ! wait_for_server 1800; then
        echo "FAILED: Full server did not start for tp=$tp" | tee -a "$full_log"
        kill_server
    else
        run_benchmark "benchmark/config_full_bench.json" "$full_log" "$save_baseline"
        kill_server
    fi

    echo ""
    echo "--- Full Results (tp=$tp, $tp_label) ---"
    grep "^avg_prefill_tok_per_s:\|^correctness_pct:\|^compile_time_s:" "$full_log" 2>/dev/null || true
    echo ""
}

restore_repo() {
    log "Restoring repository to original state (HEAD)"
    git -C "$VLLM_FORK" checkout 353e1ab --quiet 2>/dev/null || true
    setup_nkilib_seqlen "$NKILIB_FINAL_SEQLEN" 2>/dev/null || true
    git -C "$REPO_ROOT" checkout HEAD -- run/config.env run/serve_medium.bash 2>/dev/null || true
    echo "  Repository restored to HEAD state"
}

# Trap to restore on exit
trap restore_repo EXIT

# Main execution
log "BASELINE BENCHMARK SUITE — Starting $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

case "$RUN_TP" in
    1)   run_timepoint 1 ;;
    2)   run_timepoint 2 ;;
    3)   run_timepoint 3 ;;
    all)
        run_timepoint 1
        # Skip TP2 and TP3: reliable data already exists
        # TP2 medium: 844.7 tok/s (results.tsv "MEDIUM BASELINE", two consistent measurements)
        # TP2 full:   364.5 tok/s (results.tsv "898a396" + "f3e5fd8", two independent runs match)
        # TP3 medium: 4269.0 tok/s (results.tsv "phase3_final", confirmed across 3 runs: 4258/4274/4269)
        # TP3 full:   1533.2 tok/s (results.tsv "full_val_final2", 3 runs: 1521/1528/1533)
        #
        # Uncomment below to re-run for consistency:
        # run_timepoint 2
        # run_timepoint 3
        ;;
    *)
        echo "Usage: $0 [1|2|3|all]"
        exit 1
        ;;
esac

# Print summary
log "SUMMARY"
echo ""
printf "%-20s %-15s %-15s\n" "Time-point" "Medium (TP=4)" "Full (TP=8)"
printf "%-20s %-15s %-15s\n" "----------" "------------" "----------"
for tp_label in before_r1 after_r1 after_r2; do
    medium_tps=$(grep "^avg_prefill_tok_per_s:" "$RESULTS_DIR/medium_${tp_label}.log" 2>/dev/null | awk '{print $2}') || true
    full_tps=$(grep "^avg_prefill_tok_per_s:" "$RESULTS_DIR/full_${tp_label}.log" 2>/dev/null | awk '{print $2}') || true
    [[ -z "$medium_tps" ]] && medium_tps="N/A"
    [[ -z "$full_tps" ]] && full_tps="N/A"
    printf "%-20s %-15s %-15s\n" "$tp_label" "$medium_tps" "$full_tps"
done
echo ""
echo "Baseline logits saved to:"
echo "  benchmark/baseline_logits_medium.json"
echo "  benchmark/baseline_logits_full.json"
echo ""
log "DONE — $(date '+%Y-%m-%d %H:%M:%S')"
