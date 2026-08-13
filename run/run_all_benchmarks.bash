#!/usr/bin/env bash
set -euo pipefail

# Master benchmark orchestration script.
# Runs baseline and optimized benchmarks for all models using 64 NeuronCores.
#
# Phase 1: Baseline benchmarks (uses opt-baseline/ directory)
# Phase 2: Optimized benchmarks (uses opt/ directory)
#
# Core allocation per experiment set:
#   OSS-20B official:   TP=8,  cores 0-7,    port 8100
#   OSS-20B full:       TP=8,  cores 8-15,   port 8101
#   OSS-120B official:  TP=16, cores 16-31,  port 8102
#   OSS-120B full:      TP=16, cores 32-47,  port 8103
#   Qwen3-VL official:  TP=16, cores 48-63,  port 8104
#   (Qwen3-VL full runs after oss-120b official finishes, reuse cores 16-23)
#
# Usage:
#   bash run/run_all_benchmarks.bash [baseline|optimized|all]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PHASE="${1:-all}"

OPT_DIR="/dev3/zigeng/bc/opt"
BASELINE_DIR="/dev3/zigeng/bc/opt-baseline"
RESULTS_DIR="$REPO_ROOT/benchmark/all_results"
mkdir -p "$RESULTS_DIR"

run_benchmark() {
    local container="$1"
    local cores="$2"
    local src_dir="$3"
    local port="$4"
    local serve_script="$5"
    local bench_config="$6"
    local result_name="$7"

    echo ">>> Starting: $result_name (container=$container, cores=$cores, port=$port)"

    # Launch container
    bash "$REPO_ROOT/docker/run_bench.bash" "$container" "$cores" "$src_dir" "$port"

    # Start server inside container
    docker exec -d "$container" bash -c \
        "cd /dev3/zigeng/bc/opt && PORT=$port bash $serve_script > /tmp/server.log 2>&1"

    # Wait for server
    echo "    Waiting for server on port $port..."
    local ready=0
    for i in $(seq 1 600); do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            ready=1
            echo "    Server ready after ${i}x5s"
            break
        fi
        sleep 5
    done

    if [[ $ready -eq 0 ]]; then
        echo "    ERROR: Server failed to start. Logs:"
        docker exec "$container" cat /tmp/server.log | tail -50
        docker stop "$container" &>/dev/null || true
        docker rm "$container" &>/dev/null || true
        return 1
    fi

    # Run benchmark (generate baseline logits if not exist, then measure)
    docker exec "$container" bash -c \
        "cd /dev3/zigeng/bc/opt && python benchmark/prefill_bench.py --config $bench_config --save-baseline" \
        > "$RESULTS_DIR/${result_name}.log" 2>&1 || true

    # Extract results
    local tok_s=$(grep "^avg_prefill_tok_per_s:" "$RESULTS_DIR/${result_name}.log" 2>/dev/null | awk '{print $2}')
    local correctness=$(grep "^correctness_pct:" "$RESULTS_DIR/${result_name}.log" 2>/dev/null | awk '{print $2}')
    echo "    Result: ${tok_s:-FAIL} tok/s, ${correctness:-N/A}% correctness"
    echo "$result_name,$tok_s,$correctness" >> "$RESULTS_DIR/summary.csv"

    # Cleanup
    docker stop "$container" &>/dev/null || true
    docker rm "$container" &>/dev/null || true
    echo "    Done: $result_name"
}

echo "=========================================="
echo "  Multi-Model Benchmark Suite"
echo "  Phase: $PHASE"
echo "  Date: $(date -Iseconds)"
echo "=========================================="
echo ""

echo "model,tok_per_s,correctness_pct" > "$RESULTS_DIR/summary.csv"

if [[ "$PHASE" == "baseline" || "$PHASE" == "all" ]]; then
    echo ""
    echo "=== PHASE 1: Baseline Benchmarks ==="
    echo ""

    # Run baseline experiments in parallel batches
    # Batch 1: OSS-20B (8 cores) + OSS-120B (16 cores) + Qwen3-VL (16 cores)
    run_benchmark "bl-oss20b-off" "0,1,2,3,4,5,6,7" "$BASELINE_DIR" 8100 \
        "run/serve_oss20b_official.bash" "benchmark/config_oss20b_official.json" "baseline_oss20b_official" &
    PID1=$!

    run_benchmark "bl-oss120b-off" "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" "$BASELINE_DIR" 8102 \
        "run/serve_oss120b_official.bash" "benchmark/config_oss120b_official.json" "baseline_oss120b_official" &
    PID2=$!

    run_benchmark "bl-qwen3vl-off" "48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63" "$BASELINE_DIR" 8104 \
        "run/serve_qwen3vl_official.bash" "benchmark/config_qwen3vl_official.json" "baseline_qwen3vl_official" &
    PID3=$!

    wait $PID1 $PID2 $PID3 || true

    # Batch 2: Full configs
    run_benchmark "bl-oss20b-full" "0,1,2,3,4,5,6,7" "$BASELINE_DIR" 8100 \
        "run/serve_oss20b_full.bash" "benchmark/config_oss20b_full.json" "baseline_oss20b_full" &
    PID4=$!

    run_benchmark "bl-oss120b-full" "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" "$BASELINE_DIR" 8103 \
        "run/serve_oss120b_full.bash" "benchmark/config_oss120b_full.json" "baseline_oss120b_full" &
    PID5=$!

    run_benchmark "bl-qwen3vl-full" "32,33,34,35,36,37,38,39" "$BASELINE_DIR" 8105 \
        "run/serve_qwen3vl_full.bash" "benchmark/config_qwen3vl_full.json" "baseline_qwen3vl_full" &
    PID6=$!

    wait $PID4 $PID5 $PID6 || true

    echo ""
    echo "=== Baseline phase complete ==="
fi

if [[ "$PHASE" == "optimized" || "$PHASE" == "all" ]]; then
    echo ""
    echo "=== PHASE 2: Optimized Benchmarks ==="
    echo ""

    # Batch 1: Official configs
    run_benchmark "opt-oss20b-off" "0,1,2,3,4,5,6,7" "$OPT_DIR" 8100 \
        "run/serve_oss20b_official.bash" "benchmark/config_oss20b_official.json" "optimized_oss20b_official" &
    PID1=$!

    run_benchmark "opt-oss120b-off" "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" "$OPT_DIR" 8102 \
        "run/serve_oss120b_official.bash" "benchmark/config_oss120b_official.json" "optimized_oss120b_official" &
    PID2=$!

    run_benchmark "opt-qwen3vl-off" "48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63" "$OPT_DIR" 8104 \
        "run/serve_qwen3vl_official.bash" "benchmark/config_qwen3vl_official.json" "optimized_qwen3vl_official" &
    PID3=$!

    wait $PID1 $PID2 $PID3 || true

    # Batch 2: Full configs
    run_benchmark "opt-oss20b-full" "0,1,2,3,4,5,6,7" "$OPT_DIR" 8100 \
        "run/serve_oss20b_full.bash" "benchmark/config_oss20b_full.json" "optimized_oss20b_full" &
    PID4=$!

    run_benchmark "opt-oss120b-full" "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" "$OPT_DIR" 8103 \
        "run/serve_oss120b_full.bash" "benchmark/config_oss120b_full.json" "optimized_oss120b_full" &
    PID5=$!

    run_benchmark "opt-qwen3vl-full" "32,33,34,35,36,37,38,39" "$OPT_DIR" 8105 \
        "run/serve_qwen3vl_full.bash" "benchmark/config_qwen3vl_full.json" "optimized_qwen3vl_full" &
    PID6=$!

    wait $PID4 $PID5 $PID6 || true

    echo ""
    echo "=== Optimized phase complete ==="
fi

echo ""
echo "=========================================="
echo "  ALL BENCHMARKS COMPLETE"
echo "  Results: $RESULTS_DIR/summary.csv"
echo "=========================================="
cat "$RESULTS_DIR/summary.csv"
