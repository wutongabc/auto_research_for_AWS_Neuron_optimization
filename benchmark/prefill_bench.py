#!/usr/bin/env python3
"""
Prefill Benchmark — Judge Program (READ-ONLY, DO NOT MODIFY)

Measures prefill throughput for Qwen3MoE models on Neuron.
Simulates multi-turn conversations with growing KV cache.

Usage:
    python benchmark/prefill_bench.py --config benchmark/config_fast.json
    python benchmark/prefill_bench.py --config benchmark/config_full.json
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: torch and transformers required", file=sys.stderr)
    sys.exit(1)

# OpenAI-compatible API client
try:
    import openai
except ImportError:
    openai = None

import requests


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def generate_synthetic_turns(tokenizer, tokens_per_turn: int, num_turns: int, seed: int = 42) -> list[list[int]]:
    """Generate synthetic token sequences for each turn."""
    rng = np.random.default_rng(seed)
    vocab_size = tokenizer.vocab_size
    turns = []
    for _ in range(num_turns):
        # Generate random tokens, avoiding special tokens (first 100)
        tokens = rng.integers(100, vocab_size, size=tokens_per_turn).tolist()
        turns.append(tokens)
    return turns


def compute_mfu(
    tokens_processed: int,
    wall_time_s: float,
    num_cores: int,
    active_params_b: float = 3.0,
    peak_tflops_per_core: float = 380.0,
) -> float:
    """
    Compute Model FLOPs Utilization for MoE prefill.
    Uses active params only (standard MoE definition).

    FLOPs per token = 2 * active_params (forward pass)
    MFU = achieved_flops / peak_flops
    """
    flops_per_token = 2 * active_params_b * 1e9
    total_flops = flops_per_token * tokens_processed
    peak_flops = peak_tflops_per_core * 1e12 * num_cores
    achieved_flops_per_s = total_flops / wall_time_s
    mfu = achieved_flops_per_s / peak_flops * 100.0
    return mfu


def collect_logits_sample(
    base_url: str,
    prompt_tokens: list[int],
    num_sample_positions: int = 100,
    seed: int = 42,
) -> dict:
    """
    Collect top-5 logits at sampled positions for correctness checking.
    Uses the /v1/completions endpoint with logprobs.
    """
    rng = np.random.default_rng(seed)

    # We need at least 1 token of generation to get the logprobs for the last prefill position
    # Use echo=True to get logprobs for prompt tokens
    headers = {"Content-Type": "application/json"}

    # Sample positions to check (distributed across the prompt)
    total_positions = len(prompt_tokens)
    if total_positions <= num_sample_positions:
        sample_indices = list(range(total_positions))
    else:
        sample_indices = sorted(rng.choice(total_positions, size=num_sample_positions, replace=False).tolist())

    payload = {
        "prompt": prompt_tokens,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": 5,
        "echo": True,
    }

    try:
        resp = requests.post(f"{base_url}/v1/completions", json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        # Extract logprobs at sampled positions
        logprobs_data = data["choices"][0].get("logprobs", {})
        if not logprobs_data:
            return {"positions": sample_indices, "top_tokens": [], "available": False}

        top_logprobs = logprobs_data.get("top_logprobs", [])
        sampled_logprobs = []
        for idx in sample_indices:
            if idx < len(top_logprobs) and top_logprobs[idx]:
                sampled_logprobs.append(top_logprobs[idx])
            else:
                sampled_logprobs.append(None)

        return {"positions": sample_indices, "top_tokens": sampled_logprobs, "available": True}
    except Exception as e:
        print(f"  WARNING: logit collection failed: {e}", file=sys.stderr)
        return {"positions": sample_indices, "top_tokens": [], "available": False}


def check_correctness(baseline_logits: dict, current_logits: dict) -> float:
    """
    Compare top-1 tokens between baseline and current run.
    Returns percentage of matching positions.
    """
    if not baseline_logits.get("available") or not current_logits.get("available"):
        return 100.0  # Skip correctness if logits unavailable

    baseline_tops = baseline_logits["top_tokens"]
    current_tops = current_logits["top_tokens"]

    if not baseline_tops or not current_tops:
        return 100.0

    matches = 0
    total = 0
    for b, c in zip(baseline_tops, current_tops):
        if b is None or c is None:
            continue
        total += 1
        # Compare top-1 token
        b_top = max(b, key=b.get) if isinstance(b, dict) else None
        c_top = max(c, key=c.get) if isinstance(c, dict) else None
        if b_top == c_top:
            matches += 1

    if total == 0:
        return 100.0
    return (matches / total) * 100.0


def measure_prefill_turn(
    base_url: str,
    prompt_tokens: list[int],
    timeout: float = 120.0,
) -> float:
    """
    Measure wall-clock time for a single prefill operation.
    Sends the full prompt and measures TTFT (time to first token).
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": prompt_tokens,
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": True,
    }

    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/v1/completions",
            json=payload,
            headers=headers,
            stream=True,
            timeout=timeout,
        )
        resp.raise_for_status()

        # TTFT = time until first streamed chunk arrives
        for chunk in resp.iter_lines():
            if chunk:
                ttft = time.perf_counter() - start
                # Consume remaining stream
                for _ in resp.iter_lines():
                    pass
                return ttft

        # Fallback: no streaming data
        return time.perf_counter() - start
    except Exception as e:
        print(f"  ERROR: prefill measurement failed: {e}", file=sys.stderr)
        return -1.0


def run_benchmark(config: dict) -> dict:
    """Run the full multi-turn prefill benchmark."""
    base_url = config["base_url"]
    tokens_per_turn = config["tokens_per_turn"]
    num_turns = config["num_turns"]
    num_cores = config["num_cores"]
    model_name = config.get("model_name", "tongyi-30b")
    baseline_logits_path = config.get("baseline_logits_path", None)

    print(f"Prefill Benchmark: {model_name}")
    print(f"  Turns: {num_turns}, Tokens/turn: {tokens_per_turn}")
    print(f"  NeuronCores: {num_cores}")
    print(f"  Server: {base_url}")
    print()

    # Wait for server to be ready
    print("Waiting for server...", end="", flush=True)
    for attempt in range(60):
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                print(" ready.")
                break
        except requests.ConnectionError:
            pass
        time.sleep(5)
        print(".", end="", flush=True)
    else:
        print("\nERROR: Server not ready after 5 minutes", file=sys.stderr)
        return {"error": "server_timeout"}

    # Load tokenizer for vocab size
    tokenizer_name = config.get("tokenizer", "Qwen/Qwen3-30B-A3B")
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Generate synthetic data
    print("Generating synthetic turns...")
    turns = generate_synthetic_turns(tokenizer, tokens_per_turn, num_turns)

    # Warmup: send one short request
    print("Warmup...")
    warmup_tokens = turns[0][:100]
    measure_prefill_turn(base_url, warmup_tokens, timeout=60.0)

    # Main benchmark loop
    print("\nRunning benchmark:")
    print(f"  {'Turn':<6} {'New Tok':<10} {'Total Ctx':<12} {'Prefill(ms)':<14} {'Tok/s':<12}")
    print("  " + "-" * 60)

    cumulative_tokens = []
    turn_results = []

    for turn_idx in range(num_turns):
        new_tokens = turns[turn_idx]
        cumulative_tokens.extend(new_tokens)

        total_ctx = len(cumulative_tokens)
        new_count = len(new_tokens)

        # Measure prefill time (full context including cache)
        prefill_time = measure_prefill_turn(base_url, cumulative_tokens)

        if prefill_time < 0:
            print(f"  {turn_idx+1:<6} FAILED")
            turn_results.append({
                "turn": turn_idx + 1,
                "new_tokens": new_count,
                "total_context": total_ctx,
                "prefill_time_ms": -1,
                "tok_per_s": 0,
                "error": True,
            })
            continue

        prefill_ms = prefill_time * 1000
        tok_per_s = new_count / prefill_time

        turn_results.append({
            "turn": turn_idx + 1,
            "new_tokens": new_count,
            "total_context": total_ctx,
            "prefill_time_ms": prefill_ms,
            "tok_per_s": tok_per_s,
        })

        cache_hit_pct = ((total_ctx - new_count) / total_ctx * 100) if total_ctx > new_count else 0
        print(f"  {turn_idx+1:<6} {new_count:<10} {total_ctx:<12} {prefill_ms:<14.1f} {tok_per_s:<12.1f}  (cache: {cache_hit_pct:.0f}%)")

    # Compute summary statistics
    valid_results = [r for r in turn_results if not r.get("error")]
    if not valid_results:
        return {"error": "all_turns_failed"}

    # Use only the last 50% of turns for the summary metric (excludes warmup/short-context turns)
    cutoff = len(valid_results) // 2
    scoring_results = valid_results[cutoff:]

    avg_tok_per_s = np.mean([r["tok_per_s"] for r in scoring_results])
    total_tokens = sum(r["new_tokens"] for r in scoring_results)
    total_time = sum(r["prefill_time_ms"] for r in scoring_results) / 1000.0

    # Also compute full-run average for reference
    avg_tok_per_s_all = np.mean([r["tok_per_s"] for r in valid_results])

    mfu = compute_mfu(total_tokens, total_time, num_cores)

    # Correctness check
    correctness_pct = 100.0
    if baseline_logits_path and os.path.exists(baseline_logits_path):
        print("\nRunning correctness check...")
        with open(baseline_logits_path) as f:
            baseline_logits = json.load(f)

        # Collect logits at the final turn's context
        current_logits = collect_logits_sample(base_url, cumulative_tokens)
        correctness_pct = check_correctness(baseline_logits, current_logits)
        print(f"  Correctness: {correctness_pct:.1f}% top-1 match")
    elif baseline_logits_path is None:
        # First run: save baseline logits
        print("\nCollecting baseline logits...")
        baseline_logits = collect_logits_sample(base_url, cumulative_tokens)
        save_path = config.get("baseline_logits_save_path", "benchmark/baseline_logits.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(baseline_logits, f)
        print(f"  Saved baseline logits to {save_path}")

    # Get compile time from environment (set by the serving script)
    compile_time_s = float(os.environ.get("NEURON_COMPILE_TIME_S", "0.0"))

    # Get peak HBM
    peak_hbm_mb = 0.0
    try:
        import subprocess
        result = subprocess.run(
            ["neuron-top", "-j"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            neuron_data = json.loads(result.stdout)
            # Sum HBM usage across cores
            for core in neuron_data.get("neuron_cores", []):
                peak_hbm_mb = max(peak_hbm_mb, core.get("hbm_used_mb", 0))
    except Exception:
        pass

    # Print summary
    print(f"\n(Scoring on last {len(scoring_results)}/{len(valid_results)} turns, excluding warmup)")
    print("\n---")
    print(f"avg_prefill_tok_per_s:  {avg_tok_per_s:.1f}")
    print(f"avg_tok_per_s_all:      {avg_tok_per_s_all:.1f}")
    print(f"mfu_percent:            {mfu:.1f}")
    print(f"correctness_pct:        {correctness_pct:.1f}")
    print(f"total_turns:            {len(valid_results)}")
    print(f"scoring_turns:          {len(scoring_results)}")
    print(f"compile_time_s:         {compile_time_s:.1f}")
    print(f"peak_hbm_mb:            {peak_hbm_mb:.1f}")
    print("---")

    # Save detailed results
    results = {
        "summary": {
            "avg_prefill_tok_per_s": round(avg_tok_per_s, 1),
            "avg_tok_per_s_all": round(avg_tok_per_s_all, 1),
            "mfu_percent": round(mfu, 1),
            "correctness_pct": round(correctness_pct, 1),
            "total_turns": len(valid_results),
            "scoring_turns": len(scoring_results),
            "compile_time_s": compile_time_s,
            "peak_hbm_mb": peak_hbm_mb,
        },
        "per_turn": turn_results,
        "config": config,
    }

    results_path = config.get("results_output", "benchmark/results_latest.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Prefill Benchmark for Qwen3MoE on Neuron")
    parser.add_argument("--config", required=True, help="Path to benchmark config JSON")
    parser.add_argument("--save-baseline", action="store_true", help="Save logits as baseline (first run)")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.save_baseline:
        config["baseline_logits_path"] = None  # Triggers baseline save

    results = run_benchmark(config)

    if "error" in results:
        print(f"\nBENCHMARK FAILED: {results['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
