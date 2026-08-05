# Neuron Prefill Optimization

This is an autonomous optimization loop for Tongyi-30B-A3B (Qwen3MoE) prefill throughput on AWS Trainium 2.

## Setup

To set up a new optimization run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `aug3`). The branch `opt/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b opt/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `program.md` — this file. The rules of the game.
   - `benchmark/prefill_bench.py` — the judge program. **Do not modify.**
   - `benchmark/config_fast.json` — fast model config. **Do not modify.**
   - `benchmark/config_medium.json` — medium model config (loop default). **Do not modify.**
   - `benchmark/config_full.json` — full model config. **Do not modify.**
   - `run/serve_fast.bash` — launch script for fast model. You CAN modify.
   - `run/serve_medium.bash` — launch script for medium model. You CAN modify.
   - `run/config.env` — environment variables. You CAN modify.
   - `docker/run.bash` — docker launch script. **Do not modify.**
4. **Verify environment**: Check that the docker container `neuron-prefill` can start:
   ```bash
   bash docker/run.bash -d
   ```
5. **Establish baseline**: Run both vanilla and patched configurations on the fast model. The better-performing one becomes the baseline; the other is recorded as a reference floor.
6. **Initialize results.tsv**: Create `results.tsv` with the header row and baseline entry.
7. **Confirm and go**: Confirm setup looks good, then begin the loop.

## Architecture

**Model**: Alibaba-NLP/Tongyi-DeepResearch-30B-A3B (Qwen3MoE)
- 30B total params, 3B active per token
- 128 experts, top-8 routing, 48 layers
- Hidden: 2048, MoE intermediate: 768, GQA (32 heads, 4 KV heads)

**Three configurations**:
- **Fast model** (quick iteration): TP=4, max_model_len=16384, 1200 tokens/turn, ~13 turns
- **Medium model** (loop): TP=4, max_model_len=32768, 3000 tokens/turn, 10 turns — shows long-context attention behavior with same compile time as fast
- **Full model** (end validation): TP=8, max_model_len=131072, 3000 tokens/turn, ~42 turns

**Hardware**: trn2, NeuronCores 0-7 (8 cores total). Fast model uses 4 cores, full model uses all 8.

## Metrics

The benchmark reports:
1. **prefill_tok_per_s**: new_tokens / wall_clock_prefill_time (end-to-end, includes attention over cached KV)
2. **mfu_percent**: Model FLOPs Utilization using active params only. Peak = 380 TFLOPS/core BF16.
3. **correctness**: Top-5 logit comparison against baseline. Pass if >99% of positions match top-1.
4. **Per-turn curve**: tok/s at each context length (turn).

The single summary number for keep/discard decisions is the **average prefill_tok_per_s across all turns**.

## What you CAN modify

### Phase 1 — Launch Parameters (budget: 2 hours)
- `run/serve_fast.bash` — serving script for fast model (quick iteration)
- `run/serve_medium.bash` — serving script for medium model (loop default)
- `run/config.env` — environment variables (bucket sizes, batch sizes, scheduling flags)

### Phase 2 — vLLM Model Code (budget: 4 hours)
- Full vLLM-neuron fork at `/dev3/zigeng/bc/vllm-neuron/` (mounted read-write into container at `/opt/vllm-neuron`)
- You can modify ANY file in this fork: model code, worker, attention backend, functional layers, fx passes, etc.
- Targets: attention implementation, MoE routing, chunked prefill, scheduling, graph compilation hints
- Be more creative to change whatever you want in this vLLM-neuron code. Above targets are just suggestions. You are not forced to follow these targets. You can change whatever you want in this vLLM-neuron code.

### Phase 3 — NKI Kernels (budget: 6 hours)
- Full nkilib fork at `nkilib-fork/` (mounted read-write into container at `/opt/conda/lib/python3.13/site-packages/nkilib/`)
- You can modify ANY file in this fork: attention kernels, MoE kernels, utils, quantization, etc.
- Key directories:
  - `nkilib-fork/core/attention/` — segmented attention, KV-parallel attention, fused attention
  - `nkilib-fork/core/moe/moe_tkg/` — MoE expert dispatch (previously optimized)
  - `nkilib-fork/core/utils/` — shared utilities
  - `nkilib-fork/core/quantization/` — quantization kernels
- Targets: MoE expert dispatch, attention kernels, custom fused operations
- Be more creative to change whatever you want in this Neuron Kernel code. Above targets are just suggestions. You are not forced to follow these targets. You can change whatever you want in this kernel code.

## What you CANNOT modify

- `benchmark/` — the entire benchmark folder is read-only. It is the ground truth.
- `docker/run.bash` — the container configuration is fixed.
- `program.md` — this file.

## The Goal

**Maximize average prefill tok/s** while maintaining correctness (>99% top-1 logit match against baseline).

**Don't be afraid to go big.** The Neuron software stack is immature — there are likely significant performance wins hiding behind default configurations, unoptimized code paths, and unnecessary abstractions. Both small parameter tweaks and large structural changes are welcome:
- Simple parameter changes (bucket sizes, batch configs, compilation flags) are fine and encouraged
- But also feel free to rip out entire code paths, rewrite subsystems, or restructure the MoE dispatch if you think it will help
- Question framework defaults — many were designed for GPUs, not Neuron
- If you can DELETE code and get the same performance, that's the best kind of win

## Output Format

The benchmark prints a summary after each run:

```
---
avg_prefill_tok_per_s:  12345.6
avg_tok_per_s_all:      13000.2
mfu_percent:            42.3
correctness_pct:        99.8
total_turns:            13
scoring_turns:          6
compile_time_s:         312.4
peak_hbm_mb:            24576.0
---
```

NOTE: `avg_prefill_tok_per_s` uses only the **last 50% of turns** (long-context, steady-state). This is the scoring metric. `avg_tok_per_s_all` includes all turns for reference.

Extract the key metric: `grep "^avg_prefill_tok_per_s:" run.log`

## Logging Results

Log every experiment to `results.tsv` (tab-separated). Columns:

```
commit	phase	tok_per_s	mfu	correctness	compile_s	status	description
```

- commit: short git hash (7 chars)
- phase: 1, 2, or 3
- tok_per_s: average prefill tok/s (0.0 for crashes)
- mfu: MFU percentage (0.0 for crashes)
- correctness: percentage top-1 match (0.0 for crashes)
- compile_s: compilation time in seconds
- status: `keep`, `discard`, or `crash`
- description: short text of what this experiment tried

Example:
```
commit	phase	tok_per_s	mfu	correctness	compile_s	status	description
a1b2c3d	1	12345.6	42.3	99.9	0.0	keep	baseline (patched)
b2c3d4e	1	12567.8	43.1	99.8	0.0	keep	increase prefill chunk size to 4096
c3d4e5f	1	12100.0	41.5	99.9	0.0	discard	reduce bucket count
d4e5f6g	2	13200.0	45.2	99.7	285.0	keep	fuse QKV projection in prefill
```

## The Experiment Loop

The loop operates in three phases with fixed time budgets. Use a timer to track elapsed time per phase.

### Phase Transitions

- Phase 1 → Phase 2: after 2 hours elapsed
- Phase 2 → Phase 3: after 4 hours elapsed
- Phase 3 → End: after 6 hours elapsed
- At the very end (after Phase 3): run the full model validation once

Auto-advance between phases. Do NOT stop or ask the human.

### Within Each Phase

LOOP until time budget exhausted:

1. Look at the current git state and results.tsv history
2. Propose and implement an experimental idea (edit the appropriate files for this phase)
3. `git commit` the change
4. Compile (if needed) and run the benchmark:
   ```bash
   docker exec neuron-prefill bash -c "cd /dev3/zigeng/bc/opt && bash run/serve_medium.bash > logs/server.log 2>&1 & sleep 30 && python benchmark/prefill_bench.py --config benchmark/config_medium.json > logs/run.log 2>&1"
   ```
   (Adjust startup wait as needed based on model load time)
5. Read results: `grep "^avg_prefill_tok_per_s:\|^correctness_pct:\|^compile_time_s:" logs/run.log`
6. If grep output is empty, the run crashed. Run `tail -n 50 logs/run.log` for the traceback.
7. Record results in results.tsv
8. **Keep/Discard logic**:
   - If tok_per_s improved AND correctness >= 99.0%: KEEP (branch advances)
   - If tok_per_s improved but correctness < 99.0%: DISCARD (correctness violation)
   - If tok_per_s equal or worse: DISCARD
   - DISCARD = `git reset --hard HEAD~1`
9. Repeat

### Final Validation

After Phase 3 completes:
1. Run the full model benchmark (TP=8, 128K context, 3000 tokens/turn):
   ```bash
   docker exec neuron-prefill bash -c "cd /dev3/zigeng/bc/opt && bash run/serve_full.bash > logs/server_full.log 2>&1 & sleep 120 && python benchmark/prefill_bench.py --config benchmark/config_full.json > logs/run_full.log 2>&1"
   ```
2. Log the final validation results in results.tsv with description "FINAL VALIDATION (full model)"
3. Print a summary comparing baseline vs final performance

## Critical Rules

- **NEVER STOP**: Once the loop begins, do NOT pause to ask the human. Run autonomously for the full 12 hours (2+4+6). The human may be asleep.
- **Respect phase boundaries**: Do not modify kernel code in Phase 1 or 2. Do not modify launch params in a way that requires kernel changes during Phase 1.
- **Compilation awareness**: Phase 1 changes should NOT require recompilation (param-only changes). If a Phase 1 change triggers recompilation, that's acceptable but try to minimize it.
- **Crashes**: If something is easy to fix (typo, import error), fix and re-run. If fundamentally broken, log crash, revert, move on.
- **Timeout**: If a single experiment takes >30 minutes total (compile + run), kill it and treat as crash.
- **Mix small and large**: Alternate between quick parameter tweaks and bigger structural changes. Small wins compound, but don't get stuck only tuning knobs — when params are exhausted, be willing to refactor aggressively.
- **Learn from CUDA**: Reference mature CUDA optimization techniques and published literature — FlashAttention, FlashDecoding, PagedAttention, MegaBlocks MoE, Triton kernel patterns, DeepSpeed-MoE, etc. Many of these ideas can be adapted to Neuron's architecture (e.g., tiling strategies, operator fusion, memory access patterns). If a technique is well-proven on GPU, consider how the same principle applies to NeuronCores' SBUF/PSUM/DMA model.
- **Think harder**: If you run out of ideas within a phase, re-read the model architecture, look at the existing patches in BrowseComp-Plus for inspiration, read the vLLM-neuron source code to find bottlenecks, think about what's fundamentally different about Neuron vs GPU that the code isn't accounting for.
