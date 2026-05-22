# HW2: LLM Inference Optimization

## Goal

Write the fastest possible autoregressive generation loop for a tiny decoder-only transformer. You'll start from a slow baseline, identify what's wasting time, and apply a series of optimizations to reach a significant speedup.

## Grading (60 points total)

| Part                                              | Points |
| ------------------------------------------------- | -----: |
| `profile()` — produces a usable Chrome trace      |      5 |
| `generate_optimized()` — runs and times the loop  |      5 |
| `optimized_loop()` — speedup over the V0 baseline |     30 |
| Writeup (changes, per-fix speedups, biggest win)  |     20 |

The `optimized_loop` points are awarded by the **measured speedup** that `time_generation` reports against the V0 slow baseline on the target GPU (L40S):

| Speedup vs V0 | Points |
| ------------- | -----: |
| < 1.5×        |      0 |
| ≥ 1.5×        |     12 |
| ≥ 3× (Good)   |     22 |
| ≥ 4× (Great)  |     30 |

If you cannot reach the next tier, partial credit is still awarded for the lower tier you do hit.

## Setup

The model is a randomly-initialized 2-layer Llama built from a `LlamaConfig` (`hidden_size=2048`, `intermediate_size=6144`, 8 attention heads, 8 KV heads, `vocab_size=4096`). It runs on CUDA with synthetic random token IDs — no tokenizer, no pretrained weights — so the trace stays focused on the generation loop itself rather than I/O or model loading.

Two run lengths are used:

- **Profiling** runs `PROFILE_STEPS = 12` decode steps so the trace stays small enough to navigate in Perfetto.
- **Timing** runs `MAX_NEW_TOKENS = 128` decode steps from a `PROMPT_LEN = 1024` prompt, which is what the speedup numbers are measured against.

Each "step" is one forward pass through the model, so the slow baseline does 12 forward passes per profile and 128 per timed run.

## File Layout

- `hw2_task.py`: **your only file to edit** — implement all three functions described below
- `utils.py`: provided helpers — `build_model`, `slow_loop`, `time_generation`, `get_input_ids` (do not modify)
- `results/`: output directory for Chrome trace files

## What to do

1. Implement the three functions in `hw2_task.py`:
   - `profile(loop_fn, model, input_ids, trace_name)` — wrap `loop_fn` with `torch.profiler`, print the summary table, and export a Chrome trace to `results/trace_name`.
   - `optimized_loop(model, input_ids, n_steps)` — starts as a copy of `slow_loop`. Make it as fast as possible. Changes may span the loop body and the model loading in `generate_optimized()`.
   - `generate_optimized()` — build the tiny Llama (consider dtype and other loading options too), then call `profile` and `time_generation` on `optimized_loop`, and **return the elapsed time** from `time_generation` so `main()` can print a speedup.
2. From the repository root, run:

   ```bash
   python3 hw2/hw2_task.py
   ```

   The script prints the two `time_generation` lines and then a `SUMMARY` block with the computed speedup, e.g.:

   ```
   Slow: 128 tokens in 21.34s (6.0 tok/s)
   Optimized: 128 tokens in 4.91s (26.1 tok/s)
   ============================================================
   SUMMARY
   ============================================================
     Slow:       21.34s
     Optimized:   4.91s
     Speedup:     4.35x  (vs V0 slow baseline)
   ```

   The `Speedup` line is the number that determines your grade tier (see the table above). It also writes two Chrome traces to `hw2/results/`:
   - `v0_slow_trace.json`
   - `v1_optimized_trace.json` (or whatever name you pass)
3. **Cross-check with the traces.** Open both at [ui.perfetto.dev](https://ui.perfetto.dev) to confirm your fixes are actually showing up: the optimized trace should have a much denser GPU stream and far fewer per-step CPU↔GPU sync points than V0.
4. Fill in the **Writeup** comment block at the bottom of `hw2_task.py` (see below).

## Constraints

- Stay within **PyTorch** and the libraries already pinned in `requirements.txt`. Anything built into PyTorch or `transformers` is fair game — figuring out *which* built-ins are worth reaching for is part of the exercise.
- **Do not** use dedicated inference engines such as vLLM, TensorRT-LLM, TGI, SGLang, llama.cpp, ExLlama, or similar.
- `utils.py` is provided and must not be modified — the slow baseline, model builder, and timing helpers must stay identical so your speedup numbers are comparable to the targets above.

## Background

### torch.profiler

`torch.profiler` records every PyTorch operator, CUDA kernel launch, and GPU kernel execution. See the [official docs](https://docs.pytorch.org/docs/2.11/profiler.html) for the full API. It produces two types of output:

1. **Summary table** — sorted by CPU or CUDA time, shows call counts and averages. Good for finding expensive operators.
2. **Chrome trace** — a JSON file you open at [ui.perfetto.dev](https://ui.perfetto.dev). Shows a timeline of all events on CPU threads and GPU streams.

### Reading Chrome Traces

When you open a trace in Perfetto you'll see several rows. The two most important are:

- **The CPU thread** (`python <pid>` → main thread) — a nested stack of colored bars, one per PyTorch operator. The outermost bars are high-level ops (e.g. `aten::linear`); inner bars are what they decompose into. The `aten::` prefix is PyTorch's C++ tensor library ("A Tensor Library") — every built-in op like `add`, `matmul`, or `item` lives in that namespace. At the very bottom of each stack you'll usually find a `cuda_runtime` event such as `cudaLaunchKernel` — the CPU handing off work to the GPU driver.
- **The GPU stream** (`stream N`) — where CUDA kernels actually execute on the hardware. A kernel bar here starts slightly after its `cudaLaunchKernel` on the CPU side.

The relationship between these rows tells you a lot:

```
cpu_op: aten::some_op
  └── cuda_runtime: cudaLaunchKernel   ← CPU hands off, then moves on immediately
                                              ↓  (async gap)
GPU stream: some_kernel                ← GPU executes it later
```

Because launches are asynchronous, a healthy trace has the CPU and GPU rows both densely filled and overlapping. When the CPU thread has long spans with **no `cudaLaunchKernel` at the bottom**, the CPU is doing real computation itself instead of delegating to the GPU — and the GPU stream goes quiet.


**Trust the trace for structure, not for wall-clock.** `torch.profiler` pays a few microseconds of bookkeeping per aten op, which is negligible when ops are large but can dominate the CPU timeline when there are hundreds of tiny ops per step. Use the trace to find *what* to fix (kernel names and shapes, launch patterns, per-step sync points, redundant work); use the unprofiled `time_generation` numbers to confirm *how much* the fix actually helps.

**You don't have to lead with the trace.** Reading the baseline loop and asking "is anything here obviously wasteful?" is a perfectly valid approach for this homework — you can form a hypothesis from the code alone, apply the change, measure with `time_generation`, and use the trace to *understand* or *confirm* the speedup rather than to discover it. What matters is that each claim in your writeup is backed by either a trace observation or a timing measurement.

## Writeup

In the comment block at the bottom of `hw2_task.py`, briefly describe:

- What you changed and why
- The speedup each fix contributed
- Which fix had the biggest impact and why
