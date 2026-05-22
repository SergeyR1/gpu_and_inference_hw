# HW1: GPU Roofline Model - Understanding Memory-Bound vs Compute-Bound Kernels

## Goal

Build intuition for why some GPU kernels are fast and others are not by plotting real measurements on a roofline diagram.

## Grading (40 points total)

| Part                                              | Points |
| ------------------------------------------------- | -----: |
| Part 1a — `lowest_ai_fn`                          |      4 |
| Part 1b — `make_compute_fn` (eager + compiled)    |      8 |
| Part 2  — `benchmark_fn` (CUDA events)            |      8 |
| Part 3  — `compute_elementwise_metrics`           |      8 |
| Part 4  — Writeup (Q1–Q4, 3 pts each)             |     12 |

Full credit on Parts 1–3 requires that `python3 hw1/hw1_task.py` runs to completion and produces a roofline plot whose points sit *below* the theoretical ceiling, with the compiled `ops-K` series moving rightward (higher AI) as `K` grows.

## File Layout

- `hw1_task.py`: task entrypoint
- `hw1_task_impl.py`: functions you implement (TODOs)
- `hw1_runtime.py`: provided benchmark/measurement/plotting utilities
- `results/`: output directory for the roofline plot and JSON data

## What to do

1. Fill in the four TODOs in `hw1/hw1_task_impl.py` (Parts 1a, 1b, 2, 3).
2. From the repository root, run:

   ```bash
   python3 hw1/hw1_task.py
   ```

   The script prints a per-row line for each measured point (with timing, AI, and TFLOP/s) and writes:
   - `hw1/results/roofline.png` — the roofline plot
   - `hw1/results/roofline_data.json` — the raw measurements

3. **Inspect the results.**
4. Answer Q1–Q4 (Part 4) in the comment block at the bottom of `hw1_task_impl.py`.


## Background

Every GPU kernel does two things: move data (bytes) and compute (FLOPs).
The **arithmetic intensity** (AI) = FLOPs / Bytes tells you which is the bottleneck:

- Low AI: limited by memory bandwidth -> "memory-bound"
- High AI: limited by compute throughput -> "compute-bound"

The roofline model draws both ceilings on a log-log plot:

`achievable FLOP/s = min(peak_compute, bandwidth * AI)`

The crossover is the **ridge point**. Kernels left of it are memory-bound; right of it are compute-bound. This is the key concept for understanding GPU performance.

## GPU Specs

This homework works out of the box on `H100` and `L40S`.

For the default `H100 SXM` roofline used here (FP32, no Tensor Cores):

- Peak FP32 compute: ~67 TFLOP/s
- HBM3 bandwidth: ~3.35 TB/s
- Ridge point: 67000 / 3350 ~= 20 FLOP/Byte
- Spec reference: [NVIDIA H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/)

If you are running on an `L40S`, the runtime already includes its FP32 and bandwidth settings:

- Spec reference: [NVIDIA L40S GPU](https://www.nvidia.com/en-us/data-center/l40s/)

If you have a different GPU, look up its peak FP32 throughput and memory bandwidth, then add a new entry to `GPU_SPECS` in `hw1/hw1_runtime.py`.

## A note on `torch.compile`

In eager mode, PyTorch executes each operation immediately as Python reaches it, instead of first capturing a larger fused computation graph. So writing `acc = acc * x + x` in a Python loop typically launches separate GPU kernels for the element-wise ops each iteration, repeatedly touching global memory.

For this simple, static loop, `torch.compile` can capture and fuse the work into a much smaller number of kernels (often one), keeping intermediates in registers and making arithmetic-intensity control meaningful. In more complex code, graph breaks can prevent full fusion.

The plotted arithmetic intensity for the `ops-K` operations uses two byte-traffic models. For the compiled version, use the fused-kernel model: one read, one write, and `2K` FLOPs per element. For eager PyTorch, estimate the separate multiply and add operations in each loop iteration: intermediates are materialized, more data is moved, and the true arithmetic intensity is much lower. This is why the eager points do not move right on the roofline the same way the compiled points do.

## A note on cuda events

- [https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html)
- [https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch)

## Logistics

- Requires: 1x H100 or 1x L40S
