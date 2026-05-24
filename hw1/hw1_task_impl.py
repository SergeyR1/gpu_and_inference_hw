import statistics
import torch


# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    # clone = one read + one write, effectively 0 arithmetic
    return x.clone()


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        # FMA-style loop: each iteration contributes 2 FLOPs/element (mul + add)
        # acc is kept alive across iterations => torch.compile can fuse everything
        acc = x
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    return torch.compile(fn) if compiled else fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup: triggers torch.compile on first call, warms caches
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Measure rep runs with CUDA events
    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # milliseconds

    return statistics.median(times)


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    """
    total_flops: num_elements * num_ops * 2  (2 FLOPs per FMA per element)

    Byte traffic models:
      compiled: fused kernel — each element read once, written once
                bytes = num_elements * 2 * bytes_per_element
                AI   = total_flops / bytes

      eager: each loop iteration launches separate mul + add kernels,
             materializing intermediates. Per iteration:
               mul: read acc (4B) + read x (4B) + write tmp (4B) = 12B
               add: read tmp (4B) + read x (4B) + write acc (4B) = 12B
               => 24 bytes/element/iteration
             total bytes = num_elements * num_ops * 24
             AI is roughly 2/24 = 1/12 FLOP/Byte — essentially flat w.r.t. num_ops
    """
    total_flops = num_elements * num_ops * 2

    if variant == "compiled":
        total_bytes = num_elements * 2 * bytes_per_element  # fused: 1 read + 1 write
    else:  # eager
        # 2 separate kernels per iteration, each with 3 tensor accesses (4B each)
        total_bytes = num_elements * num_ops * 2 * 3 * bytes_per_element

    ai = total_flops / total_bytes
    achieved_flops = total_flops / (ms * 1e-3)

    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
#
# A1. The compiled kernel fuses all num_ops FMA iterations into a single GPU
# kernel, so the tensor x is read once and the result written once regardless
# of how many operations are performed. Because runtime stays roughly flat
# (~0.211–0.213 ms across 1–64 ops), the GPU is spending the same wall-clock
# time but doing proportionally more useful FLOPs per byte transferred. That is
# exactly the definition of higher arithmetic intensity, and higher AI on the
# memory-bound slope of the roofline means higher observed TFLOP/s — even
# though the kernel itself is not getting "faster" in absolute time.
#
# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
#
# A2. Two reasons:
# (1) Tile occupancy / wave quantization. A 1024×1024 matmul with FP32 CUBLAS
#     tiles does not perfectly fill all 132 SMs of the H100 — the last "wave"
#     of tiles may leave many SMs idle, wasting a large fraction of the
#     available compute time. The measured 32.2 TFLOP/s vs the 53.2 TFLOP/s of
#     the compiled elementwise kernel reflects this underutilization.
# (2) cuBLAS autotuning / kernel selection overhead. At 1024×1024, cuBLAS may
#     select a kernel variant that is not optimally tuned for this exact size on
#     this GPU, whereas the compiled elementwise kernel is a perfectly simple
#     bandwidth-saturating pattern.
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
#
# A3. At 64 ops the arithmetic intensity is 16 FLOP/Byte, just below the H100
# ridge point of ~20 FLOP/Byte, and runtime is ~0.212 ms — the kernel is still
# largely memory-bound. At 128 ops the arithmetic intensity reaches 32 FLOP/Byte,
# crossing the ridge point, and runtime jumps to 0.323 ms. This means the kernel
# has exhausted the available memory bandwidth and is now limited by raw compute
# throughput — it has transitioned from memory-bound to compute-bound. The
# runtime increase reflects the GPU needing more time to execute the additional
# FLOPs, rather than the data transfer being the limiting factor.
#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
#
# A4. In eager mode PyTorch executes each iteration of the FMA loop as two
# separate GPU kernels (one for the multiply, one for the add), materializing
# intermediate tensors in global memory each time. The estimated byte traffic
# grows linearly with num_ops (24 bytes per element per iteration), while the
# FLOPs also grow linearly — so the arithmetic intensity stays flat at
# ~1/12 FLOP/Byte regardless of K. All eager points therefore cluster at the
# same low-AI position on the x-axis. Runtime, however, grows linearly with K
# (0.45 ms at K=1 → 67.2 ms at K=128), so the points also stay at the same
# low TFLOP/s. The compiled points, by contrast, move rightward along the
# memory-bound slope because fusion eliminates the intermediate traffic.
