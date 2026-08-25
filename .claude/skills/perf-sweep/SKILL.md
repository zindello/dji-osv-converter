---
name: perf-sweep
description: Profile and optimise the stitch pipeline — the measurement discipline, per-stage benchmark recipe, and the list of already-measured dead ends so they aren't retried.
---

# Performance work on the stitch pipeline

## Measurement recipe

- Warm up once, then time ≥12 iterations, with `torch.mps.synchronize()` /
  `torch.cuda.synchronize()` OUTSIDE the loop (throughput, not latency).
- Benchmark stages in isolation AND the full `stitch_yuv`; stage sums lie
  when allocator/lazy effects shift.
- For end-to-end, use `convert(progress=...)` timestamps and report the last
  100 frames (setup, seek-decode and encoder spin-up pollute short runs).
- Expect ±10% run-to-run variance on MPS; never claim a win inside it.

## Correctness gate (non-negotiable)

Before optimising, save baseline outputs (packed Y/UV for ≥3 spread frames,
or load the pre-change module from `git show HEAD:...` side-by-side). After:
bit-equal, or quantify exactly (max LSB, % pixels, where they live) and get
the user's sign-off in the report. "Looks the same" is not a result.

## Current budget (8K, M3/MPS) — docs/architecture.md keeps the live copy

decode 52.6 | upload/split ~11 | warp 13.2 | blend 8.4 | level grid 5.7 |
level sample 8.2+2.4 | pack+D2H ~10 → static stitch ~60 ms; the default
auto seam repair adds ~65 ms CPU (DIS flow 6 + seam DP 10 + estimator 15 +
gathers/syncs) → ~127 ms stitch. E2e pipelined: ~73 ms with
`seam_flow=False`, ~212 ms sustained under defaults (fanless-M3 throttling
included). CUDA numbers live in docs/architecture.md (V100: 36 ms/frame
pre-seam-merge; re-measure under merged defaults).

## Measured dead ends — do not retry without new evidence

- Threading the CPU map build (memory-bound: 2.0 → 2.9 s).
- Disk-caching remap tables (885 MB/resolution raw for ~2 s).
- Batching the two lens warps into one grid_sample (pre-`cat` eats the win).
- fp16 image planes (MPS grid_sample isn't dtype-bound; no speedup).
- fp16 sampling grids (ulp ≈ 4 px at 8K — never do this).
- `(N,3)@(3,3)` matmul orientation on MPS (7× slower than `(3,3)@(3,N)`).
- Interpolating chroma grids (corrupts the wrap meridian — decimate `[::2]`).
- torch.compile of uint16 casts (inductor MPS backend lacks the dtype).
- Eager (unfused) elementwise chains at 8K (kernel-launch dominated; only
  compiled fusions or single big ops pay).
- D2H fetch on the encoder thread (63 → 74 ms on MPS: cross-thread `.cpu()`
  serializes on the kernel queue — fetch stays on the stitch thread).
- CPU-path concurrency, every variant: stage overlap 912 → 1078 ms,
  frame-parallel stitch threads 0.79× at 2 workers. numpy/OpenCV already
  saturate the cores; the CPU path stays serial.
- Frame batching / multi-frame-in-flight on the GPU: the M3 GPU is already
  saturated by one 8K frame (two frames' kernels scale at exactly 2.01×) —
  there is no idle GPU to harvest. Open on CUDA only: overlapping the
  ~13 ms of H2D/D2H with compute via pinned memory + side streams.
