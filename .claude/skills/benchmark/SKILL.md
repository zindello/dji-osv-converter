---
name: benchmark
description: Record pipeline speed for this platform in docs/benchmarks.md — run tools/benchmark.py per engine and update the table. Use after performance changes and on first contact with new hardware (e.g. the CUDA/V100 box).
---

# Benchmarking the pipeline

`docs/benchmarks.md` is the canonical per-platform speed record; the ONLY
way numbers get into it is `tools/benchmark.py`. Run it after any
performance-relevant change and once per new machine/GPU.

## Run (one row per engine)

```bash
python tools/benchmark.py CLIP.OSV --e2e 150          # best GPU (cuda > mps)
python tools/benchmark.py CLIP.OSV --device cpu --frames 4
python tools/benchmark.py CLIP.OSV --reference --frames 3 --e2e 9
```

Each prints a ready-to-paste markdown row. Use the worst-case tumble clip
(or any clip with continuous real motion so levelling recomputes every
frame). Standard workload is 8K (7680×3840) 10-bit `rocksteady_horizon`.

## Recording rules

- **Replace** the platform's existing row, don't append — the table records
  the current code; git keeps history. Add a new row only for new hardware.
- State the platform as chip + OS (e.g. "Apple M3 (macOS 15)", "NVIDIA
  V100 (CUDA 12, Linux)"); put anything anomalous in the notes column.
- ±10% run-to-run variance is normal — never record a "win" inside it, and
  don't churn the table for sub-10% drift.
- `--e2e` is only meaningful where `convert()` would pick that engine: the
  auto-selected GPU, or `--reference` (the CPU fallback). torch-CPU gets
  "—" (convert never auto-selects it).
- Keep `docs/architecture.md`'s per-stage budget in sync separately — it
  comes from per-stage profiling (see the `perf-sweep` skill), not from
  this tool.
- After editing the table, also refresh the headline numbers in
  `docs/index.md` and `README.md` if the shipping-path row changed.

## Interpreting

Serial e2e ≈ decode + stitch (they run back to back); the pipelined floor
is max(decode, stitch). A big stitch win that barely moves e2e means the
platform is decode-bound — that's expected, not a failed optimisation.
