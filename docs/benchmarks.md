# Benchmarks

Conversion speed per platform, measured by
[`tools/benchmark.py`](../tools/benchmark.py) — the canonical record; add
rows whenever the pipeline meets new hardware.

**Workload**: 8K (7680×3840) 10-bit output, `rocksteady_horizon` levelling,
from a 2×3840×3840 @ 29.97 fps OSV (the worst-case tumble clip). **setup**
is the one-time cost before the first frame; **per frame** is steady-state
wall time converting (decode + stitch + encode, shipping defaults —
pipelined on GPU, serial on CPU).

| platform | GPU/CPU | setup | per frame |
|---|---|---|---|
| Apple M3 (macOS 15) | GPU | ~8 s | **212 ms (4.7 fps)** |
| Apple M3 (macOS 15) | CPU | 3 s | 972 ms (1.0 fps) |
| NVIDIA V100 (CUDA 12.6, Linux) | GPU | 32 s | **54 ms (18.6 fps)** |
| NVIDIA V100 (CUDA 12.6, Linux) | CPU | 23 s | 5334 ms (0.2 fps) |

The V100 GPU row is NVDEC-bound and content-dependent: the worst-case
benchmark clip decodes hard and encodes easily, while typical outdoor
footage flips that — there the segment-parallel NVENC default (3 sessions,
`enc_sessions`) delivers **36 ms/frame (28 fps, 0.79× realtime)** measured
over the full 4724-frame base clip, with the single NVDEC engine at
94-100% (the hardware wall). Short `--e2e` windows under-report
multi-session throughput (the pipeline drains its session backlogs after
the last frame); use ≥450 frames.

The default includes the auto seam repair (photometric match + parallax
flow + dynamic seam), which engages on nearly all real handheld footage —
the operator is close to the camera, so the lenses genuinely disagree at
the seam. `seam_flow=False` restores the static-blend fast path: **73 ms
(13.6 fps)** on the M3. Sustained full-clip numbers on the fanless M3
include thermal throttling; short bursts run faster.

## Adding a row

```bash
python tools/benchmark.py CLIP.OSV --e2e 150              # GPU row
python tools/benchmark.py CLIP.OSV --reference --e2e 9    # CPU row
```

Each run prints a ready-to-paste markdown row. Use a clip with real motion
(levelling active every frame), state the platform (chip, OS), and replace a
platform's existing rows rather than appending — this table records the
current code, not history (git keeps the history). Per-stage and per-engine
numbers (torch-CPU parity runs, stitch-only timings) live in
[architecture](architecture.md), not here.
