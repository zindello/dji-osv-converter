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
| Apple M3 (macOS 15) | GPU | ~7 s | **73 ms (13.6 fps)** |
| Apple M3 (macOS 15) | CPU | 3 s | 972 ms (1.0 fps) |
| NVIDIA V100 (Linux) | GPU | *pending* | *pending* |

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
