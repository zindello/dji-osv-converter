# Benchmarks

Conversion speed per platform, all measured the same way by
[`tools/benchmark.py`](../tools/benchmark.py) — the canonical record; add a
row whenever the pipeline meets new hardware.

**Workload**: 8K (7680×3840) 10-bit output, `rocksteady_horizon` levelling,
from a 2×3840×3840 @ 29.97 fps OSV (the worst-case tumble clip). **setup**
is the one-time cost before the first frame; **per frame** is steady-state
wall time converting (decode + stitch + encode, serial).

| platform | engine | setup | per frame |
|---|---|---|---|
| Apple M3 (macOS 15) | torch/**mps** | ~10 s | **102 ms (9.8 fps)** |
| Apple M3 (macOS 15) | torch/**cpu** | ~9 s | 493 ms (stitch only¹) |
| Apple M3 (macOS 15) | numpy reference | 3 s | 972 ms (1.0 fps) |
| NVIDIA V100 (CUDA) | torch/**cuda** | *pending* | *pending* |

¹ `convert()` never auto-selects torch-CPU (the CPU fallback is the numpy
reference path), so its number is the stitch alone — kept for cross-device
parity work.

## Adding a row

```bash
python tools/benchmark.py CLIP.OSV --e2e 150          # best GPU on this box
python tools/benchmark.py CLIP.OSV --device cpu --frames 4
python tools/benchmark.py CLIP.OSV --reference --frames 3 --e2e 9
```

Each run prints a ready-to-paste markdown row. Use a clip with real motion
(levelling active every frame), state the platform (chip, OS), and replace a
platform's existing row rather than appending — this table records the
current code, not history (git keeps the history). Per-stage numbers live in
[architecture](architecture.md), not here.
