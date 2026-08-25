# Benchmarks

Speed of the shipping pipeline per platform and engine, all measured the
same way by [`tools/benchmark.py`](../tools/benchmark.py) — these numbers
are the canonical record; add a row whenever the pipeline meets new
hardware.

**Workload**: 8K (7680×3840) 10-bit output, `rocksteady_horizon` levelling,
from a 2×3840×3840 @ 29.97 fps OSV (the worst-case tumble clip).

- **stitch** — steady-state ms/frame for the stitch alone (frames
  pre-decoded, one device sync around the loop).
- **e2e (serial)** — `convert()` wall time per frame: decode + stitch +
  encode back to back, steady tail of the run.
- **setup** — one-time cost before the first frame (calibration + map build
  + GPU init/`torch.compile`).

| platform | engine | stitch ms/frame | e2e serial ms/frame | setup | notes |
|---|---|---|---|---|---|
| Apple M3 (macOS 15) | torch/**mps** | 67.5 | 102 (9.8 fps) | ~10 s | shipping path; serial e2e is decode-bound (VideoToolbox decode ~53 ms) |
| Apple M3 (macOS 15) | torch/**cpu** | 493 | — | ~9 s | fallback/parity engine; `convert()` never auto-selects it (`available()` is False on cpu) |
| Apple M3 (macOS 15) | numpy reference | 1205 | 972 (1.0 fps) | 3 s | correctness reference (`Stitcher360.stitch`); its e2e is the 8-bit BGR path |
| NVIDIA V100 (CUDA) | torch/**cuda** | *pending* | *pending* | | NVENC encode + threaded pipelining auto-enable on CUDA |

Reading the table:

- Short hot-loop microbenchmarks (as in the per-stage budget in
  [architecture](architecture.md)) read ~10% faster than the tool's longer
  loop; treat the tool's numbers as the comparable ones and the budget as
  the per-stage breakdown.
- Serial e2e ≈ decode + stitch. The planned concurrency stage overlaps them:
  the pipelined floor is max(decode, stitch) per platform.
- MPS vs torch-CPU output parity is ≤0.13 of a 10-bit LSB — the engines are
  interchangeable for correctness, only speed differs.

## Adding a row

```bash
python tools/benchmark.py CLIP.OSV --e2e 150          # best GPU on this box
python tools/benchmark.py CLIP.OSV --device cpu --frames 4
python tools/benchmark.py CLIP.OSV --reference --frames 3 --e2e 9
```

Each run prints a ready-to-paste markdown row. Use a clip with real motion
(levelling active every frame), state the platform (chip, OS), and replace a
platform's existing row rather than appending duplicates — this table
records the current code, not history (git keeps the history).
