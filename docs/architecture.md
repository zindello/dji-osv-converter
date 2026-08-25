# Architecture

## Module map

```
dji_osv_converter/
├── constants.py       every value NOT from the OSV, with provenance
├── protobuf_wire.py   schema-less protobuf wire reader
├── osv_calibration.py header sample → CameraCalib (intrinsics, distortion,
│                      extrinsics, LSC) + mount rotation (1.3.1)
├── metadata.py        per-frame quats/gravity + high-rate track (read_highrate)
├── project.py         DJI's f-theta model → fisheye↔equirect remap tables
├── calibration.py     CameraMap dataclass (shared)
├── orientation.py     equirect direction grids & rotations (CPU)
├── stitcher.py        CPU reference stitch (warp/exposure/blend variants)
├── stitch_gpu.py      the shipping GPU stitch (CUDA / Apple MPS), 10-bit YUV
├── pipeline.py        Stitcher360: ties calibration+metadata → per-frame R
├── decode.py          in-process HEVC decode (PyAV, VideoToolbox hwaccel)
├── video.py           convert(): decode → stitch → encode → 360 tag
└── spherical.py       inject sv3d/V1 spherical metadata into the MP4
```

## Data flow (video convert)

```
OSV ─ decode.iter_pairs ──► two yuv420p10le fisheyes  (VideoToolbox, ~53 ms)
                              │
Stitcher360.maps_for ───────► │  static remap tables (built once, ~3.5 s)
Stitcher360.levelling_for ──► │  per-frame rotation R (per mode)
                              ▼
              GpuStitcher.stitch_yuv          (~96 ms at 8K on M3 MPS)
                 warp both lenses (grid_sample, Y full / UV half res)
                 lerp blend across the seam band
                 rotate by R (compiled fused grid build + grid_sample)
                 pack p010le → CPU
                              ▼
             hevc_videotoolbox encode (hvc1, 0.25 bpp) + audio copy
                              ▼
             spherical.inject_spherical  (sv3d + V1 uuid boxes)
```

The whole stitch stays 10-bit YUV on the GPU; the only transfers are one
fisheye pair up and one packed frame down per frame.

## Performance budget (8K, Apple M3, MPS)

| stage | ms/frame |
|---|---|
| decode (separate hardware) | 52.6 |
| upload + split YUV | ~11 |
| warp ×2 lenses | 13.2 |
| blend (lerp) | 8.4 |
| level grid build (fused, separable) | 5.7 |
| level grid_sample | 8.2 + 2.4 decimate |
| pack (fused) + uint16 device→host | ~10 |
| **stitch total** | **~59** |
| end-to-end, pipelined (default on GPU) | **~73 (13.6 fps)** |
| end-to-end, serial (`pipeline=False`) | ~104 (9.6 fps) |

Decode/stitch/encode run on three threads on the GPU path (decode prefetch +
encoder queue in `video.py`); decode fully hides under the stitch, and the
residual gap to the ~60 ms stitch floor is queue handoff + the encoder-side
frame copy. Output is frame-identical to the serial path (verified by
decoding both).

Notable implementation facts:

- The level-grid build is `torch.compile`-fused AND separable: the direction
  field is `[cos(lat)sin(lon), sin(lat), −cos(lat)cos(lon)]`, so the kernel
  reads four 1-D sin/cos tables (~90 KB) instead of 354 MB of precomputed
  rays (13.5 → 5.7 ms). Recomputing the products in f32 shifts ≤0.0004% of
  output pixels by ≥1 LSB, only where the sampled pano's pole lands (see the
  code comment). The fallback is a `(3,3)@(3,N)` matmul — that orientation
  matters: `(N,3)@(3,3)` is 7× slower on MPS.
- The p010 pack: float math is compile-fused (the uint16 cast can't be —
  inductor's MPS backend lacks the dtype) and the cast goes straight to
  uint16, halving D2H bytes vs the old int32 route. Pack 37 → 10 ms,
  bit-identical. `_u16` probes backend support and falls back to int32.
- Measured and rejected: batching the two lens warps into one `grid_sample`
  (the pre-`cat` copies eat the 0.5 ms win) and fp16 image planes
  (MPS `grid_sample` isn't dtype-bound — no speedup, only precision risk).
- The longitude wrap needs no `remainder`: `atan2` already lands in [−π, π].
- Chroma grids are **decimated** (`[::2]`) from the full grid, never
  interpolated — averaging across the ±1 wrap discontinuity corrupts the wrap
  meridian.
- Concurrency, measured (M3): GPU stage-pipelining WINS (104 → 73 ms) and is
  the default. Everything else LOSES: moving the D2H fetch to the encoder
  thread (63 → 74 ms — cross-thread `.cpu()` serializes on the MPS kernel
  queue), CPU stage overlap (912 → 1078 ms) and CPU frame-parallel stitch
  threads (0.79× at 2) — numpy/OpenCV already saturate the cores, so the
  CPU path stays serial.
- The GPU is **saturated**, not idle: two frames' kernels cost exactly 2.01×
  one frame's (81.6 vs 40.5 ms) — a single 8K frame fills the M3 GPU, so
  batching/multi-frame-in-flight has nothing to harvest. The ~60 ms stitch
  is 40.5 compute + 6 upload + 7 D2H + ~5 launch overhead; on MPS the
  transfers share the one kernel queue and cannot overlap compute. Going
  faster on this hardware means less work per frame, not more parallelism.
  On CUDA the copy engines CAN overlap (pinned memory + `non_blocking` on
  side streams) — an open V100-phase item worth ~13 ms.
- One-time setup is ~6.5 s: ~3.5 s map build + ~3 s `torch.compile`. The
  radial polynomial is Horner-evaluated (bit-identical maps, 2.4× faster than
  the naive `theta**n` form). Threading the two lens builds was **measured as
  a loss** (2.0 → 2.9 s — the build is memory-bandwidth-bound); a raw disk
  cache of the maps was rejected at 885 MB per resolution for a ~2 s saving.
- Cross-device parity: the same frame stitched on MPS vs torch-CPU differs by
  at most 0.13 of a 10-bit level (interpolation rounding) — the math is
  device-independent, which is the CUDA-readiness guarantee short of the box.
- Encoding is portable: `convert()` auto-selects VideoToolbox → NVENC →
  libx265, with the packed→planar repack handled losslessly for software
  encoders (libx265 muxes as `hev1`, cosmetic: QuickLook-only issue, and macOS
  gets VideoToolbox anyway).

## CPU reference path

`Stitcher360.stitch()` mirrors the pipeline on the CPU (with multiband and
optical-flow seam options) for stills, debugging, and machines without a GPU.
It is the semantic reference the GPU path is checked against.
