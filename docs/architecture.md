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
Stitcher360.maps_for ───────► │  static remap tables (built once, ~10 s)
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
| warp ×2 lenses | 27.6 |
| blend (lerp) | 8.4 |
| level grid build (torch.compile-fused) | 14.5 |
| level grid_sample | 8.2 + 2.7 decimate |
| pack + device→host | 34.2 |
| **stitch total** | **~96** |
| end-to-end steady state | ~115 (8.7 fps) |

Notable implementation facts:

- The level-grid build is `torch.compile`-fused (91 → 6.5 ms of math; ~14 ms
  in situ). The fallback is a `(3,3)@(3,N)` matmul — that orientation matters:
  `(N,3)@(3,3)` is 7× slower on MPS.
- The longitude wrap needs no `remainder`: `atan2` already lands in [−π, π].
- Chroma grids are **decimated** (`[::2]`) from the full grid, never
  interpolated — averaging across the ±1 wrap discontinuity corrupts the wrap
  meridian.
- Remaining headroom is overlap, not compute: decode and pack+D2H are prime
  candidates for the planned concurrency stage (floor ≈ max(GPU, decode) →
  ~15 fps).

## CPU reference path

`Stitcher360.stitch()` mirrors the pipeline on the CPU (with multiband and
optical-flow seam options) for stills, debugging, and machines without a GPU.
It is the semantic reference the GPU path is checked against.
