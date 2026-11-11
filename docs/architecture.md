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
             spherical.inject_spherical  (sv3d + V1 uuid boxes; streams only the moov)
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
| **stitch total (static blend, `seam_flow=False`)** | **~59** |
| stitch with the auto seam repair engaged (the default) | ~127 |
| end-to-end, pipelined, `seam_flow=False` | ~73 (13.6 fps) |
| end-to-end, pipelined, defaults | ~212 sustained (4.7 fps) |
| end-to-end, serial (`pipeline=False`, `seam_flow=False`) | ~104 (9.6 fps) |

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
- On CUDA the copy engines DO overlap: `stitch_yuv_async` packs on-device
  and runs the D2H on a dedicated stream into a pooled pinned slot; the
  encoder thread blocks on a per-frame event and recycles the slot. On the
  V100 (PCIe, where pageable D2H alone was ~26 ms at 8K) this took the
  stitch loop 103 → 50 ms/frame, output bit-identical; the pipeline path
  uses it automatically on CUDA (`async_d2h`). Input-side H2D pinning was
  measured and skipped: staging the copy on the stitch thread makes it a
  wash (~4 ms) — it only pays if the staging copy moves to the decode
  thread (open item, matters once decode stops being the V100 bottleneck).
- CUDA decode is **decode-to-device** (`gpu_decode_fmt` → `"cuda"`): the
  hwdevice is created with `primary_ctx=1` + `is_hw_owned=True`, so NVDEC
  frames stay ON the GPU in torch's own context — 13 ms/frame, 25 ms/pair
  at engine speed, zero decode PCIe traffic; the planes are wrapped into
  torch via `__cuda_array_interface__` and cloned off the surface pool on
  a per-worker stream. How the feeding matters: the same engine fed
  frame-at-a-time through the generic hwaccel path ran 55 ms/frame at 15%
  engine utilization; libavcodec's `hevc_cuvid` (the "p010le" fallback
  path) saturates it at 22.5. Two traps encoded in `decode.py`: the
  probe container is held open for the process lifetime (FFmpeg sets
  primary-context scheduling flags at hwdevice creation; releasing the
  retain and letting torch re-create the context with different flags
  breaks all later hwdevice creations with EOPNOTSUPP), and hwaccel probes
  must be platform-gated (a VideoToolbox probe on Linux "succeeds", then
  falls back to software silently WITHOUT threading — the original
  590 ms/frame single-thread trap).
- The Python GIL is a pipeline stage: an 88 MB/frame numpy copy in the
  encoder thread (pinned slot → AVFrame) holds the GIL and stalls the
  stitch thread's kernel launches — measured as THE limiter once decode
  was fast (e2e stuck at ~95 ms with every engine <50% busy). numpy's
  memcpy does not release the GIL; torch's does. `_p010_frame` copies via
  torch: e2e 95 → 53 ms/frame.
- The CUDA pipeline is three stages (stitch → frame-build → encode+mux):
  the NVENC `encode()` call costs ~35 ms with the engine only ~30% busy —
  most of it is libav's internal pageable upload of the frame we just
  downloaded — so it gets a thread of its own, separate from the ~18 ms
  frame-build copy: e2e 53 → 46 ms/frame (22 fps), the V100's shipping
  number. Stage decomposition at 46 ms: D2H wait 0.5 (fully hidden),
  build 18, encode+mux 35, decode 25/pair. MPS keeps the two-thread
  layout its numbers were recorded with.
- Device-frame NVENC encode needs NO PyAV fork — stock PyAV passes any
  `cuda`-format frame straight through to the encoder, and reusable
  CUDA frames can be minted by decoding a tiny generated clip
  device-resident (see `tools/hwframe_encode_probe.py`). Measured on
  the V100 it is a 2.7× LOSS (55 vs 20 ms/call): Volta's per-frame
  nvEncMapInputResource/unmap costs more than the upload it saves, and
  nvenc.c routes encoder-pool frames through the identical register/map
  path, so no API (fork or otherwise) can do better on this silicon.
  Re-probe on Turing+ before assuming either way.
- One NVENC session runs on ONE engine: before segment-parallel encode
  the full base clip showed the encoder at exactly 33% in every sample —
  the V100's three engines with one pegged (~59 ms/frame on real
  content) and two idle. `enc_sessions` (default 3 on CUDA+NVENC) fixes
  it: three PERSISTENT sessions each encode every third ~2 s chunk, a
  forced IDR (`pict_type=I` + `forced-idr`) opens each chunk so chunks
  stay closed, `bf=0` keeps dts monotonic across seams, and a reorder
  muxer re-serializes chunks into the single hvc1 track (all sessions
  emit identical SPS/PPS). Sessions must be persistent — creating one
  per chunk was measured as a ~20 ms/frame loss. Cost: keyframe every
  2 s instead of ~8 s, ~+6% file size, encode quality 39-40 dB PSNR
  against the single-session encode (normal GOP-layout noise; the
  stitched frames are bit-identical). Full base clip: 310.9 → 200.7 s
  (36 ms/frame, 28 fps, 0.79× realtime), engines at GPU 80% / NVENC 66%
  / **NVDEC 94-100% — the single decode engine is now the hardware
  wall**. Short benchmark windows under-report multi-session (drain
  tail); measure ≥450 frames.
- One-time setup is ~6.5 s: ~3.5 s map build + ~3 s `torch.compile`. The
  radial polynomial is Horner-evaluated (bit-identical maps, 2.4× faster than
  the naive `theta**n` form). Threading the two lens builds was **measured as
  a loss** (2.0 → 2.9 s — the build is memory-bandwidth-bound); a raw disk
  cache of the maps was rejected at 885 MB per resolution for a ~2 s saving.
- Cross-device parity: the same frame stitched on MPS vs torch-CPU differs by
  at most 0.13 of a 10-bit level (interpolation rounding); CUDA vs torch-CPU
  is bit-identical with identity levelling, and with a live rotation 97% of
  pixels are bit-equal and 3 of 29.5M exceed 1 LSB (Triton's sin/cos in the
  compiled level grid — same class of effect as the documented ≤0.0004%
  MPS recompute delta). The math is device-independent in practice.
- V100 (CUDA 13 dropped Volta): the box needs the cu126 torch wheel —
  `pip install torch==2.13.0+cu126 --index-url
  https://download.pytorch.org/whl/cu126` after requirements (plain
  `==2.13.0` is "satisfied" by the kernel-less +cu130 wheel and every op
  fails at runtime). `torch.compile` also takes ~28 s there (vs ~3 s MPS),
  the bulk of the ~32 s setup.
- Encoding is portable: `convert()` auto-selects VideoToolbox → NVENC →
  libx265, with the packed→planar repack handled losslessly for software
  encoders (libx265 muxes as `hev1`, cosmetic: QuickLook-only issue, and macOS
  gets VideoToolbox anyway).

## Single-pass resample (`single_pass=True`)

The default stitch is two-pass: warp both lenses into the rig equirect,
blend, rotate the pano by the per-frame `R` with a second `grid_sample`.
That interpolates every output pixel twice, puts the `asin` singularity of
the second pass on the lens axes (the output's front/back centres), and —
a bug found in review — samples the pano with a (W−1)/W stretch
(`align_corners=True` maps ±1 onto the edge pixel *centres*, but the
pano's pixel for a longitude is `lon/2π·W`) with the wrap column clamped.

`GpuStitcher(single_pass=True)` pads the static uv maps and seam weights
to `(H+1, W+1)` (wrapped column, extrapolated pole row) so the unchanged
level grid indexes them exactly, samples each lens's uv map at the level
grid and the fisheye once at that uv. The seam stack keeps its semantics:
the band rows are warped into rig space for the detectors, and the fields
it produces are gathered only at the in-band output pixels (sparse). It
costs extra gathers — +47 ms static / +140 ms with the seam stack on the
M3 (whose GPU one 8K frame saturates); on the V100 the estimate is
+8–15 ms, under the NVDEC wall (to be measured). `lens_interp` selects
the luma sampling kernel: `bicubic` is torch's a = −0.75 cubic
(≡ cv2 `INTER_CUBIC`), a mildly *sharpening* kernel — more accurate than
bilinear on fine texture, less on smooth content (sub-LSB), free on the
GPU; `catmull` is the accurate Catmull-Rom cubic built from 9 bilinear
fetches (quality renders only). Numbers: `IMPROVEMENTS.md`.

## Setup amortisation

The rig geometry (maps + seam weights) is cached process-wide by a hash of
the calibration and output geometry (`pipeline.geometry_key`); the
`GpuStitcher` is kept across `convert()` calls while geometry and options
match; `convert(compile_cache_dir=)` / `set_compile_cache_dir()` points
inductor's and Triton's on-disk caches at persistent storage so the
`torch.compile` cost is paid once per *image*, not per process.
`Converter` wraps this for batch use. Second file from the same unit:
13.4 s → 1.2 s setup on the M3.

## CPU reference path

`Stitcher360.stitch()` mirrors the pipeline on the CPU (with multiband and
optical-flow seam options) for stills, debugging, and machines without a GPU.
It is the semantic reference the GPU path is checked against.
