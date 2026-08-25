# dji_osv_converter — project instructions

DJI Osmo 360 `.OSV` → equirectangular 8K (7680×3840) 10-bit 360 MP4, matching
or beating DJI Studio's own exports, with **everything derived from the OSV
file itself**. Pure-Python library + reference tools. Reverse-engineered; no
DJI code or software is used at runtime.

## Hard rules (user-mandated, non-negotiable)

- **Never invoke the `ffmpeg` binary.** All decode/encode is libavcodec
  in-process via PyAV.
- **No DJI software/code at runtime.** The OSV file is the only input.
- **"Constants are bad."** Anything readable from the OSV is read from *each
  file* (per-unit lens calibration, extrinsics, camera↔IMU mount, frame rate,
  resolution). What cannot be derived lives in
  `dji_osv_converter/constants.py` with `[MEASURED]` / `[CONVENTION]` /
  `[ASSUMED]` provenance tags — never scattered through the code. Do not
  assume all cameras share the reference unit's values.
- **Device-agnostic GPU.** torch only; must run on CUDA and Apple MPS, with a
  working torch-CPU and pure-numpy fallback.

## Validation discipline (how claims get accepted here)

- Validate against **DJI Studio's own exports**, frame-by-frame. OSV frame N
  aligns 1:1 with DJI's `exported_360_plain.mp4` frame N (frame 10 is the
  standing comparison baseline).
- Minimum **6 wide-spaced frames** per comparison — never cherry-picked.
- **Measure the actual render, never the analytic ideal.** Several plausible
  metrics failed here; see `docs/validation.md` for the graveyard.
- The user watching the rendered video is the **final gate**. Nothing is
  "solved" before that.
- Around refactors: capture a bit-exact output baseline first, verify against
  it after. This has caught real bugs more than once.

## Architecture — read these before touching the pipeline

`docs/` is the source of truth: `architecture.md` (module map, data flow,
perf budget), `levelling.md` (the stabilisation math and its landmines),
`osv-format.md` (every reverse-engineered field), `constants.md`,
`validation.md`. `docs/api/` is pdoc-generated — **never hand-edit**; the
pre-commit hook regenerates it.

Facts that repeatedly mattered (full detail in docs/):

- Render convention: the GPU samples output direction `d` at `R @ d`; the
  output pole `[0,1,0]` shows rig direction `R @ [0,1,0]`.
- IMU-body → equirect bridge: `bridge = C_UP_FORMAT @ mount`, `mount` read
  per-file (OSV header 1.3.1, ~13° magnetic-mount skew), `C_UP_FORMAT` a
  format constant with det = −1 (intentional).
- RockSteady: `R = bridge · rawᵀ · smooth · bridgeᵀ · K_PLAIN` — the
  conjugation makes raw cancel exactly; `K_PLAIN` stays on the RIGHT.
- Horizon lock: up **and** fwd from the **same instantaneous** quaternion
  (mixing a smoothed heading in caused a 56° yaw swing).
- The fisheyes arrive **ISP-processed in camera** (devignetted,
  exposure-matched): LSC and exposure correction are OFF by default; enabling
  them double-corrects and paints arcs across the seam.
- A complete rolling-shutter subsystem exists behind `video.use_rs = False`;
  it measurably worsens results on this camera — leave it off.
- **Seam repair** (`seam_flow="auto"`, the default — `docs/seam.md`): the
  lenses genuinely disagree near the camera. Three band-limited corrections,
  each detector-gated: model-free photometric match (2-pass — tight-gate
  estimators have a SELECTION BIAS where mismatch is worst), DIS flow morph
  (band-limited; full-frame flow made seams worse), and a dynamic seam gated
  to real parallax (ungated it draws a jumping ring around the sphere).
  Verify photometric fixes with a mask fixed BEFORE correction.
- Quality achieved (worst-case clip): rendered jitter 0.027 calm / 0.071
  moving °/frame vs DJI 0.046 / 0.118; horizon lock holds yaw tighter than
  DJI under violent motion.

## Fresh-clone setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
git config core.hooksPath .githooks   # REQUIRED: pre-commit = tests + doc regen
```

Sample footage lives in `osv_example/` — **gitignored**, transfer it
separately (each clip dir holds the `.OSV` plus DJI Studio reference exports
`exported_360_plain.mp4` / `exported_360.mp4`). The worst-case test clip is
`osv_example/pitching-rolling-worst-case-scenario/CAM_20260824110935_0001_D.OSV`.

## Commands

```bash
python -m unittest discover -s tests            # unit tests (no sample data)
sh tools/gen_docs.sh                            # regenerate docs/api
python examples/convert_video.py IN.OSV out.mp4 --levelling rocksteady_horizon
python tools/validate_against_dji.py            # frame comparison vs DJI export
```

## Performance state (V100 / CUDA)

Validated 2026-08-25 (branch `cuda-validation`): stitch ran unmodified
(bit-parity vs torch-CPU), NVENC + hvc1 + spherical verified, DJI
comparison passed per `docs/validation.md`. **36 ms/frame (28 fps,
0.79× realtime) on the full base clip** — 2× the M3 — with the single
NVDEC engine at 94-100% (the hardware wall); segment-parallel NVENC
(`enc_sessions`, 3 persistent sessions, forced-IDR chunks, reorder
muxer) recovered the card's other two encode engines. Under that:
decode-to-device (`gpu_decode_fmt` → `"cuda"`: NVDEC frames stay on the
GPU in torch's primary context, 25 ms/pair; the generic hwaccel path fed
the same engine at 15% utilization — CUDA decode lives in `nvdec.py`),
async D2H on a side stream (`stitch_yuv_async`), GIL-releasing torch
copies in the encoder thread (numpy's 88 MB memcpy held the GIL and
stalled kernel launches — e2e 95 → 53 from that one change), and a
three-stage CUDA pipeline (stitch → frame-build → encode+mux; the NVENC
call's internal re-upload is ~35 ms and deserves its own thread: 53 →
46). Platform traps: the box needs the **cu126 torch wheel** (CUDA 13
dropped Volta; the default wheel "imports fine" and fails at first
kernel); hwaccel probes must be platform-gated; the primary-ctx probe
container is held open on purpose (context-flags trap — see `nvdec.py`).
V100 GPU compute floor is ~10.5 ms/frame; the remaining e2e gap is
libav's pageable NVENC upload (device-frame encode is the roadmap item).

## Performance state (8K, Apple M3 / MPS)

~60 ms/frame static stitch (~127 ms with the default auto seam repair
engaged); e2e pipelined **~212 ms/frame sustained under defaults (4.7 fps)**
or ~73 ms (13.6 fps) with `seam_flow=False` — decode/stitch/encode pipelined
on threads, the GPU default; static output verified frame-identical to
serial. Sustained full-clip numbers on the fanless M3 include thermal
throttling. ~7-8 s one-time setup (map build + torch.compile). Budget in
`docs/architecture.md`, per-platform table in `docs/benchmarks.md`
(tools/benchmark.py is the canonical instrument).
Measured-and-rejected: threading the map build (memory-bound), disk-caching
maps (885 MB/resolution), batched lens warps (pre-`cat` eats the win), fp16
planes (MPS grid_sample not dtype-bound), D2H fetch on the encoder thread
(cross-thread `.cpu()` serializes on the MPS kernel queue), and EVERY form of
CPU concurrency (stage overlap 912→1078 ms; frame-parallel threads 0.79× —
numpy/OpenCV already saturate the cores; CPU stays serial). MPS quirks:
`(3,3)@(3,N)` matmul is 7× faster than `(N,3)@(3,3)`; inductor's MPS backend
can't codegen uint16 (cast eagerly). Chroma grids are decimated `[::2]`,
never interpolated (wrap meridian corruption).

## Current roadmap

1. **Multi-camera data** (awaited from the user): re-verify every
   `[MEASURED]` constant (`K_PLAIN`, `C_UP_FORMAT`, seam ratio 0.038) and the
   resolution/frame-rate handling against other units, rates and modes.
2. **Faster CUDA decode** (see `docs/ROADMAP.md`): per-stream decode
   threads / NVDEC+sw hybrid / decode-to-device; only matters on
   single-NVDEC parts — newer NVIDIA GPUs have 2-5 engines.
3. **Seam CPU work → builder stage** (V100): the auto seam repair adds
   ~65 ms of CPU (DIS flow + seam DP + estimator) on the stitch thread;
   it rides `stitch_yuv_async` already, so moving it into the CUDA
   pipeline's builder stage recovers the V100's throughput. Re-measure
   both benchmark rows under merged defaults afterwards.

Known accepted gaps: rim motion blur under extreme tumble (source-limited,
present in DJI's exports too); libx265 fallback muxes `hev1` not `hvc1`
(QuickLook-only cosmetic); `[MEASURED]` constants verified on one unit so far.
