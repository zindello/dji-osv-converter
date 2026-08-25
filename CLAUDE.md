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

## Performance state (8K, Apple M3 / MPS)

~59 ms/frame stitch, ~110 ms end-to-end serial (decode 52.6 + stitch back to
back; pipelined floor ≈ 59 ms → ~17 fps), ~7 s one-time setup (map build +
torch.compile). Budget table in `docs/architecture.md`. Measured-and-rejected:
threading the map build (memory-bound), disk-caching maps (885 MB/resolution),
batched lens warps (pre-`cat` eats the win), fp16 planes (MPS grid_sample not
dtype-bound). MPS quirks: `(3,3)@(3,N)` matmul is 7× faster than
`(N,3)@(3,3)`; inductor's MPS backend can't codegen uint16 (cast eagerly).
Chroma grids are decimated `[::2]`, never interpolated (wrap meridian
corruption).

## Current roadmap

1. **Concurrency** (on the M3): overlap decode (~53 ms) / stitch (~96 ms) /
   pack+D2H (~34 ms) / encode. Floor ≈ max(GPU, decode) → ~15 fps.
   Threaded pipelining already auto-enables on CUDA (`pipeline=True` path in
   `video.py`); MPS measured faster serial.
2. **CUDA validation** (on the V100 box): `pick_device` → cuda, NVENC encode
   (auto-selected), the pipeline path, real perf numbers. Cross-device parity
   already verified MPS vs torch-CPU to 0.13 of a 10-bit LSB.
3. **Multi-camera data** (awaited from the user): re-verify every
   `[MEASURED]` constant (`K_PLAIN`, `C_UP_FORMAT`, seam ratio 0.038) and the
   resolution/frame-rate handling against other units, rates and modes.

Known accepted gaps: rim motion blur under extreme tumble (source-limited,
present in DJI's exports too); libx265 fallback muxes `hev1` not `hvc1`
(QuickLook-only cosmetic); `[MEASURED]` constants verified on one unit so far.
