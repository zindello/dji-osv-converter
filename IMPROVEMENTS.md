# Improvements on `seam-parallax` — quality, speed, correctness (Nov 2026)

Branch `fable/improvements`, based on `origin/seam-parallax`. Nothing was
pushed; no media is committed (`ab_out/` is gitignored). Every change is
switchable and off by default unless stated; the A/B viewer
(`tools/ab_compare.py` → `ab_out/index.html`) renders each one alone and
all together for Josh's eyes.

All measurements below are from an Apple M3 (8 cores, 16 GB, MPS) unless
labelled; the production target (V100, Ivy Bridge host) must be
re-measured with the commands in §"What must be re-measured on the V100".

## Summary table

| # | Change | Switch (default) | Measured effect | Risk |
|---|---|---|---|---|
| 1 | Streaming spherical-metadata injection | `inject_spherical(path, streaming=True)` | peak RSS growth < 64 MB on a 3 GB file (was ~3× file size); byte-identical | low: tested both box orders |
| 2 | Single-pass resample + level-grid fix | `convert(single_pass=True)` / `--single-pass` (off) | sharpness +30–40 % (Laplacian var.); exact wrap column; no pole singularity; matches an f64 reference to <0.5 LSB | medium: +98 ms/frame static, +186 ms with seam stack on the M3 (GPU-saturated); V100 estimate +10–15 ms |
| 3 | Lens sampling kernels | `lens_interp="bicubic"\|"catmull"` (bilinear) | bicubic: free on MPS (62 vs 60 ms), +30 % sharpness, **sharpening kernel** (less accurate on smooth content, sub-LSB); catmull: accurate, ~9× warp cost | low (clamped); bicubic can ring on hard edges |
| 4 | Setup amortisation | `reuse=True` (on), `compile_cache_dir=`, `Converter` | second file: setup 13.4 s → 1.2 s on the M3 (maps cached, inductor cache warm) | low |
| 5 | Partial lens-shading correction | `lsc=<alpha>` (0 = off) | measured residual rim falloff is ≥ 1.0 (no darkening) → **default 0 justified**; exploratory α = 0.3 in the viewer | none at default |
| 6 | Correctness batch | see §6 | colour tags carried; range guard; no silent CPU encode; hold-last levelling; empty-track crash fixed; explicit MP4 container | low |
| 7 | Encoder options | `enc_options=encoder_options(...)`, `bpp=`, `enc_inflight_frames=` | defaults unchanged; NVENC AQ/lookahead need the V100 to validate; in-flight 8K frames bounded to 1 chunk (~6 GB) instead of 3 | none at default |
| 8 | Seam stack on the target host | measured CPU cost per component (below) | see §8 | — |

## Prior review re-checked against `seam-parallax`

The review (`osv_converter_review.md`) was written against `main`.
Against `seam-parallax`:

- **§2.1.3 (parallax flow at the seam)** — done on the branch and more:
  DIS flow (band-limited, consistency-gated), a dynamic seam gated to real
  parallax, and a 2-pass photometric match. Moot.
- **§2.2 (brightness gradient)** — the branch's photometric v2 addresses
  the *inter-lens* mismatch (band-limited). The *common-mode* residual
  vignette the review hypothesised is **not there** (§5 below): measured
  ≥ 1.0 toward the rim. Resolved by measurement, not by code.
- **§2.1.1 / §2.5.2 (two-pass resample, level-grid stretch)** — still
  present on the branch (`stitch_gpu._stitch_dev`): fixed here (§2).
- **§2.1.2 (bicubic)** — not on the branch; added (§3), with the
  important caveat that torch's bicubic is a = −0.75.
- **§2.4.1 (setup)** — not on the branch; added (§4).
- **§2.5.1 (colour tags), §2.5.3 (inject_spherical RAM), §2.5.4 (audio),
  §2.5.5 (K_PLAIN fallback, codec fallback)** — all still present on the
  branch; fixed/verified here (§1, §6).
- **§2.4.3 (session queue memory)** — still present; bounded (§7).
- **§2.4.4 (NVENC knobs)** — still absent; added as options (§7).
- **Seam CPU cost** — the branch's "seam CPU diet" (DIS + 960-col DP)
  landed; the review's concern about the Ivy Bridge host remains and is
  quantified in §8.
- `docs/validation.md` still references a `tools/ab_seam.py` that is not in
  the tree; `tools/ab_compare.py` supersedes it.

## 1. Streaming `inject_spherical`

`dji_osv_converter/spherical.py`. The old function read the whole MP4 into
`bytes`, copied it into a `bytearray` and wrote `bytes(buf)` back: three
copies of the file. An 8K master is ~1.6 GB/min at 0.25 bpp, so a 16-minute
clip (~26 GB) would need ~78 GB and be OOM-killed.

Now only the box headers and the `moov` are read. `moov` after `mdat`
(libav's default): truncate at the moov and rewrite it (O(moov) I/O and
memory). `moov` before `mdat` (faststart): the media must move, so the
file is rebuilt beside the original with an 8 MB-chunk streamed copy and
`os.replace`d (never whole-file in RAM; needs the file's size in free
disk temporarily — the only case where that happens).

Switch: `inject_spherical(path, streaming=False)` is the original edit.
Tests (`tests/test_spherical.py`): byte-identical to the original on
libav-muxed files in both orders, still decodes, stco shifted correctly on
a synthetic faststart file, and **peak RSS growth < 64 MB while injecting a
3 GB sparse file** (subprocess-measured).

## 2. Single-pass resample (and the level-grid scale bug)

`dji_osv_converter/stitch_gpu.py`: `_sp_init`, `_sp_sample_lens`,
`_sp_seam_fields`, `_stitch_single_pass`; dispatch in `_stitch_dev`.
Switch: `GpuStitcher(single_pass=True)`, `convert(single_pass=True)`,
`examples/convert_video.py --single-pass`. Default off.

What was wrong in the two-pass path (confirmed, not just from the review):

- every output pixel was bilinearly interpolated twice (warp into the rig
  equirect, then the levelling `grid_sample`);
- the level grid `gx = lon/π` under `align_corners=True` samples pixel
  `(gx+1)/2·(W−1)` but the pano's pixel for that longitude is
  `(gx+1)/2·W`: a (W−1)/W stretch (up to 1 px at the far edge) and the wrap
  column clamped (`padding_mode="border"`) instead of wrapped;
- the second pass's `asin` singularity sits where the rig poles land: the
  lens axes, i.e. the front/back centres of the output.

How it is done now: the static uv maps (lens pixel per rig direction) and
the seam weights are padded to `(H+1, W+1)` — a wrapped column plus a
pole row extrapolated from the last two rows (the pixel offset from the
lens axis is linear in the polar angle there, so the extrapolation lands
on the pole's one direction; tested). Under `align_corners=True` the
unchanged level grid then indexes these maps *exactly*. Per frame: sample
each lens's uv map at the level grid (bilinear — the map is smooth; its
interpolation error at 8K is ~1e-3 px, measured 0.001 px on the synthetic
rig away from the rim), then sample the fisheye once. The seam stack is
unchanged in meaning: the band rows of each lens are warped into rig space
(about 1/8 of the sphere), the estimators run on them as before, and the
resulting fields (flow morph in rig px, photometric offset, dynamic-weight
delta) are gathered **only at the in-band output pixels** (nonzero +
point-grid `grid_sample` + `index_put`; bit-identical to the dense
version, −110 ms on the M3). Chroma is sited per MPEG-2 "left" (the
two-pass reused luma-normalised coordinates on the half plane: a ≤ 0.5
chroma-px radial scale error at the rim). Exposure matching and
rolling-shutter (both off by default) fall back to the two-pass path.

Tests (`tests/test_single_pass.py`, synthetic rig, torch-CPU): against an
independent f64 numpy reference (`build_map(R_extra=R)` + `cv2.remap`),
mean error < 0.5 LSB and less than half the two-pass error; a one-column
yaw rolls the panorama by exactly one column; identity equals the two-pass
to 1 LSB; the stack is band-limited; chroma matches a correctly sited
reference.

Real footage, vs the bit-exact baseline planes captured before any change
(base clip f10/f2000, worst-case f230; plain + rocksteady_horizon; seam
off/auto) — Laplacian variance of 512-px luma crops:

| crop (base f10, rocksteady_horizon) | baseline | single-pass | single-pass + bicubic |
|---|---|---|---|
| front centre | 293 | 409 | 546 |
| seam left | 1018 | 1155 | 1605 |
| seam right | 186 | 213 | 290 |
| sky | 390 | 564 | 748 |

Mean |Δ| vs baseline 4–5 LSB, 63–66 % of pixels differ by ≥ 1 LSB (the
baseline was softer and stretched). Against DJI at 3840 px width the
difference is invisible (NCC 0.958 both), as expected for sub-pixel
sharpness through a 2× downscale — the 1:1 crops in the viewer are where
to look.

Cost (M3/MPS, one stitcher resident, 8 frames, base clip f2000):

| | two-pass (baseline) | single-pass | single-pass + bicubic |
|---|---|---|---|
| static blend (`seam_flow=False`) | 60 ms | 158 ms | 164 ms |
| auto seam repair (default) | 126 ms | 312 ms | 329 ms |

The M3's GPU is saturated by one 8K frame (docs), so the extra gathers
(two 2-channel uv gathers per lens + the weight gather) cost what they
cost. On the V100 an 8K single-channel gather is ~1–2 ms, so the estimate
is +8–15 ms per frame against a compute floor of 10.5 ms under a 36 ms
NVDEC wall — i.e. still hidden. **Unverified until run there.** A further
M3 win would be computing the uv analytically in the fused level-grid
kernel instead of gathering the maps (removes the 2-channel gathers and
the map upload); not done.

## 3. Lens sampling kernels

`GpuStitcher(lens_interp=)`, `convert(lens_interp=)`, `--lens-interp`.
Luma only; chroma stays bilinear. Default `"bilinear"`.

Measured on an analytic signal (1-D probe and the synthetic-rig test):

| kernel | smooth content (period ~60 px) RMS err | fine texture (period ~10 px) |
|---|---|---|
| bilinear | 0.18 LSB | 5.8 LSB |
| `"bicubic"` (torch, a = −0.75 ≡ cv2 INTER_CUBIC) | **0.52 LSB** (worse) | 3.4 LSB |
| `"catmull"` (Catmull-Rom, a = −0.5) | < bilinear | < bilinear |

So torch's "bicubic" is a mildly sharpening kernel, not a more accurate
interpolator: it wins on fine texture and loses (sub-LSB) on smooth
content, with overshoot at hard edges that the pack's range clamp bounds
(tested). It is free on the M3 (62 vs 60 ms, two-pass) and raises
Laplacian variance ~30 % (two-pass) / ~90 % with single-pass. Catmull-Rom
is the accurate cubic; torch doesn't expose it, so it is built from 9
bilinear fetches: 1.3 s/frame on the M3 — a quality-render option only.
`tests/test_single_pass.py` pins both characters so nobody flips the
default on the strength of the name.

Recommendation: judge `bicubic` in the viewer on the seam/periphery
crops and the sky (ringing on hard edges is the thing to look for). If
accepted for offcuts, it is the cheap one on the V100 too (16-tap native
kernel).

## 4. Setup amortisation

- `pipeline.geometry_key()` — sha256 over every calibration field the
  projection reads + output geometry; `Stitcher360.maps_for()` serves maps
  and seam weights from a bounded process-wide cache keyed by it
  (`RIG_CACHE_ENTRIES = 2`).
- `video._gpu_stitcher()` keeps one `GpuStitcher` across `convert()` calls
  (`reuse=True` default; `clear_gpu_cache()`).
- `set_compile_cache_dir()` / `convert(compile_cache_dir=)` — points
  `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` at persistent storage
  *before the first compile* (inductor pins the directory on first use and
  otherwise uses the temp dir, which a pod loses). Inductor's FX-graph
  cache is on by default in torch 2.15, so the ~28 s V100 compile becomes a
  warm load on the next process. Existing env is honoured.
- `Converter(**defaults).convert(src, dst, **overrides)` for many files in
  one process.

Measured (M3): first window setup 13.4 s (maps 9.5 s + GPU 5.3 s incl.
compile; `from_file` ~1 s); the next stitcher for the same unit 1.2 s. The
map build is the single largest CPU cost at setup here (9.5 s with 16 GB;
the docs' 3.5 s was a 2 s-class machine with more RAM) — on Ivy Bridge
expect 10–20 s, paid once per unit per process with the cache. GPU map
building was not done: MPS has no f64, and on the V100 the cache makes it
a once-per-night cost. Tests: `tests/test_setup_cache.py`.

## 5. Brightness gradient — measured, default stays 0

Method (scripts in the session scratchpad; results in
`ab_out/_sky_falloff.json`, `ab_out/_luma_vs_theta.json`): seven clips
(the base sample plus six from the camera library), 6–8 spread frames
each, both lenses, in the fisheye domain (no stitch). A plain mean-luma vs
θ profile is scene-dominated (centre/rim ratios 0.6–7× between clips) and
useless. The second estimator keeps only sky pixels (bright, bluish, low
gradient), bins them by *elevation in the camera frame* and θ, normalises
each elevation row by its θ = 40–52° bins, and averages the ratios:

```
theta:     40   44   48   52   56   60   64   68   72   76   80   84   88   92
all mean 0.99 1.00 1.01 1.01 1.02 1.03 1.04 1.04 1.04 1.05 1.05 1.06 1.06 1.05
all std  0.02 0.01 0.02 0.04 0.06 0.08 0.09 0.11 0.13 0.13 0.14 0.14 0.15 0.16
factory  1.01 1.00 0.99 0.98 0.96 0.95 0.93 0.91 0.89 0.87 0.85 0.82 0.81 0.81
```

There is no common-mode darkening toward the rim; if anything the ISP
slightly over-lifts it (+5 % at 80–92°, within a 0.15 std). The implied
alpha against the factory curve is *negative* (−0.2…−0.6), so a partial
positive LSC would make the gradient worse. Per lens the picture is
asymmetric and scene-dependent (clip 0005: lens0 0.94, lens1 1.14 at 92°)
— that is inter-lens, lighting-direction mismatch, which the branch's
band-limited photometric match already handles. Conclusion: the review's
hypothesis is not supported; `lsc` default 0. The plumbing
(`lsc_gain_map(alpha=)`, `convert(lsc=0.3)`) exists, and an α = 0.3
variant is in the viewer for a visual verdict only.

## 6. Correctness

`dji_osv_converter/video.py`, `pipeline.py`, `stitch_gpu.py`,
`tests/test_video.py`, `tests/test_pipeline.py`, `tests/test_stitch_gpu.py`.

- **Colour tags**: `color_range/primaries/trc/colorspace` copied from the
  source (`_source_colour`), onto the mux stream and every NVENC session
  context. The OSV is BT.709 limited (1/1/1/1); the output was untagged.
  `colour_tags=False` restores that. Verified on a real 6-frame convert.
- **Range guard**: `GpuStitcher(yuv_range="limited"|"full")` from the
  source's declared range sets the pack clamp (64–940/960 vs 0–1023) and
  every black-level centring (LSC, exposure, seam estimator); a full-range
  source through the old fixed clamp would have been crushed silently.
  Range 0 (unspecified) is treated as limited, as before.
- **Audio start alignment** (review §2.5.4, verified with numbers): the
  OSV has no edit list; video, audio and data streams all start at pts 0.
  Copying packets preserves the source's A/V relationship exactly. With
  `start=N` the first kept packet is the first at or after t₀ = N/fps:
  `start=100` → first audio pts 608/48000 = 12.67 ms, which is exactly
  where that audio sits relative to frame 100 (one AAC frame is 21.3 ms,
  so the lead-in silence is ≤ 21 ms, never early). DJI's own export instead
  re-encodes audio with a −44 ms priming edit list. No change needed.
- **End-of-clip jump**: `levelling_for()` now holds the last orientation
  when the metadata track is shorter than the video (was a `K_PLAIN` snap);
  an empty track no longer crashes `smoothed_rotation_track`.
- **No silent CPU encode**: `convert(codec=None)` auto-selects
  VideoToolbox/NVENC only; `libx265` needs `allow_software=True` or an
  explicit `codec=`. The message says so.
- **Explicit MP4 container** (`_open_output`): PyAV guesses from the
  extension, so `out.mp4.partial` failed; now `format="mp4"` always.
  Tested with `.partial`, extension-less and `.bin` names, including the
  360 tag afterwards.
- Found and fixed in passing: `smoothed_rotation_track` on an empty track.

## 7. Encoder settings and queue memory

- `encoder_options(codec, preset=, tune=, aq=, lookahead=, multipass=,
  rc=, extra=)` → `convert(enc_options=...)`; applied to the mux stream and
  the session contexts. NVENC-only knobs never leak onto other encoders.
  Defaults unchanged. `bpp=` overrides `BITRATE_BPP` (`bitrate=` wins).
- `enc_inflight_frames` (default one chunk + 2× the stage queue): a single
  semaphore bounds the 8K p010 frames alive across *all* session queues —
  ~6 GB worst case instead of ~18 GB (the old bound was per session).
  Steady state (~1 deep) is unaffected.

These can only be validated on NVIDIA — see the V100 section.

## 8. Seam stack on the target host (Ivy Bridge)

Per-frame CPU cost of the seam stack's components on this M3 (from
`ab_out/_cpucost_and_timing3.log`; base clip f2000, band 364 × 7680 →
pooled 364 × 1920, 5.6 MB downloaded per frame):

| component | ms (8 threads) | ms (1 thread) |
|---|---|---|
| photometric estimator (`seam_luma_offsets`) | 4.6 | 4.7 |
| grays prep | 1.3 | — |
| DIS flow pair (`seam_flow_small`) | 11.2 → **~6 after this branch** (two-threaded, identical output) | 11.9 |
| seam DP (`seam_path`, 960 cols) | 9.8 | 10.4 |
| seam gate | 2.2 | — |
| **total** | **29.1 → ~24** | |

It is effectively single-threaded Python/numpy/OpenCV work on the stitch
thread (the 1-thread numbers match), so core count does not help it;
only per-core speed does.

Ivy Bridge estimate: the E5-2600 v2 class runs roughly 2–2.5× slower per
core than the M3 on this kind of numpy/OpenCV work (AVX only, no AVX2, ~2.5
GHz). Estimate: 29 ms × 2.2 ≈ **64 ms/frame** before this branch — which
matches the ~65 ms the docs already recorded on the V100 box — and
~53 ms after the DIS threading. Both exceed the 36 ms NVDEC wall, so with
`seam_flow="auto"` engaged (it engages on nearly all handheld footage) the
seam stack, not NVDEC, is the V100's limiter at roughly 15–19 fps instead
of 28. Mitigations already in the branch: DIS instead of
Farneback, 960-column DP. If it threatens throughput there, the cheap
levers in order: (1) `seam_flow="auto"` already skips flow/DP on frames
without parallax (but the estimator always runs); (2) the estimator could
run at 1/8 instead of 1/4 pooling (its output is smooth by construction);
(3) move the CPU work to the pipeline's builder stage (the roadmap item).
None of these was implemented here — the measurement says what the budget
is; the decision needs the V100 run.

## A/B windows and the viewer

`tools/ab_compare.py` → `ab_out/index.html` (`open ab_out/index.html`).
Windows (plain levelling so the seam meridians are fixed at x = W/4 and
3W/4 and DJI's plain export is the 1:1 reference where it exists):

| window | clip | frame | why |
|---|---|---|---|
| `base_f10` | `osv_example/CAM_20260819204151_0011_D.OSV` | 10 | the standing comparison frame; DJI export |
| `worst_f230` | `pitching-rolling-worst-case-scenario/CAM_20260824110935_0001_D.OSV` | 230 | mid-tumble; DJI export |
| `light_0011_f10` | `Movies/Osmo360/CAM_20261030165035_0011_D.OSV` | 10 | brightest second in the library (LRF mean luma 158/255) |
| `dark_0021_f1349` | `Movies/Osmo360/CAM_20261101111038_0021_D.OSV` | 1349 | darkest second in the library (71/255; there is no night footage) |
| `normal_0017_f1229` | `Movies/Osmo360/CAM_20261101105955_0017_D.OSV` | 1229 | steady mid-tone handheld (117/255) |

Selection: mean grey per second of every `.LRF` proxy
(`ab_out/_lrf_luma.json`). Variants: `baseline`, `seam_off`,
`single_pass`, `bicubic`, `catmull`, `lsc_partial` (α = 0.3),
`all` (single-pass + bicubic), `all_catmull`, `dji`. Regions: full
(3840 px), and 1024×768 1:1 crops at front, seam_left, seam_right,
periphery, sky_front, sky_seam, ground. Diff images are pre-rendered
against the baseline (browsers refuse canvas readback of `file://`
images; the live pair diff works when the folder is served over HTTP).

### Results (numbers from `ab_out/manifest.json`; the images are the verdict)

Stitch ms/frame on the M3 with one resident stitcher (auto seam engaged,
final code): baseline 136, seam_off 62, single_pass 284, bicubic 136,
catmull 1156, lsc_partial 160, all 283, all_catmull 1407. (The isolated
runs in §2 give 265 ms for single-pass; this tool has decode/JPEG work
alongside, and MPS varies ±10 % run to run.)

Laplacian variance of the luma (higher = more high-frequency energy;
sharper *or* noisier — the crops decide which), per 1:1 crop:

| window / variant | full | front | seam L | seam R | periphery | sky seam | ground |
|---|---|---|---|---|---|---|---|
| base_f10 baseline | 151 | 15 | 47 | 27 | 81 | 38 | 35 |
| base_f10 single_pass | 175 | 17 | 69 | 37 | 118 | 52 | 45 |
| base_f10 bicubic | 179 | 20 | 59 | 33 | 105 | 47 | 46 |
| base_f10 catmull | 171 | 18 | 55 | 32 | 98 | 44 | 43 |
| base_f10 all (sp+bicubic) | 209 | 23 | 92 | 47 | 160 | 68 | 62 |
| base_f10 **DJI** | 84 | 11 | 16 | 11 | 29 | 11 | 20 |
| worst_f230 baseline | 447 | 617 | 154 | 258 | 168 | 6 | 170 |
| worst_f230 single_pass | 561 | 769 | 277 | 436 | 326 | 10 | 263 |
| worst_f230 all | 781 | 1220 | 421 | 654 | 528 | 14 | 426 |
| worst_f230 **DJI** | 210 | 270 | 34 | 59 | 27 | 2 | 56 |
| light_0011 baseline / single_pass / all | 237 / 298 / 404 | 106 / 137 / 218 | 238 / 464 / 751 | 2 / 2 / 2 | 341 / 630 / 999 | 32 / 57 / 82 | 107 / 150 / 229 |
| dark_0021 baseline / single_pass / all | 1061 / 1324 / 1763 | 262 / 316 / 479 | 98 / 146 / 205 | 267 / 427 / 627 | 208 / 341 / 490 | 222 / 396 / 552 | 112 / 155 / 236 |
| normal_0017 baseline / single_pass / all | 200 / 251 / 339 | 93 / 117 / 185 | 43 / 77 / 118 | 69 / 118 / 183 | 57 / 107 / 169 | 1 / 2 / 2 | 127 / 204 / 328 |

Reading it: the single-pass gain is largest exactly where the two-pass
path softened most — the seam meridians and the lens periphery (+45–90 %)
— and smaller at the front centre (+15–25 %), as the geometry predicts
(the level pass resampled everywhere, the stretch grew toward the
edges). Bicubic adds a uniform ~+30 % (sharpening). In the dark window
every variant's increase at the seams includes noise; look at the
`dark_0021` seam crops before reading any of it as detail. DJI's own
render is roughly half the baseline's Laplacian variance everywhere:
the converter was already sharper than DJI Studio before this work.

Sky gradient (peak-to-peak of the smoothed sky-band column means, %):
unchanged by single-pass/bicubic (they move no energy at that scale);
`lsc_partial` (α = 0.3) *raises* it (base 163.1 → 165.1 global; light
27.4 → 30.2), consistent with §5: there is no residual falloff to lift.

Seam-band mismatch (mean |A−B| over the inner band, 10-bit, before the
stack → after photometric fix + morph), baseline / single-pass:
base_f10 75.5 → 57.0 / 57.3; worst_f230 56.1 → 56.8 / 56.7; light 47.4 →
46.0 / 45.7; dark 78.9 → **86.8** / 87.0; normal 35.6 → 33.3 / 34.5.
Single-pass reproduces the two-pass stack within 0.3 (same estimators on
the same band warps). A finding about the branch's stack itself, not
this work: on the dark window the post-stack mismatch is *higher* than
before it, and on the worst-case frame it is flat — the morph/photometric
pass is not reducing pixel disagreement there (noise-driven flow in a
dark band is the likely cause; the dark seam crops are where to look).
`seam_off` scores slightly *higher* similarity to DJI than the stack
(base NCC 0.9599 vs 0.9580; worst 0.9786 vs 0.9736): DJI's band differs
from both, and the stack moves ours away from DJI's where parallax is
real — the metric cannot tell "closer to DJI" from "correct"; that was
settled on the branch by eye (docs/seam.md).

## Validation vs DJI (≥ 6 wide-spaced frames, the project's rule)

`tools/validate_against_dji.py --gpu [--single-pass --lens-interp bicubic]`
renders through `GpuStitcher` at 8K (the shipping path) and scores at 3840
px after yaw alignment and tone matching.

Plain levelling vs each clip's plain export, 7 frames spread over the
export (base: frames 10 56 103 150 196 243 290; worst: 10 116 223 330 436
543 650), rendered on MPS at 8K through `GpuStitcher`:

| clip | variant | NCC | SSIM | edge-NCC | PSNR dB | MAE |
|---|---|---|---|---|---|---|
| base | baseline (two-pass, bilinear) | 0.9471 | 0.8029 | 0.6105 | 22.96 | 9.47 |
| base | single_pass | **0.9473** | 0.8026 | 0.6103 | **23.00** | **9.45** |
| base | all (single_pass + bicubic) | 0.9467 | 0.7993 | 0.6062 | 22.94 | 9.51 |
| worst | baseline | **0.9449** | **0.7527** | 0.6924 | **22.98** | **9.10** |
| worst | single_pass | 0.9439 | 0.7457 | **0.6939** | 22.89 | 9.21 |
| worst | all (single_pass + bicubic) | 0.9426 | 0.7372 | 0.6910 | 22.78 | 9.39 |

Per frame the worst-frame NCC is 0.906 (base f196) for all three variants
alike — the same frames are hard for all of them (motion), i.e. no
variant regresses any frame. Reading: **single-pass is neutral on the
DJI-similarity metrics** (±0.001 NCC, within frame-to-frame noise; edge
NCC up on the worst clip). **Bicubic lowers similarity slightly and
consistently** (−0.002 NCC, −0.015 SSIM on the worst clip): it adds
high-frequency energy DJI's softer render does not have, and these
metrics measure similarity to DJI, not quality. The project's standing
claims (NCC ≥ 0.927 worst frame, "matches or beats DJI" on geometry and
stabilisation) hold for every variant; whether bicubic's sharpening is
"better" is for the 1:1 crops. The rendered-jitter metric is untouched by
these changes (same `R` per frame, verified by the one-column-yaw test).

## What must be re-measured on the V100 (exact commands)

In the converter's venv on the V100 box (cu126 torch wheel, per
`docs/architecture.md`), on the base clip:

```bash
# 1. per-frame cost of single-pass / bicubic under the NVDEC wall (>=450 frames)
python tools/benchmark.py CLIP.OSV --e2e 450                      # baseline row
python - <<'EOF'
from dji_osv_converter import convert
import time
for kw in ({}, {"single_pass": True}, {"lens_interp": "bicubic"},
           {"single_pass": True, "lens_interp": "bicubic"}):
    ts = []
    convert("CLIP.OSV", "/tmp/v.mp4", count=450, levelling="rocksteady_horizon",
            audio=False, spherical=False, progress=lambda d, t: ts.append(time.time()), **kw)
    import numpy as np; d = np.diff(ts)[150:]
    print(kw, f"{d.mean()*1000:.1f} ms/frame")
EOF
nvidia-smi dmon -s u      # alongside: NVDEC/NVENC/SM utilisation; the question is whether
                          # SM stays under the NVDEC wall with single_pass on

# 2. setup: cold vs warm compile cache
rm -rf /tmp/osvcache; time python -c "from dji_osv_converter import Converter; Converter(compile_cache_dir='/tmp/osvcache').convert('CLIP.OSV','/tmp/a.mp4',count=5)"
time python -c "from dji_osv_converter import Converter; Converter(compile_cache_dir='/tmp/osvcache').convert('CLIP.OSV','/tmp/a.mp4',count=5)"

# 3. NVENC options: quality per bit (PSNR/SSIM of the decoded output vs the stitched
#    planes, the docs' 39-40 dB method) and whether the encoders become the wall
python - <<'EOF'
from dji_osv_converter import convert
from dji_osv_converter.video import encoder_options
for opts, bpp in ((None, 0.25), (encoder_options("hevc_nvenc", preset="p5", tune="hq", aq=True, lookahead=16), 0.25),
                  (encoder_options("hevc_nvenc", preset="p5", tune="hq", aq=True, lookahead=16, multipass="qres"), 0.20)):
    convert("CLIP.OSV", f"/tmp/enc_{bpp}_{bool(opts)}.mp4", count=450, enc_options=opts, bpp=bpp, audio=False)
EOF
# then PSNR/SSIM vs the 0.25-bpp default and file sizes; watch nvidia-smi dmon for NVENC at 100%

# 4. peak host RSS during a long convert (inject_spherical is now O(moov)):
/usr/bin/time -v python examples/convert_video.py LONG.OSV /tmp/long.mp4 --levelling rocksteady_horizon 2>&1 | grep "Maximum resident"

# 5. the DJI comparison on the V100 render (the "better than DJI" claim):
python tools/validate_against_dji.py BASE.OSV exported_360_plain.mp4 10 120 230 340 450 560 670 --gpu
python tools/validate_against_dji.py BASE.OSV exported_360_plain.mp4 10 120 230 340 450 560 670 --gpu --single-pass --lens-interp bicubic

# 6. seam stack CPU per frame on Ivy Bridge (compare with section 8):
python /path/to/seam_cpu_cost.py CLIP.OSV 2000     # the script is in the session scratchpad; tools/benchmark.py --e2e covers the e2e effect
```

## Tried and reverted / not done

- Dense application of the seam fields in the single-pass path (first
  version): +290 ms on the M3; replaced by the sparse version (bit-identical).
- Three stitchers resident at once for the A/B: 2–13 s/frame from memory
  pressure on 16 GB; the tool keeps one resident.
- GPU map building: MPS lacks f64; with the geometry cache the CPU build
  is paid once per unit. Not done.
- Analytic per-frame uv in the fused kernel (would remove the map gathers
  and the padded-map memory): not done; the M3 would gain most from it.
- Catmull-Rom with fewer than 9 fetches (the 5-tap game approximation
  drops the corner terms): not done, the exact kernel was wanted.
- Partial LSC with a non-zero default: the measurement says no.
- `docs/api` was not regenerated (pdoc is not in the venv; the pre-commit
  hook was not installed in this clone so the commits stay cherry-pickable
  without generated-HTML churn). Run `sh tools/gen_docs.sh` after
  cherry-picking.

## Commits on `fable/improvements` (oldest first)

- `312f1f9` chore: ignore ab_out/ (A/B renders are footage-derived and never committed)
- `8af97b8` spherical: stream the metadata injection (O(moov) memory, never the whole file)
- `54f7934` video: mux the output as MP4 explicitly, whatever the path's extension
- `71b86f6` video/pipeline: colour tags, range guard, no silent software encode, encoder options, bounded in-flight frames, hold-last levelling
- `ca63863` stitch: single-pass resample (levelling composed into the lens maps)
- `f3a5ecb` stitch: lens sampling kernels - bicubic (torch, a=-0.75) and Catmull-Rom
- `7599afa` setup: amortise it - geometry cache by calibration hash, GPU stitcher reuse, persistent compile cache, Converter
- `bcc25d4` lsc: partial lens-shading correction (alpha), default 0
- `931503e` stitch: apply the single-pass seam fields sparsely (band pixels only)
- `dbc62a1` tools: A/B comparison generator + offline viewer; validate_against_dji can score GPU variants
- `897d87a` perf: fuse the single-pass uv->grid conversion; run the two DIS directions concurrently
- `fc002f4` stitch: seam_stats pre/post use the same statistic (abs-then-pool mean |A-B| over the inner band)
- (this commit) docs: IMPROVEMENTS.md, README, architecture, validation, roadmap, CLAUDE.md
