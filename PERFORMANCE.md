# Performance: why production ran at 4.5 fps, and what changed

Branch `fable/gpu-offload`, from `ee1e170` (the commit production runs).
Written 13–15 November 2026.

**Read this first.** Nothing here has run on an NVIDIA card. Every number
marked *measured* was measured on the development Mac (M3, 8 cores, Apple MPS,
VideoToolbox) or is quoted from the V100 measurements already in
`docs/architecture.md`. Every V100 figure for the new code is a *prediction*,
and says so. `tools/v100_bench.py` exists to replace the predictions with
measurements; the command is at the end.

## Summary

- Production's 222 ms per frame is spent on **one thread**: the stitch thread.
  Decode, frame building, encode and mux run on their own threads and wait for
  it. The stitch thread does two things the 28 fps pipeline of August did not:
  it runs the seam repair on the CPU (OpenCV and numpy, with five downloads
  from the card that each stop and wait for it), and it runs a heavier
  single-pass sampler.
- The seam repair now runs **on the card, in torch**, with no download, no
  upload and no wait in the frame (`seam_engine="gpu"`, the default on CUDA and
  MPS). The CPU version is still there (`seam_engine="cpu"`).
- Measured on the M3 at production settings: **3.34 → 5.65 fps** end to end
  (299 → 177 ms per frame); the stitch thread's CPU time fell from 43.2 to
  9.9 ms per frame. Seam off is 9.35 fps on the same machine.
- **Predicted on the V100: 10–14 fps, most likely about 12** (4.5 today), and
  15–19 fps with `seam_flow` off. The reasoning is in "Predicted V100 speed".
  The hard ceiling stays at about 28 fps: the card has one NVDEC engine and
  two 3840×3840 streams to decode.
- What is left is GPU time in the single-pass sampler, which this branch did
  not restructure. "Next steps" ranks what would cut it.
- Output outside the seam belt is **bit-identical** to `ee1e170`. Inside the
  belt the pixels differ, because the optical flow is a different algorithm;
  seam quality by every measure we have is level with the CPU version. Numbers
  and the viewer are under "Quality".
- A bug in the shipped seam repair turned up on the way: the morph moves each
  lens *away* from the other. The fix is opt-in (`seam_morph="corrected"`),
  because it changes the look. See "The morph direction".

## Where the 222 ms went

### The pipeline, thread by thread

At `ee1e170` on the CUDA box a conversion runs these threads:

| Thread | Work per frame | Where | Host↔device copies |
|---|---|---|---|
| decode worker × 2 (one per lens) | demux; NVDEC decodes into the card's memory; planes cloned off the surface pool | CPU (demux), NVDEC engine, GPU (clone) | H2D: the compressed stream, 0.29 MB per lens (70 Mbit/s at 29.97 fps). No frame copy: frames stay on the card (`decode="cuda"`) |
| **stitch (the critical path)** | levelling lookup; unpack p010; level grid; seam repair; sample both lenses; blend; pack to p010; start the download | GPU kernels launched from Python, plus the CPU seam stack | D2H, each one a full wait for the card: two luma strips for the photometric estimate (2 × 175 KB), two greys for the flow (2 × 700 KB), the seam cost (657 KB), the band index count. H2D: offsets, two flow fields (2 × 350 KB), the seam weights |
| (side stream) | output download into a pinned slot | GPU copy engine | D2H 88.5 MB (Y 59.0 + UV 29.5), 7–9 ms at PCIe 3's 10–12 GB/s, not on the stitch thread |
| frame builder | copy the pinned slot into an `AVFrame` | CPU | host copy 88.5 MB (18 ms, measured on the V100 host in August) |
| encoder session × 3 | `hevc_nvenc` | CPU (libav's upload), NVENC engines | H2D 88.5 MB inside libavcodec, pageable; about 35 ms per call (measured, August); the three persistent sessions take the ~2 s chunks in turn |
| muxer | reorder and write | CPU | D2H: the bitstream, well under 1 MB |

With `decode="p010le"` (the fallback when the device-frame probe fails) the
decode workers instead download each lens (2 × 44.2 MB D2H inside hevc_cuvid)
and upload it again from pinned memory (2 × 44.2 MB H2D): 177 MB more per
frame across PCIe, 15–18 ms of copy-engine time, still off the stitch thread.
The profile line's `decode` field says which arrangement a run used.

### Per-stage accounting of the stitch thread

Columns: *M3* is measured on the Mac with the profile in sync mode (the device
is synchronised around every stage, so the time is that stage's alone), clip
`0017` from frame 1150, 70 frames, production settings. *CPU* is the thread's
CPU time inside the stage. *V100* is the prediction for the production box.

Scaling used, and why:

- **CPU work × 2.0–2.5.** The Ivy Bridge Xeons are 2–2.5 times slower per core
  than the M3 on numpy and OpenCV (the figure given for this host). The
  dynamic programme is a Python loop over 960 columns, interpreter-bound, so
  if anything it sits at the top of that range.
- **GPU kernels × 0.46–0.54 of the M3's time.** This ratio is *calibrated, not
  measured*: it is the value that makes this table add up to the observed
  222 ms. An independent check: in August the V100 ran the two-pass stitch at
  36 ms per frame with the SM at 80 % (≤ 29 ms of kernels), where the M3 needs
  40.5 ms for the same kernels: a ratio of at most 0.71. So 0.5 is plausible
  but not established.

| Stage (`ee1e170`, seam on the CPU) | Runs on | M3 ms | of which CPU | V100 + Ivy Bridge, predicted ms |
|---|---|---:|---:|---:|
| `st.split` unpack 2 × p010 to float planes | GPU | 19.0 | 0.6 | 9–10 |
| `st.level_grid` per-frame rotation grid | GPU (compiled) | 8.6 | 0.3 | 4–5 |
| `st.seam_warp` both lenses onto the band | GPU | 6.8 | 0.9 | 3–4 |
| `st.seam_est` download; photometric offsets (numpy); DIS flow × 2 (OpenCV); upload | **CPU** + 2 waits | 26.9 | 22.7 | 47–59 |
| `st.seam_fields` smooth and upsample the flow | GPU | 26.3 | 0.8 | 12–14 |
| `st.seam_dp` download cost; seam path (numpy loop); upload | **CPU** + 1 wait | 22.5 | 15.8 | 35–43 |
| `st.band_index` `nonzero` over the band (a wait) and sparse map build | GPU + 1 wait | 40.2 | 0.2 | 18–22 |
| `st.sample` both lenses, luma and chroma | GPU | 95.9 | 0.4 | 44–52 |
| `st.blend` | GPU | 37.5 | 0.2 | 17–20 |
| `st.pack` to p010 | GPU (compiled) | 8.0 | 0.1 | 4 |
| `st.d2h` | GPU copy | 7.9 | 0.2 | 0 (asynchronous on CUDA) |
| Python and torch launching about 300 kernels | CPU | (in the rows above) | 4.1 | 8–10 |
| **Total** | | **301.9** | 42.6 | **201–243, centred on 222 by construction** |

What this model says and does not say:

- It reaches 222 ms only because the GPU ratio was fitted to it. With the
  ratio left at the 0.71 upper bound from August the same table gives
  265–285 ms, so the uncalibrated model *overshoots* by a fifth to a quarter. The split
  between "CPU seam stack" (roughly 80–100 ms) and "GPU kernels" (roughly
  115–135 ms) is therefore uncertain by tens of milliseconds either way.
- A check that needs no new code: if the model is right, the SM utilisation
  of a production run at `ee1e170` is about **53–61 %**, and the process uses
  about one core on the stitch thread (CUDA's waits spin) plus under one core
  across the builder, encoders and OpenCV's flow threads, which matches the
  observed "about two cores".
- Three things the Mac cannot show, any of which would move the split: decode
  running as `p010le` rather than `cuda`; the allocator churning on the CPU
  path's variable-size sparse tensors (the 9–11 GB swing in production points
  that way); and the GIL being held by the builder's and encoders' copies
  while the stitch thread wants to launch kernels. The profile line reports
  all three (`decode`, `osv-mem`, and `wall` against `cpu` per stage).

## What changed

`git log ee1e170..fable/gpu-offload` lists the commits.

**1. Per-stage timing, always on** (`dji_osv_converter/profile.py`). Every
conversion logs two lines:

    osv-profile: {"frames": 450, "fps": ..., "ms_per_frame": ..., "cpu_cores": ..., "stages": {...}, "config": {...}}
    osv-mem: {"peak_rss_mib": ..., "gpu_peak_alloc_mib": ..., "gpu_peak_reserved_mib": ...}

Each stage has `wall`, `cpu` (that thread's CPU time, so `wall − cpu` is time
blocked) and, for GPU stages on CUDA, `gpu` from CUDA events resolved lazily,
with no synchronisation added. It costs a few microseconds per stage.
`convert(profile="sync")` gives exact per-stage GPU time at the price of
serialising the pipeline; `profile=False` turns it off.

**2. The seam stack in torch** (`dji_osv_converter/seam_gpu.py`). The three
CPU pieces, ported to run on the stitcher's device:

- *Photometric offsets*: the same algorithm as the numpy version, to within
  0.02 of a 10-bit level.
- *Optical flow*: OpenCV's DIS has no torch equivalent, so this is a new
  estimator: both directions in one batch, coarse to fine, mean-normalised
  Lucas–Kanade with a Gaussian window, then the same forward–backward gate and
  anisotropic smoothing as before. See "The flow".
- *Seam path*: the exact minimum-cost path of the CPU version (the test
  compares it with a reference dynamic programme on six shapes), computed in
  parallel: single-column steps are composed pairwise into blocks, then the
  blocks are walked in order. That is 255 kernel launches a frame with blocks
  of 64 columns (the CUDA setting) and 526 with blocks of 16 (MPS, where it
  measured fastest), instead of a 960-iteration Python loop on the host.
- The `auto` decisions (is the photometric correction needed, does the flow
  engage) are 0/1 tensors multiplied in, so the host never reads a value back.

**3. The application folded into the single-pass maps**
(`dji_osv_converter/stitch_gpu.py`, `seam_engine="gpu"`). The CPU path built a
sparse set of band pixels each frame (`nonzero`, a wait, and tensors whose
size changes every frame). The new path rewrites the band rows of each lens's
persistent sampling map in place and keeps one persistent two-channel map for
the blend weight and the photometric correction, so every tensor has a fixed
shape and the frame has no host round trip. `seam_stats` is resolved lazily,
only when someone reads it.

**4. No stream synchronisation in the frame.** A `torch.tensor(...,
device="cuda")` built per frame is a pageable host→device copy, and that
synchronises the stream. With one at the start of a frame, the next frame's
seam phase (hundreds of tiny kernels, limited by how fast the host can launch
them) cannot overlap the previous frame's heavy kernels. The new path's scale
factors are built once and the level grid's rotation is filled by scalar.

**5. Options, with Josh's defaults unchanged.**

| Option | Default | Meaning |
|---|---|---|
| `seam_engine` | `None`: `"gpu"` on CUDA and MPS with `single_pass`, else `"cpu"` | where the seam stack runs |
| `seam_morph` | `"legacy"` | `"corrected"` fixes the morph direction (changes the look) |
| `profile` | `True` | the two log lines |

Levelling, resolution, bit depth, `single_pass`, `lens_interp`, `seam_flow`,
`allow_software`, the compile cache, spherical metadata and audio passthrough
are untouched.

### What moved to the GPU, what stayed, and why

| Work | Before | Now |
|---|---|---|
| Photometric offsets | CPU (numpy), after a download | GPU |
| Optical flow, both directions | CPU (OpenCV DIS), after a download | GPU (Lucas–Kanade in torch) |
| Seam path | CPU (numpy loop), after a download | GPU (parallel min-plus) |
| Seam gate, feather, morph fields | GPU, with sparse indexing and a wait | GPU, fixed shapes, no wait |
| Decode | NVDEC into device memory (already) | unchanged |
| p010 unpack, colour handling, level grid, sampling, blend, p010 pack | GPU (already) | unchanged |
| Output download | asynchronous, pinned, side stream (already) | unchanged |
| Frame build (88.5 MB host copy) | CPU, own thread | **stays**: libav's NVENC wants a system-memory frame; see below |
| NVENC upload inside libavcodec | CPU + PCIe, three session threads | **stays**: see below |
| Demux, mux, audio passthrough | CPU | stays: byte shuffling, under 2 ms a frame |
| Levelling lookup | CPU, 0.2 ms | stays: nine floats per frame |

### Evaluated and not adopted

- **Encoding from device memory** (CUDA frames into `hevc_nvenc`, which would
  remove the download, the builder's copy and libav's upload: 265 MB of
  copies per frame). Stock PyAV can already do it, and it was measured on this
  V100 in August: **2.7 times slower per call** (55 ms against 20 ms), because
  Volta's NVENC pays more to map a CUDA resource each frame than it saves on
  the upload (`tools/hwframe_encode_probe.py`, `docs/architecture.md`). The
  cost is in the encoder's resource mapping, not in PyAV, so PyNvVideoCodec's
  encoder would pay it too (that last step is an inference; it was not run).
  All three copies are off the stitch thread already. Worth re-probing on the
  T10 (Turing), where mapping is cheap.
- **PyNvVideoCodec for decode.** What it offers, frames as device tensors, the
  code already has through PyAV's hardware frames at the engine's full speed
  (25 ms per pair, the NVDEC wall). It would add a native dependency whose
  wheels have not been checked on a CPU without AVX2. Not run: no CUDA here.
- **torchaudio's CUDA decoder/encoder.** As far as I know its media I/O was
  removed in torchaudio 2.9 in favour of TorchCodec, whose decoder returns
  8-bit RGB and would throw away the 10-bit source. This is from memory of
  those projects' documentation: it was not looked up or run in this work, so
  treat it as a reason not to spend time there, not as a finding.
- **More CUDA streams and queues.** The pipeline already has a decode stream
  per lens, a download stream and bounded queues between every stage. The
  stitch itself is one dependency chain per frame; a second stitch stream
  would only interleave two frames on the same SMs. On the Mac, with the seam
  stack on the device, every other stage idles waiting for the stitch (the
  encoder thread's `enc.idle` was 169 ms of a 183 ms frame in the last run),
  so queues are not what limits it.
- **`torch.compile` for the seam stack.** Not done: the kernels are tiny and
  launch-bound, compile time would add to the 60 s set-up, and there was no
  sm_70 card to measure on. The existing compiled kernels (level grid, pack)
  are unchanged. If the V100 report shows `st.seam_est` and `st.seam_dp` with
  `cpu` close to `wall` and large, this is the next thing to try.

## Before and after

### M3, production settings, measured

End to end (`convert()`, clip `0017`, 100 frames from frame 1150, pipelined):

| | fps | ms/frame | stitch thread CPU ms/frame | peak RSS MiB | MPS allocated / driver pool MiB |
|---|---:|---:|---:|---:|---:|
| `ee1e170` path (`seam_engine="cpu"`) | 3.34 | 299.4 | 43.2 | 5,070 | 1,561 / 6,153 |
| new default (`seam_engine="gpu"`) | **5.65** | **177.0** | **9.9** | 3,960 | 1,815 / 4,132 |
| new default, `seam_flow=False` | 9.35 | 106.9 | 1.7 | 5,693 | 1,794 / 4,105 |

Per stage, sync mode (70 frames; milliseconds per frame, the stage's own time):

| Stage | `ee1e170` path | New default | Note |
|---|---:|---:|---|
| `st.split` | 19.0 | 17.7 | |
| `st.level_grid` | 8.6 | 8.3 | |
| `st.seam_warp` | 6.8 | 2.2 | |
| `st.seam_est` | 26.9 (22.7 CPU) | 18.1 (3.9 CPU) | on MPS the new one is limited by per-kernel overhead, not arithmetic |
| `st.seam_fields` | 26.3 | 12.5 | |
| `st.seam_dp` | 22.5 (15.8 CPU) | 25.0 (1.9 CPU) | the same: 526 tiny kernels at MPS's ~45 µs each |
| `st.band_index` / `st.band_write` | 40.2 | 9.0 | |
| `st.sample` | 95.9 | 53.0 | the sparse seam application was inside this stage |
| `st.blend` | 37.5 | 25.0 | |
| `st.pack` | 8.0 | 8.4 | |
| `st.d2h` | 7.9 | 11.0 | |
| **Stitch thread, wall / CPU** | **299.6 / 42.6** | **190.3 / 8.4** | |
| **Frame** | 301.9 (3.31 fps) | 192.4 (5.20 fps) | |

### Predicted V100 speed

All estimates. Using the calibrated GPU ratio (0.46–0.54 of the M3's time) and
the CPU factor (2.0–2.5):

| Part of the new frame | M3 measured ms | V100 predicted ms | Basis |
|---|---:|---:|---|
| Sampler: split, level grid, sample, blend, pack | 112.4 | 52–61 | GPU ratio |
| Seam, the heavy kernels: warp, fields, band write | 23.7 | 11–13 | GPU ratio |
| Seam, the tiny kernels: estimator and seam path, under 730 launches | 43.1 | 7–10 on the card | ~10 µs of card time per tiny kernel (an assumption, not a measurement); MPS's ~45 µs each does not carry over |
| Host: launching about 800 kernels | 8.4 CPU | 13–20 CPU, overlapping the card's work | 8 µs per launch on the M3 × 2.0–2.5 |
| **Stitch stream per frame** | 190.3 | **70–84, or 83–104 if the launches cannot overlap** | |

So **10–14 fps, most likely about 12**, against 4.5 (the table's extremes are
9.6 and 14.3). With `seam_flow` off:
52–61 ms on the card plus launches, **15–19 fps**. If the V100 is faster
relative to the M3 than the calibration assumes (possible: these kernels are
memory-bound and the V100 has several times the M3's memory bandwidth), the
numbers improve towards the decode ceiling of about 28 fps, and no further.

The prediction is wrong in a known direction if any of these hold, and the
harness shows each: the compiled level grid synchronises the stream when
called with Python floats (then `st.level_grid` shows `wall` far above `cpu`);
launches cost more than 25 µs each on this host (then `st.seam_est` and
`st.seam_dp` show tens of milliseconds of `cpu`); or decode is not running as
`cuda`.

### Memory

Not measurable here: MPS keeps no peak counters and the Mac has none of the
NVDEC and NVENC allocations. What can be said:

- The new path adds 254 MiB of persistent tensors (1,561 → 1,815 MiB allocated
  on MPS: the two-channel weight-and-correction map and the band statics).
- It removes the per-frame variable-size sparse tensors. The MPS driver pool,
  the allocator's high-water mark, fell by 2,021 MiB (6,153 → 4,132).
- Production's peak at `ee1e170` was 11,263 MiB on the card. If none of the
  transient saving carries over to CUDA the new peak is about 11,520 MiB; if
  all of it does, about 9,250 MiB. **Both are inside the 12,000 MiB lease**,
  the first with less than 500 MiB to spare. This is an estimate until the
  harness's "card peak" column is read.

## The flow

OpenCV's DIS flow runs only on the CPU. The replacement is a pyramidal
Lucas–Kanade solver written for this job: band-sized greys (91 × 1920 at 8K),
`SEAM_LK_LEVELS = 2` levels, `SEAM_LK_ITERS = 3` iterations a level, a
Gaussian window (`SEAM_LK_WINDOW_SIGMA = 4.0`), mean-normalised patches so a
brightness difference between the lenses is not read as motion, and a damping
term (`SEAM_LK_DAMP = 4.0`) that leaves low-texture areas alone. The constants
were chosen by a sweep on the five test windows (recorded in `constants.py`).

Measured on the five windows, 10-bit levels of inner-band mismatch after the
morph (lower is better), and the mean flow magnitude in pixels:

| Window | DIS, corrected morph | LK, corrected morph | DIS, legacy morph | LK, legacy morph | DIS mean flow | LK mean flow |
|---|---:|---:|---:|---:|---:|---:|
| base_f10 | 45.62 | **44.10** | 57.28 | 57.80 | 1.83 | 3.57 |
| worst_f230 | 42.74 | **42.07** | 56.66 | 56.58 | 3.80 | 4.77 |
| light_0011_f10 | 38.49 | **33.70** | 45.70 | 48.61 | 3.10 | 6.01 |
| dark_0021_f1349 | 58.63 | **57.30** | 86.98 | 87.99 | 3.80 | 3.83 |
| normal_0017_f1229 | 26.96 | **25.93** | 34.47 | 34.95 | 2.42 | 3.40 |

With the morph applied in the right direction, LK aligns the lenses slightly
better than DIS on every window. Under the shipped (legacy) direction a larger
flow moves the lenses further the wrong way, so LK's figure is up to 3 levels
worse on one window (light) and within about 1 elsewhere. LK's flows are
larger than DIS's on every window (last two columns); that is what the
numbers above reward under the corrected morph and punish under the legacy
one. The unit test gives it a known shift of (3, 1) px in both directions and
requires a mean error under 0.35 px.

## The morph direction

The seam repair is meant to warp each lens half-way towards the other. At
`ee1e170` it warps each lens half-way **away**: the two flow fields are used
the wrong way round. Evidence:

- Real footage, mean inner-band mismatch in 10-bit levels:

  | Window | Raw | Photometric only | Shipped morph | Flows swapped |
  |---|---:|---:|---:|---:|
  | base_f10 | 75.50 | 52.45 | 57.29 | **45.62** |
  | worst_f230 | 56.12 | 52.40 | 56.68 | **42.71** |
  | light_0011_f10 | 47.43 | 42.82 | 45.68 | **38.53** |
  | dark_0021_f1349 | 78.90 | 75.65 | 86.96 | **58.63** |
  | normal_0017_f1229 | 35.58 | 32.22 | 34.46 | **26.95** |

  The shipped morph is worse than doing nothing after the photometric step on
  every window; swapped, it is better on every window.
- Synthetic rig, one world seen by both lenses with lens 1 displaced 6 px
  (`tests/test_seam_engine.py`, `test_morph_direction`): mismatch 69.07
  before; 99.39 after the shipped morph; 32.87 after the corrected one.
- Against DJI's own export (SSIM, the two windows that have one): shipped
  0.8346 and 0.8027; seam stack off 0.8378 and 0.8081; corrected 0.8389 and
  0.8057 (CPU engine).

The seam *path* still hides most of this, which is why it went unseen. The
fix changes the look of every clip, so it is **off by default**:
`seam_morph="corrected"` on either engine. Recommendation: look at the
`cpu_corrected` and `gpu_corrected` rows in the viewer and, if Josh agrees,
make it the default.

## Quality

Tool: `tools/ab_compare.py --set perf`, at full 8K on the five standing
windows (light, dark, normal, and the two with a DJI Studio export), plain
levelling so the seams sit at fixed places. The baseline is production's path
at `ee1e170`. Viewer: **`ab_out/perf_ab/index.html`** (gitignored; open it in
a browser). Pick `baseline` as A and `gpu_engine` as B.

New default against the `ee1e170` path. "Belt" is the 30 % of the frame within
the seam band; levels are 10-bit:

| Window | Outside the belt, max ΔY | Belt PSNR Y dB | Belt mean ΔY | Belt pixels > 4 levels | SSIM, seam crops L / R | Sharpness, seam crops L / R (old → new) | Mismatch after the stack (old → new) |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_f10 | **0** | 31.60 | 4.95 | 12.3 % | 0.9986 / 0.9985 | 69 / 37 → 69 / 37 | 57.29 → 57.80 |
| worst_f230 | **0** | 40.22 | 1.96 | 8.5 % | 0.9971 / 0.9907 | 277 / 436 → 277 / 435 | 56.68 → 56.58 |
| light_0011_f10 | **0** | 34.33 | 2.60 | 9.1 % | 0.9904 / 0.9997 | 464 / 2 → 463 / 2 | 45.68 → 48.61 |
| dark_0021_f1349 | **0** | 25.25 | 12.90 | 20.1 % | 0.9894 / 0.9678 | 146 / 427 → 147 / 423 | 86.96 → 87.99 |
| normal_0017_f1229 | **0** | 24.93 | 11.28 | 10.4 % | 0.9885 / 0.9941 | 77 / 118 → 77 / 118 | 34.46 → 34.95 |

Whole-frame PSNR against the old output: Y 36.8 / 45.4 / 39.5 / 30.4 / 30.1 dB,
UV 51.5 / 59.4 / 57.0 / 50.6 / 44.6 dB, in the same order.

Against DJI's export (old → new): base_f10 SSIM 0.8346 → 0.8335, PSNR 23.68 →
23.53 dB, NCC 0.9582 → 0.9568; worst_f230 SSIM 0.8027 → 0.8027, PSNR 25.44 →
25.45 dB, NCC 0.9728 → 0.9729.

How to read that:

- **Outside the belt nothing changed**, to the bit, on every window. With
  `seam_flow` off the two engines are bit-identical everywhere (unit test).
- Inside the belt the engines disagree, and the low PSNR on the dark and
  normal windows needs explaining. On the normal window the median difference
  is 0 levels, the 90th percentile 6, the 99th 362: the mean is carried by
  under 1 % of pixels. They are at the nadir, where the camera's own mount
  sits centimetres from both lenses and no seam can be right; the two engines
  put the join through it at different places, and the projection stretches
  that across the bottom rows. At the horizon the two outputs are
  indistinguishable by eye (checked on crops of the normal window).
- For scale: switching the seam stack off moves the belt by 15.6 / 10.9 / 9.1
  / 17.0 / 11.8 levels on the same windows. The engines differ from each
  other by less than that on every window.
- The application path itself is exact. With the CPU's DIS flow injected into
  the new path the output matches the old to 85.7 dB (Y) and 100.8 dB (UV),
  at most 3 levels, on base_f10; 83.9 and 99.2 dB on the normal window. The
  differences in the table are the flow algorithm, nothing else.
- The honest negatives: against DJI on base_f10 the new default is 0.001 SSIM
  and 0.15 dB lower, and the post-stack mismatch is up to 3 levels higher
  (light). Both follow from the legacy morph direction amplifying LK's larger
  flow; with `seam_morph="corrected"` the new engine has the lowest mismatch
  on all five windows.

This is five frames. It has not been judged in motion: temporal stability of
the seam (flicker of the path from frame to frame) was not measured for either
engine, and the outputs in `ab_out/perf/` (100 frames each engine, clip
`0017`) should be watched before this ships.

## Tests

`python -m unittest discover -s tests`: 135 tests, 4 skipped (one CUDA only,
three that need `nvidia-ml-py`, which the Mac's environment doesn't have), on
torch-CPU and MPS. New: `tests/test_profile.py` (12), `tests/test_seam_gpu.py`
(13: blur, photometric offsets against numpy, seam-path optimality at block
sizes 2, 16 and 64, gate and path against the CPU code, flow recovery), `tests/test_seam_engine.py`
(13: options, seam-off bit-identity, band restore, application against the CPU
path, morph direction, rotation invariance), `tests/test_v100_bench.py` (44:
the harness's own functions against the real package, real PyAV frames and a
real hardware encoder session, real child processes, and `nvidia-ml-py`'s
names and types when it is importable).

## Reproducing

All from the repository root, with the project's virtual environment.

    # tests
    python -m unittest discover -s tests

    # a conversion at production settings; the two log lines appear at the end
    python - <<'EOF'
    import logging; logging.basicConfig(level=logging.INFO)
    from dji_osv_converter.video import convert
    convert("CLIP.OSV", "out.mp4", levelling="rocksteady_horizon", single_pass=True,
            seam_flow="auto", count=100,
            seam_engine="cpu")        # the ee1e170 path; drop this line for the new default
    EOF
    # add profile="sync" for the per-stage GPU times of the tables above

    # quality A/B and the viewer (five windows × five variants at 8K)
    python tools/ab_compare.py --set perf --out ab_out/perf_ab
    open ab_out/perf_ab/index.html

    # on the V100 (inside the converter's image, holding the GPU lease)
    python tools/v100_bench.py /path/to/CLIP.OSV --report /path/to/v100_report.md
    # every variant, diagnostics included (about twice as long):
    python tools/v100_bench.py /path/to/CLIP.OSV --report /path/to/v100_report.md --all
    # which source a copy of the harness is (the report prints the same digest)
    python tools/v100_bench.py --code-id

`v100_bench.py` runs 450 frames per variant, each variant in its own process,
and writes its converted files to a new temporary folder (or `--out-dir`),
removing each one when its variant ends unless `--keep` is given.

Before the first variant it makes every preflight check and, if any fail,
stops with all of them named in one message: the clip, NVML and the card index,
the report and output folders and their free space, the package's names and
`convert()`'s arguments for the chosen variants, NVDEC on the clip, CUDA, a
kernel on the card, an 8K 10-bit frame built the way the converter builds
them, and that frame through an `hevc_nvenc` session set up as the converter
sets its own up.

A variant fails if its profile shows any other device, decoder or encoder, or
a setting other than the one the variant asked for; if a harness switch
(`new_p010`'s decode format, `new_dp16`'s block size) was never reached by the
conversion; if NVML gave no sample, or NVDEC or NVENC utilisation never left
zero; or if it is still running after `--timeout` seconds (default 1800), when
it is killed so a hang cannot hold the card.

## Not verified without the V100

1. **None of the new code has run on CUDA.** It is device-agnostic torch and
   passes on torch-CPU and MPS, but sm_70 kernels, cu126 and the compiled
   kernels' interaction with the new path are untested.
2. Every V100 timing and the fps prediction.
3. Peak card memory against the 12,000 MiB lease.
4. `SEAM_DP_BLOCK_CUDA = 64` is an assumption; the `new_dp16` variant tests it.
5. Whether anything in the frame still synchronises the stream on CUDA.
6. The parts of the harness that need the card. Off it, all ten variants
   were converted for real through the harness's own functions (40 frames
   each, VideoToolbox in place of NVDEC and NVENC), the whole run was made
   with only the three card checks let through, and the preflight's frame and
   encoder session were run with a real hardware encoder. What has not run:
   the NVDEC probes succeeding, `torch.cuda`'s description of the card and the
   kernel check on it, the `hevc_nvenc` session at 8K, NVML's answers inside
   the container (whether the encoder and decoder counters are readable
   there), `new_p010` through `hevc_cuvid`, and the block size `new_dp16`
   replaces (64 only on CUDA).
7. Seam stability in motion, for either engine.

## Next steps, in order of expected gain

1. Run the harness; read `new_sync` against `baseline_sync`. Everything below
   depends on which stage the card actually spends its time in.
2. **The sampler** (predicted three quarters of the new frame). Each lens is
   sampled over the whole 8K frame although it covers half of it plus the
   belt. Sampling each lens only where its weight is non-zero would remove
   about 40 % of `st.sample` and most of `st.blend` outside the belt. It needs
   tiles or a compacted pixel list that follows the per-frame rotation.
3. Fuse blend and pack into one compiled kernel (three full-frame passes
   become one).
4. If the host's launches turn out to be the limiter: compile the estimator
   and the seam path, or cut their kernel count.
5. Past about 28 fps nothing helps but a second decode engine: the T10, or
   decoding the two lenses on two cards.
