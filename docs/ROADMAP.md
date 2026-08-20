# Roadmap — OSV → 360° Equirectangular MP4

**Status:** 2026-08-20. Approach chosen: **full independent pipeline (Option B)**, using
DJI's shipped calibration *data* (not its code).

---

## The key discovery

DJI Studio ships **precomputed fisheye→equirectangular remap lookup tables** in its app bundle:

```
/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework/
    Versions/A/Resources/Stitcher-Mac.bundle/case_00/
        remap_0.bin, remap_1.bin      <- 56 MB each, the projection
        mask_00.png, mask_01.png      <- blend masks (3840×3840)
        param_list.json               <- camera serial -> calibration set
        intrinsic_extrinsic.json      <- lens model params
        intrinsic_extrinsic_list/     <- 32 per-unit calibration sets
        seam_overlap_ratio.json       <- 0.038
```

**Decoded format of `remap_N.bin`** (verified empirically, not assumed):

| Property | Value |
|---|---|
| Layout | `(1920, 3840, 2)` — rows, cols, (x,y) |
| Type | `int32`, fixed-point, **scale 1/128** |
| Meaning | For each output equirect pixel, the source pixel in that camera's fisheye |
| Invalid | Large negative/out-of-range sentinel values |
| Output grid | 3840×1920 equirectangular (half of final 7680×3840) |

Evidence it's correct: column 0 ≈ column 3839 (longitude wraparound), adjacent-sample
delta 0.0002 (smooth), and — decisively — applying it to a real frame produces a
recognisable panorama (`output/proof/01_MY_OUTPUT_remap_only.jpg`).

**Why this matters:** projection is the hard part of 360 stitching, and it's solved with
DJI's own geometry — so output geometry matches DJI's, not an approximation of it.

`0.038 × 7680 = 291.8 ≈ 292` — exactly the seam strip height seen in DJI Studio via lldb.
Independent confirmation the numbers are consistent.

---

## Target output spec (from DJI's own export, `osv_example/exported_360.mp4`)

| Property | Value |
|---|---|
| Resolution | 7680 × 3840 |
| Codec | HEVC (`hvc1`), Main 10 |
| Pixel format | `yuv420p10le` |
| Frame rate | 30 fps CFR |
| Colour | bt709 / bt709 / bt709, tv range |
| Spherical metadata | `projection=equirectangular`, yaw/pitch/roll = 0 |
| Audio | AAC |

We have DJI's real export of our exact test file — **ground truth for every stage.**

---

## Known-unsolved (honest list)

1. **Orientation.** Our output is rotated ~90° vs DJI's, and DJI applies horizon levelling
   (likely IMU/gravity from the OSV metadata stream). Not yet characterised.
2. **Resolution.** Remap grid is 3840×1920; target is 7680×3840. Map is smooth so it should
   upsample cleanly, but this needs verifying against DJI output, not assumed.
3. **Which calibration set.** `param_list.json` lists 32 per-serial sets. Need to read the
   camera serial from OSV metadata and pick correctly; `case_00`/default may be wrong for
   this unit and cause visible seam misalignment.
4. **Seam blending quality.** Feathered blend is straightforward; matching DJI's
   exposure-compensated multi-band blend is harder.
5. **Performance.** Nothing optimised yet. Real-time is a *later* goal, not a Phase 1 goal.

---

## Phases — each ends in something you can look at and judge

Rule: **no phase is "done" on my say-so.** Each ends with an artifact you inspect.
We do not start phase N+1 until you've signed off on N.

---

### Phase 1 — One correct still frame ⭐ highest risk, do first

Produce a single frame that is a genuine, properly-oriented 360 panorama.

Work: apply remap tables → apply masks (fixes the black bands) → resolve orientation
against DJI reference → feather-blend the seam.

**You check:** `output/phase1/frame_0000.jpg` side by side with
`output/proof/02_DJI_REFERENCE.jpg`.

**Pass:** loads in a 360 viewer, pans smoothly, horizon level, no black bands, seam not
obviously broken. Geometry should closely match DJI's.

**Fail/stop condition:** if orientation cannot be resolved to match DJI, stop and reassess
before building anything on top.

---

### Phase 2 — Full resolution + calibration selection

Upsample to 7680×3840; read camera serial from OSV metadata and select the right
calibration set.

**You check:** `output/phase2/frame_0000.png` (full 7680×3840) vs the DJI reference frame,
plus a numeric difference report and a zoomed crop of the seam region.

**Pass:** full-res output, sharp, seam alignment visibly no worse than DJI's.

---

### Phase 3 — Seam blend quality

Multi-band/feathered blending with exposure compensation across the 292 px overlap.

**You check:** `output/phase3/seam_comparison.jpg` — our seam vs DJI's, zoomed, on frames
chosen to be hard (high contrast across the seam).

**Pass:** no visible seam line, no exposure step between hemispheres.

---

### Phase 4 — Short video clip

10 seconds (~300 frames), correct container/codec/metadata.

**You check:** `output/phase4/clip_10sec.mp4` — plays in a 360 player, pans correctly,
no temporal flicker or seam wobble between frames. Verify against
`osv_example/e2e_10sec.mp4`.

**Pass:** stable, correct 10-second 360 video. `ffprobe` shows the target spec above.

---

### Phase 5 — Full file + audio

All 4724 frames, AAC audio muxed, spherical metadata written.

**You check:** full-length MP4, opened in your normal 360 workflow (DaVinci Resolve).

**Pass:** end-to-end equivalent to DJI Studio's export.

---

### Phase 6 — Performance (only after correctness)

GPU-accelerate the remap (the map is static — ideal for GPU), pipeline decode/stitch/encode.

**You check:** measured throughput on your M3, correctness re-verified unchanged.

---

## Ground rules (learned the hard way)

1. **Every claim is backed by an artifact you can open.** No "it works" without an image.
2. **Never invent an API or a parameter.** Verify against the binary or live capture.
3. **Tests must assert on real content**, not just array shapes. The old suite reported
   PASS on two-fisheye-circles garbage.
4. **One phase at a time.** Doing everything at once is what produced the archive folder.
5. **Ground truth always available** — `exported_360.mp4` is DJI's export of our exact
   test file. Compare against it at every stage.
