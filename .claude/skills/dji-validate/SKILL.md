---
name: dji-validate
description: Validate stitcher output against DJI Studio's own exports — the project's acceptance methodology for any quality claim (levelling, seam, colour, stabilisation).
---

# Validating against DJI

Any claim of "matches DJI" or "better than DJI" must survive this process.
Shortcuts here have produced confident wrong answers repeatedly.

## Rules

1. **Compare the actual render**, never an analytic ideal. Render frames with
   the real pipeline (`GpuStitcher.stitch_yuv` → `p010_to_bgr`, or
   `convert()` with a count), then measure those pixels.
2. **Frame alignment**: OSV frame N ↔ DJI `exported_360_plain.mp4` frame N,
   1:1 (frame 10 is the standing baseline). DJI's stabilised export aligns
   the same way.
3. **At least 6 wide-spaced frames** covering calm AND violent motion. Never
   accept a metric from one frame or one segment.
4. **The user watching the rendered video is the final gate.** Produce the
   MP4 and stop; do not declare success first.

## Instruments that work (see docs/validation.md for the failures)

- **Orientation delta**: ORB features between our frame and DJI's same frame
  → recover relative rotation (reliable even at 120° yaw offset); or
  up-vector Procrustes against DJI-extracted orientations.
- **Jitter**: median phase-correlation shift over 5 spread crops between
  consecutive rendered frames, reported °/frame; compute the same number for
  DJI's export and compare (reference: ours 0.027/0.071, DJI 0.046/0.118
  calm/moving on the worst-case clip).
- **Seam/exposure**: side-by-side web A/B page (tools/compare.html pattern),
  yaw-aligned via ORB (`R' = R @ Rrᵀ`) when framings differ.
- **Regression around refactors**: save packed (Y, UV) planes for spread
  frames BEFORE the change; assert bit-equality (or quantify the LSB delta)
  after. This has caught transcription bugs a visual check missed.

## Known metric traps (all failed here — do not reuse)

Sky-tilt sinusoid fits, sky-position tracking, ORB angle·sin(axis-tilt),
gamma-sweep scoring (ranked DJI worse than us). If a metric disagrees with
frame-by-frame eyeballs, the metric is wrong.
