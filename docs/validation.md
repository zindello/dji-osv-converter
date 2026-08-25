# Validation methodology

Every quality claim in this project is measured against **DJI Studio's own
export of the same OSV file** — the only available ground truth — and the
final gate is always a human watching the render.

## Ground rules (each learned the hard way)

1. **Measure the actual render, never the analytic ideal.** The smoothed
   trajectory's jitter and the rendered output's jitter differed by 20× until
   a composition bug was found.
2. **Validate a metric against known ground truth before trusting it.**
   Sky-boundary tilt, sky-position, and ORB axis-angle tilt metrics all gave
   confidently wrong answers on real footage. The metrics that survived:
   - **up-vector Procrustes** against ORB-extracted DJI orientations
     (same-frame, same-scene ORB is reliable even at 120° of yaw);
   - **median phase-correlation over spread crops** for jitter (sub-pixel,
     robust to a contaminated region);
   - **dense per-patch motion fields** to *see* what kind of error remains.
3. **Compare six wide-spaced frames minimum, never a favourable pair** — and
   render A/B pages (see `tools/`) so the comparison is repeatable.
4. **Bit-exact regression baselines around refactors.** The cleanup pass
   captured stitched planes for every levelling mode before touching code and
   required them unchanged after (it caught a mis-transcribed constant
   immediately).

## Tools

- `tools/validate_against_dji.py` — geometry/structure scoring vs a DJI
  export (yaw- and tone-compensated).
- `tools/ab_seam.py` — renders seam A/B images with an amplified diff.
- `tools/compare.html`, `tools/view360.html` — interactive slider / 360°
  in-browser comparison pages.
- `tests/test_pipeline.py` — the mathematical properties each mode was
  validated against, runnable without sample footage.

## Reference results (worst-case tumble clip, vs DJI Studio)

| claim | ours | DJI |
|---|---|---|
| plain layout constancy through a tumble | <2° spread | — |
| horizon-lock up-vector match | 0.24° mean / 0.7° max | (definition) |
| rocksteady rendered jitter (calm / moving) | **0.027 / 0.071** | 0.046 / 0.118 |
| horizon-locked yaw | pinned (slow gyro drift only) | wanders under violent motion |

## CUDA port (V100, 2026-08)

The CUDA path was accepted on the same methodology, not by analogy:

- **Bit-parity with the reference**: CUDA vs torch-CPU stitch is
  bit-identical under identity levelling; with a live rotation 97% of
  pixels are bit-equal and 3 px of 29.5M exceed 1 LSB (compiled sin/cos —
  the same recompute class as the documented MPS delta). NVDEC decode is
  bit-identical to software decode; the p010-native ingest and the async
  D2H path are each bit-identical to the paths they replace.
- **Against DJI** (worst-case clip, `rocksteady_horizon`, 7 wide-spaced
  frames, yaw-aligned + tone-matched): mean NCC 0.954 / SSIM 0.72 /
  PSNR 23.9 dB, worst frame NCC 0.927. Rendered jitter (median
  phase-correlation, 5 equator crops): **0.045 calm / 0.181 moving**
  °/frame vs DJI's 0.409 / 0.485 on the same frames (DJI's export lets
  yaw wander in the tumble; the gyro-quartile split is calm = lowest,
  moving = highest quartile of raw angular speed).
- The M3 render found on the V100 box, scored through the identical
  script, gives the same frame-level numbers on 6 of 7 frames (e.g. NCC
  0.944/0.975/0.968 on frames 10/230/340 vs CUDA's 0.951/0.981/0.975) —
  the port changes nothing DJI-visible. That file fails alignment on
  frame 120 and shows 1.3/6.7 °/frame rendered jitter, i.e. it predates
  the heading fix; the CUDA render at HEAD is strictly better than it.
