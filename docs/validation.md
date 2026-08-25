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
