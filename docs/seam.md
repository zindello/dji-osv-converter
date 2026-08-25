# Seam repair

Why the seam can never be perfect with a static blend: the two lenses sit a
few centimetres apart. For distant content their projections agree; for
anything near the camera they genuinely disagree — geometrically (parallax:
median 0.2 px but p90 ~23 px on real footage; a bar ~1 m away steps visibly
at the seam) and photometrically (the in-camera ISP matches the lenses under
even lighting but not asymmetric lighting). In handheld footage the operator
is *always* near the camera, so this is the common case, not the corner
case.

`seam_flow="auto"` (the `convert()` default) runs three corrections, each
gated by a per-frame detector so it engages only where its problem exists.
Everything operates inside the blend band (±margin) — the rest of the sphere
is untouched by construction.

## 1. Photometric match (`stitcher.seam_luma_offsets`)

Measured on the shed clip (bright lights into one lens): a luma-dependent
transfer difference, +20/1024 in shadows fading to −3 in highlights, varying
along the band, reaching **70/1024 in dark content** (rim stray light —
present on both sample units' footage). Model hunt, each step measured and
rejected: scalar gain under-corrects (+9.6 → +6.4); additive-in-linear-light
glare under-corrects (+6..10). The shipped estimator is **model-free**: a
16-bin per-luma offset LUT plus a smooth 2-D residual field, estimated only
from *geometrically agreeing* overlap pixels — in two passes, because a
single tight gate has a selection bias (where the mismatch is worst, the
pixels exceed the gate and the estimator never sees them; 77% of dark pixels
were invisible to pass 1). Split-applied to both lenses, ramped to zero at
the band margins. Results: lit shadows +19.1 → +0.4; dark content +70 → +8.
No-op on matched footage (worst-case clip: −1..−2.5).

Beware the measurement trap that cost two model iterations: verify with a
mask fixed *before* correction — a post-corrected agreement mask admits
former-disagreement pixels and reads a phantom +8 residual.

Known boundary (pinned by `tests/test_seam.py`): mismatch beyond the tight
gate is recovered only where it is *spatially coherent* (real dark regions
are); a random dark/bright interleave with per-pixel luma-dependent
mismatch is not representable — the field is spatial and the LUT only
learns what the tight gate can see.

## 2. Flow morph (`stitcher.seam_flow_small` + on-device resample)

DIS optical flow (fast preset — 5.6 ms/pair vs Farneback's 24.8 with
equal-or-better residual) between the two lenses' band crops at 1/4 res,
forward-backward-consistency gated, anisotropically smoothed (heavy along
the seam, light across — but not *too* heavy: σ=18 diluted a crossing
object's parallax vector with background flow and left a kink; σ=6 aligns
the crossing). Each lens is morphed toward the other, ramped by the blend
weights, so features coincide where they mix. The codebase's earlier
FULL-FRAME flow morph produced phantom bends worse than no correction —
band-limiting plus the gates is what makes flow safe.

## 3. Dynamic seam (`stitcher.seam_path` + `seam_gate`)

Flow cannot fix occlusion: near an object the lenses see different *sides*
of it, and the wide 292 px cross-fade dissolves it into the background
(transparency). A min-disagreement path is routed per column through the
post-morph band (DP, |dy| ≤ 1/step at 960 columns, smoothed) and the blend
becomes a 24 px feather around that path — the object comes wholly from one
lens wherever possible. **Gated to columns with real parallax** (column-mean
disagreement, LO=10 → HI=20 ramp): a narrow cut through aligned-but-
photometrically-imperfect flat areas draws a visible ring around the whole
sphere that jumps per frame (user-caught). The gate statistic must be a
*mean* — a max saturates on single noisy pixels (measured: 92% of columns
falsely flagged).

## Auto mode

One pooled download per frame feeds the photometric estimator and the
parallax detector (~16 ms total when nothing engages). Engagement cannot
pop: the photometric correction is continuous in its own magnitude, and the
dynamic seam ramps from the same threshold the detector uses. Measured
honesty: both sample clips engage nearly always — there is no threshold that
separates "needs it" from "doesn't" on handheld footage.

## Cost (M3)

Static blend 59 ms → full stack ~127 ms stitch (~212 ms/frame end-to-end
sustained, fanless-M3 throttling included). `seam_flow=False` restores the
static path. On CUDA the stack rides `stitch_yuv_async`; moving its CPU
work (flow + DP + estimator, ~65 ms) into the pipeline's builder stage is
the open optimization.

## Residue that remains (by design)

- A face-shading shift where the seam must cross a near object — the lenses
  photograph different surfaces; only depth-aware reconstruction removes it
  (DJI's stitcher shows it too).
- Real in-camera lens flare (haze around lights) — present in the source
  fisheyes; not a stitch artifact.

Tunables: `SEAM_FLOW_*`, `SEAM_DYN_*`, `SEAM_OFFSET_*`, `SEAM_AUTO_*` in
[constants](constants.md), all provenance-tagged.
