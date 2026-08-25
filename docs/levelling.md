# Levelling & stabilisation

All modes are a single per-frame rotation `R` applied to the stitched
equirect. **Render convention** (the root of several past bugs — read this
first): the GPU samples output pixel direction `d` at `R @ d`, so the output
*pole* (`d = [0,1,0]`) shows the rig direction `R @ [0,1,0]`.

Two frames matter:

- **rig frame** — the fixed camera frame the stitch lands in (from the lens
  extrinsics).
- **IMU body frame** — where the quaternions live. The per-file bridge
  between them for world-referenced vectors is
  `bridge = C_UP_FORMAT @ mount`, with `mount` read from each OSV (header
  1.3.1) and `C_UP_FORMAT` the format-level constant
  (see [constants](constants.md); det = −1 by design).

## `plain`

`R = K_PLAIN`, a constant: our equirect layout vs DJI's, measured once and
stable through a full tumble. No IMU involved — DJI's "plain" export is raw
optics.

## `rocksteady` — stabilisation only

Smooth the orientation trajectory (Gaussian matrix-mean over the quaternions,
σ = `RS_SIGMA` frames, SVD re-orthonormalised), then counter-rotate each frame
toward it:

```
R(f) = bridge · raw(f)ᵀ · smooth(f) · bridgeᵀ · K_PLAIN
```

The conjugation by `bridge` makes the raw orientation **cancel exactly**:
world→output reduces to `const · smooth(f)ᵀ`, so the output carries only the
smoothed trajectory's jitter (0.013°/frame analytic). `K_PLAIN` sits on the
*right* to preserve that cancellation — a steady camera renders exactly like
`plain`.

> **The composition matters.** The earlier form conjugated by a *fitted
> motion matrix* with the product order reversed; it applied the small
> correction about the tumbling raw axis and left 0.24–0.42°/frame of
> structural jitter that resisted every other explanation (quaternion timing,
> rolling shutter, smoother shape — all disproven experimentally). Rendered
> jitter now measures **0.027 calm / 0.071 moving vs DJI's 0.046 / 0.118**.

## `rocksteady_horizon` — the full lock

All three axes locked to a fixed, level world view. Build `R` as an explicit
orthonormal basis from the **instantaneous** quaternion:

```
up  = bridge · raw(f)ᵀ · WORLD_UP          (→ output pole: horizon dead level)
fwd = bridge · raw(f)ᵀ · WORLD_FWD ⊥ up    (→ yaw pinned to a fixed heading)
R   = [ up×fwd , up , fwd ]                (columns)
```

Key facts, each earned the hard way:

- The per-frame quaternion alone matches DJI's level up-vector to **0.24°
  mean across an entire worst-case tumble** — no high-rate IMU, no
  accelerometer, no drift correction needed.
- `up` *and* `fwd` must use the **same instantaneous** quaternion. Mixing a
  smoothed heading with an instantaneous up makes the heading lag the content
  and the world swings in azimuth (measured: 56° swing, 6°/frame jerks).
- An `align_vectors`-style minimal rotation is the wrong tool: its yaw is
  uncontrolled and it flips poles for some orientations. The explicit basis
  cannot.
- Result: horizon locked level end-to-end, yaw pinned (only slow, smooth gyro
  drift) — measurably steadier than DJI, whose yaw wanders under violent
  motion.

## What is intentionally *not* corrected

- **Exposure / vignette**: the footage arrives ISP-corrected; our corrections
  are off by default ([format notes](osv-format.md)).
- **Rolling shutter**: a complete per-pixel correction exists behind
  `video.use_rs` but measured *worse* than nothing on this camera once the
  composition bug was fixed — the apparent "jello" had been that bug.
- **Rim motion blur** during extreme tumble: baked into the source exposure
  (5.6 ms shutter vs up to 14°/frame at the fisheye rim); present in DJI's
  exports too.
