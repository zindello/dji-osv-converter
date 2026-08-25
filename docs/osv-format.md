# The OSV format

A `.OSV` is a plain MP4 (ISO-BMFF). Alongside the media it carries DJI
metadata tracks whose samples are **protobuf wire format** (schema
`dvtm_oq101.proto`, unpublished — decoded structurally by
`protobuf_wire.py`; field *numbers* below are ground truth, the names are
ours). Everything in this page was reverse-engineered and verified against
DJI Studio's own exports.

## Streams

| stream | content |
|---|---|
| 2 × HEVC video | the two fisheye halves, 3840×3840 yuv420p10le, shared timeline |
| 1 × MJPEG | embedded thumbnail |
| audio | AAC (copied through on convert) |
| data stream A | **the metadata track**: one header sample + one sample per video frame (carries 3.2.9 quats *and* the high-rate track) |
| data stream B | a second track with per-frame samples; its 3.2.9 quats are NaN — not used |
| data streams C, D | not protobuf (undecoded, unused) |

Streams are found **by content**, never by index — DJI renumbers between
firmware versions.

## Header sample (calibration)

One sample carries the factory calibration. Top-level field `1` is device
info, `2` the video/calibration description.

| field | meaning |
|---|---|
| `1.1.3 / 1.1.5 / 1.1.6` | firmware version, serial, sensor version strings |
| `1.3.1` | **camera↔IMU mount rotation** — Rodrigues vector, per-unit (~13° magnetic-mount skew). Must be applied to all IMU-derived directions |
| `1.11.1` | frame rate (29.97) |
| `1.14` | sensor size (3840×3840) |
| `2.3` | stream description (size, fps, bit depth 10) |
| `2.6.1 / 2.6.2` | **per-lens calibration blocks** (below); `2.6.3/2.6.4` are zero-filled placeholders |

### Per-lens calibration block (`2.6.N`)

| field | meaning |
|---|---|
| 1, 2 | fx, fy (px) |
| 3, 4 | cx, cy (px) |
| 5–8, **15** | f-theta radial distortion k1…k5 — **five terms**; k5 is stored apart from the others, which long disguised the model |
| 10, 11 | sensor width, height |
| 12 | lens yaw layout (≈180 for lens 0, ≈0 for lens 1) |
| 13 | ≈90.5 (undecoded; suspected half-FOV-related) |
| 14 | small scalar, undecoded |
| 20 | tangential distortion p1, p2 |
| 21 | a second extrinsic quaternion, near-identical to 28 (undecoded difference) |
| 22, 23 | **factory lens-shading curve**: sample positions along a diameter + brightness profile (single-channel luminance; falls to 0.762 at the rim). NOTE: the delivered footage is already devignetted by the in-camera ISP — see below |
| 24, 25, 27 | undecoded (8, −1000, [0, 0.001]) |
| 28 | **lens extrinsic rotation** (unit quaternion, wxyz) — the one used |

### The lens model

```
theta   = atan2(sqrt(X²+Y²), Z)
theta_d = theta + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹ + k5·θ¹¹
(x',y') = theta_d · (X,Y)/√(X²+Y²)   → Brown tangential (p1,p2) → K·[x'',y'',1]
```

Verified against DJI ground truth to 2.1 / 3.9 px RMS with best-fit gain
0.9998/0.9975 — i.e. the file's own intrinsics are correct as-is.

## Per-frame metadata sample

| field | meaning |
|---|---|
| `3.2.3` | scene luma / exposure target (varies with the scene) |
| `3.2.5` | fixed gain 1.0 |
| `3.2.6` | shutter time, µs (~5625 on the reference clips) |
| `3.2.9` | **orientation quaternion** (wxyz) — DJI's fused IMU output, sampled at frame START. World-up is +Z (verified via gravity). Drift-free enough to match DJI's own levelling to 0.24° across a worst-case tumble |
| `3.2.10` | accelerometer "gravity" (x,y,z in fields 2–4). Carries translational acceleration under motion (disagrees with the quaternion by up to 52°) — **do not level from it** |
| `3.2.15` | exposure block: `.3/.4/.5` = WB gains (locked [1,1,1]), `.6` = ISO-like gain, others undecoded |
| `3.2.16` | sensor temperature (°C, rises over a take) |
| `3.3.2.1` | **high-rate orientation track**: `.1` µs timestamp, `.2` frame counter, `.3` repeated ~33 quaternions spanning the frame (~1 kHz; sample 0 == the 3.2.9 quat), `.4` constant 11.775 (undecoded) |

## What the OSV does *not* contain

Findings that shaped the pipeline:

- **The fisheyes arrive ISP-processed**: already devignetted and
  exposure-matched in camera. Re-applying the factory LSC curve or an
  exposure match *double-corrects* and paints an arc across the seam — both
  are off by default.
- **No per-channel colour/vignette calibration** and no per-lens exposure
  state — the single-channel LSC curve is the only photometric data.
- **No layout conventions**: how DJI's equirect is oriented relative to the
  rig and the IMU frame is not in the file. Those are the `[MEASURED]`
  constants in [constants.md](constants.md).
