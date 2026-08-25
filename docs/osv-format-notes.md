# Decoding the DJI Osmo 360 `.OSV` format

*Reverse-engineering DJI's dual-fisheye `.OSV` container and turning it into a
single 8K equirectangular 360 frame — the calibration, the lens model, and the
stitch — entirely from data in the file.*

The DJI Osmo 360 records a proprietary `.OSV` file: two raw fisheye video
streams plus metadata, meant to be stitched only by DJI Studio. This documents
the format well enough to stitch it yourself, and — importantly — every number
needed comes **out of the file**, so it generalises across camera units (each
is factory-calibrated differently) rather than being tuned to one.

Nothing here calls the `ffmpeg` binary. The container and all metadata are read
in pure Python; only the HEVC pixels are decoded, with libavcodec in-process.

---

## 1. The container

`.OSV` is a plain MP4/ISO-BMFF. `ffprobe` (or the pure-Python reader below)
shows eight tracks:

| # | type | codec | handler | what it is |
|---|------|-------|---------|------------|
| 0 | video | HEVC Main10 | `VideoHandler` | fisheye A, 3840×3840, 10-bit |
| 1 | video | HEVC Main10 | `VideoHandler` | fisheye B, 3840×3840, 10-bit |
| 2 | audio | AAC | `SoundHandler` | stereo 48 kHz (not spatial) |
| 3 | data | `djmd` | `CAM meta` | **calibration + per-frame orientation** |
| 4 | data | `djmd` | `CAM meta` | lighter per-frame exposure/AWB telemetry |
| 5 | data | `dbgi` | `CAM dbgi` | sensor debug (`OV68A40`), not needed |
| 6 | data | `dbgi` | `CAM dbgi` | sensor debug, not needed |
| 7 | video | MJPEG | — | 688×344 thumbnail |

Both video streams are 10-bit `yuv420p10le`, tagged `bt709`. A sidecar `.WAV`
holds the same stereo audio as 16-bit PCM.

The two fisheyes each cover **> 180°** (about 210° physically), so they overlap
in a band; stitching is projecting both into one equirectangular sphere and
blending that band.

The interesting track is **3** (`CAM meta`). Its samples are protobuf
(`dvtm_oq101.proto`). Sample 0 is a header carrying the per-unit calibration;
every subsequent sample carries that frame's orientation.

---

## 2. Reading the format without a codec

An MP4 is a tree of *boxes*; a track's samples are located through its sample
tables (`stsz` sizes, `stsc` samples-per-chunk, `stco`/`co64` chunk offsets).
That's all pure Python — see `osv_mp4.py` in the accompanying code. It returns
the raw bytes of any track's samples, verified byte-for-byte against libav.

```python
from osv_mp4 import OSVContainer
c = OSVContainer("CAM_....OSV")
meta = c.find_track(name="CAM meta")
header = c.sample_bytes(meta, 0)        # calibration
frame_meta = c.sample_bytes(meta, 600)  # orientation for frame 600
```

---

## 3. The metadata protobuf

Field *numbers* are recoverable from the protobuf wire format even without the
`.proto`; the *names* below are ours. `a.b.c` means nested field c inside b
inside a.

### Header sample (calibration)

```
1.1.1   "dvtm_oq101.proto"      schema
1.1.5   "95SXN..."              camera serial (per-unit)
1.1.10  "Osmo 360"              model
2.6.<n>                         one calibration block per lens (see below)
```

There are several `2.6.<n>` blocks; blocks whose `width`/`height` match the
recording (3840×3840) are the pair in use.

### Calibration block `2.6.<n>` — the per-unit lens calibration

| field | meaning |
|-------|---------|
| `.1 .2 .3 .4` | `fx, fy, cx, cy` (focal length, principal point, px) |
| `.5 .6 .7 .8` | radial distortion `k1 k2 k3 k4` |
| **`.15`** | radial distortion **`k5`** — *stored apart from k1..k4* |
| `.20` | tangential distortion `p1 p2` (2 floats) |
| `.10 .11` | width, height the calibration applies to |
| `.12 .13 .14` | extrinsic yaw, pitch, roll (degrees) |
| `.21` / `.28` | extrinsic orientation as a unit quaternion (w,x,y,z) — same rotation, no gimbal lock at pitch≈90° |
| `.22 .23` | 14 boundary points of the valid image circle |

The single most important gotcha: the radial model has **five** terms, and the
fifth lives in field **15**, not alongside `.5–.8`. Reading only four produces a
non-monotonic curve — the mistake that makes these coefficients look useless.

### Per-frame sample (orientation)

| field | meaning |
|-------|---------|
| `3.2.9` | orientation quaternion for this frame (w,x,y,z), rig→world |
| `3.2.10` | gravity vector (3 floats), in the rig frame |
| `3.3.2.1.3` | ~34 high-rate IMU quaternions spanning the sensor readout |

Rotating the gravity vector by the frame quaternion lands on `(0,0,−1)` to
within noise, i.e. the quaternion is a gravity-aligned world orientation. Use
either to level the horizon; the gravity vector is the simplest.

---

## 4. The lens model — DJI's f-theta

The projection was read directly out of DJI Studio's binary
(`DUALFISHEYEVIDEOEIS::CameraModel::project`). For a camera-frame ray
`(X, Y, Z)`:

```
theta   = atan2( sqrt(X^2 + Y^2), Z )
theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9 + k5*theta^11
x'      = theta_d * X / sqrt(X^2 + Y^2)
y'      = theta_d * Y / sqrt(X^2 + Y^2)
r2      = theta_d^2
x''     = x' + 2*p1*x'*y' + p2*(r2 + 2*x'^2)      # Brown tangential
y''     = y' + p1*(r2 + 2*y'^2) + 2*p2*x'*y'
u       = fx * x'' + cx
v       = fy * y'' + cy
```

Raw radians, no normalisation, no gain — five radial terms plus the two
tangential terms, all from the file. Verified against DJI's own export: this
model **out-scores a lookup table built by measuring the lens curve directly**,
which is the proof it is the real model and not merely a good fit (a model
carrying form-error cannot beat a direct measurement).

---

## 5. Stitching one frame

1. **Project** — for every pixel of a 7680×3840 equirectangular output, take its
   world direction, rotate into each lens's frame (extrinsic quaternion), and
   apply the f-theta model above to find the source fisheye pixel. Build this as
   a remap once per orientation.
2. **Level** — fold a rotation that sends the equirect "down" onto the gravity
   vector into that remap, so the horizon is level. (Gravity fixes pitch and
   roll; heading/yaw is a free choice for a 360.)
3. **Field of view** — use `180° + seam_overlap_ratio·360°` per lens. DJI ships
   `seam_overlap_ratio = 0.038`, giving 193.7° — the minimum that supplies its
   blend band. (The physical lens is wider, but its rim is the worst optically
   and isn't needed once the band is covered.)
4. **Exposure** — the two sensors meter independently and vignette differently,
   so one hemisphere is brighter and the difference varies across the seam. A
   single global gain can't fix a spatially varying difference, and a per-pixel
   gain field fights parallax. **Multi-band (Laplacian-pyramid) blending** mixes
   low frequencies over a wide transition and detail over a narrow one, so the
   exposure/vignetting difference fades invisibly without being modelled.
5. **Blend** across the ~292 px seam band (again `seam_overlap_ratio`).

The full PoC — OSV in, one 7680×3840 equirectangular PNG out — is
`poc_stitch_frame.py`, about 120 lines on top of these modules.

---

## 6. To a video file

For a 360 **video** rather than stills:

- **Orientation must be consistent frame to frame** or the sphere spins. Level
  every frame with `3.2.9`/`3.2.10`; lock heading using the frame quaternion so
  the world stays put (this is most of what "stabilisation" means for 360).
- **Encode** the stitched frames as HEVC 10-bit and write **spherical video
  metadata** (`projection=equirectangular`) so players treat it as 360.
- **Mux** the stereo audio (from the sidecar `.WAV` for quality).

---

## 7. What DJI does that this deliberately doesn't

Matching DJI's *exported* pixels is **not** the goal — several of DJI's steps
are creative choices that belong later in the edit, not in a faithful
conversion:

- **Colour grading.** DJI applies a tonal look on export. If you shoot D-Log-M
  (10-bit), that grade belongs in the edit; a conversion should preserve the
  log curve untouched. So our output intentionally does *not* match DJI's graded
  tones.
- **Electronic stabilisation (forward-view EIS).** DJI smooths the camera path.
  Locking the world orientation (above) already stabilises a 360; heavier
  smoothing re-frames the shot and is the editor's call.
- **Rolling-shutter deroll.** DJI warps each frame per-row from the IMU
  (`derollingMapHex`). The data to do this is in the file (the 34 high-rate
  quaternions + readout time), and it's implemented here — but on handheld
  footage the readout motion is ~1–2° and the skew is sub-pixel, so it's off by
  default. It matters for fast whip-pans.

There is also a small (~1.5°) fixed rotation between the accelerometer's "down"
and DJI's world "down" — a camera↔IMU mount offset (`camImuExtriQ`, flagged
*not a real value* for this unit and so defaulted internally by DJI). It is not
in the file, and a gravity-levelled horizon is objectively level, so it isn't
chased.

---

## 8. Status

Fully worked out: container, both `CAM meta` tracks (calibration + orientation),
the f-theta lens model, audio, thumbnail. **Undecoded but not needed:** the two
`dbgi` debug tracks (sensor telemetry). The conversion needs none of it.

*Read directly from the binary and the file; no vendor SDK. Camera: DJI Osmo
360, sensor OmniVision OV68A40.*
