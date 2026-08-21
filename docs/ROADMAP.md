# Roadmap — DJI Osmo 360 `.OSV` → equirectangular 360° MP4

**Status:** 2026-08-21. Working single-frame pipeline; building video output.

## Approach

**Everything is derived from the `.OSV` file** — per-unit lens calibration, the
f-theta lens model, per-frame orientation, and the camera↔IMU mount rotation.
No DJI code is called and nothing is read from the DJI Studio app bundle (an
earlier approach that reused DJI's shipped remap tables was abandoned — it was
tied to one installed bundle and didn't generalise). Only the HEVC pixels are
decoded, with libavcodec in-process via PyAV; the ffmpeg binary is never
invoked. This makes the converter self-contained and portable across camera
units and future models.

See `gist/dji-osv-360-format.md` for the reverse-engineered format writeup.

## Pipeline

```
OSVContainer / OSVDecoder  -> two 3840² 10-bit fisheye frames + metadata
read_osv_calibration       -> per-lens intrinsics + f-theta coeffs (5 radial, 2 tangential)
read_metadata              -> per-frame gravity (3.2.10) + orientation quat (3.2.9)
read_mount_rotation        -> camera↔IMU mount skew (header 1.3.1)
build_map + levelling      -> fisheye→equirect remap, gravity+mount levelled
multiband_blend            -> seam-invisible exposure/vignette blend
```

## Done

- [x] Pure-Python OSV container + metadata parsing (`osv_mp4`, `metadata`, `osv_calibration`)
- [x] In-process libavcodec decoder, 10-bit preserved (`decode`)
- [x] DJI f-theta lens model from the OSV calibration, verified vs DJI's binary (`project`)
- [x] FOV from DJI's `seam_overlap_ratio` (193.68°); multiband blending (`stitcher`)
- [x] Gravity levelling **+ camera↔IMU mount correction** (header 1.3.1) — horizon
      tilt 13° → ~2°; validated against DJI's `plain`/`rocksteady`/`horizonlevelling` exports
- [x] Single-frame PoC end-to-end (`examples/poc_stitch_frame.py`)
- [x] Codebase cleanup — removed the abandoned bundle approach and dead scaffolding

## Now: video output

- [ ] Sequential decode of all frames (both fisheyes), streamed not random-access
- [ ] Per-frame stitch (mount-corrected levelling) → HEVC 10-bit encode (PyAV, in-process)
- [ ] Spherical video metadata (equirectangular projection tag) so players treat it as 360
- [ ] Mux the stereo audio (OSV AAC track, or the sidecar `.WAV` for quality)
- [ ] Clean library API + example/CLI for `OSV → 360 MP4`

## Backlog

- [ ] **RockSteady-equivalent stabilisation.** Per-frame gravity levelling leaves a
      few degrees of horizon wobble under motion (walking acceleration in the
      accelerometer). DJI's RockSteady smooths the *orientation trajectory* using the
      gyro-integrated path, then levels. Reproduce it: smooth the orientation over time
      (frame quat 3.2.9 / the 34 high-rate IMU quats 3.3.2.1.3) rather than levelling to
      instantaneous gravity. Default-on with a flag. This removes the residual tilt and
      keeps the horizon steady across a whole clip.
- [ ] Rolling-shutter deroll (`deroll.py`, off by default) — matters only for fast whip-pans.
- [ ] GPU acceleration of the remap/blend for real-time throughput.
- [ ] Minor code review items: dedupe `quat_matrix`/`quat_to_matrix`; drop unused
      `radial_map` `theta_max` param; remove the now-dead bundle path
      (`load_calibration`, `orientation.apply_to_maps`) or fold `CameraMap` into a
      neutral module.

## Deliberately NOT done (creative choices, belong in the edit)

- Colour grading — DJI bakes a look on export; shot in D-Log-M, the log curve is preserved.
- Heavy EIS reframing beyond horizon-levelling.
