# dji_osv_converter

Convert DJI Osmo 360 `.OSV` recordings into equirectangular 360° MP4 —
stitching, stabilisation and horizon levelling included — **matching or
beating DJI Studio's own exports**, with everything derived from the OSV file
itself.

- No DJI software required (and none is called).
- No `ffmpeg` binary — decode/encode run in-process via libavcodec (PyAV),
  hardware-accelerated where available (VideoToolbox on macOS).
- GPU stitch on Apple MPS or CUDA, 10-bit YUV end to end.
- Per-unit calibration (lens model, extrinsics, camera↔IMU mount) is read from
  **each file**; the few format-level constants live in
  [`constants.py`](constants.md) with their provenance.

## Install

```bash
pip install -r requirements.txt      # av, numpy, opencv-python, torch
pip install -e .
```

## Convert a clip

```bash
python examples/convert_video.py INPUT.OSV output.mp4 --levelling rocksteady_horizon
```

or from Python:

```python
from dji_osv_converter import convert
convert("INPUT.OSV", "out.mp4", levelling="rocksteady_horizon")
```

The output is 8K (7680×3840) 10-bit HEVC by default, tagged `hvc1` with
spherical (360) metadata and the original audio — drop it straight into a 360
player, YouTube, or DaVinci Resolve.

## Levelling modes

| mode | what it does | vs DJI Studio |
|---|---|---|
| `plain` | raw optics; the image follows the camera | matches DJI's default export |
| `rocksteady` | smooths shake, keeps the camera's gross motion | **smoother** (rendered jitter 0.027/0.071 vs DJI 0.046/0.118 °/frame calm/moving) |
| `rocksteady_horizon` | locks pitch, roll **and yaw** to a fixed level world view | matches DJI's RockSteady+HorizonLevelling and **holds yaw tighter** (DJI wanders under violent motion) |
| `none` | the raw camera-rig frame (debugging) | — |

## Single frames

```python
from dji_osv_converter import Stitcher360, OSVDecoder

st = Stitcher360.from_file("INPUT.OSV")
with OSVDecoder("INPUT.OSV") as dec:
    f0, f1 = dec.get_pair(0)
pano = st.stitch(f0, f1, frame=0)      # equirectangular BGR (CPU reference path)
```

## Performance

8K 10-bit on an Apple M3 (MPS): **~96 ms/frame stitch, ~115 ms/frame
end-to-end (8.7 fps)** after a one-time ~6.5 s setup (map build + GPU
compile). See [architecture](architecture.md) for the budget.

## Documentation

- [OSV format reference](osv-format.md) (+ [historical notes](osv-format-notes.md)) — the container, metadata tracks and
  every decoded field
- [Levelling & stabilisation math](levelling.md) — how each mode works and how
  it was validated
- [Architecture](architecture.md) — module map, data flow, performance budget
- [Constants & provenance](constants.md) — every value that is *not* from the
  OSV, and why
- [Validation methodology](validation.md) — how claims against DJI are tested
- [API reference](api/index.html) — generated from the code (kept current by a
  pre-commit hook)
