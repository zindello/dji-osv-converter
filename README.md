# dji_osv_converter

Convert DJI Osmo 360 `.OSV` recordings to equirectangular 360° MP4 —
stitching, RockSteady stabilisation and HorizonLevelling included — matching
or beating DJI Studio's own exports, with everything derived from the OSV
file itself. No DJI software, no `ffmpeg` binary; libavcodec in-process and a
GPU stitch (Apple MPS / CUDA), 10-bit end to end.

```bash
pip install -r requirements.txt && pip install -e .
python examples/convert_video.py INPUT.OSV out.mp4 --levelling rocksteady_horizon
```

**Speed** (8K 10-bit, stabilised, Apple M3/MPS): ~68 ms/frame stitch,
~102 ms/frame end-to-end serial (9.8 fps). Full per-platform table — GPU,
torch-CPU and the numpy reference path, measured by `tools/benchmark.py` —
in [`docs/benchmarks.md`](docs/benchmarks.md).

**Documentation: [`docs/`](docs/index.md)** — quick start, the
reverse-engineered OSV format reference, the levelling/stabilisation math and
its validation against DJI, architecture & performance, and a generated API
reference (`docs/api/`, kept current by a pre-commit hook).

Status: see [`docs/ROADMAP.md`](docs/ROADMAP.md).
