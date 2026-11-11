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

**Speed** (8K 10-bit, stabilised + auto seam repair, Apple M3/MPS): ~8 s
setup, then ~212 ms per frame (4.7 fps); ~73 ms (13.6 fps) with the seam
repair off. Per-platform table in
[`docs/benchmarks.md`](docs/benchmarks.md), measured by
`tools/benchmark.py`.

**Options worth knowing** (all off by default, see
[`IMPROVEMENTS.md`](IMPROVEMENTS.md) for measurements): `--single-pass`
samples each fisheye once per output pixel (sharper, exact wrap column);
`--lens-interp bicubic|catmull` changes the luma sampling kernel (bicubic
is a sharpening kernel, Catmull-Rom the accurate one); `Converter(...)`
converts many files in one process with the setup paid once;
`convert(compile_cache_dir=...)` persists the torch.compile cache. The
360 tag is injected with O(moov) memory. `tools/ab_compare.py` renders an
offline A/B viewer of every variant.

**Documentation: [`docs/`](docs/index.md)** — quick start, the
reverse-engineered OSV format reference, the levelling/stabilisation math and
its validation against DJI, architecture & performance, and a generated API
reference (`docs/api/`, kept current by a pre-commit hook).

Status: see [`docs/ROADMAP.md`](docs/ROADMAP.md).
