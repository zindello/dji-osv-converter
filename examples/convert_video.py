#!/usr/bin/env python3
"""Convert a DJI Osmo 360 .OSV to an equirectangular 360 MP4.

    python examples/convert_video.py INPUT.OSV OUTPUT.mp4 [options]

Everything is derived from the OSV file - lens calibration, the f-theta model,
per-frame orientation, and the camera<->IMU mount rotation. Decode and encode
use libavcodec in-process (VideoToolbox hardware when available); the ffmpeg
binary is never invoked. The output is tagged as 360 (spherical metadata) and
carries the original audio.
"""

import argparse
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from dji_osv_converter.video import convert


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input .OSV file")
    ap.add_argument("output", help="output .mp4 file")
    ap.add_argument("--width", type=int, default=7680)
    ap.add_argument("--height", type=int, default=3840)
    ap.add_argument("--start", type=int, default=0, help="first frame")
    ap.add_argument("--count", type=int, default=None, help="frames to convert (default all)")
    ap.add_argument("--codec", default="hevc_videotoolbox",
                    help="PyAV encoder (hevc_videotoolbox, libx265, ...)")
    ap.add_argument("--bitrate", type=int, default=None, help="target bits/s")
    ap.add_argument("--8bit", dest="bit10", action="store_false", help="8-bit output")
    ap.add_argument("--no-audio", dest="audio", action="store_false")
    ap.add_argument("--multiband", dest="multiband", action="store_true",
                    help="multi-band seam blend (default: linear - identical after "
                         "exposure match, ~1s/frame faster)")
    ap.add_argument("--exposure-wb", dest="exposure_wb", action="store_true",
                    help="per-channel white-balance exposure match between the two "
                         "lenses (default: luma-only; ~15ms/frame dearer, helps in "
                         "mixed/coloured lighting)")
    ap.add_argument("--software-decode", dest="hardware", action="store_false")
    ap.add_argument("--no-gpu", dest="gpu", action="store_false",
                    help="force the CPU stitch path")
    ap.add_argument("--pipeline", dest="pipeline", action="store_true", default=None,
                    help="overlap decode/stitch/encode on threads (auto-on for CUDA; "
                         "helps on dedicated GPU + NVENC, not on shared-memory MPS)")
    args = ap.parse_args()

    t = time.time()

    def progress(done, total):
        pct = f"{100 * done / total:4.0f}%" if total else ""
        rate = done / max(time.time() - t, 1e-6)
        print(f"\r  {done}/{total or '?'} {pct}  {rate:.1f} fps", end="", flush=True)

    out = convert(args.input, args.output, width=args.width, height=args.height,
                  start=args.start, count=args.count, codec=args.codec,
                  bitrate=args.bitrate, bit10=args.bit10, audio=args.audio,
                  multiband=args.multiband, exposure_wb=args.exposure_wb,
                  hardware=args.hardware, gpu=args.gpu,
                  pipeline=args.pipeline, spherical=True, progress=progress)
    dt = time.time() - t
    print(f"\nwrote {out} in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
