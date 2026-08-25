"""Benchmark the stitch pipeline on this machine and print a docs table row.

Measures, at the requested output resolution (default 8K):

  setup     one-time cost: calibration/metadata read + map build + GPU init
            (incl. torch.compile warm-up)
  stitch    steady-state ms/frame for the stitch alone (decode excluded:
            frames are pre-decoded into memory, timing loop syncs once)
  e2e       steady-state ms/frame for convert() - decode + stitch + encode,
            serial - measured over the tail of the run (--e2e)

and prints a ready-to-paste markdown row for docs/benchmarks.md. Run once
per engine:

    python tools/benchmark.py IN.OSV                  # best GPU (cuda > mps)
    python tools/benchmark.py IN.OSV --device cpu     # torch-CPU
    python tools/benchmark.py IN.OSV --reference      # numpy/OpenCV reference
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dji_osv_converter import (            # noqa: E402
    GpuStitcher, Stitcher360, convert, iter_pairs,
)
from dji_osv_converter.stitch_gpu import pick_device  # noqa: E402


def _sync(dev) -> None:
    import torch
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def _platform_label(device: str) -> str:
    mach = platform.machine()
    if device == "cuda":
        import torch
        return f"{torch.cuda.get_device_name(0)} ({mach})"
    chip = platform.processor() or mach
    if platform.system() == "Darwin":
        import subprocess
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip() or chip
    return chip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("osv", help="input .OSV (use a clip with real motion)")
    ap.add_argument("--width", type=int, default=7680)
    ap.add_argument("--height", type=int, default=3840)
    ap.add_argument("--device", default=None,
                    help="torch device (default: best available; 'cpu' for torch-CPU)")
    ap.add_argument("--reference", action="store_true",
                    help="benchmark the numpy/OpenCV CPU reference path instead of torch")
    ap.add_argument("--frames", type=int, default=None,
                    help="stitch-loop iterations (default 24 GPU / 4 CPU)")
    ap.add_argument("--e2e", type=int, default=0, metavar="N",
                    help="also run convert() over N frames and report the steady tail")
    ap.add_argument("--levelling", default="rocksteady_horizon")
    args = ap.parse_args()

    W, H = args.width, args.height
    t0 = time.time()
    st = Stitcher360.from_file(args.osv, width=W, height=H, levelling=args.levelling)
    maps, _ = st.maps_for()

    if args.reference:
        engine, dev_label = "numpy reference", "CPU (numpy/OpenCV)"
        setup = time.time() - t0
        n = args.frames or 3
        pairs = [(i, f0.copy(), f1.copy())
                 for i, f0, f1 in iter_pairs(args.osv, start=10, count=n + 1, fmt="bgr24")]
        # multiband=False: measure what convert(gpu=False) actually ships.
        st.stitch(pairs[0][1], pairs[0][2], frame=pairs[0][0],
                  multiband=False)                                      # warm caches
        t = time.time()
        for i, f0, f1 in pairs[1:]:
            st.stitch(f0, f1, frame=i, multiband=False)
        stitch_ms = (time.time() - t) / n * 1000
    else:
        import torch
        dev = args.device or pick_device()
        g = GpuStitcher(maps, band_px=st.seam_overlap_ratio * W, width=W, height=H,
                        device=dev,
                        fisheye_size=(st.calibs[0].width, st.calibs[0].height))
        setup = time.time() - t0
        engine, dev_label = f"torch/{g.dev.type}", str(g.dev.type)
        n = args.frames or (24 if g.dev.type != "cpu" else 4)
        pairs = [(i, f0.copy(), f1.copy())
                 for i, f0, f1 in iter_pairs(args.osv, start=10, count=n + 1,
                                             fmt="yuv420p10le")]
        g.stitch_yuv(pairs[0][1], pairs[0][2], st.levelling_for(pairs[0][0]))
        _sync(g.dev)
        t = time.time()
        for i, f0, f1 in pairs[1:]:
            g.stitch_yuv(f0, f1, st.levelling_for(i))
        _sync(g.dev)
        stitch_ms = (time.time() - t) / n * 1000

    e2e_ms = None
    if args.e2e:
        ts: list[float] = []
        # Shipping defaults - incl. pipeline auto (threaded on GPU).
        convert(args.osv, "/tmp/_bench_e2e.mp4", start=0, count=args.e2e,
                levelling=args.levelling, audio=False, spherical=False,
                gpu=not args.reference,
                progress=lambda d, t_: ts.append(time.time()))
        d = np.diff(ts)
        tail = d[len(d) // 3:]                       # skip spin-up
        e2e_ms = float(tail.mean() * 1000)

    label = _platform_label(dev_label if not args.reference else "cpu")
    fps = 1000.0 / e2e_ms if e2e_ms else None
    print(f"\nplatform : {label}")
    print(f"engine   : {engine}   output {W}x{H} 10-bit, levelling {args.levelling}")
    print(f"setup    : {setup:.1f} s")
    print(f"stitch   : {stitch_ms:.1f} ms/frame")
    if e2e_ms:
        print(f"e2e      : {e2e_ms:.1f} ms/frame ({fps:.1f} fps)")
    per_frame = (f"{e2e_ms:.0f} ms ({fps:.1f} fps)" if e2e_ms
                 else f"{stitch_ms:.0f} ms (stitch only)")
    kind = "CPU" if (args.reference or dev_label == "cpu") else "GPU"
    print("\ndocs/benchmarks.md row:")
    print(f"| {label} | {kind} | {setup:.0f} s | {per_frame} |")


if __name__ == "__main__":
    main()
