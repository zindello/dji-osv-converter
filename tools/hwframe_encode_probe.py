#!/usr/bin/env python3
"""Probe: is device-frame NVENC encode a win on THIS GPU?

Stock PyAV has no API to allocate a CUDA hw frame, but it will pass any
``cuda``-format frame straight through to NVENC (``_encode_upload_frame``
passthrough). This probe MINTS reusable CUDA frames by decoding a tiny
generated clip device-resident (``is_hw_owned=True``), writes test content
into their device pointers with torch, and A/B times the encode call
against the ordinary software-frame path.

Verdict on the V100 (Volta, 2026-08): device-frame encode WORKS but is
~2.7x SLOWER per call (55 vs 20 ms at 8K) — the per-frame
nvEncMapInputResource/unmap cost on this NVENC generation exceeds the
saved upload, and frames from an encoder-owned pool (what a PyAV fork
would add) ride the exact same register/map path in nvenc.c, so no API
change can beat it on Volta. Re-run this probe on Turing+ hardware where
resource mapping is cheap; if HW wins there, wire the mint into
``convert()``'s builder stage (the ring replaces the pinned-slot pool).

Usage:
    python tools/hwframe_encode_probe.py            # 8K default
    python tools/hwframe_encode_probe.py 3840 1920  # other sizes
"""

import os
import sys
import tempfile
import time
from fractions import Fraction

import av
import numpy as np
from av.codec.hwaccel import HWAccel

W = int(sys.argv[1]) if len(sys.argv) > 2 else 7680
H = int(sys.argv[2]) if len(sys.argv) > 2 else 3840
TB = Fraction(1001, 30000)
N, WARM, RING = 60, 12, 8


def hw_owned():
    return HWAccel(device_type="cuda", options={"primary_ctx": "1"},
                   is_hw_owned=True)


def make_dummy(path):
    """A tiny black p010 HEVC clip at target size, to mint hw frames from."""
    buf = av.open(path, "w")
    vs = buf.add_stream("hevc_nvenc", rate=Fraction(30000, 1001))
    vs.width, vs.height = W, H
    vs.pix_fmt = "p010le"
    vs.time_base = TB
    black = av.VideoFrame(W, H, "p010le")
    for p, v in zip(black.planes, (64 * 64, 512 * 64)):
        np.frombuffer(p, np.uint16)[:] = v
    for i in range(2 * RING):
        black.pts = i
        for pkt in vs.encode(black):
            buf.mux(pkt)
    for pkt in vs.encode():
        buf.mux(pkt)
    buf.close()


class _CAI:
    def __init__(self, ptr, shape, strides):
        self.__cuda_array_interface__ = {
            "shape": shape, "typestr": "<u2", "data": (ptr, False),
            "strides": strides, "version": 2}


def mkenc(hw):
    kw = (dict(hwaccel=HWAccel(device_type="cuda",
                               options={"primary_ctx": "1"})) if hw else {})
    e = av.CodecContext.create("hevc_nvenc", "w", **kw)
    e.width, e.height = W, H
    e.pix_fmt = "p010le"
    e.time_base = TB
    e.framerate = Fraction(30000, 1001)
    e.bit_rate = int(W * H * 29.97 * 0.15)
    e.open()
    return e


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        dummy = os.path.join(td, "dummy.mp4")
        make_dummy(dummy)

        # Claim the primary context via FFmpeg BEFORE torch (context-flags
        # trap, see dji_osv_converter/nvdec.py) and hold the container.
        hold = av.open(dummy, hwaccel=hw_owned())
        ring = [next(iter(hold.decode(hold.streams.video[0])))]
        import torch
        torch.zeros(1, device="cuda")
        while len(ring) < RING:
            c = av.open(dummy, hwaccel=hw_owned())
            try:
                for fr in c.decode(c.streams.video[0]):
                    ring.append(fr)
                    if len(ring) >= RING:
                        break
            except Exception:
                pass                       # surface pool exhausted; reopen
            c.close()
        assert all(f.format.name == "cuda" for f in ring)
        print(f"minted {len(ring)} CUDA frames {W}x{H} "
              f"(linesize {ring[0].planes[0].line_size})")

        def fill(fr, i):
            y, uv = fr.planes
            Y = torch.as_tensor(_CAI(y.buffer_ptr, (H, W), (y.line_size, 2)),
                                device="cuda")
            UV = torch.as_tensor(_CAI(uv.buffer_ptr, (H // 2, W),
                                      (uv.line_size, 2)), device="cuda")
            vals = ((torch.rand(H, W, device="cuda") * 300 + 200 + i)
                    .to(torch.int32) * 64)
            Y.copy_(vals.to(torch.uint16))
            UV.fill_(512 * 64)
            torch.cuda.synchronize()

        enc = mkenc(True)
        times = []
        for i in range(N):
            fr = ring[i % len(ring)]
            fill(fr, i)
            fr.pts = i
            t0 = time.perf_counter()
            enc.encode(fr)
            times.append(time.perf_counter() - t0)
        enc.encode(None)
        thw = np.array(times[WARM:]) * 1000

        enc2 = mkenc(False)
        sw = av.VideoFrame(W, H, "p010le")
        yb = np.frombuffer(sw.planes[0], np.uint16).reshape(
            sw.planes[0].height, sw.planes[0].line_size // 2)
        uvb = np.frombuffer(sw.planes[1], np.uint16).reshape(
            sw.planes[1].height, sw.planes[1].line_size // 2)
        noise = ((np.random.default_rng(0).random((H, W)) * 300 + 200)
                 .astype(np.uint16)) * 64
        times = []
        for i in range(N):
            yb[:, :W] = noise + i * 64
            uvb[:] = 512 * 64
            sw.pts = i
            t0 = time.perf_counter()
            enc2.encode(sw)
            times.append(time.perf_counter() - t0)
        enc2.encode(None)
        tsw = np.array(times[WARM:]) * 1000

        print(f"HW-frame encode : mean {thw.mean():5.1f}  "
              f"p50 {np.percentile(thw, 50):5.1f}  "
              f"p90 {np.percentile(thw, 90):5.1f} ms")
        print(f"SW-frame encode : mean {tsw.mean():5.1f}  "
              f"p50 {np.percentile(tsw, 50):5.1f}  "
              f"p90 {np.percentile(tsw, 90):5.1f} ms")
        win = tsw.mean() / thw.mean()
        print(f"verdict: device frames are {win:.2f}x the software path "
              f"({'WIN - wire it into convert()' if win > 1.1 else 'no win on this GPU'})")
    # held hw frames + containers can wedge interpreter teardown; exit hard
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
