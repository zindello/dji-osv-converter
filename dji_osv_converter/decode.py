"""In-process HEVC frame decode via libavcodec (PyAV).

This links libavcodec/libavformat directly - it does NOT shell out to an
`ffmpeg` binary. Seeking to a keyframe and decoding forward makes random frame
access fast (~0.1-1.5 s) instead of decoding from the start of the file, and
10-bit is preserved end to end.

The OSV carries two HEVC fisheye streams on a shared timeline; get_pair returns
both halves for one frame index.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import av
import numpy as np

logger = logging.getLogger(__name__)


def prefetch(gen, size: int = 4):
    """Run `gen` on a background thread, yielding its items via a bounded queue.

    Lets decode overlap the consumer (GPU stitch / encode): PyAV's libav calls
    and torch GPU kernels both release the GIL, so the stages run concurrently.
    """
    import queue
    import threading

    q: "queue.Queue" = queue.Queue(maxsize=size)
    sentinel = object()

    def worker():
        try:
            for item in gen:
                q.put(item)
        except Exception as e:               # surface producer errors downstream
            q.put(e)
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def _hwaccel():
    """The platform's hardware-decode context, or None if it has none.

    VideoToolbox on macOS (~5x faster than software on Apple silicon, yields
    10-bit p010le directly); NVDEC on Linux with an NVIDIA GPU. Falls back
    silently to software. The probe must be platform-gated: HWAccel
    construction succeeds even for accelerators this machine doesn't have
    (the failure only surfaces at codec-open, as a silent software fallback),
    and callers use ``None`` to decide software-decode threading — a bogus
    context here means single-threaded software decode.
    """
    try:
        from av.codec.hwaccel import HWAccel
    except Exception:
        return None
    if sys.platform == "darwin":
        return HWAccel(device_type="videotoolbox", allow_software_fallback=True)
    if sys.platform == "linux" and Path("/dev/nvidiactl").exists():
        return HWAccel(device_type="cuda", allow_software_fallback=True)
    return None


def _video_streams(container):
    vids = [s for s in container.streams.video
            if s.codec_context.name != "mjpeg"]
    if len(vids) < 2:
        raise ValueError(f"expected 2 HEVC video streams, found {len(vids)}")
    return vids[:2]


def iter_pairs(path: str | Path, start: int = 0, count: int | None = None,
               bit16: bool = False, hardware: bool = True, fmt: str | None = None):
    """Yield ``(index, fisheye0, fisheye1)`` decoding both streams sequentially.

    Streaming decode - far faster than seeking per frame for a whole clip. Each
    fisheye stream is decoded in its own container so the two advance in
    lockstep by decode order. ``hardware=True`` uses VideoToolbox when present.

    ``fmt`` sets the pixel format each frame is returned in; if None it is
    ``bgr24`` (uint8, OpenCV-ready) or ``rgb48le`` (uint16) per ``bit16``. Pass
    ``fmt="yuv420p10le"`` to keep the native 10-bit YUV (planar, uint16) with no
    colour conversion - the cheap, precision-preserving path for the GPU stitch.
    """
    if fmt is None:
        fmt = "rgb48le" if bit16 else "bgr24"
    hw = _hwaccel() if hardware else None
    ca = av.open(str(path), hwaccel=hw) if hw else av.open(str(path))
    cb = av.open(str(path), hwaccel=hw) if hw else av.open(str(path))
    try:
        sa = _video_streams(ca)[0]
        sb = _video_streams(cb)[1]
        if not hw or sys.platform != "darwin":
            # Threading must also be on when a hw context exists but ends up
            # falling back to software at codec-open (hwaccel failures are
            # silent). Harmless under real NVDEC decode; kept off only for
            # VideoToolbox, where the shipping M3 path never set it.
            sa.thread_type = sb.thread_type = "AUTO"
        if start > 0:
            fps = float(sa.average_rate)
            for s, c in ((sa, ca), (sb, cb)):
                c.seek(int(start / fps / s.time_base), stream=s,
                       backward=True, any_frame=False)
        for fa, fb in zip(ca.decode(sa), cb.decode(sb)):
            if fa.pts is None or fb.pts is None:
                continue
            idx = int(round(float(fa.pts * sa.time_base) * float(sa.average_rate)))
            if idx < start:
                continue
            if count is not None and idx >= start + count:
                break
            yield idx, fa.to_ndarray(format=fmt), fb.to_ndarray(format=fmt)
    finally:
        ca.close()
        cb.close()


class OSVDecoder:
    """Random-access decoder for an OSV file's two fisheye streams."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.container = av.open(self.path)
        vids = [s for s in self.container.streams.video
                if s.codec_context.name != "mjpeg"]
        if len(vids) < 2:
            raise ValueError(f"{path}: expected 2 HEVC video streams, found {len(vids)}")
        self.streams = vids[:2]
        self.fps = float(self.streams[0].average_rate)
        for s in self.streams:
            s.thread_type = "AUTO"   # multi-threaded decode

    @property
    def frame_count(self) -> int:
        return self.streams[0].frames or 0

    def _decode_one(self, stream, index: int, fmt: str) -> np.ndarray:
        ts = int(index / self.fps / stream.time_base)
        self.container.seek(ts, stream=stream, backward=True, any_frame=False)
        for frame in self.container.decode(stream):
            if frame.pts is None:
                continue
            n = int(round(float(frame.pts * stream.time_base) * self.fps))
            if n < index:
                continue
            return frame.to_ndarray(format=fmt)
        raise RuntimeError(f"{self.path}: could not decode frame {index} "
                           f"on stream {stream.index}")

    def get_pair(self, index: int, bit16: bool = False):
        """Decode fisheye 0 and 1 at `index`.

        Returns (img0, img1). bit16=False -> uint8 BGR (for OpenCV); bit16=True
        -> uint16 RGB with the full 10-bit range preserved. Each stream is
        seeked and decoded independently (PyAV frames do not carry their
        stream, and the streams share a timeline).
        """
        fmt = "rgb48le" if bit16 else "bgr24"
        return (self._decode_one(self.streams[0], index, fmt),
                self._decode_one(self.streams[1], index, fmt))

    def close(self) -> None:
        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def read_pair(path: str | Path, index: int = 0, bit16: bool = False):
    """Convenience: decode one frame pair and close."""
    with OSVDecoder(path) as d:
        return d.get_pair(index, bit16=bit16)


__all__ = ["OSVDecoder", "read_pair"]
