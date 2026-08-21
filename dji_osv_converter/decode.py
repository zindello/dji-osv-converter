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
from pathlib import Path

import av
import numpy as np

logger = logging.getLogger(__name__)


def _hwaccel():
    """A VideoToolbox hardware-decode context, or None if unavailable.

    Hardware decode is ~5x faster than software on Apple silicon and yields
    10-bit (p010le) frames directly. Falls back silently to software.
    """
    try:
        from av.codec.hwaccel import HWAccel
        return HWAccel(device_type="videotoolbox", allow_software_fallback=True)
    except Exception:
        return None


def _video_streams(container):
    vids = [s for s in container.streams.video
            if s.codec_context.name != "mjpeg"]
    if len(vids) < 2:
        raise ValueError(f"expected 2 HEVC video streams, found {len(vids)}")
    return vids[:2]


def iter_pairs(path: str | Path, start: int = 0, count: int | None = None,
               bit16: bool = False, hardware: bool = True):
    """Yield ``(index, fisheye0, fisheye1)`` decoding both streams sequentially.

    Streaming decode - far faster than seeking per frame for a whole clip. Each
    fisheye stream is decoded in its own container so the two advance in
    lockstep by decode order. ``hardware=True`` uses VideoToolbox when present.

    ``bit16=False`` -> uint8 BGR (OpenCV-ready); ``bit16=True`` -> uint16 RGB
    with the full 10-bit range preserved.
    """
    fmt = "rgb48le" if bit16 else "bgr24"
    hw = _hwaccel() if hardware else None
    ca = av.open(str(path), hwaccel=hw) if hw else av.open(str(path))
    cb = av.open(str(path), hwaccel=hw) if hw else av.open(str(path))
    try:
        sa = _video_streams(ca)[0]
        sb = _video_streams(cb)[1]
        if not hw:
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
