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
import queue
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


def gpu_decode_fmt(hardware: bool = True, path=None) -> str:
    """Cheapest 10-bit format the GPU stitch can ingest on this platform.

    With a ``path`` to probe, the best NVDEC arrangement is chosen:
    ``"cuda"`` (decode-to-device: frames stay on the GPU as torch tensors,
    zero PCIe decode traffic) when the primary-context hwdevice works, else
    ``"p010le"`` (hevc_cuvid to system memory, unpacked on-device). Without
    a path, the conservative ``"p010le"`` is returned on NVIDIA Linux -
    iter_pairs falls back internally if cuvid is unusable. Everywhere else
    planar is already the cheap layout (VideoToolbox frames convert
    losslessly; software decode emits planar directly).

    Call this BEFORE any torch CUDA work when probing (the primary-context
    hwdevice cannot be created once torch has initialized the context).
    """
    if hardware and sys.platform == "linux" and Path("/dev/nvidiactl").exists():
        from . import nvdec
        if path is not None and nvdec.cuda_frames_usable(path):
            return "cuda"
        return "p010le"
    return "yuv420p10le"


def _plane_u16(plane, width: int) -> np.ndarray:
    """A (height, width) uint16 view of a frame plane, stride handled.

    ``width`` is in uint16 samples: the luma width for Y, but 2x the reported
    plane width for p010's UV plane (U and V interleave in one row).
    """
    ls = plane.line_size // 2
    a = np.frombuffer(plane, np.uint16).reshape(plane.height, ls)
    return a[:, :width]


def _frame_p010(fr) -> tuple[np.ndarray, np.ndarray]:
    """(Y, UV) uint16 arrays in p010 layout from a decoded frame.

    The fast path returns ZERO-COPY views of the p010le planes NVDEC
    downloaded (both planes are row-contiguous; ``np.frombuffer`` keeps the
    frame alive through ``.base``, so the views stay valid while referenced).
    The fallback (hwaccel silently fell back to software, so frames arrive
    planar) shifts and interleaves on CPU - slower, but keeps the format
    contract with the stitcher intact.
    """
    if fr.format.name == "p010le":
        return (_plane_u16(fr.planes[0], fr.width),
                _plane_u16(fr.planes[1], fr.width))
    arr = fr.to_ndarray(format="yuv420p10le")
    fh, fw = fr.height, fr.width
    Y = arr[:fh] << 6
    UV = np.empty((fh // 2, fw), np.uint16)
    UV[:, 0::2] = arr[fh:fh + fh // 4].reshape(fh // 2, fw // 2) << 6
    UV[:, 1::2] = arr[fh + fh // 4:].reshape(fh // 2, fw // 2) << 6
    return Y, UV


def _video_streams(container):
    vids = [s for s in container.streams.video
            if s.codec_context.name != "mjpeg"]
    if len(vids) < 2:
        raise ValueError(f"expected 2 HEVC video streams, found {len(vids)}")
    return vids[:2]


def _open_stream(path, si: int, start: int, hwaccel=None):
    """Open one fisheye stream for streaming decode from ``start``.

    Returns ``(container, stream, rate, time_base)`` — the shared preamble of
    every decode worker.
    """
    c = av.open(str(path), hwaccel=hwaccel) if hwaccel else av.open(str(path))
    s = _video_streams(c)[si]
    rate, tb = float(s.average_rate), s.time_base
    if start > 0:
        c.seek(int(start / rate / tb), stream=s, backward=True, any_frame=False)
    return c, s, rate, tb


def _frame_index(pts, tb, rate) -> int:
    """Frame number for a pts on this stream's timeline."""
    return int(round(float(pts * tb) * rate))


def _iter_pairs_threaded(path, start: int, count: int | None, worker,
                         wargs=()):
    """Drive one decode worker per fisheye stream, zipping their outputs.

    Both NVDEC paths use this: decode (and any staging copies) run on the
    two worker threads and overlap each other and the consumer.
    """
    import threading

    from .constants import DECODE_WORKER_QUEUE

    stop = threading.Event()
    qa: "queue.Queue" = queue.Queue(maxsize=DECODE_WORKER_QUEUE)
    qb: "queue.Queue" = queue.Queue(maxsize=DECODE_WORKER_QUEUE)
    workers = [
        threading.Thread(target=worker,
                         args=(path, 0, start, count, qa, stop, *wargs), daemon=True),
        threading.Thread(target=worker,
                         args=(path, 1, start, count, qb, stop, *wargs), daemon=True),
    ]
    for w in workers:
        w.start()

    def next_item(q):
        item = q.get()
        if isinstance(item, Exception):
            raise item
        return item                       # None (end) or (idx, arr)

    try:
        a, b = next_item(qa), next_item(qb)
        while a is not None and b is not None:
            if a[0] == b[0]:
                yield a[0], a[1], b[1]
                a, b = next_item(qa), next_item(qb)
            elif a[0] < b[0]:             # streams should stay lockstep; skip
                a = next_item(qa)         # any solo frame defensively
            else:
                b = next_item(qb)
    finally:
        stop.set()
        for q in (qa, qb):                # unblock workers stuck on a full queue
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break


def iter_pairs(path: str | Path, start: int = 0, count: int | None = None,
               bit16: bool = False, hardware: bool = True, fmt: str | None = None,
               alloc=None):
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
    if fmt == "cuda":
        from . import nvdec
        if not (hardware and nvdec.cuda_frames_usable(path)):
            raise RuntimeError("fmt='cuda' requires device-resident NVDEC "
                               "(pick the format with gpu_decode_fmt(path))")
        yield from _iter_pairs_threaded(path, start, count, nvdec.cuda_worker)
        return
    if fmt == "p010le" and hardware:
        from . import nvdec
        if nvdec.cuvid_usable(path):
            yield from _iter_pairs_threaded(path, start, count,
                                            nvdec.cuvid_worker, (alloc,))
            return
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
        to_arr = _frame_p010 if fmt == "p010le" else (
            lambda fr: fr.to_ndarray(format=fmt))
        from . import profile as _profile
        prof = _profile.active()
        pairs = zip(ca.decode(sa), cb.decode(sb))
        while True:
            with prof.stage("decode.hw"):            # demux + decode, both streams
                pair = next(pairs, None)
            if pair is None:
                break
            fa, fb = pair
            if fa.pts is None or fb.pts is None:
                continue
            idx = _frame_index(fa.pts, sa.time_base, float(sa.average_rate))
            if idx < start:
                continue
            if count is not None and idx >= start + count:
                break
            with prof.stage("decode.to_array"):
                item = idx, to_arr(fa), to_arr(fb)
            yield item
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
