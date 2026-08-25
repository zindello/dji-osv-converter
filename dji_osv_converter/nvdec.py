"""NVDEC (NVIDIA) decode paths — the CUDA-specific side of :mod:`.decode`.

Everything here is Linux + NVIDIA only and reached lazily through
``decode.gpu_decode_fmt`` / ``decode.iter_pairs``; nothing imports this
module at package-import time, so torch-less and non-NVIDIA installs never
touch it. Two tiers, best first:

**Decode-to-device** (``fmt="cuda"``): the hwdevice is created with
``primary_ctx=1`` + ``is_hw_owned=True``, so decoded frames stay ON the GPU
in the SAME CUDA context torch uses — no 44 MB/frame download, no re-upload,
no cross-context serialization against the stitch kernels. 13 ms/frame
single stream, 25 ms for the fisheye pair (engine speed) on the V100.
Frames are cloned off the NVDEC surface pool immediately (device-to-device,
~0.1 ms) so the decoder never starves for surfaces. The primary-context
hwdevice MUST be created before torch initializes CUDA (FFmpeg sets context
flags, which fails on an already-active context) — ``convert()`` resolves
the decode format before it builds the GpuStitcher for exactly this reason.

**hevc_cuvid to system memory** (``fmt="p010le"`` fallback): the generic
hwaccel route feeds NVDEC one frame at a time and leaves the engine ~85%
idle (55 ms/frame measured on the V100). libavcodec's dedicated hevc_cuvid
decoder keeps the engine's parser and surface queue full: 22.5 ms/frame at
3840x3840x10bit with the engine pegged at 100%, 33 ms for the fisheye PAIR
when each stream decodes on its own thread. Output lands in system memory
as p010le, which the GPU stitch ingests raw (``input_fmt``).
"""

from __future__ import annotations

import logging
import queue
import sys
from pathlib import Path

import av
import numpy as np

from .constants import CUVID_PROBE_PACKETS
from .decode import _frame_index, _frame_p010, _open_stream, _video_streams

logger = logging.getLogger(__name__)


def _nvidia_linux() -> bool:
    return sys.platform == "linux" and Path("/dev/nvidiactl").exists()


def _queue_put(q, item, stop, timeout=0.25) -> bool:
    """Blocking put that gives up when the consumer has gone away."""
    while not stop.is_set():
        try:
            q.put(item, timeout=timeout)
            return True
        except queue.Full:
            continue
    return False


# --- decode-to-device --------------------------------------------------------

_HW_PRIMARY = None       # HWAccel | False (probed, unusable) | None (unprobed)


def hwaccel_primary():
    """A device-resident CUDA hwaccel on torch's primary context, or None."""
    global _HW_PRIMARY
    if _HW_PRIMARY is None:
        _HW_PRIMARY = False
        if _nvidia_linux():
            try:
                from av.codec.hwaccel import HWAccel
                _HW_PRIMARY = HWAccel(device_type="cuda",
                                      options={"primary_ctx": "1"},
                                      is_hw_owned=True)
            except Exception:
                logger.info("primary-ctx CUDA hwdevice unavailable",
                            exc_info=True)
    return _HW_PRIMARY or None


_CUDA_FRAMES_OK: bool | None = None
_CUDA_CTX_HOLD = None    # keeps FFmpeg's primary-context retain alive (below)


def cuda_frames_usable(path) -> bool:
    """Probe (once): device-resident NVDEC decode works for this file AND
    torch can wrap the frames (importable, CUDA up).

    On success the probe container is deliberately KEPT OPEN for the process
    lifetime: FFmpeg's hwdevice sets scheduling flags on the CUDA primary
    context at creation. If its retain is released and torch then initializes
    the context with different flags, every later hwdevice creation fails
    with EOPNOTSUPP. Holding one reference keeps the flags in place, so
    per-worker hwdevices keep working after torch spins up.
    """
    global _CUDA_FRAMES_OK, _CUDA_CTX_HOLD
    if _CUDA_FRAMES_OK is None:
        _CUDA_FRAMES_OK = False
        hw = hwaccel_primary()
        if hw is not None:
            try:
                import torch
                if torch.cuda.is_available():
                    c = av.open(str(path), hwaccel=hw)
                    s = _video_streams(c)[0]
                    fr = next(iter(c.decode(s)))
                    _CUDA_FRAMES_OK = (fr.format.name == "cuda"
                                       and fr.planes[0].buffer_ptr != 0)
                    if _CUDA_FRAMES_OK:
                        _CUDA_CTX_HOLD = c
                    else:
                        c.close()
            except Exception:
                logger.info("device-resident NVDEC probe failed", exc_info=True)
    return _CUDA_FRAMES_OK


class _CudaPlane:
    """__cuda_array_interface__ view of a device frame plane (uint16)."""

    def __init__(self, plane, rows: int, cols: int):
        self.__cuda_array_interface__ = {
            "shape": (rows, cols), "typestr": "<u2",
            "data": (plane.buffer_ptr, False),
            "strides": (plane.line_size, 2), "version": 2}


def cuda_worker(path, si: int, start: int, count: int | None, q, stop):
    """Decode one fisheye stream with frames staying on the GPU.

    Emits (idx, (Y, UV)) torch CUDA uint16 tensors in p010 layout
    (top-aligned samples, interleaved chroma), cloned off the NVDEC
    surfaces on a private stream.
    """
    import torch
    try:
        side = torch.cuda.Stream()
        c, s, rate, tb = _open_stream(path, si, start, hwaccel=hwaccel_primary())
        fh, fw = s.codec_context.height, s.codec_context.width
        for fr in c.decode(s):
            if stop.is_set():
                break
            if fr.pts is None:
                continue
            idx = _frame_index(fr.pts, tb, rate)
            if idx < start:
                continue
            if count is not None and idx >= start + count:
                break
            with torch.cuda.stream(side):
                Y = torch.as_tensor(_CudaPlane(fr.planes[0], fh, fw),
                                    device="cuda").clone()
                UV = torch.as_tensor(_CudaPlane(fr.planes[1], fh // 2, fw),
                                     device="cuda").clone()
            side.synchronize()           # surface safe to release after this
            if not _queue_put(q, (idx, (Y, UV)), stop):
                break
        c.close()
    except Exception as e:
        _queue_put(q, e, stop)
    finally:
        _queue_put(q, None, stop)


# --- hevc_cuvid to system memory ---------------------------------------------

_CUVID_OK: bool | None = None


def cuvid_usable(path) -> bool:
    """Probe (once per process) that hevc_cuvid exists AND decodes this file."""
    global _CUVID_OK
    if _CUVID_OK is None:
        _CUVID_OK = False
        if _nvidia_linux():
            try:
                c = av.open(str(path))
                s = _video_streams(c)[0]
                ctx = av.CodecContext.create("hevc_cuvid", "r")
                ctx.extradata = s.codec_context.extradata
                for i, pkt in enumerate(c.demux(s)):
                    if pkt.dts is None:
                        continue
                    if any(fr.format.name == "p010le" for fr in ctx.decode(pkt)):
                        _CUVID_OK = True
                        break
                    if i > CUVID_PROBE_PACKETS:   # multi-packet latency; cap
                        break
                c.close()
            except Exception:
                logger.info("hevc_cuvid probe failed; falling back", exc_info=True)
    return _CUVID_OK


def cuvid_worker(path, si: int, start: int, count: int | None, q, stop,
                 alloc=None):
    """Decode one fisheye stream via hevc_cuvid, emitting (idx, p010 planes).

    With ``alloc`` (a pinned-slot allocator, see GpuStitcher.input_pool) the
    planes are staged into pinned memory HERE, on the decode thread, so the
    stitch thread gets a true async upload instead of a pageable copy.
    """
    try:
        c, s, rate, tb = _open_stream(path, si, start)
        ctx = av.CodecContext.create("hevc_cuvid", "r")
        ctx.extradata = s.codec_context.extradata

        def payload(fr):
            planes = _frame_p010(fr)
            if alloc is None:
                return planes
            slot = alloc()
            np.copyto(slot.Y, planes[0])
            np.copyto(slot.UV, planes[1])
            return slot

        def emit(fr) -> bool:            # True = reached count, stop decoding
            if fr.pts is None:
                return False
            idx = _frame_index(fr.pts, tb, rate)
            if idx < start:
                return False
            if count is not None and idx >= start + count:
                return True
            return not _queue_put(q, (idx, payload(fr)), stop)

        done = False
        for pkt in c.demux(s):
            if done or stop.is_set():
                break
            if pkt.dts is None:
                continue
            for fr in ctx.decode(pkt):
                if emit(fr):
                    done = True
                    break
        if not done and not stop.is_set():
            for fr in ctx.decode(None):  # drain the decoder
                if emit(fr):
                    break
        c.close()
    except Exception as e:               # surface in the consumer
        _queue_put(q, e, stop)
    finally:
        _queue_put(q, None, stop)
