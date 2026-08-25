"""Convert a DJI Osmo 360 ``.OSV`` to an equirectangular 360 MP4.

Ties the pieces together: stream-decode both fisheyes (VideoToolbox when
available), stitch each frame with mount-corrected levelling, encode to HEVC
in-process (auto-selecting the best HEVC encoder present: VideoToolbox on
macOS, NVENC on CUDA boxes, ``libx265`` anywhere), copy the audio without
re-encoding, and tag the result as 360 (spherical metadata).

    from dji_osv_converter.video import convert
    convert("CAM_....OSV", "out.mp4", count=300)     # first 300 frames

Everything is derived from the OSV; the ffmpeg binary is never invoked.
"""

from __future__ import annotations

import logging
import queue
import threading
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from .constants import BITRATE_BPP
from .pipeline import Stitcher360
from .decode import iter_pairs, prefetch, gpu_decode_fmt
from .spherical import inject_spherical
from . import stitch_gpu

logger = logging.getLogger(__name__)


def _default_codec() -> str:
    """First available HEVC encoder on this machine.

    VideoToolbox (macOS) > NVENC (NVIDIA/CUDA boxes) > libx265 (software,
    everywhere). Keeps ``convert()`` runnable unmodified across platforms.
    """
    for name in ("hevc_videotoolbox", "hevc_nvenc", "libx265"):
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:
            continue
    return "libx265"


def _pick_pix_fmt(codec: str, bit10: bool) -> str:
    # Hardware encoders take the packed 10-bit layout (which is also exactly
    # what GpuStitcher emits); software encoders take planar.
    if "videotoolbox" in codec or "nvenc" in codec:
        return "p010le" if bit10 else "nv12"
    return "yuv420p10le" if bit10 else "yuv420p"


def convert(input_osv: str | Path, output_mp4: str | Path, *,
            width: int = 7680, height: int = 3840,
            start: int = 0, count: int | None = None,
            codec: str | None = None, bit10: bool = True,
            bitrate: int | None = None,
            hardware: bool = True, multiband: bool = False, exposure: bool = False,
            exposure_wb: bool = False, levelling: str = "plain", lsc: bool = False,
            audio: bool = True, spherical: bool = True, gpu: bool = True,
            pipeline: bool | None = None, progress=None) -> str:
    """Stitch an OSV to an equirectangular 360 MP4 and return the output path.

    ``count`` limits the number of frames (None = whole clip); ``start`` skips
    to a frame. ``codec`` is any PyAV video encoder ("hevc_videotoolbox",
    "libx265", "hevc_nvenc", ...); None (default) auto-selects the best HEVC
    encoder present on this machine. ``progress(done, total)`` is called per
    frame if given.

    ``exposure`` (inter-lens exposure match) and ``lsc`` (factory devignette)
    both default OFF: the OSV fisheyes are already ISP-processed in-camera
    (devignetted AND exposure-matched), so applying our own corrections on top
    DOUBLE-corrects and paints a bright/dark arc across the overlap band -
    clearly visible against sky, absent from DJI's own export. Measured: the
    true delivered vignette is ~flat (0-8%) vs the factory curve's 24%, and the
    exposure match pushes the inter-lens sky step from +1.6 to +28 (10-bit).
    Leaving both off matches DJI and also drops ~60 ms/frame. Re-enable only for
    footage that is NOT already ISP-corrected (a different camera/source).

    ``multiband`` defaults off for video: the linear feathered blend is visually
    indistinguishable from multi-band on this (aligned, ISP-matched) footage and
    ~1s/frame cheaper. Turn it on for the last ounce of seam quality on stills.
    """
    input_osv, output_mp4 = str(input_osv), str(output_mp4)
    if codec is None:
        codec = _default_codec()
    stitcher = Stitcher360.from_file(input_osv, width=width, height=height,
                                     levelling=levelling)

    # The GPU (CUDA or Apple MPS) runs the whole stitch on-device - several x
    # the CPU linear path. It uses linear blending (multiband forces the CPU
    # path). Falls back to CPU if torch/GPU is unavailable.
    use_gpu = gpu and not multiband and stitch_gpu.available()
    # Rolling-shutter correction is OFF: what looked like readout jello was the
    # old rocksteady composition applying its correction about a tumbling axis.
    # With the fixed bridge-conjugated rocksteady formula the rendered jitter is 0.027
    # deg/frame calm / 0.071 moving (DJI: 0.046 / 0.118) and every RS
    # configuration measurably WORSENS it (0.06-0.15). The full RS pipeline
    # (tau/Jacobian tables, high-rate track, per-pixel grid shift) remains in
    # place behind this flag for future non-DJI sources.
    use_rs = False
    if use_gpu:
        # 10-bit YUV in the decoder's cheapest layout: planar, or NVDEC's
        # native p010 (skips a ~70 ms/frame swscale; unpacked on-device).
        gpu_fmt = gpu_decode_fmt(hardware)
        maps, _ = stitcher.maps_for()
        gpu_stitch = stitch_gpu.GpuStitcher(
            maps, band_px=stitcher.seam_overlap_ratio * width,
            width=width, height=height,
            lsc_gain=stitcher.lsc_gains() if lsc else None,
            rs_tables=stitcher.rs_tables() if use_rs else None,
            fisheye_size=(stitcher.calibs[0].width, stitcher.calibs[0].height),
            input_fmt=gpu_fmt)

    src = av.open(input_osv)
    vid = next(s for s in src.streams.video if s.codec_context.name != "mjpeg")
    fps = Fraction(vid.average_rate).limit_denominator(90000)
    total = vid.frames or 0
    n_frames = count if count is not None else (total - start if total else None)
    a_src = src.streams.audio[0] if (audio and src.streams.audio) else None

    out = av.open(output_mp4, "w")
    vs = out.add_stream(codec, rate=fps)
    vs.width, vs.height = width, height
    vs.pix_fmt = _pick_pix_fmt(codec, bit10)
    # One tick per frame at the TRUE rate: 29.97 must not collapse to 1/30 or
    # the video plays 1.001x fast and drifts against its audio.
    vs.time_base = Fraction(fps.denominator, fps.numerator)
    # Tag HEVC as hvc1 (parameter sets in the sample description) rather than the
    # libav default hev1 (in-band): macOS QuickLook/Finder/QuickTime only decode
    # hvc1, so hev1 gives no thumbnail and won't play there. DJI uses hvc1.
    if "hevc" in codec or "h265" in codec:
        try:
            vs.codec_tag = "hvc1"
        except (AttributeError, ValueError):
            vs.codec_context.codec_tag = int.from_bytes(b"hvc1", "little")
    if bitrate:
        vs.bit_rate = bitrate
    elif "videotoolbox" in codec or "nvenc" in codec:
        # Hardware encoders are rate-driven, not quality-driven.
        vs.bit_rate = int(width * height * float(fps) * BITRATE_BPP)
    else:
        vs.options = {"crf": "18"}

    a_out = out.add_stream_from_template(a_src) if a_src is not None else None

    # Audio first (small); libav interleaves it with the video that follows.
    if a_out is not None:
        t0, t1 = start / float(fps), (start + (n_frames or 0)) / float(fps)
        for pkt in src.demux(a_src):
            if pkt.dts is None:
                continue
            t = float(pkt.pts * a_src.time_base) if pkt.pts is not None else 0.0
            if t < t0:
                continue
            if n_frames is not None and t >= t1:
                break
            pkt.pts = int(round((t - t0) / a_src.time_base))
            pkt.dts = pkt.pts
            pkt.stream = a_out
            out.mux(pkt)

    # The GPU path stays 10-bit YUV end to end: decode in the platform's
    # cheapest 10-bit layout (planar, or NVDEC-native p010), stitch on-device,
    # emit p010le planes straight to the encoder - no CPU colour conversion,
    # no 8-bit truncation. The CPU path stays 8-bit BGR.
    dec_fmt = gpu_fmt if use_gpu else "bgr24"

    def _stitch(idx, f0, f1, gpu_async=False):
        R = stitcher.levelling_for(idx)
        if use_gpu:
            rs = stitcher.rs_args_for(idx) if use_rs else None
            fn = gpu_stitch.stitch_yuv_async if gpu_async else gpu_stitch.stitch_yuv
            return fn(f0, f1, R, exposure=exposure,
                      wb=exposure_wb, rs=rs)          # (Y, UV) p010, or a handle
        return stitcher.stitch(f0, f1, frame=idx, exposure=exposure, multiband=multiband)

    def _p010_frame(planes) -> "av.VideoFrame":
        Y, UV = planes
        fr = av.VideoFrame(width, height, "p010le")
        for plane, data in zip(fr.planes, (Y, UV)):
            buf = np.frombuffer(plane, np.uint16).reshape(plane.height, plane.line_size // 2)
            buf[:, :data.shape[1]] = data
        return fr

    def _encode(pts, result):
        if use_gpu:
            if isinstance(result, stitch_gpu._AsyncPlanes):
                # blocks until the side-stream D2H lands, then copies out of
                # the pinned slot (the copy into the AVFrame happens anyway)
                fr = _p010_frame(result.planes())
                result.release()
            else:
                fr = _p010_frame(result)
            if vs.pix_fmt != "p010le":
                # Software encoders (libx265) take planar yuv420p10le; the
                # p010le->planar repack is lossless (same 10-bit samples).
                fr = fr.reformat(format=vs.pix_fmt)
        else:
            fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(result),
                                            format="bgr24").reformat(format=vs.pix_fmt)
        fr.pts = pts
        for pkt in vs.encode(fr):
            out.mux(pkt)

    # Overlap decode / stitch / encode on separate threads. Pays whenever the
    # stitch runs on a GPU: decode (~53 ms, VideoToolbox/libav) then hides
    # under the stitch. Measured on M3/MPS: 104 -> 73 ms/frame (this flipped
    # from a loss to a win once the stitch dropped under decode+encode).
    # On the CPU path every concurrency variant measured is a REGRESSION -
    # stage overlap 912 -> 1078 ms, frame-parallel stitch threads 0.79x at 2 -
    # because numpy/OpenCV already saturate the cores; CPU stays serial.
    # (Also measured: moving the D2H fetch to the encoder thread on MPS is a
    # loss, 63 -> 74 ms - cross-thread .cpu() serializes on the kernel queue.)
    if pipeline is None:
        pipeline = use_gpu

    done = 0
    if pipeline:
        enc_q: "queue.Queue" = queue.Queue(maxsize=4)
        enc_done, enc_err = object(), []

        def _encoder():
            try:
                while True:
                    item = enc_q.get()
                    if item is enc_done:
                        return
                    _encode(*item)
            except Exception as e:                 # noqa: BLE001
                enc_err.append(e)

        enc_thread = threading.Thread(target=_encoder, daemon=True)
        enc_thread.start()
        # On CUDA the encoder thread receives async handles: the packed
        # planes travel device->pinned-host on a side stream while the
        # default stream is already stitching the next frame (MPS cannot -
        # one kernel queue - so it keeps the synchronous fetch).
        gpu_async = use_gpu and gpu_stitch.async_d2h
        for idx, f0, f1 in prefetch(iter_pairs(input_osv, start=start, count=count,
                                               fmt=dec_fmt, hardware=hardware), size=4):
            enc_q.put((done, _stitch(idx, f0, f1, gpu_async=gpu_async)))
            done += 1
            if progress:
                progress(done, n_frames)
            if enc_err:
                raise enc_err[0]
        enc_q.put(enc_done)
        enc_thread.join()
        if enc_err:
            raise enc_err[0]
    else:
        for idx, f0, f1 in iter_pairs(input_osv, start=start, count=count,
                                      fmt=dec_fmt, hardware=hardware):
            _encode(done, _stitch(idx, f0, f1))
            done += 1
            if progress:
                progress(done, n_frames)

    for pkt in vs.encode():   # flush encoder
        out.mux(pkt)
    out.close()
    src.close()

    if spherical:
        inject_spherical(output_mp4)
    logger.info("wrote %s: %d frames %dx%d %s %s%s", output_mp4, done, width, height,
                codec, str(gpu_stitch.dev) if use_gpu else "CPU",
                " +audio" if a_out is not None else "")
    return output_mp4


__all__ = ["convert"]
