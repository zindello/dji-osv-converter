"""Convert a DJI Osmo 360 ``.OSV`` to an equirectangular 360 MP4.

Ties the pieces together: stream-decode both fisheyes (VideoToolbox when
available), stitch each frame with mount-corrected levelling, encode to HEVC
in-process (hardware ``hevc_videotoolbox`` by default, ``libx265`` as a
portable fallback), copy the audio without re-encoding, and tag the result as
360 (spherical metadata).

    from dji_osv_converter.video import convert
    convert("CAM_....OSV", "out.mp4", count=300)     # first 300 frames

Everything is derived from the OSV; the ffmpeg binary is never invoked.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from .pipeline import Stitcher360
from .decode import iter_pairs
from .spherical import inject_spherical
from . import stitch_mps

logger = logging.getLogger(__name__)


def _pick_pix_fmt(codec: str, bit10: bool) -> str:
    if "videotoolbox" in codec:
        return "p010le" if bit10 else "nv12"
    return "yuv420p10le" if bit10 else "yuv420p"


def convert(input_osv: str | Path, output_mp4: str | Path, *,
            width: int = 7680, height: int = 3840,
            start: int = 0, count: int | None = None,
            codec: str = "hevc_videotoolbox", bit10: bool = True,
            bitrate: int | None = None,
            hardware: bool = True, multiband: bool = False, exposure: bool = True,
            audio: bool = True, spherical: bool = True, gpu: bool = True,
            progress=None) -> str:
    """Stitch an OSV to an equirectangular 360 MP4 and return the output path.

    ``count`` limits the number of frames (None = whole clip); ``start`` skips
    to a frame. ``codec`` is any PyAV video encoder ("hevc_videotoolbox",
    "libx265", "hevc_nvenc", ...). ``progress(done, total)`` is called per
    frame if given.

    ``multiband`` defaults off for video: after the exposure match the linear
    feathered blend is visually indistinguishable from multi-band (peak diff
    ~1/255) and ~1s/frame cheaper. Turn it on for the last ounce of seam
    quality on stills.
    """
    input_osv, output_mp4 = str(input_osv), str(output_mp4)
    stitcher = Stitcher360.from_file(input_osv, width=width, height=height)

    # Metal (PyTorch MPS) runs the whole stitch on the Apple GPU - ~5x the CPU
    # linear path. It uses linear blending (multiband would force the CPU path).
    use_mps = gpu and not multiband and stitch_mps.available()
    if use_mps:
        maps, _ = stitcher.maps_for()
        mps = stitch_mps.MPSStitcher(
            maps, band_px=stitcher.seam_overlap_ratio * width,
            width=width, height=height)

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
    vs.time_base = Fraction(1, fps.numerator // fps.denominator if fps.denominator == 1
                            else int(round(float(fps))))
    if bitrate:
        vs.bit_rate = bitrate
    elif "videotoolbox" in codec:
        vs.bit_rate = int(width * height * float(fps) * 0.10)  # ~0.1 bpp
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

    done = 0
    for idx, f0, f1 in iter_pairs(input_osv, start=start, count=count,
                                  bit16=False, hardware=hardware):
        if use_mps:
            pano = mps.stitch(f0, f1, stitcher.levelling_for(idx), exposure=exposure)
        else:
            pano = stitcher.stitch(f0, f1, frame=idx,
                                   exposure=exposure, multiband=multiband)
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(pano), format="bgr24")
        frame = frame.reformat(format=vs.pix_fmt)
        frame.pts = done
        for pkt in vs.encode(frame):
            out.mux(pkt)
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
                codec, "MPS" if use_mps else "CPU",
                " +audio" if a_out is not None else "")
    return output_mp4


__all__ = ["convert"]
