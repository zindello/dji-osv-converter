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

from .constants import (BITRATE_BPP, ENC_CHUNK_SECONDS, PIPELINE_QUEUE_FRAMES,
                        gpu_engines)
from .pipeline import Stitcher360
from .decode import iter_pairs, prefetch, gpu_decode_fmt
from .spherical import inject_spherical
from . import stitch_gpu

logger = logging.getLogger(__name__)


def _default_codec(allow_software: bool = False) -> str:
    """First available hardware HEVC encoder on this machine.

    VideoToolbox (macOS) > NVENC (NVIDIA/CUDA boxes). ``libx265`` is chosen
    only with ``allow_software=True``: an 8K software encode runs at a few
    frames per MINUTE on the target hosts, so a silent fallback (the old
    behaviour) turns a lost NVENC into a days-long job nobody asked for.
    Pass ``codec="libx265"`` explicitly to use it deliberately.
    """
    for name in ("hevc_videotoolbox", "hevc_nvenc"):
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:
            continue
    if allow_software:
        return "libx265"
    raise RuntimeError("no hardware HEVC encoder (hevc_videotoolbox / hevc_nvenc) "
                       "is available; pass codec='libx265' or allow_software=True "
                       "to accept a software encode")


# Colour description fields carried from the source to the output; the OSV
# is tagged BT.709 limited range (prim/trc/matrix 1, range 1) and the output
# was previously untagged (players guessed BT.709 for UHD; Resolve/YouTube
# prefer explicit tags, DJI's own export carries them).
_COLOUR_FIELDS = ("color_range", "color_primaries", "color_trc", "colorspace")


def _source_colour(cc) -> dict:
    """The source codec context's colour description, zeros/unspecified dropped."""
    out = {}
    for f in _COLOUR_FIELDS:
        try:
            v = int(getattr(cc, f))
        except (TypeError, ValueError, AttributeError):
            continue
        # 0 = unspecified range; 2 = unspecified primaries/trc/matrix
        if (f == "color_range" and v != 0) or (f != "color_range" and v not in (0, 2)):
            out[f] = v
    return out


def _apply_colour(cc, tags: dict) -> None:
    for f, v in tags.items():
        try:
            setattr(cc, f, v)
        except (AttributeError, ValueError):
            pass


def _yuv_range(cc) -> str:
    """'limited' or 'full' from the source's declared range (0 = assume limited)."""
    try:
        return "full" if int(cc.color_range) == 2 else "limited"
    except (TypeError, ValueError, AttributeError):
        return "limited"


def encoder_options(codec: str, *, preset: str | None = None, tune: str | None = None,
                    aq: bool | None = None, lookahead: int | None = None,
                    multipass: str | None = None, rc: str | None = None,
                    extra: dict | None = None) -> dict:
    """libav private options for the video encoder. ``None`` = libav default.

    Everything here changes the encoder only, never the stitched frames.
    For NVENC: ``preset`` p1..p7 (p7 slowest/best), ``tune`` hq/ll/ull,
    ``aq`` enables spatial + temporal adaptive quantisation, ``lookahead``
    sets ``rc-lookahead`` frames, ``multipass`` disabled/qres/fullres,
    ``rc`` vbr/cbr/constqp. Which of these pay at 0.25 bpp on an 8K
    panorama can only be measured on the NVIDIA box (see IMPROVEMENTS.md
    for the exact test); the defaults are unchanged until then.
    For software/VideoToolbox encoders only ``extra`` applies.
    """
    opts: dict = {}
    if "nvenc" in codec:
        if preset:
            opts["preset"] = preset
        if tune:
            opts["tune"] = tune
        if aq is not None:
            opts["spatial-aq"] = "1" if aq else "0"
            opts["temporal-aq"] = "1" if aq else "0"
        if lookahead is not None:
            opts["rc-lookahead"] = str(int(lookahead))
        if multipass:
            opts["multipass"] = multipass
        if rc:
            opts["rc"] = rc
    if extra:
        opts.update({k: str(v) for k, v in extra.items()})
    return opts


def _open_output(path: str) -> "av.container.OutputContainer":
    """Open the output as an MP4 regardless of its extension.

    PyAV guesses the container from the extension; a temporary name such as
    ``out.mp4.partial`` (what a worker writes before publishing) would fail
    or mux something else. The converter always produces an MP4, so say so.
    """
    return av.open(path, "w", format="mp4")


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
            bitrate: int | None = None, bpp: float | None = None,
            hardware: bool = True, multiband: bool = False, exposure: bool = False,
            exposure_wb: bool = False, levelling: str = "plain", lsc: bool = False,
            seam_flow: bool | str = "auto",
            audio: bool = True, spherical: bool = True, gpu: bool = True,
            pipeline: bool | None = None, enc_sessions: int | None = None,
            allow_software: bool = False, enc_options: dict | None = None,
            colour_tags: bool = True, enc_inflight_frames: int | None = None,
            single_pass: bool = False,
            progress=None) -> str:
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

    ``seam_flow`` fixes the seam where the two lenses genuinely disagree:
    an inter-lens photometric match (asymmetric lighting leaves a real
    mismatch, worst in dark content), an optical-flow morph (parallax on
    NEAR objects - the lenses sit a few cm apart; a bar ~1 m away steps
    visibly at the seam), and a narrow dynamic seam gated to real parallax.
    The default "auto" detects per frame what is needed and costs only the
    detector (~16 ms) on frames that need nothing; True forces the full
    stack every frame (~+100 ms); False disables (the static blend).

    ``enc_sessions`` (None = auto): number of parallel NVENC encoder sessions.
    A single HEVC session runs on ONE of the GPU's NVENC engines; multi-engine
    parts (e.g. V100: 3) leave the others idle while the session is the
    pipeline's limiter. >1 encodes GOP-aligned chunks round-robin on
    independent sessions and re-serializes the closed-GOP bitstreams at mux
    (B-frames forced off so dts stays monotonic across seams). Auto reads the
    card's engine count from ``constants.GPU_ENGINES`` (unknown cards: 1);
    any count still produces correct output (sessions timeslice the engine).

    ``allow_software``: with ``codec=None`` only hardware encoders are
    auto-selected; a box without one raises instead of silently encoding 8K
    on the CPU. ``enc_options`` are libav encoder options (see
    :func:`encoder_options`); ``bpp`` overrides ``constants.BITRATE_BPP`` for
    the rate-driven hardware encoders (``bitrate`` wins if both are given).
    ``colour_tags`` copies the source's colour range/primaries/transfer/matrix
    onto the output (default on; False restores the untagged output). The
    stitch clamps to the source's declared range (limited by default, full
    when the source says so). ``enc_inflight_frames`` caps the 8K frames held
    across the encoder-session queues (None = one chunk + 2 x the stage
    queue, ~6 GB at 8K p010 - the old bound was per session, i.e. 3x that).

    ``single_pass`` (GPU path): compose the levelling with the lens maps so
    every output pixel is interpolated once instead of twice (sharper, no
    pole singularity at the front/back centres, exact wrap column); the
    seam stack still runs, on band-only warps.

    The output is always muxed as MP4 regardless of the path's extension
    (temporary names such as ``out.mp4.partial`` are fine).
    """
    input_osv, output_mp4 = str(input_osv), str(output_mp4)
    if codec is None:
        codec = _default_codec(allow_software)
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
        # 10-bit YUV in the decoder's cheapest layout: device-resident NVDEC
        # frames ("cuda", zero decode PCIe traffic), NVDEC-native p010, or
        # planar. Probed BEFORE the GpuStitcher touches torch/CUDA - the
        # shared-primary-context hwdevice can't be created afterwards.
        gpu_fmt = gpu_decode_fmt(hardware, input_osv)

    src = av.open(input_osv)
    vid = next(s for s in src.streams.video if s.codec_context.name != "mjpeg")
    src_colour = _source_colour(vid.codec_context)
    yuv_range = _yuv_range(vid.codec_context)

    if use_gpu:
        maps, _ = stitcher.maps_for()
        gpu_stitch = stitch_gpu.GpuStitcher(
            maps, band_px=stitcher.seam_overlap_ratio * width,
            width=width, height=height,
            lsc_gain=stitcher.lsc_gains() if lsc else None,
            rs_tables=stitcher.rs_tables() if use_rs else None,
            fisheye_size=(stitcher.calibs[0].width, stitcher.calibs[0].height),
            input_fmt=gpu_fmt, yuv_range=yuv_range,
            single_pass=single_pass)
    fps = Fraction(vid.average_rate).limit_denominator(90000)
    total = vid.frames or 0
    n_frames = count if count is not None else (total - start if total else None)
    a_src = src.streams.audio[0] if (audio and src.streams.audio) else None

    out = _open_output(output_mp4)
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
        vs.bit_rate = int(width * height * float(fps) * (bpp or BITRATE_BPP))
    else:
        vs.options = {"crf": "18"}
    enc_opts = dict(enc_options or {})
    if enc_opts:
        vs.options = {**(vs.options or {}), **enc_opts}
    if colour_tags:
        _apply_colour(vs.codec_context, src_colour)

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
            return fn(f0, f1, R, exposure=exposure, wb=exposure_wb, rs=rs,
                      seam_flow=seam_flow)            # (Y, UV) p010, or a handle
        return stitcher.stitch(f0, f1, frame=idx, exposure=exposure,
                               multiband=multiband, flow=bool(seam_flow))

    def _p010_frame(planes) -> "av.VideoFrame":
        Y, UV = planes
        fr = av.VideoFrame(width, height, "p010le")
        torch = stitch_gpu.torch
        for plane, data in zip(fr.planes, (Y, UV)):
            buf = np.frombuffer(plane, np.uint16).reshape(plane.height, plane.line_size // 2)
            if torch is not None:
                # torch's memcpy releases the GIL; numpy's does not, and an
                # 88 MB/frame GIL hold here stalls the stitch thread's kernel
                # launches (measured as the pipeline's limiter on the V100).
                src = data if isinstance(data, torch.Tensor) else torch.from_numpy(data)
                torch.from_numpy(buf)[:, :src.shape[1]].copy_(src)
            else:
                buf[:, :data.shape[1]] = data
        return fr

    def _build_frame(result) -> "av.VideoFrame":
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
            return fr
        return av.VideoFrame.from_ndarray(np.ascontiguousarray(result),
                                          format="bgr24").reformat(format=vs.pix_fmt)

    def _encode(pts, result):
        fr = _build_frame(result)
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
    multi_enc = False
    if pipeline:
        enc_q: "queue.Queue" = queue.Queue(maxsize=PIPELINE_QUEUE_FRAMES)
        enc_done, enc_err = object(), []

        # On CUDA the frame build (event wait + pinned->AVFrame copy,
        # ~18 ms) and the encode call (~35 ms - libav nvenc re-uploads the
        # frame internally) are each big enough to deserve their own stage:
        # stitch -> build -> encode+mux on three threads (measured 59 -> ~40
        # ms/frame). Elsewhere the two-thread layout stays (the M3 numbers
        # were recorded with it, and its encode path is the lighter one).
        gpu_async = use_gpu and gpu_stitch.async_d2h
        n_sess = enc_sessions
        if n_sess is None:
            if gpu_async and "nvenc" in codec:
                # Per-card engine count (constants.GPU_ENGINES) - one HEVC
                # session saturates one NVENC engine.
                name = stitch_gpu.torch.cuda.get_device_name(gpu_stitch.dev)
                n_sess = max(1, gpu_engines(name)["nvenc"])
            else:
                n_sess = 1
        multi_enc = gpu_async and n_sess > 1 and "nvenc" in codec
        stages = []
        if multi_enc:
            # Segment-parallel NVENC: one HEVC session occupies ONE NVENC
            # engine (the V100 has three; a lone session pegs its engine at
            # ~59 ms/frame on real content while the others idle). Frames go
            # to n_sess independent sessions in GOP-sized chunks; each chunk
            # is a closed GOP (fresh session => leading IDR, B-frames off so
            # dts stays monotonic across seams) and the muxer re-serializes
            # chunks in order. Sessions with identical settings emit
            # identical SPS/PPS, so one extradata serves the whole track.
            chunk = max(1, int(round(ENC_CHUNK_SECONDS * float(fps))))
            sess_qs = [queue.Queue(maxsize=chunk + 2 * PIPELINE_QUEUE_FRAMES)
                       for _ in range(n_sess)]
            mux_q: "queue.Queue" = queue.Queue()
            # Total 8K frames alive across ALL session queues. Steady state
            # is ~1 deep per session; this bounds the worst case (a stalled
            # encoder) to one chunk's worth instead of one per session
            # (3 x 68 x 88 MB = 18 GB before). Acquired by the builder,
            # released by the session that encoded the frame; a dead stage
            # is noticed through enc_err so the builder cannot hang on it.
            inflight = threading.Semaphore(
                enc_inflight_frames or (chunk + 2 * PIPELINE_QUEUE_FRAMES))

            def _builder():
                try:
                    while True:
                        item = enc_q.get()
                        if item is enc_done:
                            for q in sess_qs:
                                q.put(enc_done)
                            return
                        pts, result = item
                        while not inflight.acquire(timeout=0.5):
                            if enc_err:
                                raise enc_err[0]
                        fr = _build_frame(result)
                        fr.pts = pts
                        sess_qs[(pts // chunk) % n_sess].put(fr)
                except Exception as e:             # noqa: BLE001
                    enc_err.append(e)
                    for q in sess_qs:
                        q.put(enc_done)

            def _new_session():
                ctx = av.CodecContext.create(codec, "w")
                ctx.width, ctx.height = width, height
                ctx.pix_fmt = vs.pix_fmt
                ctx.time_base = vs.time_base
                ctx.framerate = fps
                ctx.bit_rate = vs.bit_rate
                ctx.options = {"bf": "0", "forced-idr": "1", **enc_opts}
                if colour_tags:
                    _apply_colour(ctx, src_colour)
                ctx.flags |= av.codec.context.Flags.global_header
                return ctx

            def _session(k):
                # ONE persistent session per worker - NVENC session creation
                # costs enough that churning one per chunk measurably LOSES
                # (~20 ms/frame on the short benchmark). Chunk starts get a
                # forced IDR instead, which keeps every chunk closed; with
                # bf=0 the packet stream is in pts order, so a packet from a
                # later chunk proves the earlier ones are complete.
                pending: dict[int, list] = {}

                def emit_before(cid):
                    for c in sorted(c for c in pending if c < cid):
                        mux_q.put((c, pending.pop(c)))

                try:
                    ctx = _new_session()
                    while True:
                        fr = sess_qs[k].get()
                        if fr is enc_done:
                            for pkt in ctx.encode(None):
                                pending.setdefault(pkt.pts // chunk, []).append(pkt)
                            emit_before(float("inf"))
                            mux_q.put((enc_done, bytes(ctx.extradata or b"")))
                            return
                        if fr.pts % chunk == 0:
                            # chunk start -> forced IDR (closed chunk)
                            fr.pict_type = av.video.frame.PictureType.I
                        try:
                            pkts = list(ctx.encode(fr))
                        finally:
                            del fr
                            inflight.release()
                        for pkt in pkts:
                            cid = pkt.pts // chunk
                            pending.setdefault(cid, []).append(pkt)
                            emit_before(cid)
                except Exception as e:             # noqa: BLE001
                    enc_err.append(e)
                    mux_q.put((enc_done, b""))

            def _muxer():
                try:
                    expected, buf, ends, xset = 0, {}, 0, False

                    def drain():
                        nonlocal expected
                        while expected in buf:
                            for pkt in buf.pop(expected):
                                pkt.stream = vs
                                out.mux(pkt)
                            expected += 1

                    while ends < n_sess:
                        cid, payload = mux_q.get()
                        if cid is enc_done:
                            ends += 1
                            if not xset and payload:
                                vs.codec_context.extradata = payload
                                xset = True
                            continue
                        buf[cid] = payload
                        drain()
                    drain()
                except Exception as e:             # noqa: BLE001
                    enc_err.append(e)

            stages = [_builder] + \
                [(lambda k=k: _session(k)) for k in range(n_sess)] + [_muxer]
        elif gpu_async:
            frame_q: "queue.Queue" = queue.Queue(maxsize=PIPELINE_QUEUE_FRAMES)

            def _builder():
                try:
                    while True:
                        item = enc_q.get()
                        if item is enc_done:
                            frame_q.put(enc_done)
                            return
                        pts, result = item
                        fr = _build_frame(result)
                        fr.pts = pts
                        frame_q.put(fr)
                except Exception as e:             # noqa: BLE001
                    enc_err.append(e)
                    frame_q.put(enc_done)

            def _encoder():
                try:
                    while True:
                        fr = frame_q.get()
                        if fr is enc_done:
                            return
                        for pkt in vs.encode(fr):
                            out.mux(pkt)
                except Exception as e:             # noqa: BLE001
                    enc_err.append(e)

            stages = [_builder, _encoder]
        else:
            def _encoder():
                try:
                    while True:
                        item = enc_q.get()
                        if item is enc_done:
                            return
                        _encode(*item)
                except Exception as e:             # noqa: BLE001
                    enc_err.append(e)

            stages = [_encoder]

        threads = [threading.Thread(target=fn, daemon=True) for fn in stages]
        for t in threads:
            t.start()
        # On CUDA the downstream threads receive async handles: the packed
        # planes travel device->pinned-host on a side stream while the
        # default stream is already stitching the next frame (MPS cannot -
        # one kernel queue - so it keeps the synchronous fetch).
        pool = gpu_stitch.input_pool() if use_gpu else None

        def _feed(item):
            # Bounded put that keeps checking for a dead consumer: a stage
            # thread that died leaves its queue full forever, and a plain
            # put() would deadlock the producer instead of surfacing the
            # error (seen live when a builder exception killed the run).
            while True:
                if enc_err:
                    raise enc_err[0]
                try:
                    enc_q.put(item, timeout=0.5)
                    return
                except queue.Full:
                    continue

        for idx, f0, f1 in prefetch(iter_pairs(input_osv, start=start, count=count,
                                               fmt=dec_fmt, hardware=hardware,
                                               alloc=pool.alloc if pool else None),
                                    size=PIPELINE_QUEUE_FRAMES):
            _feed((done, _stitch(idx, f0, f1, gpu_async=gpu_async)))
            done += 1
            if progress:
                progress(done, n_frames)
        enc_q.put(enc_done)
        for t in threads:
            t.join()
        if enc_err:
            raise enc_err[0]
    else:
        for idx, f0, f1 in iter_pairs(input_osv, start=start, count=count,
                                      fmt=dec_fmt, hardware=hardware):
            _encode(done, _stitch(idx, f0, f1))
            done += 1
            if progress:
                progress(done, n_frames)

    if not multi_enc:
        for pkt in vs.encode():   # flush encoder (unused in multi-session)
            out.mux(pkt)
    out.close()
    src.close()

    if spherical:
        inject_spherical(output_mp4)
    logger.info("wrote %s: %d frames %dx%d %s %s%s", output_mp4, done, width, height,
                codec, str(gpu_stitch.dev) if use_gpu else "CPU",
                " +audio" if a_out is not None else "")
    return output_mp4


__all__ = ["convert", "encoder_options"]
