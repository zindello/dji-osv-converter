#!/usr/bin/env python3
"""Measure this GPU's NVENC/NVDEC engine counts for constants.GPU_ENGINES.

Self-contained: synthesizes its own noise footage, needs no OSV.

Primary signal - ENGINE UTILIZATION under ONE saturating session: NVML
reports encoder/decoder utilization aggregated over all engines, so one
pegged session reads ~100/engines percent (this is exactly how the V100's
3 NVENC engines were discovered: a lone convert() session sat at 33% in
every sample). engines = round(100 / median_util).

Secondary signal - THROUGHPUT SCALING with k parallel sessions, printed as
corroboration: aggregate fps grows until k exceeds the engine count. It is
deliberately NOT used for the verdict; a single underfed session makes it
ambiguous (measured on the V100: 2 decode sessions gave 1.33x on ONE
engine, because one session left the engine ~70% busy).

Prints a ready-to-paste GPU_ENGINES row. Run on every new card BEFORE
adding a row; keep the utilization/scaling output in the commit message.
Sanity-checked on the V100: nvenc=3, nvdec=1, both matching real-convert
utilization logs.

Usage:
    python tools/gpu_engine_probe.py
    python tools/gpu_engine_probe.py --max-sessions 6   # scaling table depth
"""

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction

import av
import numpy as np

# A session must be engine-bound, not submission-bound: fisheye-sized noise
# frames are expensive enough that one feeder thread saturates one engine.
RES = 3840
FPS = Fraction(30000, 1001)
MIN_UTIL = 15            # below this, the workload failed to load the engine


# ---- utilization sampling ---------------------------------------------------

def _nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None, None


def _read_util(kind: str) -> int | None:
    """Instantaneous encoder/decoder utilization %, NVML or nvidia-smi."""
    mod, handle = _nvml()
    if handle is not None:
        fn = (mod.nvmlDeviceGetEncoderUtilization if kind == "encoder"
              else mod.nvmlDeviceGetDecoderUtilization)
        return fn(handle)[0]
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu=utilization.{kind}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def _sample_util(kind: str, stop: threading.Event, out: list):
    while not stop.is_set():
        u = _read_util(kind)
        if u is not None:
            out.append(u)
        time.sleep(0.2)


# ---- workloads --------------------------------------------------------------

def _noise_frames(seed: int) -> list:
    frames = []
    for s in (seed, seed + 1):           # alternate: defeat static-scene skip
        rng = np.random.default_rng(s)
        fr = av.VideoFrame(RES, RES, "p010le")
        y = np.frombuffer(fr.planes[0], np.uint16).reshape(
            fr.planes[0].height, fr.planes[0].line_size // 2)
        y[:, :RES] = rng.integers(64, 940, size=(RES, RES),
                                  dtype=np.uint16) * 64
        np.frombuffer(fr.planes[1], np.uint16)[:] = 512 * 64
        frames.append(fr)
    return frames


def _encoder():
    e = av.CodecContext.create("hevc_nvenc", "w")
    e.width = e.height = RES
    e.pix_fmt = "p010le"
    e.time_base = Fraction(1, int(round(float(FPS))))
    e.framerate = FPS
    e.bit_rate = int(RES * RES * float(FPS) * 0.15)
    e.open()
    return e


def _encode_worker(wid: int, frames: int, done: list):
    """One NVENC session. Own frame objects: pts must not race across threads."""
    enc = _encoder()
    src = _noise_frames(2 * wid + 1)
    for i in range(frames):
        fr = src[i % 2]
        fr.pts = i
        enc.encode(fr)
    enc.encode(None)
    done.append(frames)


def _make_clip(path: str, frames: int):
    out = av.open(path, "w")
    try:
        vs = out.add_stream("hevc_nvenc", rate=FPS)
        vs.pix_fmt = "p010le"
    except Exception:
        vs = out.add_stream("libx265", rate=FPS)
        vs.pix_fmt = "yuv420p10le"
    vs.width = vs.height = RES
    vs.time_base = Fraction(1, int(round(float(FPS))))
    src = _noise_frames(1)
    for i in range(frames):
        fr = src[i % 2]
        fr.pts = i
        fr2 = fr if vs.pix_fmt == "p010le" else fr.reformat(format=vs.pix_fmt)
        for pkt in vs.encode(fr2):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    out.close()


def _decode_worker(wid: int, clip: str, done: list):
    """One cuvid session (wid unused; uniform worker signature)."""
    c = av.open(clip)
    s = c.streams.video[0]
    ctx = av.CodecContext.create("hevc_cuvid", "r")
    ctx.extradata = s.codec_context.extradata
    n = 0
    for pkt in c.demux(s):
        if pkt.dts is None:
            continue
        n += sum(1 for _ in ctx.decode(pkt))
    n += sum(1 for _ in ctx.decode(None))
    c.close()
    done.append(n)


# ---- measurement ------------------------------------------------------------

def _run_k(k: int, worker, wargs, kind: str):
    """k parallel sessions; returns (aggregate_fps, median_util_percent)."""
    done: list = []
    threads = [threading.Thread(target=worker, args=(i, *wargs, done))
               for i in range(k)]
    stop = threading.Event()
    samples: list = []
    sampler = threading.Thread(target=_sample_util, args=(kind, stop, samples))
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    sampler.start()
    for t in threads:
        t.join()
    stop.set()
    sampler.join()
    fps = sum(done) / (time.perf_counter() - t0)
    core = samples[2:-1] if len(samples) > 5 else samples   # drop ramp edges
    util = statistics.median(core) if core else None
    return fps, util


def _infer(kind: str, k_runner, max_sessions: int) -> int:
    fps1, util1 = k_runner(1)
    print(f"  {kind} x1: {fps1:6.1f} fps, util {util1}%", flush=True)
    if util1 is None or util1 < MIN_UTIL:
        print(f"  {kind}: utilization unreadable or engine underfed; "
              f"NOT inferring - fix the workload first", flush=True)
        return 0
    engines = max(1, min(8, round(100 / util1)))
    # corroborate: scaling table + a confirm run at k=engines
    for k in range(2, max_sessions + 1):
        fps, util = k_runner(k)
        note = "  <- confirm" if k == engines else ""
        print(f"  {kind} x{k}: {fps:6.1f} fps, util {util}%{note}", flush=True)
    return engines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sessions", type=int, default=4)
    ap.add_argument("--frames", type=int, default=48,
                    help="frames per session per measurement")
    args = ap.parse_args()

    try:
        import torch
        name = (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None)
    except Exception:
        name = None
    print(f"device: {name or 'unknown (no torch/CUDA)'}", flush=True)

    print("NVENC:", flush=True)
    try:
        _encoder()
        nvenc = _infer(
            "nvenc",
            lambda k: _run_k(k, _encode_worker, (args.frames,), "encoder"),
            args.max_sessions)
    except Exception as e:
        print(f"  hevc_nvenc unavailable ({type(e).__name__}): 0 engines",
              flush=True)
        nvenc = 0

    print("NVDEC:", flush=True)
    td = tempfile.mkdtemp()
    clip = os.path.join(td, "probe.mp4")
    _make_clip(clip, args.frames)
    try:
        probe: list = []
        _decode_worker(0, clip, probe)
        assert probe and probe[0] >= args.frames
        nvdec = _infer(
            "nvdec",
            lambda k: _run_k(k, _decode_worker, (clip,), "decoder"),
            args.max_sessions)
    except Exception as e:
        print(f"  hevc_cuvid unavailable ({type(e).__name__}): cannot probe",
              flush=True)
        nvdec = 0

    key = "<short-name>"
    if name:
        for tok in name.replace("-", " ").split():
            if any(ch.isdigit() for ch in tok) and not tok.upper().endswith("GB"):
                key = tok
                break
    print(f"\nmeasured: nvenc={nvenc} nvdec={nvdec}")
    print("GPU_ENGINES row (verify the key matches only this card):")
    print(f'    "{key}": {{"nvenc": {nvenc}, "nvdec": {nvdec}}},')
    return 0


if __name__ == "__main__":
    sys.exit(main())
