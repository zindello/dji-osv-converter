#!/usr/bin/env python3
"""Benchmark the conversion on the real NVIDIA card and write a Markdown report.

Written for the production box (Tesla V100, CUDA 12.x userspace, PyAV with
``hevc_cuvid``/``hevc_nvenc``), where nothing in PERFORMANCE.md could be
measured. One clip, a few hundred frames, several variants; each variant
runs in its own process (fresh peak-memory counters, and the device decode
has to be probed before torch starts CUDA) while this process samples NVML
every 0.5 s.

    python tools/v100_bench.py /path/to/CLIP.OSV --report /path/to/report.md
    python tools/v100_bench.py CLIP.OSV --report report.md --frames 450 --all
    python tools/v100_bench.py CLIP.OSV --report report.md --variants baseline new
    python tools/v100_bench.py --code-id      # which source this copy is

Variants (the first three are the comparison PERFORMANCE.md asks for):

  baseline       production as it ran at ee1e170: the seam stack on the CPU
  new            the new default (the seam stack on the card)
  new_seam_off   the new default with seam_flow off: the stitch without seam work
  new_sync       ``new`` with the profile in sync mode: true GPU ms per stage
                 (the card is synchronised around every stage, so its fps is
                 not the pipeline's)
  baseline_sync  the same for the baseline
  ref_static     plain levelling, two-pass, seam off: the layout that measured
                 28 fps on this card in August; shows whether the box still does

and with ``--all`` (or named in ``--variants``):

  new_dp16       dynamic-seam block of 16 columns instead of 64 on CUDA
  new_corrected  seam_morph="corrected" (opt-in look change)
  new_p010       decode through hevc_cuvid to system memory instead of to the device
  new_enc1       one NVENC session instead of three

Nothing falls back quietly. Before any variant runs, a preflight makes every
check it can and the run stops with a non-zero exit that names **each** check
that failed, not just the first: the clip, NVML, the report and output
folders, the variants' arguments against ``convert()``, NVDEC on this clip,
CUDA and a kernel on the card, the ``hevc_nvenc`` encoder, and 8K 10-bit
frames through it. A variant then fails if its profile shows a decode path
other than NVDEC, an encoder other than ``hevc_nvenc`` or a device other
than CUDA, if a harness switch it asked for was never reached, if NVML gave
no sample or saw no decode or encode activity, or if it runs past
``--timeout``. Every output file is written under ``--out-dir`` and removed
again (that one file, by its path) unless ``--keep`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_S = 0.5
PREFLIGHT_TIMEOUT_S = 900.0       # the probes take seconds; this only bounds a hang
VARIANT_TIMEOUT_S = 1800.0        # 450 frames at 4.5 fps plus set-up is under 200 s

# Josh's production settings (offcuts' 360 converter); every variant starts here.
PRODUCTION = dict(levelling="rocksteady_horizon", width=7680, height=3840, bit10=True,
                  single_pass=True, lens_interp="bilinear", seam_flow="auto",
                  allow_software=False, codec="hevc_nvenc", hardware=True)

# name -> (description, convert() overrides, harness switches)
VARIANTS = {
    "baseline": ("ee1e170 path: seam stack on the CPU", {"seam_engine": "cpu"}, {}),
    "new": ("new default: seam stack on the card", {}, {}),
    "new_seam_off": ("new default, seam_flow off", {"seam_flow": False}, {}),
    "new_sync": ("new default, sync-mode profile (GPU ms per stage; fps not comparable)",
                 {"profile": "sync"}, {}),
    "baseline_sync": ("ee1e170 path, sync-mode profile (fps not comparable)",
                      {"seam_engine": "cpu", "profile": "sync"}, {}),
    "ref_static": ("plain levelling, two-pass, seam off (the 28 fps layout of August)",
                   {"levelling": "plain", "single_pass": False, "seam_flow": False}, {}),
    "new_dp16": ("new default, dynamic-seam block 16", {}, {"dp_block": 16}),
    "new_corrected": ("new default, seam_morph=corrected (opt-in look change)",
                      {"seam_morph": "corrected"}, {}),
    "new_p010": ("new default, decode via hevc_cuvid to system memory", {},
                 {"decode": "p010le"}),
    "new_enc1": ("new default, one NVENC session", {"enc_sessions": 1}, {}),
}
DEFAULT = ["baseline", "new", "new_seam_off", "new_sync", "baseline_sync", "ref_static"]
SWITCHES = ("decode", "dp_block")

# What the harness reaches into the package for, by name: checked in the
# preflight, so a rename is reported before the card's time is spent.
PACKAGE_NAMES = (("video", "convert"), ("video", "gpu_decode_fmt"),
                 ("video", "_p010_av_frame"), ("video", "_pick_pix_fmt"),
                 ("decode", "gpu_decode_fmt"), ("nvdec", "cuvid_usable"),
                 ("seam_gpu", "SeamGpu"), ("profile", "last_profile"),
                 ("profile", "format_table"), ("constants", "BITRATE_BPP"))

# The pynvml (nvidia-ml-py) names the sampler calls.
NVML_CALLS = ("nvmlInit", "nvmlDeviceGetCount", "nvmlDeviceGetHandleByIndex",
              "nvmlDeviceGetName", "nvmlDeviceGetUUID", "nvmlSystemGetDriverVersion",
              "nvmlDeviceGetMemoryInfo", "nvmlDeviceGetUtilizationRates",
              "nvmlDeviceGetEncoderUtilization", "nvmlDeviceGetDecoderUtilization",
              "NVMLError")


class BenchError(RuntimeError):
    """Something the benchmark needs is missing: reported, never worked around."""


def code_id() -> str:
    """Short digest of the source this copy runs (the package and this file).

    The harness is often unpacked into an image with no git checkout; the
    digest says which source that was. Compare it with ``--code-id`` run in
    the repository at the commit that was meant to ship.
    """
    h = hashlib.sha256()
    files = sorted((ROOT / "dji_osv_converter").glob("*.py")) + [Path(__file__).resolve()]
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def convert_kwargs(name: str, frames: int, start: int, production: dict | None = None) -> dict:
    """The arguments ``convert()`` gets for a variant."""
    kw = dict(PRODUCTION if production is None else production,
              start=start, count=frames, profile=True)
    kw.update(VARIANTS[name][1])
    return kw


def static_problems(names: list[str]) -> list[str]:
    """What is wrong between this harness and the package, without running anything."""
    import importlib
    bad, mods = [], {}
    for mod, attr in PACKAGE_NAMES:
        if mod not in mods:
            try:
                mods[mod] = importlib.import_module(f"dji_osv_converter.{mod}")
            except Exception as e:
                mods[mod] = None
                bad.append(f"dji_osv_converter.{mod} does not import: {type(e).__name__}: {e}")
        if mods[mod] is not None and not hasattr(mods[mod], attr):
            bad.append(f"dji_osv_converter.{mod}.{attr} does not exist (the harness uses it)")
    conv = getattr(mods.get("video"), "convert", None)
    params = inspect.signature(conv).parameters if conv is not None else None
    for name in names:
        if params is not None:
            unknown = sorted(set(convert_kwargs(name, 1, 0)) - set(params))
            if unknown:
                bad.append(f"variant {name}: convert() takes no argument named "
                           f"{', '.join(unknown)}")
        unknown = sorted(set(VARIANTS[name][2]) - set(SWITCHES))
        if unknown:
            bad.append(f"variant {name}: the harness has no switch named {', '.join(unknown)}")
    return bad


# ---------------------------------------------------------------- child side

def _need_cuda():
    import torch
    if not torch.cuda.is_available():
        raise BenchError("CUDA is not available to torch (torch.cuda.is_available() is False)")
    return torch


def card_info(torch) -> dict:
    """What torch says the card is."""
    major, minor = torch.cuda.get_device_capability(0)
    return {"gpu": torch.cuda.get_device_name(0), "capability": f"{major}.{minor}",
            "total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 20),
            "arch_list": torch.cuda.get_arch_list()}


def kernel_check(torch, device: str = "cuda", card: dict | None = None) -> None:
    """A kernel runs on the device and gives the right answer."""
    try:
        x = torch.arange(1024, device=device, dtype=torch.float32)
        got = float((x * 2.0).sum().item())
    except Exception as e:
        why = ""
        if card and card.get("arch_list"):
            arch = "sm_" + card["capability"].replace(".", "")
            if arch not in card["arch_list"]:
                why = (f" (this torch build has no kernels for {arch}; it was built for "
                       f"{' '.join(card['arch_list'])})")
        raise BenchError(f"a kernel did not run on {device}: "
                         f"{type(e).__name__}: {e}{why}") from e
    if got != 1023.0 * 1024.0:
        raise BenchError(f"a kernel on {device} gave a wrong answer "
                         f"({got}, not {1023.0 * 1024.0})")


def _decode_probe(clip: str, want: str | None = None) -> str:
    """The NVDEC arrangement for this clip, probed before torch starts CUDA.

    ``want="p010le"`` (the ``new_p010`` variant) asks only for hevc_cuvid and
    leaves the device-frame probe, which keeps a decoder open, alone.
    """
    from dji_osv_converter import nvdec
    if want == "p010le":
        if nvdec.cuvid_usable(clip):
            return want
        raise BenchError(f"NVDEC (hevc_cuvid to system memory) cannot decode {clip}")
    from dji_osv_converter.decode import gpu_decode_fmt
    fmt = gpu_decode_fmt(True, clip)
    if fmt == "cuda":
        return fmt
    if fmt == "p010le" and nvdec.cuvid_usable(clip):
        return fmt
    raise BenchError(f"NVDEC cannot decode {clip}: neither device frames nor hevc_cuvid work "
                     f"(gpu_decode_fmt gave {fmt!r})")


def p010_test_frame(w: int, h: int, level: int = 512):
    """A flat grey p010le frame, built by the converter's own frame builder.

    (Not ``av.VideoFrame.from_ndarray``: PyAV has no p010le path there, which
    is what stopped the first run on the card.)
    """
    import numpy as np
    from dji_osv_converter import video
    Y = np.full((h, w), level << 6, np.uint16)
    UV = np.full((h // 2, w), 512 << 6, np.uint16)
    return video._p010_av_frame(w, h, (Y, UV))


def encode_probe(codec: str, w: int, h: int, frames: int = 2) -> int:
    """Encode a few 10-bit frames in memory with an encoder session set up the
    way ``convert()`` sets its own up; returns the bytes that came out."""
    import av
    from dji_osv_converter import video
    from dji_osv_converter.constants import BITRATE_BPP
    ctx = av.CodecContext.create(codec, "w")
    ctx.width, ctx.height = w, h
    ctx.pix_fmt = video._pick_pix_fmt(codec, True)
    ctx.time_base = Fraction(1, 30)
    ctx.framerate = Fraction(30, 1)
    ctx.bit_rate = int(w * h * 30 * BITRATE_BPP)
    ctx.options = {"bf": "0", "forced-idr": "1"}
    ctx.flags |= av.codec.context.Flags.global_header
    size = 0
    for i in range(frames):
        fr = p010_test_frame(w, h)
        fr.pts = i
        if i == 0:
            fr.pict_type = av.video.frame.PictureType.I
        size += sum(pkt.size for pkt in ctx.encode(fr))
    size += sum(pkt.size for pkt in ctx.encode(None))
    if not size:
        raise BenchError(f"{codec} took {frames} frames of {w}x{h} and returned no packet")
    return size


def child_preflight(clip: str, names: list[str]) -> dict:
    """Make every check; return what was learnt and every check that failed."""
    info: dict = {"python": platform.python_version(), "cpus": os.cpu_count(),
                  "affinity": (len(os.sched_getaffinity(0))
                               if hasattr(os, "sched_getaffinity") else None)}
    errors: list[str] = []

    def check(what: str, fn):
        try:
            return True, fn()
        except BenchError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"{what}: {type(e).__name__}: {e}")
        return False, None

    errors += static_problems(names)

    # NVDEC first: the device-frame probe must run before torch starts CUDA
    ok, fmt = check("the NVDEC probe failed", lambda: _decode_probe(clip))
    if ok:
        info["decode"] = fmt
    if any(VARIANTS[n][2].get("decode") == "p010le" for n in names):
        check("the hevc_cuvid probe failed", lambda: _decode_probe(clip, "p010le"))

    try:
        import torch
    except Exception as e:
        errors.append(f"torch does not import: {type(e).__name__}: {e}")
    else:
        info.update(torch=torch.__version__, torch_cuda=torch.version.cuda)
        ok, _ = check("CUDA did not start", _need_cuda)
        if ok:
            ok, card = check("torch could not describe the card", lambda: card_info(torch))
            if ok:
                info.update(card)
            check("the CUDA kernel check failed", lambda: kernel_check(torch, "cuda", card))

    w, h, codec = PRODUCTION["width"], PRODUCTION["height"], PRODUCTION["codec"]
    try:
        import av
    except Exception as e:
        errors.append(f"PyAV does not import: {type(e).__name__}: {e}")
    else:
        info.update(pyav=av.__version__,
                    libavcodec=".".join(map(str, av.library_versions.get("libavcodec", ()))))
        frame_ok, _ = check(f"a {w}x{h} p010le frame could not be built",
                            lambda: p010_test_frame(w, h))
        codec_ok, _ = check(f"this PyAV build has no {codec} encoder",
                            lambda: av.codec.Codec(codec, "w"))
        if frame_ok and codec_ok:
            check(f"NVENC could not encode a {w}x{h} 10-bit frame",
                  lambda: encode_probe(codec, w, h))
        else:
            errors.append(f"NVENC was not tried on a {w}x{h} 10-bit frame "
                          "(no encoder or no frame, see above)")
    return {"info": info, "errors": errors}


def apply_switches(sw: dict, applied: list) -> None:
    """Set a variant's harness-only switches on the real modules.

    Each one appends to ``applied`` when the conversion actually reaches it,
    so a switch that silently did nothing can be told from one that worked.
    """
    from dji_osv_converter import seam_gpu, video
    if sw.get("decode"):
        want = sw["decode"]

        def forced_fmt(*_a, **_k):
            applied.append(("decode", want))
            return want
        video.gpu_decode_fmt = forced_fmt           # convert() reads it from the module
    if sw.get("dp_block"):
        block, init = int(sw["dp_block"]), seam_gpu.SeamGpu.__init__

        def init_with_block(self, *a, **k):
            init(self, *a, **k)
            self.dp_block = block                   # read by seam_path on every frame
            applied.append(("dp_block", block))
        seam_gpu.SeamGpu.__init__ = init_with_block


def switch_problems(sw: dict, applied: list) -> list[str]:
    """Switches a variant asked for that the conversion never reached."""
    want = [(k, int(v) if k == "dp_block" else v) for k, v in sw.items()]
    return [f"the harness switch {k}={v!r} was never reached by the conversion"
            for k, v in want if (k, v) not in applied]


def run_conversion(name: str, clip: str, out: str, frames: int, start: int,
                   production: dict | None = None) -> dict:
    """One conversion with the variant's arguments; its profile and memory."""
    from dji_osv_converter import profile, video
    kw = convert_kwargs(name, frames, start, production)
    t0 = time.time()
    video.convert(clip, out, **kw)
    total = time.time() - t0
    rep = profile.last_profile()
    if rep is None:
        raise BenchError("the conversion published no profile")
    return {"profile": rep["profile"], "memory": rep["memory"], "total_s": round(total, 2),
            "table": profile.format_table(rep["profile"]), "out_bytes": os.path.getsize(out)}


def card_problems(p: dict, fmt: str) -> list[str]:
    """Why a finished conversion's profile is not a run on the card."""
    cfg, bad = p["config"], []
    if not str(cfg.get("device", "")).startswith("cuda"):
        bad.append(f"stitch device is {cfg.get('device')!r}, not CUDA")
    if cfg.get("codec") != "hevc_nvenc":
        bad.append(f"encoder is {cfg.get('codec')!r}, not hevc_nvenc")
    if cfg.get("decode") != fmt or "decode.nvdec" not in p["stages"] or "decode.hw" in p["stages"]:
        bad.append(f"decode did not run on NVDEC as {fmt!r} (format {cfg.get('decode')!r}, "
                   f"stages {sorted(k for k in p['stages'] if k.startswith('decode.'))})")
    return bad


def override_problems(name: str, p: dict) -> list[str]:
    """Settings a variant asked for that its profile says it did not run with."""
    over, cfg, bad = VARIANTS[name][1], p["config"], []
    for key, want in over.items():
        if key == "profile":
            if p["mode"] != want:
                bad.append(f"profile mode is {p['mode']!r}, the variant asked for {want!r}")
        elif key in cfg and cfg[key] != want:
            bad.append(f"{key} ran as {cfg[key]!r}, the variant asked for {want!r}")
    return bad


def child_variant(name: str, clip: str, out: str, frames: int, start: int) -> dict:
    sw = VARIANTS[name][2]
    fmt = _decode_probe(clip, sw.get("decode"))     # before torch touches CUDA
    applied: list = []
    apply_switches(sw, applied)
    _need_cuda()
    res = run_conversion(name, clip, out, frames, start)
    bad = (card_problems(res["profile"], fmt) + override_problems(name, res["profile"])
           + switch_problems(sw, applied))
    if bad:
        raise BenchError("; ".join(bad))
    res["switches"] = sorted(set(applied))
    return res


def child_main(args) -> int:
    try:
        if args.child == "preflight":
            res = child_preflight(args.clip, args.variants or list(VARIANTS))
        else:
            res = child_variant(args.child, args.clip, args.out, args.frames, args.start)
    except BenchError as e:
        res = {"error": str(e)}
    except Exception as e:                           # a crash is a result too
        import traceback
        res = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    Path(args.json).write_text(json.dumps(res))
    return 2 if (res.get("error") or res.get("errors")) else 0


# --------------------------------------------------------------- parent side

class Nvml:
    """NVML handle of the one card the benchmark runs on."""

    def __init__(self, index: int | None):
        try:
            import pynvml
        except ImportError as e:
            raise BenchError("pynvml is not installed: utilisation cannot be sampled") from e
        missing = [n for n in NVML_CALLS if not hasattr(pynvml, n)]
        if missing:
            raise BenchError(f"this pynvml has no {', '.join(missing)}: install nvidia-ml-py")
        self.n = pynvml
        s = lambda v: v.decode() if isinstance(v, bytes) else str(v)
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count == 0:
                raise BenchError("NVML sees no GPU")
            if index is None and count > 1:
                raise BenchError(f"NVML sees {count} GPUs: pass --gpu INDEX (the benchmark "
                                 "pins its processes to that card)")
            if index is not None and not 0 <= index < count:
                raise BenchError(f"--gpu {index}: NVML sees {count} GPU(s), numbered from 0")
            self.index = index or 0
            self.count = count
            self.h = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            self.name = s(pynvml.nvmlDeviceGetName(self.h))
            self.uuid = s(pynvml.nvmlDeviceGetUUID(self.h))
            self.driver = s(pynvml.nvmlSystemGetDriverVersion())
        except pynvml.NVMLError as e:
            raise BenchError(f"NVML did not initialise: {type(e).__name__}: {e}") from e
        #: metric -> why it could not be read (the first failure)
        self.unreadable: dict[str, str] = {}

    def sample(self) -> dict:
        """One reading; a metric NVML will not give is left out and noted."""
        n, out = self.n, {}
        for key, fn in (("mem", lambda: n.nvmlDeviceGetMemoryInfo(self.h).used / 2 ** 20),
                        ("sm", lambda: n.nvmlDeviceGetUtilizationRates(self.h).gpu),
                        ("membus", lambda: n.nvmlDeviceGetUtilizationRates(self.h).memory),
                        ("enc", lambda: n.nvmlDeviceGetEncoderUtilization(self.h)[0]),
                        ("dec", lambda: n.nvmlDeviceGetDecoderUtilization(self.h)[0])):
            try:
                out[key] = float(fn())
            except Exception as e:                   # NVMLError, or a changed return type
                self.unreadable.setdefault(key, f"{type(e).__name__}: {e}")
        return out


def _stats(samples: list[dict], key: str) -> dict | None:
    v = [s[key] for s in samples if key in s]
    if not v:
        return None
    v.sort()
    return {"mean": sum(v) / len(v), "p50": v[len(v) // 2], "max": v[-1], "n": len(v)}


def run_child(argv: list[str], env: dict, nvml: Nvml | None, timeout: float,
              script: str | Path | None = None) -> tuple[dict, list[dict], str]:
    """Run one child; sample NVML while it lives. Returns (result, samples, log tail).

    The child stays in this process's group, so whatever stops the benchmark
    by its group stops the child (and frees the card) too.
    """
    script = str(Path(__file__).resolve() if script is None else script)
    with tempfile.TemporaryDirectory(prefix="v100_bench_") as td:
        js = os.path.join(td, "result.json")
        log = os.path.join(td, "child.log")
        samples: list[dict] = []
        done = threading.Event()

        def sampler():
            while not done.wait(SAMPLE_S):
                s = nvml.sample()
                s["t"] = time.time()
                samples.append(s)

        th = threading.Thread(target=sampler, daemon=True) if nvml else None
        timed_out = False
        with open(log, "w") as lf:
            proc = subprocess.Popen([sys.executable, script, *argv, "--json", js],
                                    env=env, stdout=lf, stderr=subprocess.STDOUT)
            if th:
                th.start()
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                rc = proc.wait()
            finally:
                done.set()
                if th:
                    th.join()
        with open(log, errors="replace") as lf:
            tail = "".join(lf.readlines()[-25:])
        res = None
        if os.path.exists(js):
            try:
                res = json.loads(Path(js).read_text())
            except ValueError:
                res = None
        if timed_out:
            res = {"error": f"still running after {timeout:.0f} s (--timeout): killed"}
        elif not isinstance(res, dict):
            res = {"error": f"the process exited with code {rc} before writing a result"}
        elif rc != 0 and not (res.get("error") or res.get("errors")):
            res["error"] = f"the process exited with code {rc}"
        return res, samples, tail


def preflight_argv(clip: str, names: list[str]) -> list[str]:
    return ["--child", "preflight", clip, "--variants", *names]


def variant_argv(name: str, clip: str, out: str | Path, frames: int, start: int) -> list[str]:
    return ["--child", name, clip, "--out", str(out), "--frames", str(frames),
            "--start", str(start)]


def _f(v, d=1, dash="n/a"):
    return dash if v is None else f"{v:.{d}f}"


def _util(st: dict | None) -> str:
    return "unreadable" if st is None else f"{st['mean']:.0f} / {st['max']:.0f}"


def write_report(path: Path, head: dict, pre: dict, rows: list[dict], failures: list[str]) -> None:
    L = [f"# Conversion benchmark on {head['gpu']}", "",
         f"Written by `tools/v100_bench.py` on {head['when']}.", "",
         "| | |", "|---|---|",
         f"| clip | `{head['clip']}`, {head['frames']} frames from frame {head['start']} |",
         f"| code | {head['code']} |",
         f"| card | {head['gpu']} ({head['uuid']}), compute capability {pre.get('capability')}, "
         f"{pre.get('total_mib')} MiB, driver {head['driver']} |",
         f"| userspace | torch {pre.get('torch')} (CUDA {pre.get('torch_cuda')}; kernels for "
         f"{' '.join(pre.get('arch_list') or [])}), PyAV {pre.get('pyav')} "
         f"(libavcodec {pre.get('libavcodec')}), Python {pre.get('python')} |",
         f"| host | {platform.node()}, {platform.machine()}, {pre.get('cpus')} CPUs seen, "
         f"{pre.get('affinity')} usable |",
         f"| decode arrangement | `{pre.get('decode')}` "
         "(`cuda` = NVDEC frames stay on the card; `p010le` = hevc_cuvid to system memory) |",
         f"| card memory in use before the first variant | {_f(head['idle_mib'], 0)} MiB |", "",
         "Settings: Josh's production settings (`rocksteady_horizon`, 7680x3840, 10-bit, "
         "`single_pass`, bilinear, `seam_flow=\"auto\"`, `hevc_nvenc`, no software fallback) "
         "except where a variant says otherwise.", ""]
    if failures:
        L += ["## Failures", ""] + [f"- **{f}**" for f in failures] + [""]
    L += ["## Summary", "",
          "fps and ms/frame are steady-state (the profile leaves out the warm-up frames). "
          "Utilisation is NVML's, sampled every 0.5 s while the variant's process was alive "
          "(set-up included), shown as mean / max in per cent. \"Card peak\" is NVML's used "
          "memory (`nvmlDeviceGetMemoryInfo().used`), everything on the card included; the "
          "torch columns are this process's allocator.", "",
          "| variant | fps | ms/frame | CPU cores | set-up s | card peak MiB | torch peak "
          "alloc / reserved MiB | peak RSS MiB | SM % | NVENC % | NVDEC % | memory bus % |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if "error" in r["res"]:
            L.append(f"| `{r['name']}` | failed | | | | | | | | | | |")
            continue
        p, m, s = r["res"]["profile"], r["res"]["memory"], r["samples"]
        mem = _stats(s, "mem")
        L.append(f"| `{r['name']}` | {p['fps']:.2f} | {p['ms_per_frame']:.1f} | "
                 f"{p['cpu_cores']:.2f} | {p['setup_s']:.1f} | "
                 f"{_f(mem and mem['max'], 0)} | {_f(m.get('gpu_peak_alloc_mib'), 0)} / "
                 f"{_f(m.get('gpu_peak_reserved_mib'), 0)} | {_f(m.get('peak_rss_mib'), 0)} | "
                 f"{_util(_stats(s, 'sm'))} | {_util(_stats(s, 'enc'))} | "
                 f"{_util(_stats(s, 'dec'))} | {_util(_stats(s, 'membus'))} |")
    L.append("")
    for r in rows:
        L += [f"## `{r['name']}`: {VARIANTS[r['name']][0]}", ""]
        if "error" in r["res"]:
            L += [f"**Failed: {r['res']['error']}**", "", "Last lines of its log:", "",
                  "```", r["tail"].rstrip(), "```", ""]
            continue
        p = r["res"]["profile"]
        L += [f"{p['frames']} frames ({p['warmup_frames']} warm-up left out), profile mode "
              f"`{p['mode']}`, {r['res']['total_s']} s in all, output "
              f"{r['res']['out_bytes'] / 2 ** 20:.0f} MiB, {len(r['samples'])} NVML samples.", ""]
        if r["res"].get("switches"):
            L += ["Harness switches the conversion ran with: "
                  + ", ".join(f"`{k}={v}`" for k, v in r["res"]["switches"]) + ".", ""]
        L += ["Milliseconds per frame. `wall` is time on the stage's own thread, `cpu` that "
              "thread's CPU time (`wall` - `cpu` is time blocked), `gpu` the stitch stream's "
              "time for the stage: exact in sync mode; in events mode the time on the stream "
              "between this stage's last kernel and the previous stage's.", "",
              r["res"]["table"], "",
              "Configuration: `" + json.dumps(p["config"], sort_keys=True) + "`", ""]
    path.write_text("\n".join(L))


def _code_label() -> str:
    try:
        rev = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, check=True).stdout.strip()
        where = f"`{rev}`"
    except Exception:
        from dji_osv_converter import __version__ as v
        where = f"dji_osv_converter {v}, no git checkout"
    return f"{where}, source digest `{code_id()}` (`tools/v100_bench.py --code-id`)"


def folder_problems(report: Path, out_dir: Path, frames: int, files: int) -> list[str]:
    """The report and output folders exist, can be written, and have the room."""
    from dji_osv_converter.constants import BITRATE_BPP
    bad = []
    for what, d in (("report", report.parent), ("output", out_dir)):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            bad.append(f"the {what} folder {d} cannot be created: {e}")
            continue
        if not os.access(d, os.W_OK | os.X_OK):
            bad.append(f"the {what} folder {d} is not writable")
    if out_dir.is_dir():
        # the encoder is rate-driven: bits per pixel x pixels x frames, half as much again
        need = int(PRODUCTION["width"] * PRODUCTION["height"] * BITRATE_BPP / 8
                   * frames * files * 1.5)
        free = shutil.disk_usage(out_dir).free
        if free < need:
            bad.append(f"the output folder {out_dir} has {free / 2 ** 20:.0f} MiB free; "
                       f"{files} file(s) of {frames} frames need about {need / 2 ** 20:.0f} MiB")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip", nargs="?", help="a DJI .OSV file")
    ap.add_argument("--report", help="Markdown report to write (required)")
    ap.add_argument("--frames", type=int, default=450)
    ap.add_argument("--start", type=int, default=0, help="first frame")
    ap.add_argument("--variants", nargs="*", default=None, help=f"from: {' '.join(VARIANTS)}")
    ap.add_argument("--all", action="store_true", help="every variant, diagnostics included")
    ap.add_argument("--gpu", type=int, default=None,
                    help="NVML index of the card (needed only when several are visible)")
    ap.add_argument("--out-dir", default=None,
                    help="where the converted files go (default: a new temporary folder)")
    ap.add_argument("--keep", action="store_true", help="keep the converted files")
    ap.add_argument("--timeout", type=float, default=VARIANT_TIMEOUT_S,
                    help="seconds one variant may run before it is killed and reported "
                         f"(default {VARIANT_TIMEOUT_S:.0f})")
    ap.add_argument("--code-id", action="store_true",
                    help="print the digest of this copy's source and exit")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    ap.add_argument("--json", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.code_id:
        print(code_id())
        return 0
    if not args.clip:
        ap.error("a clip is required")
    if args.child:
        return child_main(args)
    if not args.report:
        ap.error("--report is required")
    names = list(VARIANTS) if args.all else (args.variants or DEFAULT)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        ap.error(f"unknown variants: {unknown}")
    if args.frames < 1 or args.start < 0 or args.timeout <= 0:
        ap.error("--frames must be at least 1, --start at least 0 and --timeout positive")
    clip = str(Path(args.clip).resolve())
    report = Path(args.report).resolve()

    # ---- preflight: every check is made, and every failure is named
    print(f"v100_bench: source digest {code_id()}; preflight ...", flush=True)
    problems: list[str] = []
    if not os.path.isfile(clip):
        problems.append(f"no such clip: {clip}")
    nvml = None
    try:
        nvml = Nvml(args.gpu)
    except BenchError as e:
        problems.append(str(e))
    env = dict(os.environ)
    if nvml is not None and nvml.count > 1:
        env["CUDA_VISIBLE_DEVICES"] = nvml.uuid      # torch and FFmpeg both follow it
    env.setdefault("PYTHONUNBUFFERED", "1")
    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="v100_bench_"))

    def drop_own_folder():
        # the temporary folder this run made, and only once it is empty again
        if not args.out_dir and not args.keep:
            try:
                out_dir.rmdir()
            except OSError:
                pass
    problems += folder_problems(report, out_dir, args.frames, len(names) if args.keep else 1)
    idle = None
    if nvml is not None:
        print(f"v100_bench: {nvml.name} ({nvml.uuid}), driver {nvml.driver}", flush=True)
        idle = nvml.sample().get("mem")
        for key, why in sorted(nvml.unreadable.items()):
            print(f"v100_bench: warning: NVML cannot read '{key}' here ({why})",
                  file=sys.stderr, flush=True)
    pre, _s, tail = run_child(preflight_argv(clip, names), env, None, PREFLIGHT_TIMEOUT_S)
    if "error" in pre:                               # the preflight itself died
        problems.append(f"the preflight process failed: {pre['error']}")
    problems += pre.get("errors", [])
    if problems:
        if tail.strip():
            print(tail, file=sys.stderr)
        drop_own_folder()
        raise SystemExit(f"v100_bench: FAILED preflight: {len(problems)} check(s) failed:\n"
                         + "\n".join(f"  - {p}" for p in problems))
    pre = pre["info"]
    print(f"v100_bench: preflight ok: decode {pre['decode']}, torch {pre['torch']}", flush=True)

    head = {"gpu": nvml.name, "uuid": nvml.uuid, "driver": nvml.driver, "clip": clip,
            "frames": args.frames, "start": args.start, "code": _code_label(),
            "when": time.strftime("%Y-%m-%d %H:%M %Z"), "idle_mib": idle}

    rows, failures = [], []
    for name in names:
        out = out_dir / f"v100_bench_{name}.mp4"
        print(f"v100_bench: {name}: {VARIANTS[name][0]} ...", flush=True)
        res, samples, tail = run_child(variant_argv(name, clip, out, args.frames, args.start),
                                       env, nvml, args.timeout)
        if "error" not in res:
            # the engines must have been seen working, not just selected
            bad = [] if samples else ["NVML gave no sample while the variant ran"]
            for key, what in (("dec", "NVDEC"), ("enc", "NVENC")):
                st = _stats(samples, key)
                if st is not None and st["max"] <= 0:
                    bad.append(f"{what} utilisation was 0 % in all {st['n']} samples: "
                               "the work did not run on the card's engine")
            if bad:
                res["error"] = "; ".join(bad)
        if "error" in res:
            failures.append(f"{name}: {res['error']}")
            print(f"v100_bench: {name} FAILED: {res['error']}", file=sys.stderr, flush=True)
        else:
            p = res["profile"]
            print(f"v100_bench: {name}: {p['fps']:.2f} fps, {p['ms_per_frame']:.1f} ms/frame",
                  flush=True)
        rows.append({"name": name, "res": res, "samples": samples, "tail": tail})
        if not args.keep and out.exists():
            out.unlink()                             # the one file this variant wrote
        write_report(report, head, pre, rows, failures)   # partial runs keep a report
    for key, why in sorted(nvml.unreadable.items()):
        failures.append(f"NVML could not read '{key}' on this driver/container ({why}): "
                        "that column is empty, and the engine check for it was not made")
    write_report(report, head, pre, rows, failures)
    drop_own_folder()
    if args.keep:
        print(f"v100_bench: converted files kept in {out_dir}", flush=True)
    print(f"v100_bench: report written to {report}", flush=True)
    if failures:
        print("v100_bench: FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
