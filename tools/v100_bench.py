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

Nothing falls back quietly. The run stops with a non-zero exit and a message
if CUDA is not up, if NVDEC cannot decode the clip, if NVENC cannot encode an
8K 10-bit frame, or if NVML is missing; and a variant fails if its profile
shows a decode path other than NVDEC, an encoder other than ``hevc_nvenc``,
or a device other than CUDA. Every output file is written under ``--out-dir``
and removed again (that one file, by its path) unless ``--keep`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_S = 0.5

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


class BenchError(RuntimeError):
    """Something the benchmark needs is missing: reported, never worked around."""


# ---------------------------------------------------------------- child side

def _need_cuda():
    import torch
    if not torch.cuda.is_available():
        raise BenchError("CUDA is not available to torch (torch.cuda.is_available() is False)")
    return torch


def _decode_probe(clip: str) -> str:
    """The NVDEC arrangement for this clip, probed before torch starts CUDA."""
    from dji_osv_converter import nvdec
    from dji_osv_converter.decode import gpu_decode_fmt
    fmt = gpu_decode_fmt(True, clip)
    if fmt == "cuda":
        return fmt
    if fmt == "p010le" and nvdec.cuvid_usable(clip):
        return fmt
    raise BenchError(f"NVDEC cannot decode {clip}: neither device frames nor hevc_cuvid work "
                     f"(gpu_decode_fmt gave {fmt!r})")


def _nvenc_probe() -> None:
    """Encode two 8K 10-bit frames with hevc_nvenc, in memory."""
    import io
    import av
    import numpy as np
    try:
        av.codec.Codec("hevc_nvenc", "w")
    except Exception as e:
        raise BenchError(f"this PyAV build has no hevc_nvenc encoder: {e}") from e
    w, h = PRODUCTION["width"], PRODUCTION["height"]
    try:
        with av.open(io.BytesIO(), "w", format="mp4") as oc:
            st = oc.add_stream("hevc_nvenc", rate=30)
            st.width, st.height, st.pix_fmt = w, h, "p010le"
            fr = av.VideoFrame.from_ndarray(np.full((h * 3 // 2, w), 512 << 6, np.uint16),
                                            format="p010le")
            for i in range(2):
                fr.pts = i
                for pkt in st.encode(fr):
                    oc.mux(pkt)
            for pkt in st.encode(None):
                oc.mux(pkt)
    except Exception as e:
        raise BenchError(f"NVENC could not encode a {w}x{h} 10-bit frame: {e!r}") from e


def child_preflight(clip: str) -> dict:
    fmt = _decode_probe(clip)                       # before torch touches CUDA
    torch = _need_cuda()
    _nvenc_probe()
    import av
    p = torch.cuda.get_device_properties(0)
    return {"decode": fmt, "gpu": p.name, "capability": f"{p.major}.{p.minor}",
            "total_mib": round(p.total_memory / 2 ** 20),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "arch_list": torch.cuda.get_arch_list(), "pyav": av.__version__,
            "libavcodec": ".".join(map(str, av.library_versions.get("libavcodec", ()))),
            "python": platform.python_version(), "cpus": os.cpu_count(),
            "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}


def child_variant(name: str, clip: str, out: str, frames: int, start: int) -> dict:
    _desc, over, sw = VARIANTS[name]
    fmt = _decode_probe(clip)                       # before torch touches CUDA
    from dji_osv_converter import profile, seam_gpu, video
    if sw.get("decode"):
        want = sw["decode"]
        if want == "p010le":
            from dji_osv_converter import nvdec
            if not nvdec.cuvid_usable(clip):
                raise BenchError("hevc_cuvid cannot decode this clip (variant needs it)")
        video.gpu_decode_fmt = lambda *a, **k: want          # harness-only switch
        fmt = want
    if sw.get("dp_block"):
        seam_gpu.SEAM_DP_BLOCK_CUDA = int(sw["dp_block"])    # read when the stitcher is built
    _need_cuda()
    kw = dict(PRODUCTION, start=start, count=frames, profile=True)
    kw.update(over)
    t0 = time.time()
    video.convert(clip, out, **kw)
    total = time.time() - t0
    rep = profile.last_profile()
    if rep is None:
        raise BenchError("the conversion published no profile")
    p, cfg = rep["profile"], rep["profile"]["config"]
    bad = []
    if not str(cfg.get("device", "")).startswith("cuda"):
        bad.append(f"stitch device is {cfg.get('device')!r}, not CUDA")
    if cfg.get("codec") != "hevc_nvenc":
        bad.append(f"encoder is {cfg.get('codec')!r}, not hevc_nvenc")
    if cfg.get("decode") != fmt or "decode.nvdec" not in p["stages"] or "decode.hw" in p["stages"]:
        bad.append(f"decode did not run on NVDEC (format {cfg.get('decode')!r}, stages "
                   f"{sorted(k for k in p['stages'] if k.startswith('decode.'))})")
    if bad:
        raise BenchError("; ".join(bad))
    return {"profile": p, "memory": rep["memory"], "total_s": round(total, 2),
            "table": profile.format_table(p), "out_bytes": os.path.getsize(out)}


def child_main(args) -> int:
    try:
        res = (child_preflight(args.clip) if args.child == "preflight"
               else child_variant(args.child, args.clip, args.out, args.frames, args.start))
    except BenchError as e:
        res = {"error": str(e)}
    except Exception as e:                           # a crash is a result too
        import traceback
        res = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    Path(args.json).write_text(json.dumps(res))
    return 2 if "error" in res else 0


# --------------------------------------------------------------- parent side

class Nvml:
    """NVML handle of the one card the benchmark runs on."""

    def __init__(self, index: int | None):
        try:
            import pynvml
        except ImportError as e:
            raise BenchError("pynvml is not installed: utilisation cannot be sampled") from e
        self.n = pynvml
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError as e:
            raise BenchError(f"NVML did not initialise: {e}") from e
        if count == 0:
            raise BenchError("NVML sees no GPU")
        if index is None and count > 1:
            raise BenchError(f"NVML sees {count} GPUs: pass --gpu INDEX (the benchmark pins "
                             "its processes to that card)")
        self.index = index or 0
        self.count = count
        self.h = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        s = lambda v: v.decode() if isinstance(v, bytes) else v
        self.name = s(pynvml.nvmlDeviceGetName(self.h))
        self.uuid = s(pynvml.nvmlDeviceGetUUID(self.h))
        self.driver = s(pynvml.nvmlSystemGetDriverVersion())
        self.unreadable: set[str] = set()

    def used_mib(self) -> float:
        return self.n.nvmlDeviceGetMemoryInfo(self.h).used / 2 ** 20

    def sample(self) -> dict:
        n, out = self.n, {"mem": self.used_mib()}
        for key, fn in (("sm", lambda: n.nvmlDeviceGetUtilizationRates(self.h).gpu),
                        ("membus", lambda: n.nvmlDeviceGetUtilizationRates(self.h).memory),
                        ("enc", lambda: n.nvmlDeviceGetEncoderUtilization(self.h)[0]),
                        ("dec", lambda: n.nvmlDeviceGetDecoderUtilization(self.h)[0])):
            try:
                out[key] = float(fn())
            except n.NVMLError:
                self.unreadable.add(key)
        return out


def _stats(samples: list[dict], key: str) -> dict | None:
    v = [s[key] for s in samples if key in s]
    if not v:
        return None
    v.sort()
    return {"mean": sum(v) / len(v), "p50": v[len(v) // 2], "max": v[-1], "n": len(v)}


def run_child(argv: list[str], env: dict, nvml: Nvml | None) -> tuple[dict, list[dict], str]:
    """Run one child; sample NVML while it lives. Returns (result, samples, log tail)."""
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
        with open(log, "w") as lf:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *argv,
                                     "--json", js], env=env, stdout=lf, stderr=subprocess.STDOUT)
            if th:
                th.start()
            rc = proc.wait()
        done.set()
        if th:
            th.join()
        with open(log, errors="replace") as lf:
            tail = "".join(lf.readlines()[-25:])
        if os.path.exists(js):
            res = json.loads(Path(js).read_text())
        else:
            res = {"error": f"the process exited with code {rc} before writing a result"}
        if rc != 0 and "error" not in res:
            res["error"] = f"the process exited with code {rc}"
        return res, samples, tail


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
          "memory, everything on the card included; the torch columns are this process's "
          "allocator.", "",
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
              f"{r['res']['out_bytes'] / 2 ** 20:.0f} MiB, {len(r['samples'])} NVML samples.", "",
              "Milliseconds per frame. `wall` is time on the stage's own thread, `cpu` that "
              "thread's CPU time (`wall` - `cpu` is time blocked), `gpu` the stitch stream's "
              "time for the stage: exact in sync mode; in events mode the time on the stream "
              "between this stage's last kernel and the previous stage's.", "",
              r["res"]["table"], "",
              "Configuration: `" + json.dumps(p["config"], sort_keys=True) + "`", ""]
    path.write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip", help="a DJI .OSV file")
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
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    ap.add_argument("--json", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return child_main(args)
    if not args.report:
        ap.error("--report is required")
    names = list(VARIANTS) if args.all else (args.variants or DEFAULT)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        ap.error(f"unknown variants: {unknown}")
    clip = str(Path(args.clip).resolve())
    if not os.path.isfile(clip):
        raise SystemExit(f"v100_bench: FAILED: no such clip: {clip}")

    try:
        nvml = Nvml(args.gpu)
    except BenchError as e:
        raise SystemExit(f"v100_bench: FAILED: {e}")
    env = dict(os.environ)
    if nvml.count > 1:
        env["CUDA_VISIBLE_DEVICES"] = nvml.uuid      # torch and FFmpeg both follow it
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"v100_bench: {nvml.name} ({nvml.uuid}), driver {nvml.driver}; preflight ...",
          flush=True)
    pre, _s, tail = run_child(["--child", "preflight", clip], env, None)
    if "error" in pre:
        print(tail, file=sys.stderr)
        raise SystemExit(f"v100_bench: FAILED preflight: {pre['error']}")
    print(f"v100_bench: preflight ok: decode {pre['decode']}, torch {pre['torch']}", flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="v100_bench_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        code = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
                              capture_output=True, text=True, check=True).stdout.strip()
        code = f"`{code}`"
    except Exception:
        from dji_osv_converter import __version__ as v
        code = f"dji_osv_converter {v} (no git checkout)"
    head = {"gpu": nvml.name, "uuid": nvml.uuid, "driver": nvml.driver, "clip": clip,
            "frames": args.frames, "start": args.start, "code": code,
            "when": time.strftime("%Y-%m-%d %H:%M %Z"), "idle_mib": nvml.used_mib()}

    rows, failures = [], []
    for name in names:
        out = out_dir / f"v100_bench_{name}.mp4"
        print(f"v100_bench: {name}: {VARIANTS[name][0]} ...", flush=True)
        res, samples, tail = run_child(["--child", name, clip, "--out", str(out),
                                        "--frames", str(args.frames),
                                        "--start", str(args.start)], env, nvml)
        if "error" not in res:
            # the engines must have been seen working, not just selected
            for key, what in (("dec", "NVDEC"), ("enc", "NVENC")):
                st = _stats(samples, key)
                if st is not None and st["max"] <= 0:
                    res["error"] = (f"{what} utilisation was 0 % in all {st['n']} samples: "
                                    "the work did not run on the card's engine")
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
        write_report(Path(args.report), head, pre, rows, failures)   # partial runs keep a report
    for key in sorted(nvml.unreadable):
        failures.append(f"NVML could not read '{key}' utilisation on this driver/container: "
                        "that column is empty, and the engine check for it was not made")
    write_report(Path(args.report), head, pre, rows, failures)
    print(f"v100_bench: report written to {args.report}", flush=True)
    if failures:
        print("v100_bench: FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
