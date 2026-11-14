#!/usr/bin/env python3
"""Render A/B comparison windows for the stitch improvements and build a viewer.

For each test window (a clip + frame) every variant is rendered at full 8K:
the baseline (two-pass stitch, bilinear, auto seam repair), each improvement
alone, all together, and DJI Studio's export where one exists (rotated into
our layout). Per variant the tool writes a downscaled full frame plus
full-resolution crops of the regions that matter (seam meridians, lens
periphery, sky band, front centre, ground), amplified difference images
against the baseline, and numeric metrics (NCC/SSIM/edge-NCC/PSNR vs DJI
where available, Laplacian-variance sharpness per crop, seam-band mismatch
from the stitcher, sky-band brightness gradient). Then it writes a
self-contained ``index.html`` (no network, no libraries) from
``tools/ab_viewer.html``.

    python tools/ab_compare.py                       # default windows, -> ab_out/
    python tools/ab_compare.py --set perf --out ab_out/perf_ab   # seam engines, old vs new
    python tools/ab_compare.py --out ab_out --device cuda --windows my_windows.json
    open ab_out/index.html

Everything under ``--out`` is footage-derived and must never be committed
(``ab_out/`` is gitignored). One stitcher is resident at a time: three 8K
stitchers at once exhausted a 16 GB machine and made timings meaningless.

Re-running on another box: pass ``--windows`` with that box's paths (the
default windows name files on the Mac this was written on) and ``--device``.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from dji_osv_converter import GpuStitcher, Stitcher360, iter_pairs  # noqa: E402
from dji_osv_converter.pipeline import clear_geometry_cache  # noqa: E402
from dji_osv_converter.stitch_gpu import p010_to_bgr, pick_device, _sync  # noqa: E402
import validate_against_dji as vd  # noqa: E402

log = logging.getLogger("ab")

W, H = 7680, 3840
EX = Path("/Users/zindello/repositories/zindello/djiclireframe/osv_example")
MOVIES = Path("/Users/zindello/Movies/Osmo360")

# Chosen by measuring mean luma per second on the camera's .LRF proxies and
# on the sample clips (tools notes in IMPROVEMENTS.md, "A/B windows"); each
# window says why it was picked. ``levelling`` is "plain" so the seam
# meridians sit at x = W/4 and 3W/4 in every frame (fixed crop regions) and
# DJI's plain export is the 1:1 frame-aligned reference.
DEFAULT_WINDOWS = [
    {"name": "base_f10", "role": "normal (DJI reference)",
     "clip": str(EX / "CAM_20260819204151_0011_D.OSV"), "frame": 10,
     "dji": str(EX / "exported_360_plain.mp4"),
     "why": "the project's standing comparison frame; DJI plain export available"},
    {"name": "worst_f230", "role": "normal, violent motion (DJI reference)",
     "clip": str(EX / "pitching-rolling-worst-case-scenario/CAM_20260824110935_0001_D.OSV"),
     "frame": 230, "dji": str(EX / "pitching-rolling-worst-case-scenario/pitch_roll_plain.mp4"),
     "why": "worst-case tumble clip mid-tumble; DJI plain export available"},
    # The three luma-selected windows: mean luma per second of every .LRF
    # proxy in the camera's library (0-255 grey): the library's brightest
    # second (158), its darkest (71 - there is no night footage; this is the
    # darkest available) and a steady mid-tone clip (117).
    {"name": "light_0011_f10", "role": "light (brightest second in the library, LRF luma 158)",
     "clip": str(MOVIES / "CAM_20261030165035_0011_D.OSV"), "frame": 10,
     "why": "brightest footage available: sky/overexposure stress for the gradient and seam"},
    {"name": "dark_0021_f1349", "role": "dark (darkest second in the library, LRF luma 71)",
     "clip": str(MOVIES / "CAM_20261101111038_0021_D.OSV"), "frame": 1349,
     "why": "darkest footage available (45 s in): shadow noise, photometric mismatch"},
    {"name": "normal_0017_f1229", "role": "normal (steady mid-tone clip, LRF luma 117)",
     "clip": str(MOVIES / "CAM_20261101105955_0017_D.OSV"), "frame": 1229,
     "why": "typical handheld footage, no DJI export"},
]

# Variants: the kwargs that differ from the baseline. ``seam_flow`` stays
# "auto" (the shipping default) everywhere except the explicit seam_off row.
LSC_PROBE_ALPHA = 0.3      # exploratory partial lens-shading (see IMPROVEMENTS.md)
VARIANTS = {
    "baseline": {"desc": "seam-parallax as is: two-pass warp+level, bilinear, auto seam repair",
                 "kw": {}},
    "seam_off": {"desc": "baseline with the seam stack off (static blend)",
                 "kw": {"seam_flow": False}},
    "single_pass": {"desc": "single-pass resample (levelling composed into the lens maps; "
                            "exact wrap column; no pole singularity)",
                    "kw": {"single_pass": True}},
    "bicubic": {"desc": "two-pass, torch bicubic (a=-0.75, sharpening kernel) lens sampling",
                "kw": {"lens_interp": "bicubic"}},
    "catmull": {"desc": "two-pass, Catmull-Rom (a=-0.5, accurate cubic) lens sampling",
                "kw": {"lens_interp": "catmull"}},
    "lsc_partial": {"desc": f"two-pass, partial factory lens-shading correction alpha={LSC_PROBE_ALPHA}"
                            " (exploratory; default stays 0)",
                    "kw": {"lsc": LSC_PROBE_ALPHA}},
    "all": {"desc": "single-pass + bicubic (the proposed 'all on'; lsc stays 0)",
            "kw": {"single_pass": True, "lens_interp": "bicubic"}},
    "all_catmull": {"desc": "single-pass + Catmull-Rom (quality render option)",
                    "kw": {"single_pass": True, "lens_interp": "catmull"}},
}

# The performance set (PERFORMANCE.md): the baseline is what production ran
# at ee1e170 (single-pass, the seam stack on the CPU), against the seam
# stack on the device and the opt-in corrected morph direction. Every
# variant names its engine, so a row means the same thing on any device.
PERF_VARIANTS = {
    "baseline": {"desc": "production at ee1e170: single-pass, auto seam repair, seam stack on "
                         "the CPU (OpenCV DIS flow, numpy seam path)",
                 "kw": {"single_pass": True, "seam_engine": "cpu"}},
    "gpu_engine": {"desc": "the new default on CUDA/MPS: single-pass, seam stack in torch on "
                           "the device (Lucas-Kanade flow), same morph direction as shipped",
                   "kw": {"single_pass": True, "seam_engine": "gpu"}},
    "seam_off": {"desc": "single-pass with the seam stack off (static blend)",
                 "kw": {"single_pass": True, "seam_flow": False, "seam_engine": "gpu"}},
    "cpu_corrected": {"desc": "CPU seam stack with seam_morph=\"corrected\" (opt-in: each lens "
                              "moves towards the other)",
                      "kw": {"single_pass": True, "seam_engine": "cpu",
                             "seam_morph": "corrected"}},
    "gpu_corrected": {"desc": "device seam stack with seam_morph=\"corrected\" (opt-in)",
                      "kw": {"single_pass": True, "seam_engine": "gpu",
                             "seam_morph": "corrected"}},
}
VARIANT_SETS = {"quality": VARIANTS, "perf": PERF_VARIANTS}

CROP_W, CROP_H = 1024, 768
REGIONS = {                       # centre (x, y) in the 7680x3840 plain layout
    "front": (W // 2, H // 2),
    "seam_left": (W // 4, H // 2),
    "seam_right": (3 * W // 4, H // 2),
    "periphery": (W // 4 + 420, H // 2 + 640),
    "sky_front": (W // 2, H // 4),
    "sky_seam": (W // 4, H // 4),
    "ground": (W // 2, 3 * H // 4),
}
FULL_W = 3840
DIFF_GAIN = 8


def crop(img: np.ndarray, centre: tuple[int, int]) -> np.ndarray:
    cx, cy = centre
    x0 = int(np.clip(cx - CROP_W // 2, 0, img.shape[1] - CROP_W))
    y0 = int(np.clip(cy - CROP_H // 2, 0, img.shape[0] - CROP_H))
    return img[y0:y0 + CROP_H, x0:x0 + CROP_W]


def lapvar(bgr: np.ndarray) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def sky_gradient(bgr: np.ndarray) -> dict:
    """Brightness profile of the sky band vs longitude: how flat is it?

    Column-mean luma over rows H/8..H/3, box-smoothed over 64 px; reports the
    peak-to-peak as % of the mean within +-20 deg of each seam meridian (the
    band where a residual lens falloff would show as a dark/bright arc) and
    over the whole width. Scene content dominates on textured skies - read
    it as a DIFFERENCE between variants of the same frame, not an absolute.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    band = g[H // 8:H // 3].mean(0)
    sm = cv2.blur(band.reshape(1, -1), (65, 1)).ravel()
    half = int(20 / 360 * W)
    seams = []
    for x in (W // 4, 3 * W // 4):
        seg = sm[x - half:x + half]
        seams.append(float(np.ptp(seg) / max(seg.mean(), 1e-6) * 100))
    return {"seam_ptp_pct": float(np.mean(seams)),
            "global_ptp_pct": float(np.ptp(sm) / max(sm.mean(), 1e-6) * 100),
            "profile": [float(v) for v in sm[::60]]}


def small(bgr: np.ndarray, width: int = FULL_W) -> np.ndarray:
    return cv2.resize(bgr, (width, width // 2), interpolation=cv2.INTER_AREA)


def jpeg(path: Path, img: np.ndarray, q: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, q])


def diff_image(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    return cv2.applyColorMap(np.clip(d.astype(np.int32) * DIFF_GAIN, 0, 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)


def read_dji_frame(path: str, index: int) -> np.ndarray:
    """Frame ``index`` of a DJI export as 8-bit BGR (PyAV, no ffmpeg binary)."""
    import av
    with av.open(path) as c:
        v = c.streams.video[0]
        v.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(v)):
            if i == index:
                return fr.to_ndarray(format="bgr24")
    raise RuntimeError(f"{path}: frame {index} not found")


def load_window(win: dict) -> dict:
    """Decode the fisheye pair and build the per-window stitcher state."""
    st = Stitcher360.from_file(win["clip"], width=W, height=H,
                               levelling=win.get("levelling", "plain"))
    idx, f0, f1 = next(iter(iter_pairs(win["clip"], start=win["frame"], count=1,
                                       fmt="yuv420p10le")))
    assert idx == win["frame"], (idx, win["frame"])
    return {"st": st, "f0": f0, "f1": f1, "R": st.levelling_for(win["frame"])}


def build_stitcher(st: Stitcher360, device: str, kw: dict) -> GpuStitcher:
    maps, _ = st.maps_for()
    lsc = kw.get("lsc", 0.0)
    return GpuStitcher(maps, band_px=st.seam_overlap_ratio * W, width=W, height=H,
                       device=device,
                       lsc_gain=st.lsc_gains(float(lsc)) if lsc else None,
                       fisheye_size=(st.calibs[0].width, st.calibs[0].height),
                       single_pass=kw.get("single_pass", False),
                       lens_interp=kw.get("lens_interp", "bilinear"),
                       # the quality set predates the device seam engine: it
                       # stays on the CPU stack so its rows keep their meaning
                       seam_engine=kw.get("seam_engine", "cpu"),
                       seam_morph=kw.get("seam_morph", "legacy"))


def _psnr10(a: np.ndarray, b: np.ndarray) -> float | None:
    """PSNR in dB between two p010 planes, on the 10-bit scale (None = identical)."""
    d = (a.astype(np.int32) - b.astype(np.int32)).astype(np.float32) / 64.0
    mse = float((d * d).mean())
    return None if mse == 0.0 else float(10 * np.log10(1023.0 ** 2 / mse))


def seam_mask(h: int, w: int, band_rows: tuple[int, int], margin_deg: float = 2.0) -> np.ndarray:
    """(h, w) bool: output pixels within the seam band, under plain levelling.

    The stitcher's seam work is confined to rows ``band_rows`` of the rig
    frame: a belt around the great circle between the lenses. Plain
    levelling turns that circle into the two meridians x = W/4 and 3W/4, so
    in the output the belt is two vertical strips that widen towards the
    poles. ``margin_deg`` covers the rig's small tilt against the output
    axes (1.5 degrees on the sample clips).
    """
    half = (band_rows[1] - band_rows[0]) / 2.0 / h * np.pi + np.radians(margin_deg)
    lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    along_axis = np.cos(lat)[:, None] * np.cos(lon)[None, :]       # d . lens axis
    return np.abs(np.arcsin(np.clip(along_axis, -1, 1))) <= half


def planes_vs_base(Y: np.ndarray, UV: np.ndarray, Yb: np.ndarray, UVb: np.ndarray,
                   mask: np.ndarray) -> dict:
    """10-bit differences of a variant's output planes against the baseline's.

    ``mask`` (see :func:`seam_mask`) marks the seam belt, reported as
    "seam"; everything else is "outside", where the seam stack must change
    nothing.
    """
    d = np.abs(Y.astype(np.int32) - Yb.astype(np.int32)) // 64       # 10-bit levels
    seam, out = d[mask], d[~mask]
    return {"psnr_y_db": _psnr10(Y, Yb), "psnr_uv_db": _psnr10(UV, UVb),
            "seam_psnr_y_db": _psnr10(Y[mask], Yb[mask]),
            "seam_mean_abs_y": float(seam.mean()), "seam_max_abs_y": int(seam.max()),
            "seam_pct_gt4": 100.0 * float((seam > 4).mean()),
            "seam_pct_of_frame": 100.0 * float(mask.mean()),
            "outside_max_abs_y": int(out.max()),
            "outside_pct_changed": 100.0 * float((out > 0).mean())}


def free_device(device: str) -> None:
    gc.collect()
    try:
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "ab_out"))
    ap.add_argument("--device", default=None, help="torch device (default: best)")
    ap.add_argument("--windows", default=None, help="JSON list of windows (see DEFAULT_WINDOWS)")
    ap.add_argument("--set", default="quality", choices=sorted(VARIANT_SETS),
                    help="variant set: quality (the stitch improvements) or perf "
                         "(seam engines, old vs new)")
    ap.add_argument("--variants", nargs="*", default=None, help="subset of variant names")
    ap.add_argument("--no-dji", action="store_true", help="skip the DJI reference even if present")
    ap.add_argument("--timing-frames", type=int, default=4,
                    help="stitch repetitions per variant for the ms/frame figure")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = Path(args.out)
    img_dir = out / "img"
    device = args.device or pick_device()
    windows = json.loads(Path(args.windows).read_text()) if args.windows else DEFAULT_WINDOWS
    variants = {k: v for k, v in VARIANT_SETS[args.set].items()
                if not args.variants or k in args.variants}
    if "baseline" not in variants:
        raise SystemExit("the baseline variant is required (diffs and alignment hang off it)")

    manifest = {"generated": time.strftime("%Y-%m-%d %H:%M"), "device": device,
                "windows": [], "regions": {k: list(v) for k, v in REGIONS.items()},
                "crop_size": [CROP_W, CROP_H], "full_width": FULL_W, "diff_gain": DIFF_GAIN,
                "variants": {k: v["desc"] for k, v in variants.items()},
                "variant_kwargs": {k: v["kw"] for k, v in variants.items()},
                "images": {}, "diffs": {}, "metrics": {}, "timing_ms": {}}

    # decode every window once (two 44 MB planes each)
    state = {}
    for win in windows:
        log.info("loading %s (%s frame %d)", win["name"], Path(win["clip"]).name, win["frame"])
        state[win["name"]] = load_window(win)
        manifest["windows"].append({k: v for k, v in win.items() if k != "dji"} |
                                   {"has_dji": bool(win.get("dji")) and not args.no_dji})

    renders: dict[str, dict[str, np.ndarray]] = {w["name"]: {} for w in windows}
    seam_stats: dict[str, dict[str, dict]] = {w["name"]: {} for w in windows}
    base_planes: dict[str, tuple] = {}
    vs_base: dict[str, dict[str, dict]] = {w["name"]: {} for w in windows}

    # variant-major: ONE stitcher resident at a time
    for vname, var in variants.items():
        kw = var["kw"]
        seam = kw.get("seam_flow", "auto")
        first = True
        for win in windows:
            s = state[win["name"]]
            t0 = time.time()
            g = build_stitcher(s["st"], device, kw)
            setup = time.time() - t0
            Y, UV = g.stitch_yuv(s["f0"], s["f1"], s["R"], seam_flow=seam)
            if first:
                _sync(g.dev)
                t = time.time()
                for _ in range(args.timing_frames):
                    g.stitch_yuv(s["f0"], s["f1"], s["R"], seam_flow=seam)
                _sync(g.dev)
                manifest["timing_ms"][vname] = {
                    "stitch_ms": (time.time() - t) / args.timing_frames * 1000,
                    "setup_s": setup, "device": device,
                    "note": "single resident stitcher, one window; MPS/CPU here is not the V100"}
                first = False
            if vname == "baseline":
                base_planes[win["name"]] = (Y, UV)
            else:
                vs_base[win["name"]][vname] = planes_vs_base(
                    Y, UV, *base_planes[win["name"]],
                    mask=seam_mask(H, W, g._sf_rows))
            bgr = p010_to_bgr(Y, UV)
            renders[win["name"]][vname] = bgr
            seam_stats[win["name"]][vname] = dict(g.seam_stats or {})
            log.info("  %-12s %-12s %.1f s setup, seam %s", vname, win["name"], setup,
                     g.seam_stats)
            del g, Y, UV
            free_device(device)
        clear_geometry_cache()

    # DJI reference, rotated into our layout using the baseline's alignment
    dji_small: dict[str, np.ndarray] = {}
    dji_R: dict[str, np.ndarray] = {}
    for win in windows:
        if not win.get("dji") or args.no_dji:
            continue
        log.info("DJI reference for %s", win["name"])
        ref = read_dji_frame(win["dji"], win["frame"])
        ours = small(renders[win["name"]]["baseline"])
        ref_s = cv2.resize(ref, (FULL_W, FULL_W // 2), interpolation=cv2.INTER_AREA)
        R, score = vd.find_alignment(ours, ref_s)
        dji_R[win["name"]], dji_small[win["name"]] = R, ref_s
        ref_full = ref if ref.shape[1] == W else cv2.resize(ref, (W, H), interpolation=cv2.INTER_CUBIC)
        renders[win["name"]]["dji"] = vd.apply_rotation(ref_full, R.T)   # into our layout
        seam_stats[win["name"]]["dji"] = {}
        manifest["variants"]["dji"] = ("DJI Studio export, same frame, rotated into our layout "
                                       f"(alignment score {score:.3f}); its own tone grade")
        log.info("  alignment score %.4f", score)

    # images + metrics
    for win in windows:
        wn = win["name"]
        manifest["images"][wn], manifest["diffs"][wn], manifest["metrics"][wn] = {}, {}, {}
        base = renders[wn]["baseline"]
        for vname, bgr in renders[wn].items():
            paths = {"full": f"img/{wn}/{vname}/full.jpg"}
            jpeg(out / paths["full"], small(bgr))
            dpaths = {}
            if vname != "baseline":
                dpaths["full"] = f"img/{wn}/{vname}/diff_full.jpg"
                jpeg(out / dpaths["full"], diff_image(small(bgr), small(base)), 88)
            m = {"sharp": {}, "seam": seam_stats[wn].get(vname, {})}
            for rname, centre in REGIONS.items():
                c = crop(bgr, centre)
                paths[rname] = f"img/{wn}/{vname}/{rname}.jpg"
                jpeg(out / paths[rname], c)
                m["sharp"][rname] = lapvar(c)
                if vname != "baseline":
                    dpaths[rname] = f"img/{wn}/{vname}/diff_{rname}.jpg"
                    jpeg(out / dpaths[rname], diff_image(c, crop(base, centre)), 88)
            m["sharp"]["full"] = lapvar(small(bgr))
            if vname in vs_base[wn]:
                m["vs_base"] = dict(vs_base[wn][vname])
                for rname in ("seam_left", "seam_right"):
                    g8 = [cv2.cvtColor(crop(x, REGIONS[rname]), cv2.COLOR_BGR2GRAY)
                          for x in (bgr, base)]
                    m["vs_base"][f"ssim_{rname}"] = vd._ssim(g8[0], g8[1])
            m["gradient"] = sky_gradient(bgr)
            if wn in dji_small and vname != "dji":
                aligned = vd.match_tone(vd.apply_rotation(small(bgr), dji_R[wn]), dji_small[wn])
                m["dji"] = vd.compare(aligned, dji_small[wn])
            manifest["images"][wn][vname] = paths
            manifest["diffs"][wn][vname] = dpaths
            manifest["metrics"][wn][vname] = m
            if "vs_base" in m:
                log.info("  %-12s %-12s vs baseline: %s", wn, vname,
                         {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in m["vs_base"].items()})
            log.info("  %-12s %-12s sharp(front)=%.0f seam=%s dji=%s", wn, vname,
                     m["sharp"]["front"], m["seam"] or "-",
                     {k: round(v, 3) for k, v in m["dji"].items() if k in ("ncc", "ssim")}
                     if "dji" in m else "-")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    template = (ROOT / "tools/ab_viewer.html").read_text()
    (out / "index.html").write_text(template.replace("/*__DATA__*/null", json.dumps(manifest)))
    log.info("wrote %s (open it in a browser)", out / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
