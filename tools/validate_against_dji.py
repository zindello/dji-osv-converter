#!/usr/bin/env python3
"""Validate our stitch against DJI Studio's own export of the same file.

DJI's export is ground truth for geometry. Two differences are NOT errors and
are compensated before scoring:

  * Yaw. Gravity levelling fixes pitch and roll only; heading is arbitrary.
    We solve for the rotation that best aligns the two before comparing.
  * Tone. DJI grades its output (MltTonalEnhance / filter_normalize were both
    observed in a live stack trace). We match brightness per channel first.

Reported metrics are therefore about *geometry and structure*, which is what
the stitcher is responsible for.

Usage:
    python tools/validate_against_dji.py INPUT.OSV REFERENCE.mp4 [frames...]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dji_osv_converter.orientation import directions, dirs_to_uv, rot_matrix  # noqa: E402
from dji_osv_converter.pipeline import Stitcher360  # noqa: E402

log = logging.getLogger("validate")


def extract_frame(path: Path, index: int, stream: int | None = None) -> np.ndarray:
    """Pull a single frame out of a video with ffmpeg."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "f.png"
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(path)]
        if stream is not None:
            cmd += ["-map", f"0:v:{stream}"]
        cmd += ["-vf", f"select=eq(n\\,{index})", "-frames:v", "1", "-update", "1", str(out)]
        subprocess.run(cmd, check=True)
        img = cv2.imread(str(out), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not extract frame {index} from {path}")
    return img


def _norm(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def find_alignment(ours: np.ndarray, ref: np.ndarray,
                   coarse_size: tuple[int, int] = (512, 256),
                   fine_width: int = 1920) -> tuple[np.ndarray, float]:
    """Solve for the rotation mapping our panorama onto the reference.

    Refined to well under a pixel: at 3840 px width one degree of yaw is
    10.7 px, so a 0.5 deg search step leaves a ~5 px residual that shows up as
    edge-only differences and understates true agreement. The coarse pass runs
    small for speed, then refinement runs at `fine_width` with steps down to
    0.01 deg (~0.05 px at 3840).
    """
    def prep(img, W, H):
        return (cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA),
                _norm(cv2.cvtColor(cv2.resize(ref, (W, H), interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2GRAY)))

    def make_score(W, H):
        src, tgt = prep(ours, W, H)
        base = directions(H, W)

        def score(y, p, r):
            d = base @ rot_matrix(y, p, r).T
            u, v = dirs_to_uv(d, W, H)
            q = cv2.remap(src, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            return float((_norm(cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)) * tgt).mean())
        return score

    coarse = make_score(*coarse_size)
    best = (-9.0, (0.0, 0.0, 0.0))
    for y in range(0, 360, 10):
        for p in range(-40, 41, 10):
            for r in range(-40, 41, 10):
                s = coarse(y, p, r)
                if s > best[0]:
                    best = (s, (float(y), float(p), float(r)))

    fine = make_score(fine_width, fine_width // 2)
    best = (fine(*best[1]), best[1])
    for step in (4.0, 1.0, 0.25, 0.06, 0.015):
        improved = True
        while improved:                      # walk downhill at this step size
            improved = False
            y0, p0, r0 = best[1]
            for dy in (-step, 0.0, step):
                for dp in (-step, 0.0, step):
                    for dr in (-step, 0.0, step):
                        if dy == dp == dr == 0.0:
                            continue
                        cand = (y0 + dy, p0 + dp, r0 + dr)
                        s = fine(*cand)
                        if s > best[0] + 1e-6:
                            best = (s, cand)
                            improved = True
    return rot_matrix(*best[1]), best[0]


def apply_rotation(img: np.ndarray, R: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    d = directions(h, w) @ R.T
    u, v = dirs_to_uv(d, w, h)
    return cv2.remap(img, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def match_tone(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Monotonic per-channel histogram match, src -> ref.

    DJI grades its export with a nonlinear tone curve (MltTonalEnhance /
    filter_normalize, seen in the live stack trace). A gain+offset cannot
    absorb that, so residual brightness masquerades as geometric error. A
    monotonic histogram match applies whatever 1-D curve best aligns our tones
    to DJI's - it cannot move a pixel spatially, so geometry is untouched and
    what remains after it is genuinely structural.
    """
    out = np.empty_like(src)
    for c in range(3):
        a = src[..., c].ravel()
        b = ref[..., c].ravel()
        a_vals, a_idx, a_cnt = np.unique(a, return_inverse=True, return_counts=True)
        b_vals, b_cnt = np.unique(b, return_counts=True)
        a_cdf = np.cumsum(a_cnt).astype(np.float64) / a.size
        b_cdf = np.cumsum(b_cnt).astype(np.float64) / b.size
        lut = np.interp(a_cdf, b_cdf, b_vals)
        out[..., c] = lut[a_idx].reshape(src[..., c].shape).astype(np.uint8)
    return out


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM on luma, Gaussian-windowed (Wang et al.)."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    win = (11, 11); sig = 1.5
    mu_a = cv2.GaussianBlur(a, win, sig); mu_b = cv2.GaussianBlur(b, win, sig)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.GaussianBlur(a * a, win, sig) - mu_a2
    sb = cv2.GaussianBlur(b * b, win, sig) - mu_b2
    sab = cv2.GaussianBlur(a * b, win, sig) - mu_ab
    ssim = ((2 * mu_ab + C1) * (2 * sab + C2)) / ((mu_a2 + mu_b2 + C1) * (sa + sb + C2))
    return float(ssim.mean())


def compare(ours: np.ndarray, ref: np.ndarray) -> dict:
    a = cv2.cvtColor(ours, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(a - b)
    mse = float((diff ** 2).mean())
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 1e-9 else float("inf")

    ga = cv2.Sobel(a, cv2.CV_32F, 1, 0, 3), cv2.Sobel(a, cv2.CV_32F, 0, 1, 3)
    gb = cv2.Sobel(b, cv2.CV_32F, 1, 0, 3), cv2.Sobel(b, cv2.CV_32F, 0, 1, 3)
    ma = np.sqrt(ga[0] ** 2 + ga[1] ** 2)
    mb = np.sqrt(gb[0] ** 2 + gb[1] ** 2)
    edge_ncc = float((_norm(ma) * _norm(mb)).mean())

    return {
        "ncc": float((_norm(a) * _norm(b)).mean()),
        "ssim": _ssim(a, b),
        "edge_ncc": edge_ncc,
        "psnr_db": psnr,
        "mean_abs_err": float(diff.mean()),
        "pct_within_8": 100.0 * float((diff <= 8).mean()),
        "pct_within_16": 100.0 * float((diff <= 16).mean()),
        "pct_within_32": 100.0 * float((diff <= 32).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("osv")
    ap.add_argument("reference")
    ap.add_argument("frames", nargs="*", type=int, default=[0, 150, 600])
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--outdir", default="output/validation")
    ap.add_argument("--flow", action="store_true", help="enable flow seam blending")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    W, H = args.width, args.width // 2
    st = Stitcher360.from_file(args.osv, width=W, height=H,
                               max_frames=max(args.frames) + 30)

    print(f"\n{'frame':>6} {'NCC':>7} {'SSIM':>7} {'edgeNCC':>8} {'PSNR':>7} {'MAE':>6} "
          f"{'<=8':>7} {'<=16':>7} {'<=32':>7}")
    print("-" * 62)
    rows = []
    for n in args.frames:
        f0 = extract_frame(Path(args.osv), n, 0)
        f1 = extract_frame(Path(args.osv), n, 1)
        ours = st.stitch(f0, f1, frame=n, flow=args.flow)
        ref = cv2.resize(extract_frame(Path(args.reference), n), (W, H),
                         interpolation=cv2.INTER_AREA)

        R, coarse = find_alignment(ours, ref)
        aligned = match_tone(apply_rotation(ours, R), ref)
        m = compare(aligned, ref)
        rows.append(m)
        print(f"{n:>6} {m['ncc']:>7.4f} {m['ssim']:>7.4f} {m['edge_ncc']:>8.4f} {m['psnr_db']:>7.2f} "
              f"{m['mean_abs_err']:>6.2f} {m['pct_within_8']:>6.1f}% "
              f"{m['pct_within_16']:>6.1f}% {m['pct_within_32']:>6.1f}%")

        cv2.imwrite(str(outdir / f"frame{n:05d}_ours.jpg"),
                    cv2.resize(aligned, (2400, 1200)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(outdir / f"frame{n:05d}_dji.jpg"),
                    cv2.resize(ref, (2400, 1200)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        d = cv2.absdiff(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
        cv2.imwrite(str(outdir / f"frame{n:05d}_diff.jpg"),
                    cv2.resize(cv2.applyColorMap(np.clip(d * 4, 0, 255).astype(np.uint8),
                                                 cv2.COLORMAP_INFERNO), (2400, 1200)),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    print("-" * 62)
    avg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"{'mean':>6} {avg['ncc']:>7.4f} {avg['ssim']:>7.4f} {avg['edge_ncc']:>8.4f} {avg['psnr_db']:>7.2f} "
          f"{avg['mean_abs_err']:>6.2f} {avg['pct_within_8']:>6.1f}% "
          f"{avg['pct_within_16']:>6.1f}% {avg['pct_within_32']:>6.1f}%")
    print(f"\nwrote comparisons to {outdir}/")
    print("NCC/edge-NCC near 1.0 means the geometry matches DJI's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
