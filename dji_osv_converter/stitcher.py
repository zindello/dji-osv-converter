"""Dual-fisheye -> equirectangular stitching.

Pipeline per frame:
    1. remap each fisheye into equirectangular space (DJI's tables)
    2. blend the two using DJI's feathered masks as weights

Orientation correction is applied by rotating the sampling directions, so it
costs nothing extra at runtime - it is folded into the remap tables once.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .calibration import CameraMap

logger = logging.getLogger(__name__)


def warp(frame: np.ndarray, cam: CameraMap) -> np.ndarray:
    """Project one fisheye frame into equirectangular space."""
    return cv2.remap(frame, cam.map_x, cam.map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def _wrapped_distance(mask: np.ndarray) -> np.ndarray:
    """Distance (px) from each covered pixel to the edge of its coverage region.

    Computed on a horizontally tiled copy so the equirectangular seam at
    longitude +/-180 does not read as a coverage boundary.
    """
    h, w = mask.shape
    tiled = np.concatenate([mask, mask, mask], axis=1).astype(np.uint8)
    # border must be "outside" vertically only; pad top/bottom with zeros
    padded = cv2.copyMakeBorder(tiled, 1, 1, 0, 0, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, :]
    return dist[:, w:2 * w]


def seam_weights(cams: list[CameraMap], band_px: float = 292.0) -> np.ndarray:
    """Feathered blend weight for camera 0, in [0,1]; camera 1 gets 1 - w.

    DJI's masks are *coverage* masks, not blend weights - inside the overlap
    both are fully on. Averaging them directly double-exposes the overlap
    (visible ghosting). Instead we place the seam on the locus equidistant
    from both coverage boundaries and cross-fade over `band_px` around it,
    matching the 292 px blend band DJI itself uses (7680 * 0.038).
    """
    c0 = (cams[0].weight > 0.5) & cams[0].valid
    c1 = (cams[1].weight > 0.5) & cams[1].valid

    d0 = _wrapped_distance(c0)
    d1 = _wrapped_distance(c1)

    # >0 means deeper inside camera 0's coverage than camera 1's
    signed = d0 - d1
    w = np.clip(0.5 + signed / max(band_px, 1e-6), 0.0, 1.0).astype(np.float32)

    # Outside one camera's coverage the other must take the pixel outright.
    w[~c1] = 1.0
    w[~c0] = 0.0
    both_missing = ~c0 & ~c1
    if both_missing.any():
        # fall back to any geometrically valid sample rather than leaving a hole
        w[both_missing & cams[0].valid] = 1.0
        w[both_missing & ~cams[0].valid] = 0.0
    return w


def _wide_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Large-sigma Gaussian blur, computed cheaply on a downsampled copy.

    A direct GaussianBlur at sigma~480 needs a ~2880-tap kernel and was the
    single slowest thing in the stitch (>60s at 8K). The result is smooth by
    construction, so it is computed downsampled and scaled back - visually
    identical, dramatically faster.
    """
    h, w = img.shape[:2]
    f = max(1, int(round(sigma / 8.0)))
    if f <= 1:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def match_exposure(warped: list[np.ndarray], cams: list[CameraMap],
                   max_gain: float = 1.6, spatial: bool = True) -> list[np.ndarray]:
    """Equalise the two cameras' exposure/colour across the seam.

    The sensors meter independently AND each fisheye vignettes toward its rim,
    by different amounts. Measured in the overlap, the cam0/cam1 luma ratio is
    not a single number - it runs ~1.05 near the seam centre to ~1.9 toward the
    rim. A single scalar gain (spatial=False) therefore matches the middle and
    leaves a bright/dark gradient at the edges of the seam.

    spatial=True estimates a SMOOTH per-channel gain field from the overlap:
    each camera is nudged, by a heavily blurred local factor, so the two agree
    everywhere they see the same scene. The correction is feathered to unity at
    the edge of the overlap, so single-camera regions keep only the global gain
    and no new discontinuity is introduced. Vignetting is a lens artifact, so
    flattening it is faithful, not grading.
    """
    ov = cams[0].valid & cams[1].valid
    if ov.sum() < 5000:
        return warped

    a = warped[0].astype(np.float32)
    b = warped[1].astype(np.float32)
    lit = ov & (a.mean(axis=2) > 6) & (b.mean(axis=2) > 6)
    if lit.sum() < 5000:
        return warped

    # --- global per-channel gain: aligns overall level (unchanged behaviour) ---
    gains = []
    for ch in range(3):
        ma = float(a[..., ch][lit].mean()); mb = float(b[..., ch][lit].mean())
        ratio = (mb / ma) if (ma > 1e-3 and mb > 1e-3) else 1.0
        gains.append((float(np.clip(np.sqrt(ratio), 1/max_gain, max_gain)),
                      float(np.clip(1/np.sqrt(ratio), 1/max_gain, max_gain))))
    for ch in range(3):
        a[..., ch] *= gains[ch][0]; b[..., ch] *= gains[ch][1]

    if spatial:
        # Estimate the smooth cam0/cam1 log-ratio and DIFFUSE it across the
        # frame: a large normalised blur carries the overlap measurement into
        # each camera's exclusive region and decays to zero far away, so every
        # camera's gain map is smooth (no internal step) and vanishes where the
        # seam is irrelevant. Applied fully in the overlap - including the rim,
        # where the vignetting mismatch is largest.
        sigma = a.shape[1] / 16.0
        for ch in range(3):
            ca, cb = a[..., ch], b[..., ch]
            m = (lit & (ca > 1) & (cb > 1)).astype(np.float32)
            logr = np.zeros_like(ca)
            np.log(np.divide(ca, cb, out=np.ones_like(ca), where=m > 0),
                   out=logr, where=m > 0)
            num = _wide_blur(logr * m, sigma)
            den = _wide_blur(m, sigma)
            lr = np.zeros_like(num)
            np.divide(num, den, out=lr, where=den > 1e-4)
            lr = np.clip(lr, np.log(1/max_gain), np.log(max_gain)).astype(np.float32)
            a[..., ch] *= np.exp(-0.5 * lr)   # cam0 down half,
            b[..., ch] *= np.exp(+0.5 * lr)   # cam1 up half -> meet in the middle

    return [np.clip(a, 0, 255).astype(np.uint8), np.clip(b, 0, 255).astype(np.uint8)]


def blend(warped: list[np.ndarray], cams: list[CameraMap],
          band_px: float = 292.0, weights: np.ndarray | None = None) -> np.ndarray:
    """Cross-fade two projected frames across a narrow seam band."""
    w = seam_weights(cams, band_px) if weights is None else weights
    w3 = w[..., None]
    out = warped[0].astype(np.float32) * w3 + warped[1].astype(np.float32) * (1.0 - w3)
    return np.clip(out, 0, 255).astype(np.uint8)


def multiband_blend(warped: list[np.ndarray], cams: list[CameraMap],
                    weights: np.ndarray | None = None, band_px: float = 292.0,
                    levels: int = 6) -> np.ndarray:
    """Laplacian-pyramid (multi-band) blend across the seam.

    A residual exposure or vignetting difference between the two lenses is a
    LOW-frequency signal. Multi-band blending mixes low frequencies over a wide
    transition and high-frequency detail over a narrow one, so a smooth
    brightness difference fades across the whole overlap and never reads as a
    line - without having to measure or model it (which parallax makes
    unreliable). Fine detail still switches sharply at the seam, so there is no
    ghosting.

    Each camera's invalid region is filled with the other camera so the
    pyramids are defined across the whole blend; the mask makes that fill
    irrelevant to the result except at the coarsest levels, where it should be.
    """
    w = seam_weights(cams, band_px) if weights is None else weights
    a = warped[0].astype(np.float32)
    b = warped[1].astype(np.float32)
    v0, v1 = cams[0].valid, cams[1].valid
    # fill holes with the other camera so Laplacians don't see a black cliff
    a = np.where(v0[..., None], a, b)
    b = np.where(v1[..., None], b, a)

    def pyr_down_to(img, n):
        g = [img]
        for _ in range(n):
            g.append(cv2.pyrDown(g[-1]))
        return g

    ga, gb, gw = pyr_down_to(a, levels), pyr_down_to(b, levels), pyr_down_to(w, levels)
    out_levels = []
    for i in range(levels):
        size = (ga[i].shape[1], ga[i].shape[0])
        la = ga[i] - cv2.pyrUp(ga[i + 1], dstsize=size)
        lb = gb[i] - cv2.pyrUp(gb[i + 1], dstsize=size)
        wi = gw[i][..., None]
        out_levels.append(la * wi + lb * (1.0 - wi))
    wl = gw[levels][..., None]
    out_levels.append(ga[levels] * wl + gb[levels] * (1.0 - wl))

    img = out_levels[levels]
    for i in range(levels - 1, -1, -1):
        img = cv2.pyrUp(img, dstsize=(out_levels[i].shape[1], out_levels[i].shape[0])) + out_levels[i]
    return np.clip(img, 0, 255).astype(np.uint8)


def flow_blend(warped: list[np.ndarray], cams: list[CameraMap],
               weights: np.ndarray, scale: float = 0.25,
               max_shift: float = 80.0) -> np.ndarray:
    """Blend with optical-flow correction for parallax.

    The two lenses sit a few centimetres apart, so near objects land in
    different places in the two projections while distant ones align. Measured
    on real footage: median inter-camera displacement 0.2 px but p90 ~23 px.
    No static blend can reconcile that - the images genuinely disagree.

    So we morph across the seam: each image is progressively warped towards
    the other as the blend weight shifts, meaning features coincide wherever
    they are being mixed. This is what DJI's FlowStitchWarpFusion does.

    Flow is computed at reduced `scale` because flow fields are smooth and
    full-resolution Farneback on a 7680x3840 frame is needlessly slow.
    """
    h, w = warped[0].shape[:2]
    sw, sh = max(int(w * scale), 32), max(int(h * scale), 32)

    small = [cv2.cvtColor(cv2.resize(im, (sw, sh), interpolation=cv2.INTER_AREA),
                          cv2.COLOR_BGR2GRAY) for im in warped]
    f01 = cv2.calcOpticalFlowFarneback(small[0], small[1], None,
                                       0.5, 4, 25, 3, 7, 1.5, 0)
    f10 = cv2.calcOpticalFlowFarneback(small[1], small[0], None,
                                       0.5, 4, 25, 3, 7, 1.5, 0)

    # Forward-backward consistency. Where the two directions disagree the flow
    # is guesswork - typically flat, untextured areas - and applying it tears
    # the image. Trust only mutually consistent flow.
    gxs, gys = np.meshgrid(np.arange(sw, dtype=np.float32),
                           np.arange(sh, dtype=np.float32))
    back = cv2.remap(f10, gxs + f01[..., 0], gys + f01[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    resid = np.linalg.norm(f01 + back, axis=2)
    mag = np.linalg.norm(f01, axis=2)
    conf = np.clip(1.0 - resid / (0.5 * mag + 2.0), 0.0, 1.0).astype(np.float32)
    conf = cv2.GaussianBlur(conf, (0, 0), 3.0)

    # Smooth and gate the flow before it is amplified by the upscale.
    def clean(f):
        f = f * conf[..., None]
        f = cv2.GaussianBlur(f, (0, 0), 4.0)
        big = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
        big[..., 0] *= w / sw
        big[..., 1] *= h / sh
        return np.clip(big, -max_shift, max_shift)

    f01, f10 = clean(f01), clean(f10)

    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    a = weights[..., None]

    # Only morph inside the transition band. Where one camera owns the pixel
    # outright there is nothing to reconcile, and warping there would drag in
    # neighbouring content for no reason.
    band = (4.0 * weights * (1.0 - weights)).astype(np.float32)   # 0 at edges, 1 mid-band
    t0 = ((1.0 - weights) * band).astype(np.float32)
    t1 = (weights * band).astype(np.float32)

    m0 = cv2.remap(warped[0], gx + f01[..., 0] * t0, gy + f01[..., 1] * t0,
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m1 = cv2.remap(warped[1], gx + f10[..., 0] * t1, gy + f10[..., 1] * t1,
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    out = m0.astype(np.float32) * a + m1.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def stitch(frame0: np.ndarray, frame1: np.ndarray,
           cams: list[CameraMap], band_px: float = 292.0,
           weights: np.ndarray | None = None) -> np.ndarray:
    """Stitch a dual-fisheye pair into one equirectangular frame.

    `weights` may be precomputed once with seam_weights() and reused across
    frames - it depends only on calibration, not on image content.
    """
    warped = [warp(frame0, cams[0]), warp(frame1, cams[1])]
    return blend(warped, cams, band_px, weights)


def coverage_report(cams: list[CameraMap]) -> dict:
    """Diagnostics for verifying a calibration set without rendering."""
    v0, v1 = cams[0].valid, cams[1].valid
    tot = cams[0].weight + cams[1].weight
    return {
        "cam0_valid_pct": 100 * float(v0.mean()),
        "cam1_valid_pct": 100 * float(v1.mean()),
        "union_pct": 100 * float((v0 | v1).mean()),
        "geometric_overlap_pct": 100 * float((v0 & v1).mean()),
        "blend_overlap_pct": 100 * float(
            ((cams[0].weight > 0.01) & (cams[1].weight > 0.01)).mean()),
        "uncovered_pct": 100 * float((tot <= 1e-6).mean()),
    }


__all__ = ["warp", "blend", "multiband_blend", "flow_blend", "seam_weights",
           "stitch", "match_exposure", "coverage_report"]
