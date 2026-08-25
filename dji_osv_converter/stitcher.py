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


def band_rows(weights: np.ndarray, margin: int | None = None) -> tuple[int, int]:
    """Row range [r0, r1) of the blend band (plus context margin)."""
    from .constants import SEAM_FLOW_MARGIN_PX
    m = SEAM_FLOW_MARGIN_PX if margin is None else margin
    band = (weights > 0.001) & (weights < 0.999)
    rows = np.where(band.any(axis=1))[0]
    return max(int(rows.min()) - m, 0), min(int(rows.max()) + m + 1, weights.shape[0])


def seam_flow_small(gray_a: np.ndarray, gray_b: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Gated, band-smoothed flow a->b and b->a for the seam band (small res).

    ``gray_a/b`` are the two lenses' warped band crops as small uint8 grays
    (downscaled by SEAM_FLOW_SCALE). Returns (sh, sw, 2) fields in SMALL-res
    pixel units - callers upsample and rescale (CPU: seam_flow_fields; GPU:
    on-device interpolate).

    Guardrails, each of which the naive full-frame version lacked (that
    version bent straight objects far worse than no correction at all):
    forward-backward consistency gating, then ANISOTROPIC smoothing - heavy
    along the seam so the morph cannot vary abruptly and shear structures
    that cross it, light across the seam. The hard shift clamp is applied at
    full res by the caller.
    """
    from .constants import SEAM_FLOW_SIGMA_ACROSS, SEAM_FLOW_SIGMA_ALONG
    sh, sw = gray_a.shape
    f01 = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 3, 25, 3, 7, 1.5, 0)
    f10 = cv2.calcOpticalFlowFarneback(gray_b, gray_a, None, 0.5, 3, 25, 3, 7, 1.5, 0)
    gxs, gys = np.meshgrid(np.arange(sw, dtype=np.float32),
                           np.arange(sh, dtype=np.float32))
    back = cv2.remap(f10, gxs + f01[..., 0], gys + f01[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    resid = np.linalg.norm(f01 + back, axis=2)
    mag = np.linalg.norm(f01, axis=2)
    conf = np.clip(1.0 - resid / (0.5 * mag + 2.0), 0.0, 1.0).astype(np.float32)
    conf = cv2.GaussianBlur(conf, (0, 0), 3.0)

    def clean(f):
        f = f * conf[..., None]
        return cv2.GaussianBlur(f, (0, 0), sigmaX=SEAM_FLOW_SIGMA_ALONG,
                                sigmaY=SEAM_FLOW_SIGMA_ACROSS)

    return clean(f01), clean(f10)


def seam_flow_fields(gray_a: np.ndarray, gray_b: np.ndarray,
                     out_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Full-band-res (h, w, 2) pixel-displacement fields (see seam_flow_small)."""
    from .constants import SEAM_FLOW_MAX_SHIFT_PX
    sh, sw = gray_a.shape
    h, w = out_size
    small = seam_flow_small(gray_a, gray_b)

    def up(f):
        big = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
        big[..., 0] *= w / sw
        big[..., 1] *= h / sh
        return np.clip(big, -SEAM_FLOW_MAX_SHIFT_PX, SEAM_FLOW_MAX_SHIFT_PX)

    return up(small[0]), up(small[1])


def seam_luma_offsets(A: np.ndarray, B: np.ndarray, inner: np.ndarray,
                      lo: float, hi: float, n_lut: int, sigma: float,
                      agree_t: float) -> tuple[np.ndarray, np.ndarray]:
    """Empirical inter-lens photometric mismatch in the overlap band.

    Returns ``(lut, spatial)``: ``lut[v]`` is the median (B - A) as a
    function of luma value v (16 bins, interpolated to ``n_lut`` entries)
    and ``spatial`` a smooth per-pixel residual field at the input's
    resolution. Split-apply: ``A += (lut[A]+spatial)/2``,
    ``B -= (lut[B]+spatial)/2``, ramped inside the band.

    Two passes, both from pixels where the lenses agree geometrically:

    1. The luma LUT plus a per-column residual, gated tightly (``agree_t``).
    2. A smooth 2-D residual field with a 3x WIDER gate. Pass 1's tight gate
       has a SELECTION BIAS: where the mismatch is worst (dark content under
       asymmetric lighting: median |A-B| of 70/1024, 77% of pixels above the
       tight gate) the mismatched pixels are excluded as "disagreeing" and
       the estimate misses them. After pass 1 removes the bulk, the wide
       gate can admit them safely; heavy smoothing keeps real content
       differences (parallax - the dynamic seam's job) out of the field.

    Model-free on purpose: gain and additive-linear-glare models were tried
    and measured to under-correct. Under even lighting everything is ~0 and
    this is a no-op (worst-case clip: -1..-2.5/1024).
    """
    h0, w0 = A.shape
    nb = 16
    clamp = (hi - lo) * 0.05
    # Everything runs at an extra 4x4 downsample: the outputs are smooth by
    # construction and full-res numpy here cost 119 ms/frame (blurs and
    # nanmedian dominated); downsampled it is ~15 ms with the same result.
    ds = 4
    h, w = (h0 // ds) * ds, (w0 // ds) * ds
    blk = lambda x: x[:h, :w].reshape(h // ds, ds, w // ds, ds).mean((1, 3))
    A = blk(A.astype(np.float32))
    B = blk(B.astype(np.float32))
    inner = blk(inner.astype(np.float32)) > 0.99
    h, w = A.shape
    agree = inner & (np.abs(A - B) < agree_t)
    a_v, d_v = A[agree], (B - A)[agree]
    edges = np.linspace(lo, hi, nb + 1)
    idx = np.clip(np.digitize(a_v, edges) - 1, 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float32)
    ssum = np.bincount(idx, weights=d_v, minlength=nb)
    lut_b = np.where(cnt > 50, ssum / np.maximum(cnt, 1), np.nan)
    centers = (edges[:-1] + edges[1:]) * 0.5
    valid = ~np.isnan(lut_b)
    if not valid.any():
        return np.zeros(n_lut, np.float32), np.zeros((h0, w0), np.float32)
    lb = np.interp(centers, centers[valid], lut_b[valid]).astype(np.float32)
    lb = cv2.GaussianBlur(lb.reshape(1, -1), (0, 0), 1.5).ravel()
    lut = np.clip(np.interp(np.arange(n_lut, dtype=np.float32), centers, lb,
                            left=lb[0], right=lb[-1]), -clamp, clamp).astype(np.float32)
    res = (B - A) - lut[np.clip(A.astype(np.int64), 0, n_lut - 1)]
    o = (res * agree).sum(0) / np.maximum(agree.sum(0), 1)
    o = cv2.GaussianBlur(o.reshape(1, -1).astype(np.float32), (0, 0),
                         sigmaX=sigma / ds, sigmaY=0.1).ravel()
    o = np.clip(o, -clamp * 0.75, clamp * 0.75)

    res2 = (res - o[None, :]).astype(np.float32)
    sy = max(2.0, h / 18.0)
    field = np.zeros_like(res2)
    # Two refinement iterations: masked Gaussian smoothing dilutes the field
    # at region edges (pulls toward the surrounding zeros), so re-estimating
    # on the remaining residual converges it. The field's clamp is 2x the
    # LUT's: the dark-region mismatch reaches 70/1024 (rim stray light,
    # present on BOTH sample clips) and a 5% clamp was measured binding.
    for _ in range(2):
        r = res2 - field
        ag = inner & (np.abs(r) < 3.0 * agree_t)
        num = cv2.GaussianBlur(r * ag, (0, 0), sigmaX=sigma * 1.3 / ds, sigmaY=sy)
        den = cv2.GaussianBlur(ag.astype(np.float32), (0, 0),
                               sigmaX=sigma * 1.3 / ds, sigmaY=sy)
        field = np.clip(field + num / np.maximum(den, 1e-3),
                        -2.0 * clamp, 2.0 * clamp)
    spatial = (o[None, :] + field).astype(np.float32)
    return lut, cv2.resize(spatial, (w0, h0), interpolation=cv2.INTER_LINEAR)


def seam_path(cost: np.ndarray, out_w: int) -> np.ndarray:
    """Min-disagreement seam row per column through the blend band.

    ``cost`` is (band_h, n) - post-morph |A-B|, already masked to the true
    blend band (large values outside it). Dynamic programming finds the
    cheapest left-to-right path with |dy| <= 1 per column step; the path is
    then Gaussian-smoothed and upsampled to ``out_w`` columns.

    This is what handles OCCLUSION at the seam: optical flow aligns what both
    lenses can see, but near an object they see different sides of it, and
    any wide cross-fade dissolves the object (transparency ghosting). The
    narrow feather around this path cuts where the lenses actually agree.
    """
    from .constants import SEAM_DYN_COLS, SEAM_DYN_SIGMA
    h, _ = cost.shape
    cs = cv2.resize(cost, (SEAM_DYN_COLS, h), interpolation=cv2.INTER_AREA)
    n = cs.shape[1]
    D = cs.copy()
    for x in range(1, n):
        prev = D[:, x - 1]
        m = np.minimum(np.minimum(np.roll(prev, 1), prev), np.roll(prev, -1))
        m[0] = min(prev[0], prev[1])
        m[-1] = min(prev[-1], prev[-2])
        D[:, x] = cs[:, x] + m
    path = np.zeros(n, dtype=np.int32)
    path[-1] = int(np.argmin(D[:, -1]))
    for x in range(n - 2, -1, -1):
        p = path[x + 1]
        lo, hi = max(p - 1, 0), min(p + 2, h)
        path[x] = lo + int(np.argmin(D[lo:hi, x]))
    pf = cv2.GaussianBlur(path.astype(np.float32).reshape(1, -1),
                          (0, 0), SEAM_DYN_SIGMA).ravel()
    return np.interp(np.arange(out_w), np.linspace(0, out_w - 1, n),
                     pf).astype(np.float32)


def seam_gate(diff: np.ndarray, inner: np.ndarray, out_w: int) -> np.ndarray:
    """Per-column mix factor [0,1] between wide fade (0) and narrow cut (1).

    A narrow cut through photometrically-mismatched but ALIGNED areas (flat
    floor, walls) draws a visible line - on the sphere, a ring around the
    whole seam that jumps as the path re-solves per frame. The wide fade is
    invisible there by construction. So the narrow dynamic seam applies only
    in columns whose post-morph disagreement says the parallax is real.
    """
    from .constants import (SEAM_DYN_COLS, SEAM_DYN_GATE_HI, SEAM_DYN_GATE_LO,
                            SEAM_DYN_GATE_SIGMA)
    # Column score = MEAN blurred disagreement over the band. A max spikes on
    # single noisy pixels and saturates the gate (measured: 92% of columns
    # read "parallax" on a frame with one bar); the mean separates cleanly
    # (median ~6 on aligned flat areas vs 20-40 at real parallax).
    db = cv2.GaussianBlur(np.where(inner, diff, 0.0).astype(np.float32), (0, 0), 3)
    n = np.maximum(inner.sum(axis=0), 1)
    score = (db.sum(axis=0) / n).astype(np.float32)                # (w,)
    s = cv2.resize(score.reshape(1, -1), (SEAM_DYN_COLS, 1),
                   interpolation=cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), sigmaX=SEAM_DYN_GATE_SIGMA, sigmaY=0.1).ravel()
    g = np.clip((s - SEAM_DYN_GATE_LO) / (SEAM_DYN_GATE_HI - SEAM_DYN_GATE_LO),
                0.0, 1.0)
    return np.interp(np.arange(out_w), np.linspace(0, out_w - 1, len(g)),
                     g).astype(np.float32)


def flow_blend(warped: list[np.ndarray], cams: list[CameraMap],
               weights: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Blend with band-limited optical-flow correction for parallax.

    The two lenses sit a few centimetres apart, so near objects land in
    different places in the two projections while distant ones align. Measured
    on real footage: median inter-camera displacement 0.2 px but p90 ~23 px.
    No static blend can reconcile that - the images genuinely disagree. So
    each image is progressively morphed towards the other as the blend weight
    shifts, and features coincide wherever they are being mixed (DJI's
    FlowStitchWarpFusion does the same).

    Everything happens INSIDE the blend band (plus a context margin): outside
    it one camera owns the pixel outright and there is nothing to reconcile.
    An earlier full-frame version of this existed and made seams WORSE -
    unconstrained Farneback over the whole pano produced large phantom bends.
    """
    from .constants import (SEAM_FLOW_SCALE, SEAM_OFFSET_AGREE_T,
                            SEAM_OFFSET_SIGMA)
    scale = SEAM_FLOW_SCALE if scale is None else scale
    r0, r1 = band_rows(weights)

    def grays(A, B):
        sw, sh = max(int(A.shape[1] * scale), 32), max(int(A.shape[0] * scale), 32)
        ga = cv2.cvtColor(cv2.resize(A, (sw, sh), interpolation=cv2.INTER_AREA),
                          cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cv2.resize(B, (sw, sh), interpolation=cv2.INTER_AREA),
                          cv2.COLOR_BGR2GRAY)
        return ga, gb

    ga, gb = grays(warped[0][r0:r1], warped[1][r0:r1])
    # Veiling-glare correction (constants.SEAM_OFFSET_*): per-column additive
    # offset from geometrically AGREEING pixels, split across the lenses and
    # ramped to zero at the band margins (mirrors the GPU path; 8-bit here).
    inner_s = cv2.resize(((weights[r0:r1] > 0.001) & (weights[r0:r1] < 0.999))
                         .astype(np.float32), ga.shape[::-1]) > 0.5
    lut, spatial = seam_luma_offsets(ga.astype(np.float32), gb.astype(np.float32),
                                     inner_s, lo=0.0, hi=255.0, n_lut=256,
                                     sigma=SEAM_OFFSET_SIGMA,
                                     agree_t=SEAM_OFFSET_AGREE_T / 4.0)
    h_band, w_full = r1 - r0, warped[0].shape[1]
    sp_full = cv2.resize(spatial, (w_full, h_band), interpolation=cv2.INTER_LINEAR)
    wc_ = weights[r0:r1]
    inner_rows = np.where(((wc_ > 0.001) & (wc_ < 0.999)).any(axis=1))[0]
    ri0, ri1 = int(inner_rows.min()), int(inner_rows.max())
    rp = np.ones(h_band, np.float32)
    if ri0 > 0:
        rp[:ri0] = np.linspace(0.0, 1.0, ri0, endpoint=False)
    if ri1 < h_band - 1:
        rp[ri1 + 1:] = np.linspace(1.0, 0.0, h_band - ri1 - 1)
    w0 = warped[0].astype(np.float32)
    w1 = warped[1].astype(np.float32)
    for wimg, sgn in ((w0, 0.5), (w1, -0.5)):
        gray = cv2.cvtColor(wimg[r0:r1].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        corr = (lut[gray] + sp_full) * rp[:, None] * sgn
        wimg[r0:r1] += corr[..., None]
    warped = [np.clip(w0, 0, 255).astype(np.uint8),
              np.clip(w1, 0, 255).astype(np.uint8)]
    ga, gb = grays(warped[0][r0:r1], warped[1][r0:r1])

    A, B = warped[0][r0:r1], warped[1][r0:r1]
    h, w = A.shape[:2]
    f01, f10 = seam_flow_fields(ga, gb, (h, w))

    wc = weights[r0:r1]
    band = (4.0 * wc * (1.0 - wc)).astype(np.float32)   # 0 at edges, 1 mid-band
    t0 = ((1.0 - wc) * band).astype(np.float32)
    t1 = (wc * band).astype(np.float32)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    m0 = cv2.remap(A, gx + f01[..., 0] * t0, gy + f01[..., 1] * t0,
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m1 = cv2.remap(B, gx + f10[..., 0] * t1, gy + f10[..., 1] * t1,
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Dynamic seam + narrow feather where the lenses DISAGREE; the static wide
    # fade where they agree (see seam_path and seam_gate for why both exist).
    from .constants import SEAM_DYN_FEATHER_PX
    diff = np.abs(m0.astype(np.int16) - m1.astype(np.int16)).max(axis=2).astype(np.float32)
    inner = (wc > 0.001) & (wc < 0.999)
    cost = np.where(inner, diff, 1e6).astype(np.float32)
    seam = seam_path(cost, w)
    g = seam_gate(diff, inner, w)                       # (w,) in [0,1]
    yy = np.arange(h, dtype=np.float32)[:, None]
    wd = np.clip(0.5 + (seam[None, :] - yy) / SEAM_DYN_FEATHER_PX, 0.0, 1.0)
    if wc[0].mean() < 0.5:          # orient: give each side to the lens that owns it
        wd = 1.0 - wd
    wd = g[None, :] * wd + (1.0 - g[None, :]) * wc      # gate: narrow cut only
                                                        # where parallax is real
    a = weights[..., None].astype(np.float32)
    out = warped[0].astype(np.float32) * a + warped[1].astype(np.float32) * (1.0 - a)
    out[r0:r1] = m0.astype(np.float32) * wd[..., None] + m1.astype(np.float32) * (1.0 - wd[..., None])
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


__all__ = ["warp", "blend", "multiband_blend", "flow_blend", "band_rows",
           "seam_flow_small", "seam_flow_fields", "seam_path", "seam_gate",
           "seam_luma_offsets", "seam_weights",
           "stitch", "match_exposure", "coverage_report"]
