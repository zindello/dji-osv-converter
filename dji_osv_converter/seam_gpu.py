"""The seam stack on the device: photometric match, parallax flow, dynamic seam.

``stitcher.py`` holds the reference CPU implementation (numpy + OpenCV):
``seam_luma_offsets``, ``seam_flow_small`` (DIS optical flow), ``seam_path``
(a 960-step dynamic programme) and ``seam_gate``. On the stitch thread that
costs two device->host syncs per frame plus ~30 ms of single-threaded host
work on an M3 - several times that on the production host's Ivy Bridge
cores - during which the GPU idles. This module does the same work in torch
on the stitcher's device, with static shapes and NO host sync: every
decision the CPU code takes with an ``if`` (the ``seam_flow="auto"``
detectors) is a 0/1 tensor multiplied into the result, so the stitch thread
only queues kernels.

What is the same as the CPU code, to float rounding:

- the photometric estimator (``seam_luma_offsets``): same bins, gates,
  Gaussian kernels (OpenCV's kernel-size rule and BORDER_REFLECT_101) and
  clamps;
- the forward-backward consistency gate and the anisotropic smoothing of
  the flow;
- the dynamic-seam gate (``seam_gate``);
- the dynamic-seam path: the same recurrence solved exactly, in parallel
  (see :meth:`SeamGpu.seam_path`). Only exact ties can resolve differently.

What is different: the optical flow itself. DIS (OpenCV, CPU only) is
replaced by a dense, mean-normalised, coarse-to-fine Lucas-Kanade solve at
the same two pyramid scales DIS's fast preset uses on this band (1/8 and
1/4 of the flow gray). Its output goes through the same gate, smoothing and
clamp, so the morph has the same character; it is not bit-comparable, and
is validated by the warp residual and the A/B frames (PERFORMANCE.md).

Pure torch: runs on CUDA, MPS and the CPU device.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:                      # pragma: no cover - torch is optional
    torch = None

from .constants import (SEAM_AUTO_PHOTO_T, SEAM_DP_BLOCK, SEAM_DP_BLOCK_CUDA, SEAM_DYN_COLS, SEAM_DYN_FEATHER_PX,
                        SEAM_DYN_GATE_HI, SEAM_DYN_GATE_LO, SEAM_DYN_GATE_SIGMA,
                        SEAM_DYN_SIGMA, SEAM_FLOW_MAX_SHIFT_PX, SEAM_FLOW_SCALE,
                        SEAM_FLOW_SIGMA_ACROSS, SEAM_FLOW_SIGMA_ALONG,
                        SEAM_LK_DAMP, SEAM_LK_ITERS, SEAM_LK_LEVELS,
                        SEAM_LK_WINDOW_SIGMA, SEAM_OFFSET_AGREE_T,
                        SEAM_OFFSET_SIGMA)

# A Gaussian along an axis of at most this many samples is one matmul with a
# precomputed (n, n) matrix that has the border handling folded in; longer
# axes use pad + conv (the matrix would be large and mostly zero).
_BLUR_MATRIX_MAX = 1024


def gauss_kernel(sigma: float) -> np.ndarray:
    """OpenCV's Gaussian kernel for a float image and ``ksize=(0, 0)``."""
    k = int(round(sigma * 8 + 1)) | 1
    x = np.arange(k, dtype=np.float64) - k // 2
    g = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return g / g.sum()


def _reflect101(i: np.ndarray, n: int) -> np.ndarray:
    if n == 1:
        return np.zeros_like(i)
    period = 2 * (n - 1)
    i = np.mod(i, period)
    return np.where(i >= n, period - i, i)


def blur_matrix(n: int, sigma: float) -> np.ndarray:
    """(n, n) matrix M with ``M @ x`` = cv2.GaussianBlur of x along that axis."""
    g = gauss_kernel(sigma)
    r = len(g) // 2
    M = np.zeros((n, n), dtype=np.float64)
    rows = np.arange(n)
    for k in range(-r, r + 1):
        np.add.at(M, (rows, _reflect101(rows + k, n)), g[k + r])
    return M


class SeamGpu:
    """Device statics and the per-frame seam computations for one geometry.

    ``inner4``: (bh, W/4) bool, the inner blend band at 1/4 width (the
    stitcher's ``_sf_inner4``). ``w_band``: (bh, W) device tensor, the
    static blend weight over the band rows. ``cam0_top``: lens 0 owns the
    top of the band.
    """

    def __init__(self, inner4: np.ndarray, w_band, cam0_top: bool,
                 y_lo: float, y_hi: float, device):
        self.dev = torch.device(device)
        dev = self.dev
        self.bh, self.w4 = inner4.shape
        self.W = int(w_band.shape[-1])
        self.cam0_top = bool(cam0_top)
        self.lo, self.hi = float(y_lo), float(y_hi)
        f32 = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)
        self._bm_cache: dict = {}
        self._kern_cache: dict = {}

        # inner band at 1/4 width
        self.inner = torch.from_numpy(np.ascontiguousarray(inner4)).to(dev)[None, None]
        self.innerf = self.inner.to(torch.float32)
        self.inner_n = float(max(int(inner4.sum()), 1))
        self.inner_col = f32(np.maximum(inner4.sum(0), 1))[None, None]     # (1,1,w4)
        self.has_inner = bool(inner4.any())

        # ---- photometric estimator statics (stitcher.seam_luma_offsets) ----
        ds, nb = 4, 16
        self._ds, self._nb = ds, nb
        h, w = (self.bh // ds) * ds, (self.w4 // ds) * ds
        blk = inner4[:h, :w].astype(np.float32).reshape(h // ds, ds, w // ds, ds).mean((1, 3))
        self.inner16 = torch.from_numpy(blk > 0.99).to(dev)[None, None]
        self._ph_h = h // ds
        self._clamp = (self.hi - self.lo) * 0.05
        edges = np.linspace(self.lo, self.hi, nb + 1)
        centers = (edges[:-1] + edges[1:]) * 0.5
        # lut = interp(arange(1024), centres, lb, left=lb[0], right=lb[-1]) is
        # linear in lb: one (1024, nb) matrix
        xs = np.arange(1024, dtype=np.float64)
        self.lut_interp = f32(np.stack([np.interp(xs, centers, np.eye(nb)[j])
                                        for j in range(nb)], 1))
        ar = np.arange(nb)
        self._bin_i = torch.from_numpy(ar).to(dev)[None, :]               # candidates
        self._bin_j = torch.from_numpy(ar).to(dev)[:, None]               # bins
        self._bin_f = f32(ar)

        # ---- flow statics ----
        k = max(int(round(1.0 / SEAM_FLOW_SCALE)), 1)
        self._gk = k
        self.sh, self.sw = self.bh // k, self.w4
        # validity of each flow-gray pixel: the fraction of its rows inside
        # the inner band. Outside it one lens has no image; those pixels are
        # kept out of the Lucas-Kanade sums.
        bh4 = self.sh * k
        self._flow_mask = f32(inner4[:bh4].astype(np.float32)
                              .reshape(self.sh, k, self.w4).mean(1))[None, None]
        self._lk_levels = []
        for lv in range(SEAM_LK_LEVELS - 1, -1, -1):                      # coarse -> fine
            f = 4 * (2 ** lv)
            hl, wl = self.sh // f, self.sw // f
            if hl < 4 or wl < 16:
                continue
            m = F.avg_pool2d(self._flow_mask, f)
            ys = torch.linspace(-1.0, 1.0, hl, device=dev)
            xs_ = torch.linspace(-1.0, 1.0, wl, device=dev)
            gy, gx = torch.meshgrid(ys, xs_, indexing="ij")
            self._lk_levels.append(dict(
                f=f, h=hl, w=wl, mask=m,
                base=torch.stack([gx, gy], -1)[None],
                to_norm=torch.tensor([2.0 / max(wl - 1, 1), 2.0 / max(hl - 1, 1)],
                                     device=dev).view(1, 1, 1, 2)))
        # flow rescale factors between pyramid levels and up to the band.
        # Built once: a per-frame torch.tensor(..., device=cuda) is a
        # pageable host->device copy, which synchronises the stream
        vec2 = lambda a, b: torch.tensor([a, b], device=dev).view(1, 2, 1, 1)
        prev = None
        for lv in self._lk_levels:
            lv["up"] = None if prev is None else vec2(lv["w"] / prev["w"], lv["h"] / prev["h"])
            prev = lv
        self._lk_out = None if prev is None else vec2(self.sw / prev["w"], self.sh / prev["h"])
        self._flow_scale = vec2(self.W / self.sw, self.bh / max(self.sh, 1))
        # central-difference gradient kernels: output channels [d/dx, d/dy]
        gk = torch.zeros(2, 1, 3, 3, device=dev)
        gk[0, 0, 1, 0], gk[0, 0, 1, 2] = -0.5, 0.5
        gk[1, 0, 0, 1], gk[1, 0, 2, 1] = -0.5, 0.5
        self._grad_k = gk
        ys = torch.linspace(-1.0, 1.0, self.sh, device=dev)
        xs_ = torch.linspace(-1.0, 1.0, self.sw, device=dev)
        gy, gx = torch.meshgrid(ys, xs_, indexing="ij")
        self._fb_base = torch.stack([gx, gy], -1)[None]
        self._fb_norm = torch.tensor([2.0 / max(self.sw - 1, 1), 2.0 / max(self.sh - 1, 1)],
                                     device=dev).view(1, 1, 1, 2)

        # ---- dynamic seam statics ----
        rows = np.where(inner4.any(axis=1))[0]
        self._q0 = int(rows.min()) if rows.size else 0
        self._q1 = int(rows.max()) + 1 if rows.size else self.bh
        self._yy = torch.arange(self.bh, device=dev, dtype=torch.float32)[:, None]
        self.w_band = w_band.reshape(1, 1, self.bh, self.W)
        self._dp_diag = None
        #: columns per block of the dynamic programme (constants.SEAM_DP_BLOCK*)
        self.dp_block = SEAM_DP_BLOCK_CUDA if dev.type == "cuda" else SEAM_DP_BLOCK

    # ---- Gaussian blurs with OpenCV's conventions --------------------------

    def _bm(self, n: int, sigma: float):
        key = (n, round(float(sigma), 6))
        m = self._bm_cache.get(key)
        if m is None:
            M = torch.from_numpy(blur_matrix(n, sigma).astype(np.float32)).to(self.dev)
            m = (M, M.t().contiguous())
            self._bm_cache[key] = m
        return m

    def _kern(self, sigma: float):
        key = round(float(sigma), 6)
        k = self._kern_cache.get(key)
        if k is None:
            k = torch.from_numpy(gauss_kernel(sigma).astype(np.float32)).to(self.dev)
            self._kern_cache[key] = k
        return k

    def blur(self, x, sx: float = 0.0, sy: float = 0.0):
        """cv2.GaussianBlur(x, (0, 0), sigmaX=sx, sigmaY=sy) over the last two dims.

        A sigma of 0 leaves that axis alone (the CPU code passes 0.1 for a
        one-row image, which is the identity there too).
        """
        h, w = x.shape[-2], x.shape[-1]
        if sx > 0 and w > 1:
            if w <= _BLUR_MATRIX_MAX:
                x = x @ self._bm(w, sx)[1]
            else:
                k = self._kern(sx)
                r = k.numel() // 2
                shp = x.shape
                xp = F.pad(x.reshape(-1, 1, h, w), (r, r, 0, 0), mode="reflect")
                x = F.conv2d(xp, k.view(1, 1, 1, -1)).reshape(shp)
        if sy > 0 and h > 1:
            if h <= _BLUR_MATRIX_MAX:
                x = self._bm(h, sy)[0] @ x
            else:
                k = self._kern(sy)
                r = k.numel() // 2
                shp = x.shape
                xp = F.pad(x.reshape(-1, 1, h, w), (0, 0, r, r), mode="reflect")
                x = F.conv2d(xp, k.view(1, 1, -1, 1)).reshape(shp)
        return x

    def _area(self, x, n_out: int):
        """cv2.resize(..., INTER_AREA) along the last axis.

        An integer reduction is plain average pooling. Any other ratio
        (output widths that are not a multiple of ``4 * SEAM_DYN_COLS``) is
        one matmul with the matrix OpenCV itself produces for that size -
        the resize is linear, so resizing an identity gives it - because
        MPS has no adaptive pooling for non-divisible sizes and torch's
        differs from OpenCV's when enlarging.
        """
        n = x.shape[-1]
        if n == n_out:
            return x
        if n % n_out == 0:
            shp = x.shape
            return F.avg_pool1d(x.reshape(-1, 1, n), n // n_out).reshape(*shp[:-1], n_out)
        key = ("area", n, n_out)
        M = self._bm_cache.get(key)
        if M is None:
            import cv2
            M = cv2.resize(np.eye(n, dtype=np.float32), (n_out, n),
                           interpolation=cv2.INTER_AREA)
            M = torch.from_numpy(np.ascontiguousarray(M)).to(self.dev)
            self._bm_cache[key] = M
        return x @ M

    # ---- photometric estimator ---------------------------------------------

    def luma_offsets(self, A, B):
        """``stitcher.seam_luma_offsets`` on the device.

        ``A``/``B``: (1,1,bh,W/4) pooled band crops. Returns ``lut`` (1024,)
        and ``spatial`` (1,1,bh,W/4), not yet halved for the split apply.
        """
        ds, nb = self._ds, self._nb
        lo, hi, clamp = self.lo, self.hi, self._clamp
        h0, w0 = A.shape[-2], A.shape[-1]
        A = F.avg_pool2d(A, ds)
        B = F.avg_pool2d(B, ds)
        inner = self.inner16
        d = B - A
        ag = (inner & (d.abs() < SEAM_OFFSET_AGREE_T)).to(torch.float32)
        idx = ((A - lo) * (nb / (hi - lo))).floor().clamp(0, nb - 1).long().view(-1)
        acc = torch.zeros(2, nb, device=A.device)
        acc.index_add_(1, idx, torch.stack([ag.view(-1), (d * ag).view(-1)]))
        cnt, ssum = acc[0], acc[1]
        valid = cnt > 50
        lut_b = ssum / cnt.clamp(min=1.0)
        # np.interp over the valid bins: linear between the nearest valid
        # bin on each side, constant beyond the outermost valid bins
        vi = valid[None, :]
        left = torch.where(vi & (self._bin_i <= self._bin_j), self._bin_i, -1).amax(1)
        right = torch.where(vi & (self._bin_i >= self._bin_j), self._bin_i, nb).amin(1)
        has_l, has_r = left >= 0, right < nb
        vl = lut_b[left.clamp(min=0)]
        vr = lut_b[right.clamp(max=nb - 1)]
        t = (self._bin_f - left) / (right - left).clamp(min=1)
        lb = torch.where(has_l & has_r, vl + (vr - vl) * t, torch.where(has_l, vl, vr))
        anyv = valid.any().to(torch.float32)
        lb = self._bm(nb, 1.5)[0] @ lb
        lut = (self.lut_interp @ lb).clamp(-clamp, clamp) * anyv
        res = d - lut[A.clamp(0, 1023).long()]
        o = (res * ag).sum(-2, keepdim=True) / ag.sum(-2, keepdim=True).clamp(min=1.0)
        o = self.blur(o, sx=SEAM_OFFSET_SIGMA / ds).clamp(-clamp * 0.75, clamp * 0.75)
        res2 = res - o
        sy = max(2.0, self._ph_h / 18.0)
        sx = SEAM_OFFSET_SIGMA * 1.3 / ds
        field = torch.zeros_like(res2)
        for _ in range(2):
            r = res2 - field
            ag2 = (inner & (r.abs() < 3.0 * SEAM_OFFSET_AGREE_T)).to(torch.float32)
            nd = self.blur(torch.cat([r * ag2, ag2], 1), sx=sx, sy=sy)
            field = (field + nd[:, :1] / nd[:, 1:].clamp(min=1e-3)).clamp(-2.0 * clamp,
                                                                          2.0 * clamp)
        spatial = (o + field) * anyv
        spatial = F.interpolate(spatial, size=(h0, w0), mode="bilinear", align_corners=False)
        return lut, spatial

    # ---- optical flow -------------------------------------------------------

    def lk_flow(self, ga, gb):
        """Dense flow a->b and b->a between two flow grays, both directions at once.

        ``ga``/``gb``: (1,1,sh,sw). Returns (2,2,sh,sw): [0] = a->b, [1] =
        b->a, channels (x, y) in gray pixels, such that
        ``b(p + f01(p)) ~ a(p)``.

        Coarse-to-fine Lucas-Kanade on mean-normalised windows (as DIS
        normalises its patches, so a residual brightness offset between the
        lenses does not read as motion): at each level the moving image is
        warped by the current flow and the 2x2 normal equations - built
        from the fixed image's gradients over a Gaussian window, damped so
        textureless windows produce no update - give the correction.
        """
        T = torch.cat([ga, gb], 0)                                        # fixed
        I = torch.cat([gb, ga], 0)                                        # moving
        flow = None
        gs = lambda img, grid: F.grid_sample(img, grid, mode="bilinear",
                                             align_corners=True, padding_mode="border")
        sw_ = SEAM_LK_WINDOW_SIGMA
        for lv in self._lk_levels:
            f, hl, wl = lv["f"], lv["h"], lv["w"]
            m = lv["mask"]
            Tl, Il = F.avg_pool2d(T, f), F.avg_pool2d(I, f)
            G = F.conv2d(F.pad(Tl, (1, 1, 1, 1), mode="replicate"), self._grad_k)   # (2,2,h,w)
            gx, gy = G[:, :1], G[:, 1:]
            # windowed, masked sums: [gxgx, gxgy, gygy, gx, gy, 1]
            Wn = self.blur(torch.cat([gx * gx, gx * gy, gy * gy, G,
                                      torch.ones_like(gx)], 1) * m, sx=sw_, sy=sw_)
            wm = Wn[:, 5:6].clamp(min=1e-4)
            Wg = Wn[:, 3:5]                                               # W(m g)
            mg = Wg / wm
            sxx = Wn[:, 0:1] - Wg[:, 0:1] * mg[:, 0:1] + SEAM_LK_DAMP * wm
            sxy = Wn[:, 1:2] - Wg[:, 0:1] * mg[:, 1:2]
            syy = Wn[:, 2:3] - Wg[:, 1:2] * mg[:, 1:2] + SEAM_LK_DAMP * wm
            det = (sxx * syy - sxy * sxy).clamp(min=1e-6)
            # inverse of the damped structure tensor, rows as (2,2) channels
            inv_x = torch.cat([syy, -sxy], 1) / det                       # -> du_x
            inv_y = torch.cat([-sxy, sxx], 1) / det                       # -> du_y
            Gm = G * m
            if flow is None:
                flow = torch.zeros(2, 2, hl, wl, device=T.device)
            else:
                flow = F.interpolate(flow, size=(hl, wl), mode="bilinear",
                                     align_corners=False) * lv["up"]
            for _ in range(SEAM_LK_ITERS):
                grid = lv["base"] + flow.permute(0, 2, 3, 1) * lv["to_norm"]
                It = gs(Il, grid) - Tl
                Wq = self.blur(torch.cat([Gm * It, m * It], 1), sx=sw_, sy=sw_)
                b = Wq[:, :2] - mg * Wq[:, 2:3]
                du = torch.cat([(inv_x * b).sum(1, keepdim=True),
                                (inv_y * b).sum(1, keepdim=True)], 1)
                flow = flow - du
        if flow is None:
            # a band too small for any pyramid level (tiny test rigs): no flow
            return torch.zeros(2, 2, self.sh, self.sw, device=T.device)
        return F.interpolate(flow, size=(self.sh, self.sw), mode="bilinear",
                             align_corners=False) * self._lk_out

    def flow_small(self, ga, gb):
        """``stitcher.seam_flow_small`` on the device: gated, band-smoothed flows.

        Returns (2,2,sh,sw): [0] = a->b, [1] = b->a, small-res px.
        """
        fl = self.lk_flow(ga, gb)
        f01, f10 = fl[:1], fl[1:]
        grid = self._fb_base + f01.permute(0, 2, 3, 1) * self._fb_norm
        back = F.grid_sample(f10, grid, mode="bilinear", align_corners=True,
                             padding_mode="border")
        resid = (f01 + back).square().sum(1, keepdim=True).sqrt()
        mag = f01.square().sum(1, keepdim=True).sqrt()
        conf = (1.0 - resid / (0.5 * mag + 2.0)).clamp(0.0, 1.0)
        conf = self.blur(conf, sx=3.0, sy=3.0)
        return self.blur(fl * conf, sx=SEAM_FLOW_SIGMA_ALONG, sy=SEAM_FLOW_SIGMA_ACROSS)

    # ---- the per-frame estimate ---------------------------------------------

    def estimate(self, Ya, Yb, auto: bool) -> dict:
        """Photometric offsets and morph flows from the two band warps.

        ``Ya``/``Yb``: (1,1,bh,W) band rows of each lens in rig space.
        Returns device tensors only (no host sync):

        - ``lut`` (1024,), ``sp`` (1,1,bh,W/4): the photometric correction,
          already halved for the split apply and zeroed when ``auto`` finds
          it below ``SEAM_AUTO_PHOTO_T``;
        - ``flow`` (2,2,bh,W): morph offsets a->b and b->a in full-res px,
          clamped, zeroed when ``auto`` finds no parallax;
        - ``engaged``, ``photo``: 0/1 scalars; ``pre``: the band mismatch.
        """
        an, bn = F.avg_pool2d(Ya, (1, 4)), F.avg_pool2d(Yb, (1, 4))
        pre = (F.avg_pool2d((Ya - Yb).abs(), (1, 4)) * self.innerf).sum() / self.inner_n
        lut, sp = self.luma_offsets(an, bn)
        one = torch.ones((), device=Ya.device)
        photo = one
        if auto:
            photo = ((lut.abs().amax() > SEAM_AUTO_PHOTO_T)
                     | (sp.abs().amax() > SEAM_AUTO_PHOTO_T)).to(torch.float32)
        lut = lut * (0.5 * photo)
        sp = sp * (0.5 * photo)
        cA = an + (lut[an.clamp(0, 1023).long()] + sp)
        cB = bn - (lut[bn.clamp(0, 1023).long()] + sp)
        engaged = one
        if auto:
            col = ((cB - cA).abs() * self.innerf).sum(-2) / self.inner_col[0]
            engaged = (col.amax() * 0.25 >= SEAM_DYN_GATE_LO).to(torch.float32)
        k = self._gk
        ga = (F.avg_pool2d(cA, (k, 1)) * 0.25).clamp(0.0, 255.0)
        gb = (F.avg_pool2d(cB, (k, 1)) * 0.25).clamp(0.0, 255.0)
        fs = self.flow_small(ga, gb) * engaged                            # (2,2,sh,sw)
        fl = F.interpolate(fs, size=(self.bh, self.W), mode="bilinear", align_corners=False)
        fl = (fl * self._flow_scale).clamp(-SEAM_FLOW_MAX_SHIFT_PX, SEAM_FLOW_MAX_SHIFT_PX)
        return {"lut": lut, "sp": sp, "flow": fl, "engaged": engaged, "photo": photo,
                "pre": pre}

    # ---- dynamic seam ---------------------------------------------------------

    def seam_path(self, cs):
        """The min-cost left-to-right path with |dy| <= 1: row per column.

        ``cs``: (h, n) cost. Solves ``stitcher.seam_path``'s recurrence
        ``D[i, x] = cs[i, x] + min(D[i-1, x-1], D[i, x-1], D[i+1, x-1])``
        without its n sequential steps. A run of columns is a min-plus
        operator ``P[i, d]`` = cheapest cost from row i before the run to
        row i+d at its end; a run of 2^L columns can move at most 2^L rows,
        so P is banded. Adjacent runs compose pairwise (L doubling levels,
        each a handful of batched kernels) up to blocks of ``dp_block``
        columns; the blocks are then walked in order, and the path is
        recovered top-down from the argmins each composition kept.
        Returns (n,) int64 rows.
        """
        h, n = cs.shape
        inf = float("inf")
        K = int(self.dp_block)
        nb = -(-n // K)
        c = cs.t()
        if nb * K != n:                  # zero-cost columns: free, cannot change the path
            c = torch.cat([c, c.new_zeros(nb * K - n, h)], 0)
        P = F.pad(c, (1, 1), value=inf).unfold(1, 3, 1)                   # (m, h, 3), k = 1
        args = []
        k = 1
        while k < K:
            k1 = 2 * k + 1
            A, B = P[0::2], P[1::2]
            # Bw[b, i, dd, e'] = B[b, i + dd - k, e']: the second run, entered
            # at the row the first run left at
            Bw = F.pad(B, (0, 0, k, k), value=inf).unfold(1, k1, 1)       # (m/2,h,e',dd)
            # the composed run ends at offset e = dd + e': shift row dd right
            # by dd. Padding each row with k1 infs and re-reading the flat
            # buffer with rows one element shorter does exactly that (the
            # skew trick): no gather, no index tensor
            Bw = F.pad(Bw.permute(0, 1, 3, 2), (0, k1), value=inf)        # (m/2,h,dd,2*k1)
            Bw = Bw.flatten(2)[:, :, :k1 * (4 * k + 1)].view(-1, h, k1, 4 * k + 1)
            P, arg = (A.unsqueeze(3) + Bw).min(2)                         # (m/2,h,4k+1)
            args.append(arg)
            k *= 2
        # walk the blocks: v'[j] = min_i v[i] + P[b, i, j - i + K]
        Qf = F.pad(P.flip(2), (0, 0, K, K), value=inf)                    # (nb, h+2K, 2K+1)
        if self._dp_diag is None or self._dp_diag.shape != (h, 2 * K + 1):
            self._dp_diag = (torch.arange(h, device=self.dev)[:, None]
                             + torch.arange(2 * K + 1, device=self.dev)[None, :])
        Qf = Qf.gather(1, self._dp_diag.expand(nb, h, 2 * K + 1))         # [b, j, t]
        v = cs.new_zeros(h)
        targ = []
        for b in range(nb):
            v, t = (F.pad(v, (K, K), value=inf).unfold(0, 2 * K + 1, 1) + Qf[b]).min(1)
            targ.append(t)
        pred = torch.stack(targ) + (torch.arange(h, device=self.dev) - K)  # (nb, h) start row
        r = v.argmin().view(1)
        bounds = [r]
        for b in range(nb - 1, -1, -1):
            r = pred[b].gather(0, r)
            bounds.append(r)
        bounds = torch.cat(bounds[::-1])                                  # (nb+1,)
        starts, ends = bounds[:-1], bounds[1:]
        # top-down: split each run at the row its composition chose
        while args:
            arg = args.pop()
            k //= 2
            m = starts.numel()
            dd = arg[torch.arange(m, device=self.dev), starts, ends - starts + 2 * k]
            mids = starts + dd - k
            starts, ends = (torch.stack([starts, mids], 1).reshape(-1),
                            torch.stack([mids, ends], 1).reshape(-1))
        return ends[:n]

    def dynamic(self, Ma, Mb, engaged):
        """Dynamic-seam blend weight over the band, and the post-morph mismatch.

        ``Ma``/``Mb``: (1,1,bh,W) morphed, photometrically corrected band
        crops. Returns ``wd`` (1,1,bh,W) - the static weight where the gate
        is closed, a narrow feather around the min-disagreement path where
        it is open - and ``post`` (0-d tensor, 10-bit units).
        """
        W, w4 = self.W, self.w4
        d4 = F.avg_pool2d((Ma - Mb).abs(), (1, 4)) * 0.25                 # 8-bit units
        dm = d4 * self.innerf
        post = dm.sum() / self.inner_n * 4.0
        # gate (stitcher.seam_gate)
        score = self.blur(dm, sx=3.0, sy=3.0).sum(-2) / self.inner_col[0]  # (1,w4)
        s = self._area(score, SEAM_DYN_COLS)                              # (1,1,cols)
        s = self.blur(s, sx=SEAM_DYN_GATE_SIGMA)
        g = ((s - SEAM_DYN_GATE_LO) / (SEAM_DYN_GATE_HI - SEAM_DYN_GATE_LO)).clamp(0.0, 1.0)
        g = F.interpolate(g * engaged, size=W, mode="linear", align_corners=True)   # (1,1,W)
        # path (stitcher.seam_path) over the rows that hold the inner band
        q0, q1 = self._q0, self._q1
        cost = torch.where(self.inner, d4, 1e6)[:, :, q0:q1]
        cs = self._area(cost, SEAM_DYN_COLS)[0, 0]
        path = (self.seam_path(cs) + q0).to(torch.float32)
        pf = self.blur(path.view(1, 1, 1, -1), sx=SEAM_DYN_SIGMA).view(1, 1, -1)
        seam = F.interpolate(pf, size=W, mode="linear", align_corners=True)[0]      # (1,W)
        wd = (0.5 + (seam - self._yy) / SEAM_DYN_FEATHER_PX).clamp(0.0, 1.0)
        if not self.cam0_top:
            wd = 1.0 - wd
        wd = torch.lerp(self.w_band, wd[None, None], g[:, :, None])
        return wd, post


__all__ = ["SeamGpu", "gauss_kernel", "blur_matrix"]
