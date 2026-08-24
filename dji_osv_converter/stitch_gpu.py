"""GPU stitch - the whole pipeline resident on the device (PyTorch).

``stitcher.py`` is the CPU reference; this runs warp -> exposure -> linear
blend -> levelling entirely on the GPU, so the only CPU<->GPU transfers are one
fisheye pair up and one panorama down per frame. ``grid_sample`` is the remap
(warp and the levelling rotation); everything else is elementwise / conv tensor
ops. Output matches the CPU path to bilinear/rounding.

Backend-agnostic: uses CUDA (NVIDIA) when present, else Apple MPS (Metal), else
falls back to CPU tensors. The stitch math is identical on all three - only the
device changes. Construct only when ``available()`` is true.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:                       # torch not installed
    torch = None
    F = None

from .calibration import CameraMap
from .stitcher import seam_weights
from .orientation import directions


def pick_device(prefer: str | None = None) -> str | None:
    """Best available torch device name: 'cuda' > 'mps' > None (torch absent)."""
    if torch is None:
        return None
    if prefer and prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def available(prefer: str | None = None) -> bool:
    dev = pick_device(prefer)
    return dev is not None and dev != "cpu"


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class GpuStitcher:
    """GPU-resident remap grids, masks and weights for one OSV geometry.

    ``device`` may be "cuda", "mps", "cpu", or None/"auto" to pick the best.
    """

    def __init__(self, cams: list[CameraMap], band_px: float,
                 width: int, height: int, device: str | None = None):
        if torch is None:
            raise RuntimeError("torch is not installed")
        dev = pick_device(device)
        self.dev = torch.device(dev)
        self.W, self.H = width, height

        # warp grids: map_x/map_y (pixel coords in the source fisheye) -> the
        # normalised [-1, 1] sampling grid grid_sample expects, uploaded once.
        self.grid = []
        self.valid = []
        for c in cams:
            sh, sw = c.map_x.shape[0], c.map_x.shape[1]  # output size (H, W)
            fw = fh = 3840  # fisheye source is square; use its own size
            gx = torch.from_numpy(2.0 * c.map_x / (fw - 1) - 1.0)
            gy = torch.from_numpy(2.0 * c.map_y / (fh - 1) - 1.0)
            self.grid.append(torch.stack([gx, gy], -1).unsqueeze(0).to(self.dev))
            v = torch.from_numpy(c.valid.astype(np.float32))[None, None].to(self.dev)
            self.valid.append(v)

        w = seam_weights(cams, band_px).astype(np.float32)           # (H, W) in [0,1]
        self.w3 = torch.from_numpy(w)[None, None].to(self.dev)       # (1,1,H,W)
        # static overlap mask, pooled to 1/8 (exposure works at low res)
        self.ov8 = F.avg_pool2d(self.valid[0] * self.valid[1], 8)

        # --- YUV-native (4:2:0) statics ---
        # Chroma runs at half res, so it needs the warp grid and the seam weight
        # at half res. DECIMATE the grid (stride-2 slice), never interpolate it:
        # the levelling/warp grids carry a +-1 longitude-wrap discontinuity, and
        # averaging across it corrupts the wrap meridian.
        self.grid_half = [g[:, ::2, ::2, :].contiguous() for g in self.grid]
        self.w3h = F.avg_pool2d(self.w3, 2)                          # (1,1,H/2,W/2)
        # BT.709 limited-range centred YUV<->RGB matrices; used only by the opt-in
        # per-channel white-balance exposure path (_expose_wb).
        self._Minv = torch.tensor(                                   # centred YUV -> RGB
            [[1.164384, 0.0, 1.792741],
             [1.164384, -0.213249, -0.532909],
             [1.164384, 2.112402, 0.0]], dtype=torch.float32, device=self.dev)
        self._M = torch.tensor(                                      # RGB -> centred YUV
            [[0.182586, 0.614231, 0.062007],
             [-0.100644, -0.338572, 0.439216],
             [0.439216, -0.398942, -0.040274]], dtype=torch.float32, device=self.dev)

        # equirect ray components for the levelling rotation, kept as three
        # contiguous (H*W,) tensors. The rotation is then done with elementwise
        # ops (9 mul + 6 add) instead of an (H*W,3)@(3,3) matmul, which is
        # ~5x faster here - MPS handles that skinny matmul shape badly.
        d = directions(height, width).reshape(-1, 3).astype(np.float32)
        self.dx = torch.from_numpy(np.ascontiguousarray(d[:, 0])).to(self.dev)
        self.dy = torch.from_numpy(np.ascontiguousarray(d[:, 1])).to(self.dev)
        self.dz = torch.from_numpy(np.ascontiguousarray(d[:, 2])).to(self.dev)

    # ---- stages (all operate on (1,3,H,W) float tensors on the device) ----

    def warp(self, fisheye: np.ndarray, i: int) -> "torch.Tensor":
        src = (torch.from_numpy(np.ascontiguousarray(fisheye))
               .permute(2, 0, 1).unsqueeze(0).float().to(self.dev))
        return F.grid_sample(src, self.grid[i], mode="bilinear",
                             align_corners=True, padding_mode="zeros")

    def match_exposure(self, a, b, max_gain: float = 1.6, maxval: float = 255.0):
        """Global per-channel gain + smooth spatial gain field.

        All the masking / reductions / log run on the 1/8-downsampled images
        (the full-res versions were 80% of the cost on MPS); only the final
        gain multiply is at full res. Mirrors the CPU result to within noise.
        ``maxval`` is the pixel range (255 for 8-bit, 1023 for 10-bit).
        """
        lo, hi = np.log(1 / max_gain), np.log(max_gain)
        thr = 6.0 * maxval / 255.0                     # "lit" luma threshold
        a_s = F.avg_pool2d(a, 8)                       # (1,3,H/8,W/8)
        b_s = F.avg_pool2d(b, 8)
        lit = ((self.ov8 > 0.5) & (a_s.mean(1, keepdim=True) > thr)
               & (b_s.mean(1, keepdim=True) > thr)).float()   # (1,1,hs,ws)
        denom = lit.sum().clamp_min(1)
        ma = (a_s * lit).sum((0, 2, 3)) / denom
        mb = (b_s * lit).sum((0, 2, 3)) / denom
        ratio = torch.where((ma > 1e-3) & (mb > 1e-3), mb / ma, torch.ones_like(ma))
        ga = ratio.sqrt().clamp(1 / max_gain, max_gain).view(1, 3, 1, 1)
        a_s, b_s = a_s * ga, b_s / ga   # low-res only; ga is folded into g0/g1 below
        # smooth spatial log-ratio, diffused at low res then bilinear back up
        logr = torch.log((a_s / b_s.clamp_min(1e-3)).clamp(1e-3, 1e3)) * lit
        num = F.avg_pool2d(logr, 4)
        den = F.avg_pool2d(lit, 4).clamp_min(1e-6)
        num = F.avg_pool2d(num, 15, 1, 7)
        den = F.avg_pool2d(den, 15, 1, 7).clamp_min(1e-6)
        lr = (num / den).clamp(lo, hi)
        # Fold the global gain into the spatial gain and exp at LOW res, then
        # upsample the finished gain maps and apply with one multiply each -
        # removes two full-res exp and two full-res multiplies. The difference
        # from exp-at-full-res is a smooth <1% gain (no seam).
        g0 = ga * torch.exp(-0.5 * lr)
        g1 = (1.0 / ga) * torch.exp(+0.5 * lr)
        g0 = F.interpolate(g0, size=(self.H, self.W), mode="bilinear", align_corners=False)
        g1 = F.interpolate(g1, size=(self.H, self.W), mode="bilinear", align_corners=False)
        return a * g0, b * g1

    def blend(self, a, b):
        return a * self.w3 + b * (1.0 - self.w3)

    def _level_grid(self, R: np.ndarray) -> "torch.Tensor":
        """Full-res sampling grid (1,H,W,2) for rotating the equirect by R.

        The rotated ray is d' = dirs @ R.T, i.e. d'[j] = sum_k dirs[k]*R[j,k];
        computed elementwise on the precomputed components (no matmul).
        """
        r = R.astype(np.float64)
        nx = self.dx * r[0, 0] + self.dy * r[0, 1] + self.dz * r[0, 2]
        ny = self.dx * r[1, 0] + self.dy * r[1, 1] + self.dz * r[1, 2]
        nz = self.dx * r[2, 0] + self.dy * r[2, 1] + self.dz * r[2, 2]
        lon = torch.atan2(nx, -nz)
        lat = torch.asin(ny.clamp(-1, 1))
        gx = torch.remainder(lon / np.pi + 1.0, 2.0) - 1.0  # wrap longitude
        gy = -2.0 * lat / np.pi                             # lat +-90 -> +-1
        return torch.stack([gx, gy], -1).view(1, self.H, self.W, 2)

    def level(self, pano, R: np.ndarray):
        """Rotate the equirect by R via grid_sample, with longitude wrap."""
        grid = self._level_grid(R)
        return F.grid_sample(pano, grid, mode="bilinear",
                             align_corners=True, padding_mode="border")

    def stitch(self, f0: np.ndarray, f1: np.ndarray, R: np.ndarray,
               exposure: bool = True) -> np.ndarray:
        a = self.warp(f0, 0)
        b = self.warp(f1, 1)
        if exposure:
            a, b = self.match_exposure(a, b)
        pano = self.blend(a, b)
        if not np.allclose(R, np.eye(3), atol=1e-9):
            pano = self.level(pano, R)
        out = pano.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8)
        return out.cpu().numpy()

    # ---- 10-bit YUV-native path (BT.709 limited range) ----------------------
    #   in : yuv420p10le packed uint16 (H*3/2, W); out: p010le planes (Y, UV).
    #   The whole stitch stays in 4:2:0 YUV - no RGB round-trip. warp, blend and
    #   levelling are linear/resampling ops that work directly on the Y and
    #   half-res UV planes, so the ~180ms/frame of YUV<->RGB conversion is gone
    #   and chroma is never up- then down-sampled (sharper). Luma Y is warped at
    #   full res, chroma UV at half res on the decimated grids.

    def _split_yuv(self, packed: np.ndarray, fh: int = 3840, fw: int = 3840):
        t = torch.from_numpy(np.ascontiguousarray(packed)).to(self.dev).float()
        Y = t[:fh].view(1, 1, fh, fw)
        U = t[fh:fh + fh // 4].reshape(1, 1, fh // 2, fw // 2)
        V = t[fh + fh // 4:fh + fh // 2].reshape(1, 1, fh // 2, fw // 2)
        return Y, U, V

    def _warp_planes(self, packed: np.ndarray, i: int):
        """Warp one fisheye's YUV -> (Yw (1,1,H,W), UVw (1,2,H/2,W/2))."""
        Y, U, V = self._split_yuv(packed)
        Yw = F.grid_sample(Y, self.grid[i], mode="bilinear",
                           align_corners=True, padding_mode="zeros")
        UVw = F.grid_sample(torch.cat([U, V], 1), self.grid_half[i],
                            mode="bilinear", align_corners=True, padding_mode="zeros")
        return Yw, UVw

    def _spatial_gain(self, Ya, Yb, max_gain: float):
        """Smooth luma gain field from the overlap, at low res.

        Returns the global luma gain ``gl`` (scalar), the low-res ``lit`` mask,
        its sum ``den``, and the four upsampled scalar spatial gain maps
        (full/half res, for image a/b): g = exp(-/+ 0.5 * logratio).
        """
        lo, hi = np.log(1 / max_gain), np.log(max_gain)
        thr = 6.0 * 1023.0 / 255.0
        a_s = F.avg_pool2d(Ya, 8)
        b_s = F.avg_pool2d(Yb, 8)
        lit = ((self.ov8 > 0.5) & (a_s > thr) & (b_s > thr)).float()
        den = lit.sum().clamp_min(1)
        lma = (a_s * lit).sum()
        lmb = (b_s * lit).sum()
        ratio = torch.where((lma > 1e-3) & (lmb > 1e-3), lmb / lma,
                            torch.ones((), device=self.dev))
        gl = ratio.sqrt().clamp(1 / max_gain, max_gain)
        logr = torch.log(((a_s * gl) / (b_s / gl).clamp_min(1e-3)).clamp(1e-3, 1e3)) * lit
        num = F.avg_pool2d(logr, 4)
        dn = F.avg_pool2d(lit, 4).clamp_min(1e-6)
        num = F.avg_pool2d(num, 15, 1, 7)
        dn = F.avg_pool2d(dn, 15, 1, 7).clamp_min(1e-6)
        lr = (num / dn).clamp(lo, hi)
        s0, s1 = torch.exp(-0.5 * lr), torch.exp(0.5 * lr)
        up = lambda s, sz: F.interpolate(s, size=sz, mode="bilinear", align_corners=False)
        full, half = (self.H, self.W), (self.H // 2, self.W // 2)
        return gl, lit, den, a_s, b_s, (up(s0, full), up(s0, half),
                                        up(s1, full), up(s1, half))

    @staticmethod
    def _scale_centred(Y, UV, gY, gUV):
        """Apply a (broadcastable) gain to centred Y and UV planes."""
        return (Y - 64.0) * gY + 64.0, (UV - 512.0) * gUV + 512.0

    def _expose_luma(self, Ya, UVa, Yb, UVb, max_gain=1.6):
        """Luma-only exposure: one scalar gain field per image (the default)."""
        gl, _, _, _, _, (s0f, s0h, s1f, s1h) = self._spatial_gain(Ya, Yb, max_gain)
        Ya, UVa = self._scale_centred(Ya, UVa, gl * s0f, gl * s0h)
        Yb, UVb = self._scale_centred(Yb, UVb, (1 / gl) * s1f, (1 / gl) * s1h)
        return Ya, UVa, Yb, UVb

    def _apply_wb(self, Y, UV, C, sf, sh):
        """Scalar spatial gain s(x,y) then a constant per-channel matrix C, in YUV.

        C = M @ diag(gain) @ M^-1 is the RGB per-channel gain expressed in centred
        YUV. The Y cross-terms need chroma at full res; that correction is tiny and
        smooth, so the half-res U/V are bilinear-upsampled just for it.
        """
        y = Y - 64.0
        u = UV[:, 0:1] - 512.0
        v = UV[:, 1:2] - 512.0
        # Y's chroma cross-term is a single smooth channel; build it at half res
        # and upsample once (rather than upsampling U and V separately).
        cross = F.interpolate(C[0, 1] * u + C[0, 2] * v, size=(self.H, self.W),
                              mode="bilinear", align_corners=False)
        yh = F.avg_pool2d(y, 2)
        yp = C[0, 0] * y + cross
        up = C[1, 0] * yh + C[1, 1] * u + C[1, 2] * v
        vp = C[2, 0] * yh + C[2, 1] * u + C[2, 2] * v
        return yp * sf + 64.0, torch.cat([up * sh + 512.0, vp * sh + 512.0], 1)

    def _expose_wb(self, Ya, UVa, Yb, UVb, max_gain=1.6):
        """Per-channel white-balance exposure (opt-in): luma spatial gain plus a
        constant per-channel gain, matching the RGB path's lens-colour match."""
        _, lit, den, a_s, b_s, (s0f, s0h, s1f, s1h) = self._spatial_gain(Ya, Yb, max_gain)
        # per-channel global gain from the low-res overlap, in RGB
        UVa_s, UVb_s = F.avg_pool2d(UVa, 4), F.avg_pool2d(UVb, 4)   # UV half -> 1/8

        def to_rgb(Ys, UVs):
            y, u, v = Ys - 64.0, UVs[:, 0:1] - 512.0, UVs[:, 1:2] - 512.0
            return torch.cat([1.164384 * y + 1.792741 * v,
                              1.164384 * y - 0.213249 * u - 0.532909 * v,
                              1.164384 * y + 2.112402 * u], 1)

        ma = (to_rgb(a_s, UVa_s) * lit).sum((0, 2, 3)) / den
        mb = (to_rgb(b_s, UVb_s) * lit).sum((0, 2, 3)) / den
        ratio = torch.where((ma > 1e-3) & (mb > 1e-3), mb / ma, torch.ones_like(ma))
        ga = ratio.sqrt().clamp(1 / max_gain, max_gain)            # (3,) per channel
        Ca = self._M @ torch.diag(ga) @ self._Minv
        Cb = self._M @ torch.diag(1.0 / ga) @ self._Minv
        Ya, UVa = self._apply_wb(Ya, UVa, Ca, s0f, s0h)
        Yb, UVb = self._apply_wb(Yb, UVb, Cb, s1f, s1h)
        return Ya, UVa, Yb, UVb

    def _pack_p010(self, Y, UV):
        """(Y (1,1,H,W), UV (1,2,H/2,W/2)) -> p010le planes, top-aligned uint16."""
        H, W = self.H, self.W
        Yp = (Y[0, 0].clamp(64, 940) * 64.0).round().clamp(0, 65535)
        UVp = (UV[0].permute(1, 2, 0).reshape(H // 2, W).clamp(64, 960)
               * 64.0).round().clamp(0, 65535)
        to = lambda t: t.to(torch.int32).cpu().numpy().astype(np.uint16)
        return to(Yp), to(UVp)

    def stitch_yuv(self, y0: np.ndarray, y1: np.ndarray, R: np.ndarray,
                   exposure: bool = True, wb: bool = False):
        """Stitch a 10-bit YUV fisheye pair -> p010le planes (Y, UV) for encode.

        ``wb=True`` uses the per-channel white-balance exposure match (matches the
        two lenses' colour, ~15ms/frame dearer); the default luma-only match is
        the shipping path.
        """
        Ya, UVa = self._warp_planes(y0, 0)
        Yb, UVb = self._warp_planes(y1, 1)
        if exposure:
            expose = self._expose_wb if wb else self._expose_luma
            Ya, UVa, Yb, UVb = expose(Ya, UVa, Yb, UVb)
        Y = Ya * self.w3 + Yb * (1.0 - self.w3)
        UV = UVa * self.w3h + UVb * (1.0 - self.w3h)
        if not np.allclose(R, np.eye(3), atol=1e-9):
            gf = self._level_grid(R)
            gh = gf[:, ::2, ::2, :].contiguous()
            Y = F.grid_sample(Y, gf, mode="bilinear", align_corners=True, padding_mode="border")
            UV = F.grid_sample(UV, gh, mode="bilinear", align_corners=True, padding_mode="border")
        return self._pack_p010(Y, UV)


__all__ = ["available", "pick_device", "GpuStitcher"]
