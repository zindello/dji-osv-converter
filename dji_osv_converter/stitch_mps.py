"""Metal (PyTorch MPS) GPU stitch - the whole pipeline resident on the device.

``stitcher.py`` is the CPU reference; this runs warp -> exposure -> linear
blend -> levelling entirely on the Apple GPU via PyTorch's MPS backend, so the
only CPU<->GPU transfers are one fisheye pair up and one panorama down per
frame. ``grid_sample`` is the remap (warp and the levelling rotation);
everything else is elementwise / conv tensor ops. Output matches the CPU path
to bilinear/rounding.

Requires torch with a working MPS device; construct only when
``available()`` is true.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAVE = torch.backends.mps.is_available()
except Exception:                       # torch not installed
    torch = None
    _HAVE = False

from .calibration import CameraMap
from .stitcher import seam_weights
from .orientation import directions


def available() -> bool:
    return bool(_HAVE)


class MPSStitcher:
    """GPU-resident remap grids, masks and weights for one OSV geometry."""

    def __init__(self, cams: list[CameraMap], band_px: float,
                 width: int, height: int):
        if not _HAVE:
            raise RuntimeError("torch MPS not available")
        self.dev = torch.device("mps")
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

        # directions for the levelling rotation, as (3, H*W) on device
        d = directions(height, width).reshape(-1, 3).astype(np.float32)
        self.dirs = torch.from_numpy(d).to(self.dev)                 # (H*W, 3)

    # ---- stages (all operate on (1,3,H,W) float tensors on the device) ----

    def warp(self, fisheye: np.ndarray, i: int) -> "torch.Tensor":
        src = (torch.from_numpy(np.ascontiguousarray(fisheye))
               .permute(2, 0, 1).unsqueeze(0).float().to(self.dev))
        return F.grid_sample(src, self.grid[i], mode="bilinear",
                             align_corners=True, padding_mode="zeros")

    def match_exposure(self, a, b, max_gain: float = 1.6):
        """Global per-channel gain + smooth spatial gain field.

        All the masking / reductions / log run on the 1/8-downsampled images
        (the full-res versions were 80% of the cost on MPS); only the final
        gain multiply is at full res. Mirrors the CPU result to within noise.
        """
        lo, hi = np.log(1 / max_gain), np.log(max_gain)
        a_s = F.avg_pool2d(a, 8)                       # (1,3,H/8,W/8)
        b_s = F.avg_pool2d(b, 8)
        lit = ((self.ov8 > 0.5) & (a_s.mean(1, keepdim=True) > 6)
               & (b_s.mean(1, keepdim=True) > 6)).float()   # (1,1,hs,ws)
        denom = lit.sum().clamp_min(1)
        ma = (a_s * lit).sum((0, 2, 3)) / denom
        mb = (b_s * lit).sum((0, 2, 3)) / denom
        ratio = torch.where((ma > 1e-3) & (mb > 1e-3), mb / ma, torch.ones_like(ma))
        ga = ratio.sqrt().clamp(1 / max_gain, max_gain).view(1, 3, 1, 1)
        a, b = a * ga, b / ga
        a_s, b_s = a_s * ga, b_s / ga
        # smooth spatial log-ratio, diffused at low res then bilinear back up
        logr = torch.log((a_s / b_s.clamp_min(1e-3)).clamp(1e-3, 1e3)) * lit
        num = F.avg_pool2d(logr, 4)
        den = F.avg_pool2d(lit, 4).clamp_min(1e-6)
        num = F.avg_pool2d(num, 15, 1, 7)
        den = F.avg_pool2d(den, 15, 1, 7).clamp_min(1e-6)
        lr = (num / den).clamp(lo, hi)
        lr = F.interpolate(lr, size=(self.H, self.W), mode="bilinear", align_corners=False)
        a = a * torch.exp(-0.5 * lr)
        b = b * torch.exp(+0.5 * lr)
        return a, b

    def blend(self, a, b):
        return a * self.w3 + b * (1.0 - self.w3)

    def level(self, pano, R: np.ndarray):
        """Rotate the equirect by R via grid_sample, with longitude wrap."""
        Rt = torch.from_numpy(np.ascontiguousarray(R.T.astype(np.float32))).to(self.dev)
        d = self.dirs @ Rt                                  # (H*W, 3)
        lon = torch.atan2(d[:, 0], -d[:, 2])
        lat = torch.asin(d[:, 1].clamp(-1, 1))
        gx = (lon / np.pi)                                  # already in [-1,1]
        gy = (-2.0 * lat / np.pi)                           # lat +-90 -> +-1
        gx = torch.remainder(gx + 1.0, 2.0) - 1.0           # wrap longitude
        grid = torch.stack([gx, gy], -1).view(1, self.H, self.W, 2)
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


__all__ = ["available", "MPSStitcher"]
