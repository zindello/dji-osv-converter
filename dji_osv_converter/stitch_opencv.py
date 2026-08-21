"""OpenCL (OpenCV UMat) GPU-accelerated stitch - same algorithm as stitcher.py.

The CPU stitch in ``stitcher.py`` is the reference; this mirrors it on the GPU
via OpenCV's transparent API (``cv2.UMat``), keeping the panorama on the device
across warp -> exposure -> multi-band blend so there are no per-op CPU<->GPU
transfers. Output matches the CPU path to within blend/interpolation rounding.

Static data (remap tables, validity masks, and the whole weight pyramid - the
seam weights never change) is uploaded once. Everything falls back to the CPU
functions when OpenCL is unavailable.
"""

from __future__ import annotations

import cv2
import numpy as np

from .calibration import CameraMap
from .stitcher import seam_weights


def available() -> bool:
    return bool(cv2.ocl.haveOpenCL())


def _u(arr: np.ndarray) -> cv2.UMat:
    return cv2.UMat(np.ascontiguousarray(arr))


def _u3(single: np.ndarray) -> cv2.UMat:
    """Upload a single-channel float array as a 3-channel UMat."""
    return cv2.UMat(np.ascontiguousarray(np.repeat(single[..., None], 3, axis=2)
                                         .astype(np.float32)))


class GpuStitcher:
    """GPU-resident remap tables, validity masks and weight pyramid."""

    def __init__(self, cams: list[CameraMap], band_px: float, levels: int = 6):
        self.levels = levels
        self.mx = [_u(c.map_x) for c in cams]
        self.my = [_u(c.map_y) for c in cams]
        v0 = cams[0].valid.astype(np.float32)
        v1 = cams[1].valid.astype(np.float32)
        self.v0, self.inv0 = _u3(v0), _u3(1.0 - v0)
        self.v1, self.inv1 = _u3(v1), _u3(1.0 - v1)

        w = seam_weights(cams, band_px)          # single-channel float32
        gw = [w]
        for _ in range(levels):
            gw.append(cv2.pyrDown(gw[-1]))
        self.gw = [_u3(lvl) for lvl in gw]        # 3ch weight per pyramid level
        self.inv_gw = [_u3(1.0 - lvl) for lvl in gw]
        # pyramid level sizes as (w, h) - UMat has no .size(), and pyrUp needs
        # an explicit dstsize (cv2.pyrDown outputs ((cols+1)//2, (rows+1)//2)).
        self.sizes = [(lvl.shape[1], lvl.shape[0]) for lvl in gw]

    def warp(self, frame: np.ndarray, i: int) -> cv2.UMat:
        """Project fisheye i into equirect on the GPU -> float32 UMat."""
        src = _u(frame.astype(np.float32))
        return cv2.remap(src, self.mx[i], self.my[i], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def blend(self, a: cv2.UMat, b: cv2.UMat) -> cv2.UMat:
        """Laplacian-pyramid (multi-band) blend on the GPU."""
        # fill each camera's holes with the other so Laplacians see no cliff
        a = cv2.add(cv2.multiply(a, self.v0), cv2.multiply(b, self.inv0))
        b = cv2.add(cv2.multiply(b, self.v1), cv2.multiply(a, self.inv1))

        def pyr(img):
            g = [img]
            for _ in range(self.levels):
                g.append(cv2.pyrDown(g[-1]))
            return g

        ga, gb = pyr(a), pyr(b)
        out = []
        for i in range(self.levels):
            la = cv2.subtract(ga[i], cv2.pyrUp(ga[i + 1], dstsize=self.sizes[i]))
            lb = cv2.subtract(gb[i], cv2.pyrUp(gb[i + 1], dstsize=self.sizes[i]))
            out.append(cv2.add(cv2.multiply(la, self.gw[i]),
                               cv2.multiply(lb, self.inv_gw[i])))
        L = self.levels
        out.append(cv2.add(cv2.multiply(ga[L], self.gw[L]),
                           cv2.multiply(gb[L], self.inv_gw[L])))
        img = out[L]
        for i in range(L - 1, -1, -1):
            img = cv2.add(cv2.pyrUp(img, dstsize=self.sizes[i]), out[i])
        return img


__all__ = ["available", "GpuStitcher"]
