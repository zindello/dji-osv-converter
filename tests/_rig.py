"""A tiny synthetic two-lens rig for stitch tests that need no footage.

Two f-theta fisheyes on the rig poles (cam0 up, cam1 down), 64x64 px each,
stitched to a 128x64 equirect - small enough for the torch-CPU path to run
in milliseconds, complete enough to exercise maps, weights, the band and
the levelling grid.
"""

import numpy as np

from dji_osv_converter.constants import DEFAULT_FOV_DEG, FEATHER_FRACTION, SEAM_OVERLAP_RATIO
from dji_osv_converter.osv_calibration import CameraCalib
from dji_osv_converter.project import build_map

FW = FH = 64
W, H = 128, 64


def calibs() -> list[CameraCalib]:
    f = (FW / 2 - 1.5) / np.radians(DEFAULT_FOV_DEG / 2)       # image circle inside the sensor
    mk = lambda pitch, yaw: CameraCalib(f, f, FW / 2 - 0.5, FH / 2 - 0.5,
                                        (0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0),
                                        FW, FH, yaw, pitch, 0.0, None, None)
    return [mk(90.0, 0.0), mk(-90.0, 0.0)]


def maps(w: int = W, h: int = H):
    return [build_map(c, out_w=w, out_h=h, source="quat", order="wxyz",
                      max_fov_deg=DEFAULT_FOV_DEG,
                      feather_deg=SEAM_OVERLAP_RATIO * 360.0 * FEATHER_FRACTION)
            for c in calibs()]


def band_px(w: int = W) -> float:
    return SEAM_OVERLAP_RATIO * w


def fisheye_pair(seed: int = 0, lo: float = 64.0, hi: float = 940.0):
    """Two smooth 10-bit yuv420p10le packed fisheyes (FH*3/2, FW) uint16."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:FH, 0:FW].astype(np.float32)
    out = []
    for _ in range(2):
        a, b, c = rng.uniform(0.05, 0.3, 3)
        Y = lo + (hi - lo) * (0.5 + 0.45 * np.sin(a * xx + b * yy) * np.cos(c * yy))
        U = 512 + 200 * np.sin(0.1 * xx)
        V = 512 + 200 * np.cos(0.07 * yy)
        packed = np.empty((FH * 3 // 2, FW), np.uint16)
        packed[:FH] = np.round(Y)
        packed[FH:FH + FH // 4] = np.round(U[::2, ::2]).reshape(FH // 4, FW)
        packed[FH + FH // 4:] = np.round(V[::2, ::2]).reshape(FH // 4, FW)
        out.append(packed)
    return out
