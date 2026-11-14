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


def calibs(fw: int = FW) -> list[CameraCalib]:
    f = (fw / 2 - 1.5) / np.radians(DEFAULT_FOV_DEG / 2)       # image circle inside the sensor
    mk = lambda pitch, yaw: CameraCalib(f, f, fw / 2 - 0.5, fw / 2 - 0.5,
                                        (0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0),
                                        fw, fw, yaw, pitch, 0.0, None, None)
    return [mk(90.0, 0.0), mk(-90.0, 0.0)]


def maps(w: int = W, h: int = H, fw: int = FW):
    return [build_map(c, out_w=w, out_h=h, source="quat", order="wxyz",
                      max_fov_deg=DEFAULT_FOV_DEG,
                      feather_deg=SEAM_OVERLAP_RATIO * 360.0 * FEATHER_FRACTION)
            for c in calibs(fw)]


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


# ---- a larger rig whose two lenses see one world --------------------------
# The 128x64 rig's band is the whole frame and its lenses show unrelated
# pictures: fine for resampling tests, useless for the seam stack. This one
# is big enough for the flow pyramid (band of ~230 rows at 2048x1024) and
# renders both fisheyes from ONE smooth world texture, so the lenses agree
# in the overlap unless the test displaces one of them.

WORLD_W, WORLD_H, WORLD_FW = 2048, 1024, 512


def _world_texture(w: int, h: int, shift_px: float) -> np.ndarray:
    """A smooth 10-bit luma texture on the sphere, shifted in longitude."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    lon = (xx - shift_px) / w * 2.0 * np.pi
    lat = yy / h * np.pi
    rng = np.random.default_rng(7)
    t = np.zeros((h, w))
    for k in (5, 9, 14, 22, 31):
        a, p, q = rng.uniform(0.5, 1.0), rng.uniform(0, 6.28), rng.uniform(0, 6.28)
        t += a * np.sin(k * lon + p) * np.cos(int(rng.integers(3, 9)) * lat * 2 + q)
    t = (t - t.min()) / (t.max() - t.min())
    return 120.0 + 760.0 * t


def _splat(cam, tex: np.ndarray, fw: int) -> np.ndarray:
    """Render a fisheye by forward-splatting an equirect texture through
    the lens map (uses only build_map, so no projection convention is
    re-derived here). Returns packed yuv420p10le with neutral chroma."""
    u = np.rint(cam.map_x).astype(np.int64)
    v = np.rint(cam.map_y).astype(np.int64)
    ok = (cam.map_x >= 0) & (cam.map_y >= 0) & (u < fw) & (v < fw)
    idx = (v * fw + u)[ok]
    acc = np.bincount(idx, weights=tex[ok], minlength=fw * fw)
    cnt = np.bincount(idx, minlength=fw * fw)
    Y = np.where(cnt > 0, acc / np.maximum(cnt, 1), 64.0).reshape(fw, fw)
    packed = np.full((fw * 3 // 2, fw), 512, np.uint16)
    packed[:fw] = np.round(Y)
    return packed


def world_pair(shift_px: float = 0.0, w: int = WORLD_W, h: int = WORLD_H,
               fw: int = WORLD_FW):
    """Two fisheyes of one world; lens 1's world is displaced ``shift_px``
    output pixels along the seam (a pure parallax-like misalignment)."""
    ss = 2                                     # splat density
    big = maps(w * ss, h * ss, fw)
    return (_splat(big[0], _world_texture(w * ss, h * ss, 0.0), fw),
            _splat(big[1], _world_texture(w * ss, h * ss, shift_px * ss), fw))

