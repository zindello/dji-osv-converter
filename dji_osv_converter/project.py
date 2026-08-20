"""Build fisheye->equirectangular remap tables from camera calibration.

This replaces DJI's shipped remap_*.bin (which are generic defaults) with maps
derived from the recording camera's own factory calibration, read out of the
OSV file. Output resolution is ours to choose, so it also removes the fixed
1920x3840 grid of the shipped tables.

Lens model is the equidistant fisheye (as used by OpenCV's fisheye module):

    theta   = angle between the ray and the optical axis
    theta_d = theta * (1 + k1*t^2 + k2*t^4 + k3*t^6 + k4*t^8)
    (u, v)  = (fx*theta_d*X/rxy + cx,  fy*theta_d*Y/rxy + cy)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .calibration import CameraMap
from .osv_calibration import CameraCalib
from .orientation import directions, rot_matrix

logger = logging.getLogger(__name__)


def quat_matrix(q, order: str = "xyzw") -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q if order == "xyzw" else (q[1], q[2], q[3], q[0])
    n = np.linalg.norm([x, y, z, w])
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def extrinsic_matrix(cal: CameraCalib, source: str = "quat",
                     order: str = "xyzw", transpose: bool = False) -> np.ndarray:
    """Rotation taking rig-frame directions into this camera's frame."""
    if source == "quat" and cal.quat is not None:
        R = quat_matrix(cal.quat, order)
    else:
        R = rot_matrix(cal.yaw, cal.pitch, cal.roll)
    return R.T if transpose else R


def project_directions(d_cam: np.ndarray, cal: CameraCalib,
                       max_fov_deg: float = 220.0):
    """Project camera-frame direction vectors to fisheye pixel coordinates."""
    X, Y, Z = d_cam[..., 0], d_cam[..., 1], d_cam[..., 2]
    rxy = np.sqrt(X * X + Y * Y)
    theta = np.arctan2(rxy, Z)

    k1, k2, k3, k4 = cal.k
    t2 = theta * theta
    theta_d = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(rxy > 1e-12, theta_d / rxy, 0.0)
    u = cal.fx * scale * X + cal.cx
    v = cal.fy * scale * Y + cal.cy

    valid = (theta <= np.radians(max_fov_deg / 2.0)) \
        & (u >= 0) & (u < cal.width) & (v >= 0) & (v < cal.height) \
        & np.isfinite(u) & np.isfinite(v)
    return u.astype(np.float32), v.astype(np.float32), valid


def build_map(cal: CameraCalib, out_w: int, out_h: int,
              R_extra: np.ndarray | None = None,
              source: str = "quat", order: str = "xyzw",
              transpose: bool = False, max_fov_deg: float = 220.0,
              feather_deg: float = 8.0) -> CameraMap:
    """Build an equirectangular remap for one camera."""
    d = directions(out_h, out_w)
    R = extrinsic_matrix(cal, source, order, transpose)
    if R_extra is not None:
        R = R @ R_extra
    d_cam = d @ R.T

    u, v, valid = project_directions(d_cam, cal, max_fov_deg)

    # Coverage weight tapers off near the rim, where fisheye optics are worst.
    theta = np.arctan2(np.sqrt(d_cam[..., 0] ** 2 + d_cam[..., 1] ** 2), d_cam[..., 2])
    edge = np.radians(max_fov_deg / 2.0)
    w = np.clip((edge - theta) / max(np.radians(feather_deg), 1e-6), 0.0, 1.0)
    w = (w * valid).astype(np.float32)

    uu, vv = u.copy(), v.copy()
    uu[~valid] = -1.0
    vv[~valid] = -1.0
    return CameraMap(uu, vv, w, valid)


def build_maps(cals: list[CameraCalib], out_w: int, out_h: int, **kw) -> list[CameraMap]:
    maps = [build_map(c, out_w, out_h, **kw) for c in cals]
    union = maps[0].valid | maps[1].valid
    logger.info("built %dx%d maps: cam0 %.1f%%, cam1 %.1f%%, union %.2f%%, overlap %.1f%%",
                out_w, out_h, 100 * maps[0].valid.mean(), 100 * maps[1].valid.mean(),
                100 * union.mean(), 100 * (maps[0].valid & maps[1].valid).mean())
    return maps


__all__ = ["build_map", "build_maps", "extrinsic_matrix", "quat_matrix", "project_directions"]
