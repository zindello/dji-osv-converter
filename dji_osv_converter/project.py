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

# The OSV's own k1..k4 ARE the radial polynomial, but expressed in theta
# normalised by THETA_NORM rather than in raw radians. Applied to raw theta the
# curve is non-monotonic and lands the image rim ~770 px short.
#
# THETA_NORM was solved for against the curve recovered from DJI ground truth:
# 1.2427 rad fits best (1.9 px RMS across both cameras). Note sqrt(pi/2) =
# 1.2533 rad fits within measurement noise (1.97 px) and is the value you would
# get if the encoder writes `t2 = theta*theta/(M_PI/2)`, so the exact intended
# constant is probably that. We keep the measured value.
#
# FILE_GAIN converts fx to the polynomial's leading coefficient. Fitted
# independently per camera it came out 1.01043 and 1.01055 - agreeing to four
# decimals, which is why it is treated as one pipeline constant rather than a
# per-lens correction.
THETA_NORM = 1.2427
FILE_GAIN = 1.0105


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


def radial_map(theta: np.ndarray, cal: CameraCalib, model: str = "equidistant",
               theta_max: float = np.radians(105.0)) -> np.ndarray:
    """Map incidence angle to normalised radial distance.

    'equidistant' (theta_d = theta) reproduces the measured image circle almost
    exactly: fx * radians(105) = 1913 px against a circle of ~1916 px.

    'poly' applies cal.k as a Brown-style series in theta. That is how these
    coefficients are usually written, but for this data it is NOT valid - the
    resulting function is non-monotonic (peaks near 60 deg then collapses,
    putting the rim at 1142 px instead of 1913 px). A radial distortion curve
    must increase monotonically, so the coefficients evidently mean something
    else. Kept only for comparison.

    'poly_norm' applies the same series to theta/theta_max, which keeps the
    correction small and monotonic over the real range.
    """
    if isinstance(model, (tuple, list)) and len(model) == 3 and model[0] == "lut":
        # Non-parametric radial curve: theta -> radius, interpolated. Used to
        # test whether the 4-coefficient form is the accuracy limit.
        _, tb, rb = model
        return np.interp(theta, np.asarray(tb), np.asarray(rb),
                         left=float(rb[0]), right=float(rb[-1])) / max(cal.fx, 1e-9)
    if model == "file":
        # The k1..k4 stored in the OSV ARE the radial polynomial - but in theta
        # normalised by pi/2, not raw theta. Applied to raw theta the curve is
        # non-monotonic and puts the image rim ~770 px short; normalised, it
        # reproduces the curve recovered from DJI ground truth to 0.24 deg
        # (cam0) and 0.41 deg (cam1) RMS.
        #
        # FILE_GAIN is a constant factor found identically for both cameras
        # (1.0257 / 1.0259), so it is a property of DJI's pipeline rather than
        # of either lens.
        k1, k2, k3, k4 = cal.k
        t2 = (theta / THETA_NORM) ** 2
        return FILE_GAIN * theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    if model == "equidistant":
        return theta
    if isinstance(model, (tuple, list)) and len(model) == 2 and model[0] == "odd":
        # r = c0*th + c1*th^3 + c2*th^5 + c3*th^7, the classic fisheye series.
        # radial_map returns theta_d, and the caller multiplies by fx, so divide
        # the polynomial through by fx here.
        c = model[1]
        t = theta
        r = c[0] * t
        for i, ci in enumerate(c[1:], start=1):
            r = r + ci * t ** (2 * i + 1)
        return r / max(cal.fx, 1e-9)
    if isinstance(model, (tuple, list)):
        # explicit (a, b) series in normalised theta, fitted empirically
        a, b = model
        tn2 = (theta / theta_max) ** 2
        return theta * (1.0 + tn2 * (a + tn2 * b))
    k1, k2, k3, k4 = cal.k
    if model == "poly":
        t2 = theta * theta
    elif model == "poly_norm":
        t2 = (theta / theta_max) ** 2
    else:
        raise ValueError(f"unknown lens model {model!r}")
    return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))


def project_directions(d_cam: np.ndarray, cal: CameraCalib,
                       max_fov_deg: float = 220.0, model: str = "equidistant"):
    """Project camera-frame direction vectors to fisheye pixel coordinates."""
    X, Y, Z = d_cam[..., 0], d_cam[..., 1], d_cam[..., 2]
    rxy = np.sqrt(X * X + Y * Y)
    theta = np.arctan2(rxy, Z)

    theta_d = radial_map(theta, cal, model, np.radians(max_fov_deg / 2.0))

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
              feather_deg: float = 8.0, model: str = "equidistant",
              focal_scale: float = 1.0) -> CameraMap:
    """Build an equirectangular remap for one camera."""
    d = directions(out_h, out_w)
    R = extrinsic_matrix(cal, source, order, transpose)
    if R_extra is not None:
        R = R @ R_extra
    d_cam = d @ R.T

    if focal_scale != 1.0:
        import copy as _copy
        cal = _copy.copy(cal)
        cal.fx *= focal_scale
        cal.fy *= focal_scale
    u, v, valid = project_directions(d_cam, cal, max_fov_deg, model)

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


__all__ = ["build_map", "build_maps", "extrinsic_matrix", "quat_matrix",
           "project_directions", "radial_map", "FILE_GAIN", "THETA_NORM"]
