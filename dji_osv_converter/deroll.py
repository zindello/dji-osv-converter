"""Rolling-shutter correction (deroll).

A CMOS sensor exposes row by row over the readout time, so during camera motion
each row sees a slightly different orientation and straight lines skew. DJI
corrects this per frame from the IMU (CameraModel::derollingMapHex); we do the
same, using the ~34 high-rate orientation quaternions the OSV stores per frame
(metadata field 3.3.2.1.3).

For each output direction we find its source fisheye row, look up the camera
orientation at the instant that row was read, and project through *that*
orientation instead of one orientation for the whole frame. The result is the
frame as a global shutter would have captured it.
"""

from __future__ import annotations

import numpy as np

from .calibration import CameraMap
from .orientation import directions
from .osv_calibration import CameraCalib
from .project import extrinsic_matrix, project_directions


def _quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions, broadcasting over leading dims."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (...,3) by wxyz quaternions q (...,4)."""
    w = q[..., 0:1]
    u = q[..., 1:4]
    uv = np.cross(u, v)
    return v + 2.0 * (w * uv + np.cross(u, uv))


def _interp_quats(samples: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Linear-interpolate a stack of unit quaternions at fractional indices.

    samples: (N,4) wxyz; pos: (...,) in [0, N-1]. Sign-aligned then renormalised
    - fine for the small angles within one frame.
    """
    n = len(samples)
    i0 = np.clip(np.floor(pos).astype(np.int32), 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    t = (pos - i0).astype(np.float32)[..., None]
    q0 = samples[i0]
    q1 = samples[i1]
    # align sign so interpolation takes the short way
    flip = (np.sum(q0 * q1, axis=-1, keepdims=True) < 0)
    q1 = np.where(flip, -q1, q1)
    q = q0 * (1 - t) + q1 * t
    q /= np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q


def deroll_map(cal: CameraCalib, R_ref: np.ndarray, quats: np.ndarray,
               out_w: int, out_h: int, *, base_valid_v: np.ndarray | None = None,
               readout_span: float = 1.0, ref_frac: float = 0.5,
               flip_rows: bool = False, source: str = "quat", order: str = "wxyz",
               max_fov_deg: float = 193.68, feather_deg: float = 4.0,
               model: str = "ftheta") -> CameraMap:
    """Build a rolling-shutter-corrected remap for one camera.

    quats: (N,4) wxyz IMU orientations spanning the readout, in capture order.
    readout_span: fraction of the sample set the readout actually occupies
        (1.0 = the samples span exactly the readout).
    ref_frac: which point in the readout defines the output's reference time
        (0.5 = middle row).
    flip_rows: True if the sensor reads bottom-to-top.
    """
    E = extrinsic_matrix(cal, source, order)
    d_world = directions(out_h, out_w)
    d_rig = d_world @ R_ref.T

    # Reference orientation and per-row orientations from the IMU.
    n = len(quats)
    q_ref = _interp_quats(quats, np.array([ref_frac * (n - 1)]))[0]

    # Source fisheye row per output pixel: one un-derolled projection gives it.
    u0, v0, valid = project_directions(d_rig @ E.T, cal, max_fov_deg, model)
    row = v0 / max(cal.height - 1, 1)                    # 0..1 top..bottom
    if flip_rows:
        row = 1.0 - row
    pos = np.clip((row - ref_frac) * readout_span + ref_frac, 0.0, 1.0) * (n - 1)

    q_pix = _interp_quats(quats, pos)                    # (H,W,4)
    # correction: bring the reference-frame ray to the ray as seen at row time
    qM = _quat_mul(_quat_conj(q_pix), q_ref[None, None, :])
    d_rig_corr = _quat_rotate(qM, d_rig)

    u, v, valid = project_directions(d_rig_corr @ E.T, cal, max_fov_deg, model)

    theta = np.arctan2(np.sqrt((d_rig_corr @ E.T)[..., 0] ** 2
                               + (d_rig_corr @ E.T)[..., 1] ** 2),
                       (d_rig_corr @ E.T)[..., 2])
    edge = np.radians(max_fov_deg / 2.0)
    w = np.clip((edge - theta) / max(np.radians(feather_deg), 1e-6), 0.0, 1.0)
    w = (w * valid).astype(np.float32)
    uu, vv = u.copy(), v.copy()
    uu[~valid] = -1.0
    vv[~valid] = -1.0
    return CameraMap(uu, vv, w, valid)


__all__ = ["deroll_map"]
