"""End-to-end dual-fisheye -> levelled equirectangular pipeline.

Everything needed comes out of the OSV file itself:

    per-unit lens calibration    -> osv_calibration.read_osv_calibration
    per-frame orientation        -> metadata.read_metadata (+ read_highrate)
    camera<->IMU mount rotation  -> osv_calibration.read_mount_rotation
    fisheye -> equirect maps     -> project.build_map
    seam blending                -> stitcher.seam_weights / blend

The handful of values that are NOT in the OSV - DJI's layout conventions and
a few defaults - live in ``constants.py`` with their provenance.

Levelling modes (``Stitcher360.levelling``):

``plain``
    The constant layout convention ``K_PLAIN``; matches DJI's raw, unlevelled
    export - the image follows the camera.
``rocksteady``
    Stabilisation only: counter-rotate toward the Gaussian-smoothed orientation
    trajectory, so shake is removed while the gross tumble passes through.
    Measured smoother than DJI's own export (jitter 0.027/0.071 vs 0.046/0.118
    deg/frame calm/moving).
``rocksteady_horizon``
    HorizonLevelling: all three axes locked to a fixed level world view, from
    the per-frame IMU quaternion. Matches DJI's rocksteady+horizonlevelling and
    holds yaw tighter (DJI wanders under violent motion).
``none``
    Identity - the raw rig frame (~90 deg off DJI's layout).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import av
import cv2
import numpy as np

from .calibration import CameraMap
from .constants import (
    BITRATE_BPP,            # noqa: F401  (re-exported for callers)
    C_UP_FORMAT,
    DEFAULT_FOV_DEG,
    EXTRINSIC_ORDER,
    EXTRINSIC_SOURCE,
    FEATHER_FRACTION,
    K_PLAIN,
    QUAT_ORDER,
    RS_READOUT,
    RS_ROW_DIR,
    RS_SIGMA,
    RS_SIGN,
    RS_TABLE_DS,
    SEAM_OVERLAP_RATIO,
    TRC_GAMMA,
    TRC_GAMMA_FALLBACK,
    WORLD_FWD,
    WORLD_UP,
)
from .metadata import OSVMeta, read_metadata, quat_to_matrix
from .orientation import rotate_equirect
from .osv_calibration import CameraCalib, read_osv_calibration, read_mount_rotation
from .project import build_map, radial_map
from .stitcher import blend, multiband_blend, flow_blend, match_exposure, seam_weights, warp

logger = logging.getLogger(__name__)


def _video_size(path: str | Path) -> tuple[int, int]:
    """The fisheye streams' recording resolution, from the file itself."""
    try:
        c = av.open(str(path))
        try:
            v = next(s for s in c.streams.video
                     if s.codec_context.name != "mjpeg")
            return (v.codec_context.width, v.codec_context.height)
        finally:
            c.close()
    except Exception:
        return (3840, 3840)


def transfer_gamma(path: str | Path) -> float:
    """Encoding transfer gamma from the video stream's declared ``color_trc``."""
    try:
        c = av.open(str(path))
        try:
            v = next(s for s in c.streams.video
                     if s.codec_context.name != "mjpeg")
            trc = v.codec_context.color_trc
        finally:
            c.close()
        try:
            trc = int(trc)
        except (TypeError, ValueError):
            trc = int(getattr(trc, "value", 2))
        return TRC_GAMMA.get(trc, TRC_GAMMA_FALLBACK)
    except Exception:
        return TRC_GAMMA_FALLBACK


def lsc_gain_map(cam_map: CameraMap, calib: CameraCalib,
                 gamma: float = TRC_GAMMA_FALLBACK, alpha: float = 1.0) -> np.ndarray | None:
    """Per-output-pixel devignetting gain for one lens, or None if no LSC data.

    Inverts the f-theta radial map to recover the undistorted radius the factory
    curve (calibration fields 22/23) is authored in, looks up V, and returns
    ``(1/V)^(alpha/gamma)`` matching ``cam_map``. ``gamma`` is the file's
    encoding transfer (see :func:`transfer_gamma`); ``alpha`` scales the
    correction in log space (1 = the full factory curve, 0 = none).

    NOTE: the OSV fisheyes arrive already devignetted by the in-camera ISP, so
    applying the full curve (alpha=1) on top double-corrects (a visible bright
    arc at the seam) - it is OFF by default in :func:`video.convert`. A
    PARTIAL correction (small alpha) targets the residual falloff left by the
    ISP; see IMPROVEMENTS.md for the measured residual, which is what sets
    the justified alpha (0 unless the measurement says otherwise).
    """
    if calib.lsc is None:
        return None
    if alpha <= 0.0:
        return np.ones_like(cam_map.map_x, dtype=np.float32)
    r_lsc, v_lsc = calib.lsc
    r_sensor = np.hypot(cam_map.map_x - calib.cx, cam_map.map_y - calib.cy)
    theta_d = r_sensor / calib.fx
    # Inversion table for theta_d -> theta; the range comfortably brackets any
    # supported FOV (2.2 rad = 126 deg half-angle vs the ~97 deg actually used).
    tg = np.linspace(0.0, 2.2, 4000)
    theta = np.interp(theta_d, radial_map(tg, calib, "ftheta"), tg)
    undist_r = calib.fx * theta
    V = np.clip(np.interp(undist_r, r_lsc, v_lsc, right=v_lsc[-1]), 0.3, 1.0)
    return ((1.0 / V) ** (float(alpha) / gamma)).astype(np.float32)


def _rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle, radians)."""
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(ax)
    return ax / n * ang if n > 1e-12 else np.zeros(3)


def horizon_lock_rotation(raw_f: np.ndarray, bridge: np.ndarray) -> np.ndarray:
    """Full horizon-levelling rotation for one frame (all three axes locked).

    Locks the view to a FIXED world orientation: world-up at the output pole
    (pitch+roll level) and world-forward as the heading (yaw locked). Both come
    from the SAME instantaneous IMU quaternion via ``bridge`` (the per-file
    IMU->equirect map ``C_UP_FORMAT @ mount``) - the key to a clean yaw lock:
    using the smoothed orientation for the heading makes it lag the real
    content during fast motion and the world swings in azimuth (measured: a
    56 deg swing with 6 deg/frame jerks); the consistent instantaneous basis
    pins every world-fixed point (0 jerk), strictly steadier than DJI.

    Returned as the rig->output rotation R: the stitch samples output pixel
    direction ``d`` at ``R @ d``, so world-up as the middle column puts it at
    the output pole (``d = [0,1,0]``).
    """
    up = bridge @ (raw_f.T @ WORLD_UP)
    up = up / np.linalg.norm(up)
    fwd_ref = bridge @ (raw_f.T @ WORLD_FWD)           # fixed world heading
    fwd = fwd_ref - (fwd_ref @ up) * up
    if np.linalg.norm(fwd) < 1e-6:                     # heading ~parallel to up
        alt = WORLD_UP if abs(up[0]) > 0.9 else WORLD_FWD
        fwd = alt - (alt @ up) * up
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(up, fwd)
    right = right / np.linalg.norm(right)
    return np.stack([right, up, fwd], axis=1)          # cols = rig dirs of out axes


def smoothed_rotation_track(quats: np.ndarray, sigma: float = RS_SIGMA,
                            order: str = QUAT_ORDER):
    """Per-frame IMU rotation matrices and their Gaussian-smoothed trajectory.

    Returns ``(raw (N,3,3), smooth (N,3,3))``. Smoothing is a Gaussian-weighted
    mean of the rotation matrices, re-orthonormalised (SVD) per frame - a matrix
    low-pass that removes shake while keeping the tumble.
    """
    N = len(quats)
    if N == 0:                       # no metadata track at all
        return np.zeros((0, 3, 3)), np.zeros((0, 3, 3))
    raw = np.stack([quat_to_matrix(q, order) if np.isfinite(q).all() else np.eye(3)
                    for q in quats])
    ok = np.array([np.isfinite(q).all() for q in quats])
    half = int(3 * sigma)
    ker = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    smooth = np.empty_like(raw)
    for f in range(N):
        lo, hi = max(0, f - half), min(N, f + half + 1)
        w = ker[lo - (f - half): hi - (f - half)] * ok[lo:hi]
        s = w.sum()
        if s < 1e-6:
            smooth[f] = raw[f]
            continue
        M = np.tensordot(w / s, raw[lo:hi], axes=(0, 0))
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        smooth[f] = R if np.linalg.det(R) > 0 else U @ np.diag([1, 1, -1]) @ Vt
    return raw, smooth


def geometry_key(calibs: list[CameraCalib], width: int, height: int, fov_deg: float,
                 seam_overlap_ratio: float) -> str:
    """Stable hash of everything the static remap tables depend on.

    Files from the same unit share their calibration, so the (seconds-long)
    map build and seam-weight construction can be reused across files in a
    long-lived process. The key covers every calibration field the
    projection reads plus the output geometry; anything else (metadata,
    mount, levelling) is per file and cheap.
    """
    import hashlib
    h = hashlib.sha256()
    for c in calibs:
        h.update(repr((float(c.fx), float(c.fy), float(c.cx), float(c.cy),
                       tuple(float(k) for k in c.k), tuple(float(x) for x in (c.p or (0, 0))),
                       int(c.width), int(c.height), float(c.yaw), float(c.pitch),
                       float(c.roll), tuple(float(q) for q in c.quat) if c.quat else None)
                      ).encode())
    h.update(repr((int(width), int(height), float(fov_deg), float(seam_overlap_ratio),
                   EXTRINSIC_SOURCE, EXTRINSIC_ORDER, float(FEATHER_FRACTION))).encode())
    return h.hexdigest()


# Process-wide cache of built rig geometry (maps + seam weights) keyed by
# geometry_key: the second file from the same camera pays no map build.
# Bounded: one 8K entry is ~1.3 GB (f32 maps + masks + weights).
_RIG_CACHE: dict[str, tuple] = {}
RIG_CACHE_ENTRIES = 2


def _rig_cache_put(key: str, value: tuple) -> None:
    _RIG_CACHE[key] = value
    while len(_RIG_CACHE) > max(RIG_CACHE_ENTRIES, 1):
        _RIG_CACHE.pop(next(iter(_RIG_CACHE)))


def clear_geometry_cache() -> None:
    """Drop the process-wide rig geometry cache."""
    _RIG_CACHE.clear()


@dataclass
class Stitcher360:
    """Reusable stitcher for one OSV file.

    Build with :meth:`from_file`; everything geometric and temporal is read
    from the OSV. The heavy remap tables are built lazily on first use and
    cached on the instance.
    """

    calibs: list[CameraCalib]
    meta: OSVMeta
    width: int = 7680
    height: int = 3840
    fov_deg: float = DEFAULT_FOV_DEG
    seam_overlap_ratio: float = SEAM_OVERLAP_RATIO
    mount: np.ndarray | None = None   # camera<->IMU rotation (header 1.3.1)
    levelling: str = "plain"          # plain | rocksteady | rocksteady_horizon | none
    rs_sigma: float = RS_SIGMA        # RockSteady trajectory-smoother sigma (frames)
    gamma: float = TRC_GAMMA_FALLBACK  # encoding transfer, from the stream color_trc
    _path: str | None = None          # source OSV (for lazy high-rate IMU reads)
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: str | Path, width: int = 7680, height: int = 3840,
                  fov_deg: float | None = None, levelling: str = "plain",
                  max_frames: int | None = None) -> "Stitcher360":
        # Select the calibration blocks matching the file's ACTUAL recording
        # resolution - the header carries sets for several sensor modes.
        calibs = read_osv_calibration(path, want_size=_video_size(path))
        meta = read_metadata(path, max_frames=max_frames)
        mount = read_mount_rotation(path)
        fov = DEFAULT_FOV_DEG if fov_deg is None else fov_deg
        logger.info("%s: %s, serial %s, %d frames; FOV %.2f deg; mount %s",
                    Path(path).name, meta.model, meta.serial,
                    len(meta.quats), fov,
                    "present (1.3.1)" if mount is not None else "absent")
        return cls(calibs, meta, width, height, fov, mount=mount,
                   levelling=levelling, gamma=transfer_gamma(path),
                   _path=str(path))

    # ---- static geometry ----------------------------------------------------

    def maps_for(self, frame: int = 0) -> tuple[list[CameraMap], np.ndarray]:
        """Remap tables + seam weights in the camera-rig frame.

        These are **orientation-independent** and built once: the fisheye->rig
        projection is fixed by the calibration. Per-frame levelling is applied
        afterwards as a cheap equirect rotation. ``frame`` is accepted for API
        compatibility but does not affect the maps.
        """
        hit = self._cache.get("rig")
        if hit is not None:
            return hit
        key = self.geometry_key()
        hit = _RIG_CACHE.get(key)
        if hit is not None:
            self._cache["rig"] = hit
            return hit
        maps = [build_map(self.calibs[i], out_w=self.width, out_h=self.height,
                          source=EXTRINSIC_SOURCE, order=EXTRINSIC_ORDER,
                          max_fov_deg=self.fov_deg,
                          feather_deg=self.seam_overlap_ratio * 360.0 * FEATHER_FRACTION)
                for i in (0, 1)]
        weights = seam_weights(maps, band_px=self.seam_overlap_ratio * self.width)
        self._cache["rig"] = (maps, weights)
        _rig_cache_put(key, (maps, weights))
        return maps, weights

    def geometry_key(self) -> str:
        """Hash of the static geometry (see :func:`geometry_key`)."""
        return geometry_key(self.calibs, self.width, self.height, self.fov_deg,
                            self.seam_overlap_ratio)

    def lsc_gains(self, alpha: float = 1.0) -> list[np.ndarray] | None:
        """Per-lens devignetting gain maps (equirect), or None if no LSC data.

        ``alpha`` scales the factory curve (see :func:`lsc_gain_map`); the
        full curve is OFF by default in convert().
        """
        key = ("lsc", float(alpha))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        maps, _ = self.maps_for()
        gains = [lsc_gain_map(maps[i], self.calibs[i], self.gamma, alpha) for i in (0, 1)]
        gains = None if any(g is None for g in gains) else gains
        self._cache[key] = gains
        return gains

    # ---- orientation tracks -------------------------------------------------

    def bridge(self) -> np.ndarray:
        """Per-file IMU-body -> equirect bridge: C_UP_FORMAT @ mount.

        The mount (OSV header 1.3.1, the magnetic-mount skew, ~13 deg on the
        units seen so far) is PER-UNIT and must come from each file; only the
        format-level part is a constant. det = -1 (see constants.py).
        """
        hit = self._cache.get("bridge")
        if hit is None:
            hit = C_UP_FORMAT @ (self.mount if self.mount is not None else np.eye(3))
            self._cache["bridge"] = hit
        return hit

    def _rs_track(self):
        """Cached (raw, smooth) per-frame IMU rotations + smoothed trajectory."""
        hit = self._cache.get("rs")
        if hit is None:
            hit = smoothed_rotation_track(self.meta.quats, self.rs_sigma)
            self._cache["rs"] = hit
        return hit

    def _rs_highrate(self):
        """Cached per-frame high-rate quaternion blocks (None if absent)."""
        hit = self._cache.get("rs_hi")
        if hit is None:
            from .metadata import read_highrate
            hit = read_highrate(self._path) if self._path else []
            self._cache["rs_hi"] = hit
        return hit

    # ---- rolling-shutter correction (disabled by default; see constants) ----

    def rs_tables(self):
        """Static rolling-shutter tables per lens: (tau, J) at 1/RS_TABLE_DS res.

        ``tau`` (h, w) is the source SENSOR-ROW fraction in [0, 1] for each
        output pixel; ``J`` (h, w, 2, 3) is the sensor-plane shift (px) per
        radian of rig-frame rotation about x/y/z, built numerically from three
        perturbed projection maps. Zero where the lens has no coverage.
        """
        hit = self._cache.get("rs_tables")
        if hit is not None:
            return hit
        ds = RS_TABLE_DS
        w4, h4 = self.width // ds, self.height // ds
        eps = np.radians(0.25)
        common = dict(out_w=w4, out_h=h4, source=EXTRINSIC_SOURCE,
                      order=EXTRINSIC_ORDER, max_fov_deg=self.fov_deg,
                      feather_deg=self.seam_overlap_ratio * 360.0 * FEATHER_FRACTION)
        axes = [np.array([eps, 0, 0]), np.array([0, eps, 0]), np.array([0, 0, eps])]

        def small_rot(v):
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return np.eye(3) + vx + vx @ vx / 2.0

        tables = []
        for i in (0, 1):
            cal = self.calibs[i]
            base = build_map(cal, **common)
            J = np.zeros((h4, w4, 2, 3), np.float32)
            for a, v in enumerate(axes):
                pm = build_map(cal, R_extra=small_rot(v), **common)
                J[..., 0, a] = (pm.map_x - base.map_x) / eps
                J[..., 1, a] = (pm.map_y - base.map_y) / eps
            ok = base.valid & (np.abs(J).max(axis=(2, 3)) < 6.0 * cal.fx)
            J[~ok] = 0.0
            tau = np.clip(base.map_y / max(cal.height - 1, 1), 0.0, 1.0).astype(np.float32)
            tau[~base.valid] = 0.0
            tables.append((tau, J))
        self._cache["rs_tables"] = tables
        return tables

    def rs_args_for(self, frame: int):
        """Per-frame rolling-shutter correction: (theta (S,3) f32, t0, scale).

        ``theta[k]`` is the rig-frame rotation vector (radians) of the camera at
        sub-frame time k/(S-1) relative to frame start (where the per-frame
        quaternion and the stabilisation reference live). The GPU evaluates each
        pixel's sample position as (t0 + scale * tau) * (S-1). Returns None when
        the high-rate track is missing for this frame.
        """
        hi = self._rs_highrate()
        if frame >= len(hi) or hi[frame] is None or len(hi[frame]) < 4:
            return None
        h = hi[frame]
        R0 = quat_to_matrix(h[0], QUAT_ORDER)
        C = self.bridge()
        th = np.stack([C @ _rotvec(R0.T @ quat_to_matrix(q, QUAT_ORDER))
                       for q in h])
        th = (RS_SIGN * th).astype(np.float32)
        if RS_ROW_DIR >= 0:
            return th, 0.0, float(RS_READOUT)
        return th, float(RS_READOUT), -float(RS_READOUT)

    # ---- per-frame levelling ------------------------------------------------

    def levelling_for(self, frame: int) -> np.ndarray:
        """The rig-frame rotation to apply for this frame, per ``levelling``.

        See the module docstring for what each mode does and how it was
        validated against DJI's own exports.
        """
        if self.levelling == "none":
            return np.eye(3)
        if self.levelling == "rocksteady":
            # Counter-rotate toward the smoothed trajectory, conjugated by the
            # content bridge (C_UP_FORMAT @ mount) so the raw orientation cancels EXACTLY:
            # world->output becomes const @ smooth.T - the output follows the
            # SMOOTHED camera with the trajectory's own jitter (0.013 deg/frame).
            # (A G-conjugated composition applies the small correction about a
            # TUMBLING axis and leaves 0.24-0.42 deg/frame of structural jitter
            # - that bug was this project's longest hunt.) K_PLAIN sits on the
            # RIGHT to keep the cancellation intact; a steady camera renders
            # exactly like plain, as DJI does.
            raw, smooth = self._rs_track()
            if len(raw):
                # Past the end of the metadata track (it can run a frame or
                # two short of the video) HOLD the last orientation: the old
                # K_PLAIN fallback snapped the final frames to the raw layout.
                f = min(frame, len(raw) - 1)
                C = self.bridge()
                return C @ raw[f].T @ smooth[f] @ C.T @ K_PLAIN
            return K_PLAIN
        if self.levelling == "rocksteady_horizon":
            raw, _ = self._rs_track()
            if len(raw):
                f = min(frame, len(raw) - 1)          # hold, as above
                return horizon_lock_rotation(raw[f], self.bridge())
            return K_PLAIN
        return K_PLAIN   # "plain"

    # ---- CPU reference stitch (stills / no-GPU fallback) --------------------

    def stitch_rig(self, fisheye0: np.ndarray, fisheye1: np.ndarray,
                   flow: bool = False, exposure: bool = False,
                   multiband: bool = True) -> np.ndarray:
        """Stitch a pair into the camera-rig frame (no levelling)."""
        maps, weights = self.maps_for()
        warped = [warp(fisheye0, maps[0]), warp(fisheye1, maps[1])]
        if exposure:
            warped = match_exposure(warped, maps)
        if flow:
            return flow_blend(warped, maps, weights)
        if multiband:
            return multiband_blend(warped, maps, weights=weights)
        return blend(warped, maps, weights=weights)

    def stitch(self, fisheye0: np.ndarray, fisheye1: np.ndarray,
               frame: int = 0, flow: bool = False,
               exposure: bool = False, multiband: bool = True) -> np.ndarray:
        """Stitch one frame pair to a levelled equirectangular panorama (CPU).

        Stitches in the rig frame (fixed maps) then rotates the result per the
        levelling mode. ``flow=True`` morphs across the seam to absorb
        parallax; ``multiband=True`` uses Laplacian-pyramid blending;
        ``exposure`` matches the sensors across the seam (OFF by default - the
        OSV footage arrives exposure-matched by the in-camera ISP).
        """
        pano = self.stitch_rig(fisheye0, fisheye1, flow=flow,
                               exposure=exposure, multiband=multiband)
        R = self.levelling_for(frame)
        if not np.allclose(R, np.eye(3), atol=1e-9):
            pano = rotate_equirect(pano, R)
        return pano

    def seam_agreement(self, fisheye0: np.ndarray, fisheye1: np.ndarray,
                       frame: int = 0) -> float:
        """Inter-camera correlation in the overlap; a calibration quality check.

        Needs no reference output. 1.0 would be perfect; parallax puts a
        practical ceiling well below that for close subjects.
        """
        maps, _ = self.maps_for(frame)
        ov = maps[0].valid & maps[1].valid
        if ov.sum() < 5000:
            return float("nan")
        a = cv2.cvtColor(warp(fisheye0, maps[0]), cv2.COLOR_BGR2GRAY).astype(np.float32)[ov]
        b = cv2.cvtColor(warp(fisheye1, maps[1]), cv2.COLOR_BGR2GRAY).astype(np.float32)[ov]
        if a.std() < 1e-3 or b.std() < 1e-3:
            return float("nan")
        return float((((a - a.mean()) / a.std()) * ((b - b.mean()) / b.std())).mean())


__all__ = [
    "Stitcher360",
    "horizon_lock_rotation",
    "smoothed_rotation_track",
    "transfer_gamma",
    "lsc_gain_map",
    "geometry_key",
    "clear_geometry_cache",
]
