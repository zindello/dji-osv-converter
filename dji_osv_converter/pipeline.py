"""End-to-end dual-fisheye -> levelled equirectangular pipeline.

Everything needed comes out of the OSV file itself:

    per-unit lens calibration   -> osv_calibration.read_osv_calibration
    per-frame gravity / rotation-> metadata.read_metadata
    fisheye -> equirect maps    -> project.build_maps
    seam blending               -> stitcher.blend

DJI Studio's shipped calibration is NOT used; those sets are DJI's own
reference units and measurably fit this camera worse (inter-camera seam
agreement 0.36 vs 0.64 with the file's own calibration).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import av
import cv2
import numpy as np

from .calibration import CameraMap
from .metadata import OSVMeta, read_metadata, quat_to_matrix
from .orientation import rot_matrix, rotate_equirect
from .osv_calibration import CameraCalib, read_osv_calibration, read_mount_rotation
from .project import build_map, radial_map, extrinsic_matrix
from .stitcher import blend, multiband_blend, flow_blend, match_exposure, seam_weights, warp

logger = logging.getLogger(__name__)

# Per-camera extrinsic and intrinsic corrections. Both are now IDENTITY: every
# geometric term comes from the OSV file (see notes below). They remain as
# hooks so a caller can experiment, but the shipping pipeline adds nothing.
#
# History worth keeping, because it explains why they are gone:
#   - Once the correct f-theta lens model was decoded, the fitted focal scale
#     collapsed to 0.9995/0.9985 and the principal-point shifts to a few pixels
#     - i.e. the file's intrinsics were right all along.
#   - The ~1.8 deg / ~1.3 deg per-camera rotation turned out to be almost
#     entirely a SHARED rotation (near-identical on both cameras), so it was a
#     levelling offset, not a per-lens error. Its two parts:
#       * ~1 deg about gravity - an arbitrary heading. Gravity fixes only two
#         of three DOF; which way is "front" is a free choice for a 360.
#       * a CONSTANT tilt - the camera<->IMU mount rotation. The accelerometer
#         measures "down" in the IMU frame, which sits at a fixed skew from the
#         optical frame (the magnetic mount, ~13 deg on this unit). DJI stores
#         this in OSV header field 1.3.1 (a Rodrigues vector) as camImuExtriQ.
#         It IS recoverable from the file - read_mount_rotation() - and MUST be
#         applied, or the horizon carries that ~13 deg tilt (level at two
#         headings, tilted at the quadrants, as a 360 pans). See
#         levelling_rotation(); validated 13 deg -> ~2 deg against DJI's export.
CAM_TWEAK = {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 0.0)}
CAM_INTRINSIC_FIX = {0: (0.0, 0.0, 1.0), 1: (0.0, 0.0, 1.0)}
CAM1_TWEAK = CAM_TWEAK[1]   # retained for older callers

# Extrinsic source: the per-camera unit quaternion (OSV field .21) rather than
# yaw/pitch/roll. Both are in the file and describe the same rotation, but the
# lenses sit at pitch ~= 90 deg where yaw/pitch/roll is gimbal-locked and loses
# precision; the quaternion does not. Stored component order is w,x,y,z.
EXTRINSIC_SOURCE = "quat"
EXTRINSIC_ORDER = "wxyz"

# Field of view actually used per lens. NOT hand-tuned: it is the minimum FOV
# that supplies DJI's own blend band. DJI ships seam_overlap_ratio = 0.038 in
# Stitcher-Mac.bundle; that band spans 0.038 * 360 deg = 13.7 deg of the
# equirect, so each lens must reach 13.7/2 deg past 180, giving
# 180 + 0.038*360 = 193.7 deg. Read from the bundle at runtime; this literal is
# only the fallback if the bundle is absent. (The physical lens is ~210 deg,
# but its rim is the worst optically and is not needed once the blend band is
# covered - and DJI's own image-circle points, calib fields .22/.23, put the
# clean image at ~184-189 deg, bracketing this value.)
SEAM_OVERLAP_RATIO_FALLBACK = 0.038
DEFAULT_FOV_DEG = 180.0 + SEAM_OVERLAP_RATIO_FALLBACK * 360.0  # 193.68


def _read_seam_overlap_ratio() -> float:
    """DJI's blend-band width as a fraction of the 360 deg equirect.

    Shipped in the DualStitcher bundle. Falls back to the observed 0.038 if the
    bundle is not present (e.g. DJI Studio not installed).
    """
    import json
    from .calibration import DEFAULT_BUNDLE
    p = DEFAULT_BUNDLE / "case_00" / "seam_overlap_ratio.json"
    try:
        return float(json.loads(p.read_text())["seam_overlap_ratio"])
    except Exception:
        return SEAM_OVERLAP_RATIO_FALLBACK


def _fov_from_overlap(ratio: float) -> float:
    return 180.0 + ratio * 360.0

# Lens radial model. The k1..k4 coefficients in the file are NOT usable as a
# Brown series in theta: they produce a non-monotonic curve that puts the image
# rim at 1142 px against a measured circle of ~1916 px. Pure equidistant is
# close (rim at 1913 px), and this small series in normalised theta is a
# refinement fitted against DJI's own export of this file.
#
# CAVEAT: fitted on one camera, one recording. Improves edge agreement with DJI
# from 0.31 to 0.49 on frames 0 and 150, but only 0.23 -> 0.26 on frame 600.
# Treat as provisional until validated across more cameras.
# Odd-polynomial fisheye series r = c0*th + c1*th^3 + c2*th^5 + c3*th^7,
# recovered per camera from dense correspondence against DJI's own export:
# where our pixel at angle th shows the content DJI places at th', we know
# r_dji(th') = r_ours(th), which samples DJI's curve directly across the whole
# field rather than only at the seam. Fits to 2.8 px (cam0) and 0.93 px (cam1)
# RMS, and the leading term lands on the file's fx (1057 vs 1044, 1045 vs 1047).
#
# Equisolid, stereographic and orthographic were all ruled out (45-145 px RMS).
#
# CAVEAT: fitted for this camera against one export. The k1..k4 stored in the
# OSV do NOT reproduce these coefficients by any simple scaling, so their
# meaning is still undecoded - until it is, other cameras need their own fit.
LENS_POLY = {
    0: ("odd", [1057.15, 38.83, 5.59, -3.58]),
    1: ("odd", [1044.95, 55.84, 1.54, -3.74]),
}
# Default is the model derived from the file's own k1..k4 (see project.py),
# not the hand-fitted LENS_POLY: it scores the same (cam0 0.937 vs 0.935,
# cam1 0.943 vs 0.955) while needing nothing but the OSV itself, so it should
# generalise to other cameras. LENS_POLY is retained for comparison.
LENS_MODEL = "ftheta"
FOCAL_SCALE = 1.0


# ---- Factory lens-shading (vignetting) correction --------------------------
#
# The stitch seam sits where both lenses reach their vignetted rim, so it reads
# as a dark arc without correction. The OSV ships a factory lens-shading curve
# per lens (calibration fields 22/23; see osv_calibration._parse_lsc): a radial
# brightness profile V(r), peak 1.0 at centre, ~24% falloff by the seam. We
# divide it out - gain = (1/V)^(1/2.2). The exponent is the BT.709 transfer the
# footage is encoded in (colorspace=1): vignetting is multiplicative in linear
# light, so in the encoded domain the gain carries the transfer's power, not a
# free parameter. Applied per output pixel by the ray's radius.
#
# The curve is indexed by the UNDISTORTED radius fx*theta, NOT the distorted
# sensor radius fx*theta_d we sample the pixel at. (Indexing by the sensor
# radius put the seam 11% past the curve and over-brightened - see below.)
#
# The gamma comes from the file's declared transfer (stream color_trc), NOT a
# baked constant - see transfer_gamma().
#
# KNOWN LIMITATION: the curve only samples to ~88 deg (undistorted r=1602px) and
# the blend seam is at 90 deg (r~1639px), so the last value is HELD for the
# final ~2%. The seam centre stays a touch brighter than DJI's - imperceptible
# on real 360 footage, documented rather than chased with extrapolation.

# Encoding transfer gamma per AVCOL_TRC: SDR OETFs (BT.709/601/2020) are a ~0.45
# power, i.e. gamma ~2.2; linear is 1.0; BT.470BG is 2.8. Vignetting is linear-
# multiplicative, so the correction is applied as (1/V)^(1/gamma) in the encoded
# domain. Default 2.2 only when the stream declares nothing (unspecified SDR).
_TRC_GAMMA = {1: 2.2, 4: 2.2, 6: 2.2, 14: 2.2, 15: 2.2, 8: 1.0, 5: 2.8}


def transfer_gamma(path: str | Path) -> float:
    """Encoding transfer gamma from the video stream's declared color_trc."""
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
        return _TRC_GAMMA.get(trc, 2.2)
    except Exception:
        return 2.2


def lsc_gain_map(cam_map: CameraMap, calib: CameraCalib,
                 gamma: float = 2.2) -> np.ndarray | None:
    """Per-output-pixel devignetting gain for one lens, or None if no LSC data.

    Inverts the f-theta radial map to recover the undistorted radius the factory
    curve is authored in, looks up V, and returns ``(1/V)^(1/gamma)`` as a
    float32 array matching ``cam_map`` (equirect/rig frame). ``gamma`` is the
    file's encoding transfer (see :func:`transfer_gamma`).
    """
    if calib.lsc is None:
        return None
    r_lsc, v_lsc = calib.lsc
    r_sensor = np.hypot(cam_map.map_x - calib.cx, cam_map.map_y - calib.cy)
    theta_d = r_sensor / calib.fx
    tg = np.linspace(0.0, 2.2, 4000)                    # invert theta_d -> theta
    theta = np.interp(theta_d, radial_map(tg, calib, "ftheta"), tg)
    undist_r = calib.fx * theta
    V = np.clip(np.interp(undist_r, r_lsc, v_lsc, right=v_lsc[-1]), 0.3, 1.0)
    return ((1.0 / V) ** (1.0 / gamma)).astype(np.float32)


def align_vectors(a, b) -> np.ndarray:
    """Rotation matrix taking unit vector `a` onto unit vector `b`."""
    a = np.asarray(a, dtype=np.float64); a = a / np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64); b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(a @ b)
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# ---- Horizon levelling from the IMU orientation quaternion ------------------
#
# The OSV carries a per-frame device-orientation quaternion (metadata field
# 3.2.9) - DJI's own IMU sensor-fusion output - alongside the raw accelerometer
# "gravity" vector (3.2.10). We level from the QUATERNION, not the accelerometer:
# under handheld motion the accelerometer carries several degrees of
# translational acceleration, and that noise was the visible horizon bow.
# Validated against DJI's own "plain" export (which uses only its default
# levelling): the accelerometer path sat 6-15 deg off DJI's horizon; the
# smoothed-quaternion path below matches it to <1.3 deg across ten test frames -
# the feature-matching measurement floor. See tools/ for the harness.
#
# The chain, all derived from the OSV:
#   up_imu(f) = R(quat_f).T @ WORLD_UP           world-up in the IMU frame
#   up_imu    = moving-average over +-LEVEL_SMOOTH_HALF frames   (DJI smooths the
#               horizon; the raw per-frame quaternion jitters ~1-2 deg about it)
#   up_rig(f) = C_IMU_RIG @ mount @ up_imu(f)     mapped into our equirect frame
#   level(f)  = align_vectors(up_rig(f), +Y)      yaw-free horizon levelling
#
# WORLD_UP is +Z: R(quat) @ accelerometer lands on -Z, so world-up is +Z.
# ``mount`` (header 1.3.1) is the PER-UNIT camera<->IMU skew (magnetic mount,
# ~13 deg on this unit) - read from each OSV so it adapts per unit. C_IMU_RIG is
# the fixed FORMAT convention relating the IMU world axes to our equirect layout;
# it was recovered once as C = B @ mount^-1 from the DJI reference (B being the
# fitted IMU->rig up-vector map), so B = C_IMU_RIG @ mount reproduces it exactly.
# DJI's "plain" export is the RAW optical stitch - no levelling, no
# stabilisation: it flips and rolls with the camera. Verified on a deliberate
# pitch/roll clip - the rotation between our no-levelling rig stitch and DJI's
# plain export stays constant (~90 deg, spread <2 deg) through the whole tumble.
# So matching "plain" needs only this one constant K - NO IMU, NO mount, NO
# gravity. K is almost exactly Rx(90 deg): it is our equirect layout convention
# (how directions() lays out the sphere) versus DJI's, plus a ~1.6 deg residual
# measured against DJI's export. It is NOT in the OSV and does not need to be -
# plain is raw optics and this is purely a rendering-layout convention.
# Measured over pitch_roll_plain.mp4; validated to ~3 deg (stitch-level, not
# orientation) across extreme-tilt frames. Universal across units.
def _orthonormal(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64))
    R = U @ Vt
    return R if np.linalg.det(R) > 0 else U @ np.diag([1, 1, -1]) @ Vt


K_PLAIN = _orthonormal([
    [0.999620, 0.000390, 0.027440],
    [0.027440, 0.003060, -0.999620],
    [-0.000480, 1.000000, 0.003050],
])

QUAT_ORDER = "wxyz"
WORLD_UP = np.array([0.0, 0.0, 1.0])
WORLD_FWD = np.array([1.0, 0.0, 0.0])   # fixed world heading the yaw locks to

# HorizonLevelling: constant map from the IMU body frame to the stitch equirect
# frame for the world-up (and world-forward) vectors. Fit against DJI's
# rocksteady+horizonlevelling export: C_UP_STITCH @ (R(quat).T @ WORLD_UP) is
# DJI's level up-vector in the rig frame to 0.24 deg mean / 0.7 deg max across
# the whole worst-case tumble (per-frame quaternion, field 3.2.9 - no high-rate
# IMU or accelerometer needed; the quaternion carries no usable drift). det = -1:
# it legitimately includes a handedness flip because the equirect z-axis
# convention (dirs z = -cos(lat)cos(lon)) is left-handed vs the physical IMU
# frame. Measured universal constant, like K_PLAIN.
C_UP_STITCH = np.array([
    [0.00044549, -0.99999383, 0.00348370],
    [-0.99991755, -0.00040074, 0.01283477],
    [0.01283330, 0.00348913, 0.99991156],
])
C_IMU_RIG = np.array([
    [0.360809, -0.904583, -0.227040],
    [-0.904352, -0.398841, 0.151896],
    [-0.227955, 0.150519, -0.961967],
])
LEVEL_SMOOTH_HALF = 20   # boxcar half-window (frames) over the up-vector track

# ---- RockSteady (stabilisation) --------------------------------------------
#
# RockSteady is stabilisation only: it low-pass-filters the camera's orientation
# trajectory and counter-rotates each frame toward the smoothed version, so the
# high-frequency shake is removed while the camera's gross tumble (yaw + pitch +
# roll) passes straight through - NOTHING is locked. Measured against DJI's
# rocksteady export: frame-to-frame jitter drops from 0.47 deg (plain) to 0.11
# deg (DJI's own is 0.04), with the mean motion preserved (1.77 vs DJI 1.79).
#
# Per frame:  T(f) = G @ (Rq_smooth(f) @ Rq(f).T) @ G.T   (shake in output frame)
#             R(f) = K_PLAIN @ T(f)
# where Rq is the IMU orientation matrix, Rq_smooth its Gaussian-smoothed
# trajectory, and G = K_PLAIN @ C_IMU_RIG_MOTION @ mount is the world->output
# map. C_IMU_RIG_MOTION is the FULL (yaw-included) IMU<->rig convention - unlike
# C_IMU_RIG (fit yaw-free from the horizon up-vector), the full orientation is
# pinned by fitting the frame-to-frame motion of DJI's plain export. mount
# (header 1.3.1) makes it per-unit.
RS_SIGMA = 15.0          # Gaussian sigma (frames) of the trajectory smoother
C_IMU_RIG_MOTION = np.array([
    [-0.083031, -0.994581, 0.062567],
    [-0.982324, 0.092254, 0.162878],
    [-0.167767, -0.047937, -0.984660],
])


def horizon_lock_rotation(raw_f: np.ndarray) -> np.ndarray:
    """Full horizon-levelling rotation for one frame (all three axes locked).

    Locks the view to a FIXED world orientation: world-up at the output pole
    (pitch+roll level) and world-forward as the heading (yaw locked). Both come
    from the SAME instantaneous IMU quaternion - this is the key to a clean yaw
    lock. Using the smoothed orientation for the heading instead makes the
    smoothed frame lag the real (instantaneous) content during fast motion, so
    the world swings in azimuth (measured: a 56 deg swing, 6 deg/frame jerks);
    the consistent instantaneous basis pins every world-fixed point (0 jerk),
    which is strictly steadier than DJI (DJI actively wanders in yaw). Returned
    as the rig->output rotation R: the stitch samples output pixel dir d at
    ``R @ d``, so world-up as the middle column puts it at the pole (d=[0,1,0]).
    """
    up = C_UP_STITCH @ (raw_f.T @ WORLD_UP)
    up = up / np.linalg.norm(up)
    fwd_ref = C_UP_STITCH @ (raw_f.T @ WORLD_FWD)      # fixed world heading
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


def device_up_track(quats: np.ndarray, order: str = QUAT_ORDER) -> np.ndarray:
    """World-up expressed in the IMU frame per frame: ``R(quat).T @ WORLD_UP``.

    Rows for missing / non-unit quaternions are NaN so they drop out of the
    moving average rather than shifting every later frame.
    """
    out = np.full((len(quats), 3), np.nan)
    for i, q in enumerate(quats):
        if np.isfinite(q).all() and abs(np.linalg.norm(q) - 1.0) < 1e-2:
            out[i] = quat_to_matrix(q, order).T @ WORLD_UP
    return out


def smooth_unit_track(vecs: np.ndarray, half: int) -> np.ndarray:
    """Centred moving-average of a unit-vector track, renormalised (NaN-safe).

    This is the ring-buffer DJI appears to run on the horizon: the raw per-frame
    IMU up-vector jitters ~1-2 deg, and averaging it over a short window holds
    the horizon steady. Centred (non-causal) - fine for offline conversion,
    which has the whole track.
    """
    N = len(vecs)
    out = np.full_like(vecs, np.nan, dtype=np.float64)
    finite = np.isfinite(vecs).all(1)
    if half < 1:
        return vecs.copy()
    for i in range(N):
        lo, hi = max(0, i - half), min(N, i + half + 1)
        seg = vecs[lo:hi][finite[lo:hi]]
        if len(seg):
            v = seg.mean(0)
            n = np.linalg.norm(v)
            if n > 1e-9:
                out[i] = v / n
    return out


def up_track_from_meta(quats: np.ndarray, mount: np.ndarray | None,
                       half: int = LEVEL_SMOOTH_HALF) -> np.ndarray:
    """Smoothed world-up in the rig frame, per frame (the levelling up-vector)."""
    B = C_IMU_RIG @ (np.asarray(mount, dtype=np.float64)
                     if mount is not None else np.eye(3))
    up_imu = smooth_unit_track(device_up_track(quats), half)   # smooth in IMU frame
    return up_imu @ B.T                                        # rows: B @ up_imu


def levelling_rotation(up_rig: np.ndarray) -> np.ndarray:
    """Yaw-free horizon-levelling rotation from a rig-frame world-up vector."""
    n = np.asarray(up_rig, dtype=np.float64)
    if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-6:
        return np.eye(3)
    return align_vectors(n / np.linalg.norm(n), [0.0, 1.0, 0.0])


@dataclass
class Stitcher360:
    """Reusable stitcher for one OSV file."""

    calibs: list[CameraCalib]
    meta: OSVMeta
    width: int = 7680
    height: int = 3840
    fov_deg: float = DEFAULT_FOV_DEG
    seam_overlap_ratio: float = SEAM_OVERLAP_RATIO_FALLBACK
    lens_model: object = LENS_MODEL
    focal_scale: float = FOCAL_SCALE
    mount: np.ndarray | None = None   # camera<->IMU rotation (header 1.3.1)
    # "plain" | "rocksteady" | "rocksteady_horizon" | "horizon" | "none"
    levelling: str = "plain"
    level_smooth_half: int = LEVEL_SMOOTH_HALF
    rs_sigma: float = RS_SIGMA         # RockSteady trajectory-smoother sigma (frames)
    up_track: np.ndarray | None = None   # smoothed rig-frame world-up, per frame
    gamma: float = 2.2                # encoding transfer, from the stream color_trc
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: str | Path, width: int = 7680, height: int = 3840,
                  fov_deg: float | None = None, levelling: str = "plain",
                  level_smooth_half: int = LEVEL_SMOOTH_HALF,
                  max_frames: int | None = None) -> "Stitcher360":
        calibs = read_osv_calibration(path)
        meta = read_metadata(path, max_frames=max_frames)
        mount = read_mount_rotation(path)
        # Horizon levelling comes from the per-frame IMU orientation quaternion,
        # smoothed and mapped into the rig frame (see the levelling notes above).
        up_track = up_track_from_meta(meta.quats, mount, level_smooth_half)
        # FOV and blend band both derive from DJI's shipped seam_overlap_ratio.
        ratio = _read_seam_overlap_ratio()
        fov = _fov_from_overlap(ratio) if fov_deg is None else fov_deg
        logger.info("%s: %s, serial %s, %d frames; seam_overlap %.4f -> FOV %.2f deg; "
                    "mount %s", Path(path).name, meta.model, meta.serial,
                    len(meta.quats), ratio, fov,
                    "present (1.3.1)" if mount is not None else "absent")
        return cls(calibs, meta, width, height, fov, ratio, mount=mount,
                   levelling=levelling, level_smooth_half=level_smooth_half,
                   up_track=up_track, gamma=transfer_gamma(path))

    def maps_for(self, frame: int = 0) -> tuple[list[CameraMap], np.ndarray]:
        """Remap tables + seam weights in the camera-rig frame.

        These are **orientation-independent** and built once: the fisheye->rig
        projection is fixed by the calibration. Per-frame levelling is applied
        afterwards as a cheap equirect rotation (see ``stitch``), so the
        expensive f-theta map build no longer runs per frame. ``frame`` is
        accepted for API compatibility but does not affect the maps.
        """
        hit = self._cache.get("rig")
        if hit is not None:
            return hit

        common = dict(out_w=self.width, out_h=self.height,
                      source=EXTRINSIC_SOURCE, order=EXTRINSIC_ORDER,
                      max_fov_deg=self.fov_deg,
                      feather_deg=self.seam_overlap_ratio*360.0/3.0,
                      focal_scale=self.focal_scale)

        def prepared(i):
            """Calibration and lens model for camera i, with fitted corrections."""
            import copy as _copy
            dcx, dcy, fscale = CAM_INTRINSIC_FIX.get(i, (0.0, 0.0, 1.0))
            cal = _copy.copy(self.calibs[i])
            cal.cx += dcx
            cal.cy += dcy
            lm = self.lens_model
            model = lm[i] if isinstance(lm, dict) else lm
            if isinstance(model, tuple) and model and model[0] == "odd" and fscale != 1.0:
                model = ("odd", [c * fscale for c in model[1]])
            return cal, model

        maps = []
        for i in (0, 1):
            cal, model = prepared(i)
            maps.append(build_map(cal, R_extra=rot_matrix(*CAM_TWEAK[i]),
                                  model=model, **common))
        weights = seam_weights(maps, band_px=self.seam_overlap_ratio * self.width)
        self._cache["rig"] = (maps, weights)
        return maps, weights

    def lsc_gains(self) -> list[np.ndarray] | None:
        """Per-lens devignetting gain maps (equirect), or None if no LSC in file.

        Built once from the factory lens-shading curve; multiplied into each
        warped lens before the blend to lift the vignetted rim at the seam.
        """
        hit = self._cache.get("lsc")
        if hit is not None:
            return hit
        maps, _ = self.maps_for()
        gains = [lsc_gain_map(maps[i], self.calibs[i], self.gamma) for i in (0, 1)]
        gains = None if any(g is None for g in gains) else gains
        self._cache["lsc"] = gains
        return gains

    def _rs_track(self):
        """Cached (raw, smooth, G) for RockSteady: IMU rotations, their smoothed
        trajectory, and the world->output map G = K @ C_IMU_RIG_MOTION @ mount."""
        hit = self._cache.get("rs")
        if hit is not None:
            return hit
        raw, smooth = smoothed_rotation_track(self.meta.quats, self.rs_sigma)
        B = C_IMU_RIG_MOTION @ (self.mount if self.mount is not None else np.eye(3))
        Gmat = K_PLAIN @ B
        self._cache["rs"] = (raw, smooth, Gmat)
        return raw, smooth, Gmat

    def levelling_for(self, frame: int) -> np.ndarray:
        """The rig-frame rotation to apply for this frame, per ``levelling`` mode.

        ``plain`` (default): the constant layout convention ``K_PLAIN`` - matches
        DJI's raw, unlevelled export (the image follows the camera). ``horizon``:
        the smoothed IMU-quaternion leveller, holding the horizon flat.
        ``rocksteady``: stabilisation - counter-rotate the shake toward the
        smoothed orientation trajectory (the tumble is kept, nothing is locked).
        ``rocksteady_horizon``: HorizonLevelling - lock all three axes to a fixed
        level world view (world-up from the IMU quaternion, heading locked to the
        smoothed trajectory); matches DJI's rocksteady+horizonlevelling and holds
        yaw tighter. ``none``: identity (our raw rig frame, ~90 deg off DJI).
        """
        if self.levelling == "none":
            return np.eye(3)
        if self.levelling == "horizon":
            if self.up_track is not None and frame < len(self.up_track):
                return levelling_rotation(self.up_track[frame])
            return K_PLAIN
        if self.levelling == "rocksteady":
            raw, smooth, Gmat = self._rs_track()
            if frame < len(raw):
                T = Gmat @ (smooth[frame] @ raw[frame].T) @ Gmat.T
                return K_PLAIN @ T
            return K_PLAIN
        if self.levelling == "rocksteady_horizon":
            # HorizonLevelling: lock all three axes to a fixed level world view.
            # The world-up comes straight from the per-frame IMU quaternion mapped
            # through C_UP_STITCH (matches DJI's level up to 0.24 deg across the
            # whole worst-case tumble); the heading is locked to a fixed world
            # direction via the smoothed trajectory. The horizon stays dead level
            # and the view holds still while the camera tumbles - and it holds yaw
            # tighter than DJI, which wanders under violent motion. Validated
            # frame-by-frame against DJI's export across the full clip.
            raw, _, _ = self._rs_track()
            if frame < len(raw):
                return horizon_lock_rotation(raw[frame])
            return K_PLAIN
        return K_PLAIN   # "plain"

    def stitch_rig(self, fisheye0: np.ndarray, fisheye1: np.ndarray,
                   flow: bool = False, exposure: bool = True,
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
               exposure: bool = True, multiband: bool = True) -> np.ndarray:
        """Stitch one frame pair to a levelled equirectangular panorama.

        Stitches in the rig frame (fixed maps) then rotates the result to level
        the horizon for this frame — far cheaper than rebuilding the maps per
        frame. `flow=True` morphs across the seam to absorb parallax;
        `multiband=True` uses Laplacian-pyramid blending (seam-invisible);
        `exposure=True` equalises the two sensors across the seam.
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
    "levelling_rotation",
    "up_track_from_meta",
    "device_up_track",
    "smooth_unit_track",
    "smoothed_rotation_track",
    "K_PLAIN",
    "C_IMU_RIG",
    "C_IMU_RIG_MOTION",
    "align_vectors",
]
