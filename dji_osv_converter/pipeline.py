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

import cv2
import numpy as np

from .calibration import CameraMap
from .metadata import OSVMeta, read_metadata
from .orientation import rot_matrix, rotate_equirect
from .osv_calibration import CameraCalib, read_osv_calibration, read_mount_rotation
from .project import build_map
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


# The equirectangular sphere laid out by directions() sits Rx(-90) from the
# frame DJI's calibration (and the mount rotation, header 1.3.1) is expressed
# in. Verified: the rotation between our no-levelling stitch and DJI's raw
# ("plain") export is 90.8 deg about X, i.e. a clean Rx(-90) - the empirical
# value gives identical results, so this is our own convention, not a tuned
# constant. The mount is conjugated by it to act in our frame.
EQUIRECT_CONVENTION = _rot_x(-90.0)


def mount_skew(mount: np.ndarray) -> np.ndarray:
    """Mount rotation (header 1.3.1) expressed in our equirect convention.

    ``C @ mount @ C.T`` where ``C = EQUIRECT_CONVENTION``: a change of basis
    that re-expresses DJI's camera<->IMU rotation in the frame our gravity
    vector and stitch live in.
    """
    C = EQUIRECT_CONVENTION
    return C @ np.asarray(mount, dtype=np.float64) @ C.T


def levelling_rotation(gravity: np.ndarray,
                       mount: np.ndarray | None = None) -> np.ndarray:
    """World->rig rotation placing the horizon level, from a gravity sample.

    ``mount`` is the optional camera<->IMU mount rotation from
    :func:`osv_calibration.read_mount_rotation` (OSV header field 1.3.1). The
    accelerometer measures "down" in the IMU frame, which sits at a fixed skew
    from the optical frame (the magnetic-mount offset, ~13 deg); supplying the
    mount corrects for it and is what makes the horizon truly level. Omit it
    and the horizon carries that constant tilt.

    Yaw is left free: gravity constrains only two of three degrees of freedom,
    and there is no physically preferred heading for a 360.
    """
    g = np.asarray(gravity, dtype=np.float64)
    n = np.linalg.norm(g)
    if n < 1e-6:
        return np.eye(3)
    g = g / n
    if mount is not None:
        g = mount_skew(mount) @ g
    return align_vectors([0.0, -1.0, 0.0], g)


def smooth_gravity(grav: np.ndarray, window: int = 15) -> np.ndarray:
    """Moving-average the gravity track.

    Raw accelerometer samples contain handheld motion as well as gravity
    (observed |g| ranging 0.84-1.28), which would make the horizon jitter.
    """
    if len(grav) < 3 or window < 2:
        return grav
    window = min(window, len(grav))
    k = np.ones(window) / window
    out = np.stack([np.convolve(grav[:, i], k, mode="same") for i in range(3)], axis=1)
    # convolve tapers at the edges; restore ends to raw values
    half = min(window // 2, len(grav) // 2)
    if half:
        out[:half] = grav[:half]
        out[-half:] = grav[-half:]
    return out


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
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: str | Path, width: int = 7680, height: int = 3840,
                  fov_deg: float | None = None,
                  max_frames: int | None = None) -> "Stitcher360":
        calibs = read_osv_calibration(path)
        meta = read_metadata(path, max_frames=max_frames)
        meta.gravity = smooth_gravity(meta.gravity)
        mount = read_mount_rotation(path)
        # FOV and blend band both derive from DJI's shipped seam_overlap_ratio.
        ratio = _read_seam_overlap_ratio()
        fov = _fov_from_overlap(ratio) if fov_deg is None else fov_deg
        logger.info("%s: %s, serial %s, %d samples; seam_overlap %.4f -> FOV %.2f deg; "
                    "mount %s", Path(path).name, meta.model, meta.serial,
                    len(meta.gravity), ratio, fov,
                    "present (1.3.1)" if mount is not None else "absent")
        return cls(calibs, meta, width, height, fov, ratio, mount=mount)

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

    def levelling_for(self, frame: int) -> np.ndarray:
        """The world->rig levelling rotation for a frame (gravity + mount)."""
        if frame < len(self.meta.gravity):
            return levelling_rotation(self.meta.gravity[frame], self.mount)
        return np.eye(3)

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
    "mount_skew",
    "EQUIRECT_CONVENTION",
    "align_vectors",
    "smooth_gravity",
]
