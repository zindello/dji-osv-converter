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
from .orientation import rot_matrix
from .osv_calibration import CameraCalib, read_osv_calibration
from .project import build_map
from .stitcher import blend, seam_weights, warp

logger = logging.getLogger(__name__)

# Camera 1's extrinsics need a small empirical correction; found by maximising
# inter-camera agreement in the overlap band (gains ~0.035 NCC).
CAM1_TWEAK = (-1.4, -1.4, 1.4)

# Lens rim is optically worst and most parallax-sensitive. Cropping below the
# nominal ~210 deg FOV measurably improves seam agreement (0.53 -> 0.64).
DEFAULT_FOV_DEG = 196.0


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


def levelling_rotation(gravity: np.ndarray) -> np.ndarray:
    """World->rig rotation placing the horizon level, from a gravity sample.

    Yaw is left free: gravity constrains only two of three degrees of freedom,
    and there is no physically preferred heading.
    """
    g = np.asarray(gravity, dtype=np.float64)
    n = np.linalg.norm(g)
    if n < 1e-6:
        return np.eye(3)
    return align_vectors([0.0, -1.0, 0.0], g / n)


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
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: str | Path, width: int = 7680, height: int = 3840,
                  fov_deg: float = DEFAULT_FOV_DEG,
                  max_frames: int | None = None) -> "Stitcher360":
        calibs = read_osv_calibration(path)
        meta = read_metadata(path, max_frames=max_frames)
        meta.gravity = smooth_gravity(meta.gravity)
        logger.info("%s: %s, serial %s, %d orientation samples",
                    Path(path).name, meta.model, meta.serial, len(meta.gravity))
        return cls(calibs, meta, width, height, fov_deg)

    def maps_for(self, frame: int) -> tuple[list[CameraMap], np.ndarray]:
        """Remap tables and seam weights for a frame's orientation.

        Cached per distinct rotation; building 7680x3840 maps is expensive.
        """
        if frame < len(self.meta.gravity):
            R = levelling_rotation(self.meta.gravity[frame])
        else:
            R = np.eye(3)
        key = np.round(R, 3).tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        common = dict(out_w=self.width, out_h=self.height, source="ypr",
                      max_fov_deg=self.fov_deg, feather_deg=4.0)
        m0 = build_map(self.calibs[0], R_extra=R, **common)
        m1 = build_map(self.calibs[1], R_extra=rot_matrix(*CAM1_TWEAK) @ R, **common)
        maps = [m0, m1]
        weights = seam_weights(maps, band_px=0.038 * self.width)
        if len(self._cache) > 4:
            self._cache.clear()
        self._cache[key] = (maps, weights)
        return maps, weights

    def stitch(self, fisheye0: np.ndarray, fisheye1: np.ndarray,
               frame: int = 0) -> np.ndarray:
        maps, weights = self.maps_for(frame)
        return blend([warp(fisheye0, maps[0]), warp(fisheye1, maps[1])],
                     maps, weights=weights)

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


__all__ = ["Stitcher360", "levelling_rotation", "align_vectors", "smooth_gravity"]
