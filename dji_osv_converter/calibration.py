"""Loading of DJI's shipped stitching calibration data.

DJI Studio ships precomputed fisheye->equirectangular remap lookup tables and
per-camera blend masks inside its application bundle. This module reads them.

Nothing here calls DJI code; it only reads data files.

remap_{0,1}.bin format (determined empirically, see docs/ROADMAP.md):
    shape  (1920, 3840, 2)   rows, cols, (x, y)
    dtype  int32, fixed point with scale 1/128
    value  source pixel coordinate in that camera's 3840x3840 fisheye frame
    invalid entries fall outside [0, 3840) and are masked out
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BUNDLE = Path(
    "/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework"
    "/Versions/A/Resources/Stitcher-Mac.bundle"
)

REMAP_SHAPE = (1920, 3840, 2)   # equirect rows, cols, (x,y)
REMAP_SCALE = 128.0             # fixed point divisor
FISHEYE_SIZE = 3840             # source frame is FISHEYE_SIZE x FISHEYE_SIZE


class CalibrationError(RuntimeError):
    pass


@dataclass
class CameraMap:
    """Remap table and blend weight for one camera, in equirectangular space."""

    map_x: np.ndarray   # (H, W) float32 - source x, or -1 where invalid
    map_y: np.ndarray   # (H, W) float32 - source y, or -1 where invalid
    weight: np.ndarray  # (H, W) float32 in [0,1] - blend weight
    valid: np.ndarray   # (H, W) bool

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the equirectangular grid."""
        return self.map_x.shape[1], self.map_x.shape[0]


def _load_remap(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.fromfile(path, dtype=np.int32)
    expected = REMAP_SHAPE[0] * REMAP_SHAPE[1] * REMAP_SHAPE[2]
    if raw.size != expected:
        raise CalibrationError(
            f"{path.name}: expected {expected} int32 values, got {raw.size}. "
            "Remap format may differ in this DJI Studio version."
        )
    m = raw.reshape(REMAP_SHAPE).astype(np.float32) / REMAP_SCALE
    x, y = m[..., 0].copy(), m[..., 1].copy()

    valid = (x >= 0) & (x < FISHEYE_SIZE) & (y >= 0) & (y < FISHEYE_SIZE)
    # cv2.remap needs finite values everywhere; -1 samples the border (black)
    x[~valid] = -1.0
    y[~valid] = -1.0
    return x, y, valid


def _load_mask(path: Path, map_x, map_y, valid) -> np.ndarray:
    """Load a fisheye-space blend mask and project it into equirect space.

    The mask is defined over the fisheye frame, so it must be warped through the
    same remap as the image itself to become an equirectangular blend weight.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise CalibrationError(f"could not read mask {path}")
    if mask.shape[0] != FISHEYE_SIZE:
        mask = cv2.resize(mask, (FISHEYE_SIZE, FISHEYE_SIZE), interpolation=cv2.INTER_LINEAR)

    w = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    w = w.astype(np.float32) / 255.0
    w[~valid] = 0.0
    return w


def load_calibration(bundle: Path | str = DEFAULT_BUNDLE,
                     case: str = "case_00",
                     mask_names: tuple[str, str] = ("mask_00.png", "mask_01.png"),
                     ) -> list[CameraMap]:
    """Load remap tables and blend masks for both cameras.

    Returns one CameraMap per camera, in equirectangular space.
    """
    root = Path(bundle) / case
    if not root.is_dir():
        raise CalibrationError(
            f"calibration bundle not found: {root}\nIs DJI Studio installed?"
        )

    cams: list[CameraMap] = []
    for i, mask_name in enumerate(mask_names):
        remap_path = root / f"remap_{i}.bin"
        if not remap_path.exists():
            raise CalibrationError(f"missing {remap_path}")
        mx, my, valid = _load_remap(remap_path)
        weight = _load_mask(root / mask_name, mx, my, valid)
        cams.append(CameraMap(mx, my, weight, valid))
        logger.info("camera %d: %.1f%% valid, mean blend weight %.3f",
                    i, 100 * valid.mean(), weight[valid].mean())

    union = cams[0].valid | cams[1].valid
    if union.mean() < 0.999:
        logger.warning("equirect coverage only %.2f%% - expect gaps", 100 * union.mean())
    return cams


__all__ = ["CameraMap", "CalibrationError", "load_calibration", "DEFAULT_BUNDLE"]
