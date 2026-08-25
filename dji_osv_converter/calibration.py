"""Shared calibration data structures.

Historically this module also loaded DJI Studio's shipped remap tables from the
application bundle; that path is gone - every map is now built from the OSV
file's own factory calibration (see ``osv_calibration`` and ``project``), which
measurably fits the recording camera better than DJI's generic reference sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class CalibrationError(RuntimeError):
    pass


@dataclass
class CameraMap:
    """Remap table and blend weight for one camera, in equirectangular space.

    ``map_x``/``map_y`` give, for every output (equirect) pixel, the source
    fisheye pixel to sample (-1 where the lens has no coverage). ``weight`` is
    the coverage taper used to place the seam; ``valid`` the coverage mask.
    """

    map_x: np.ndarray   # (H, W) float32 - source x, or -1 where invalid
    map_y: np.ndarray   # (H, W) float32 - source y, or -1 where invalid
    weight: np.ndarray  # (H, W) float32 in [0,1] - blend weight
    valid: np.ndarray   # (H, W) bool

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the equirectangular grid."""
        return self.map_x.shape[1], self.map_x.shape[0]


__all__ = ["CameraMap", "CalibrationError"]
