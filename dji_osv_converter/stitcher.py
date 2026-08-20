"""Dual-fisheye -> equirectangular stitching.

Pipeline per frame:
    1. remap each fisheye into equirectangular space (DJI's tables)
    2. blend the two using DJI's feathered masks as weights

Orientation correction is applied by rotating the sampling directions, so it
costs nothing extra at runtime - it is folded into the remap tables once.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .calibration import CameraMap

logger = logging.getLogger(__name__)


def warp(frame: np.ndarray, cam: CameraMap) -> np.ndarray:
    """Project one fisheye frame into equirectangular space."""
    return cv2.remap(frame, cam.map_x, cam.map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def _wrapped_distance(mask: np.ndarray) -> np.ndarray:
    """Distance (px) from each covered pixel to the edge of its coverage region.

    Computed on a horizontally tiled copy so the equirectangular seam at
    longitude +/-180 does not read as a coverage boundary.
    """
    h, w = mask.shape
    tiled = np.concatenate([mask, mask, mask], axis=1).astype(np.uint8)
    # border must be "outside" vertically only; pad top/bottom with zeros
    padded = cv2.copyMakeBorder(tiled, 1, 1, 0, 0, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, :]
    return dist[:, w:2 * w]


def seam_weights(cams: list[CameraMap], band_px: float = 292.0) -> np.ndarray:
    """Feathered blend weight for camera 0, in [0,1]; camera 1 gets 1 - w.

    DJI's masks are *coverage* masks, not blend weights - inside the overlap
    both are fully on. Averaging them directly double-exposes the overlap
    (visible ghosting). Instead we place the seam on the locus equidistant
    from both coverage boundaries and cross-fade over `band_px` around it,
    matching the 292 px blend band DJI itself uses (7680 * 0.038).
    """
    c0 = (cams[0].weight > 0.5) & cams[0].valid
    c1 = (cams[1].weight > 0.5) & cams[1].valid

    d0 = _wrapped_distance(c0)
    d1 = _wrapped_distance(c1)

    # >0 means deeper inside camera 0's coverage than camera 1's
    signed = d0 - d1
    w = np.clip(0.5 + signed / max(band_px, 1e-6), 0.0, 1.0).astype(np.float32)

    # Outside one camera's coverage the other must take the pixel outright.
    w[~c1] = 1.0
    w[~c0] = 0.0
    both_missing = ~c0 & ~c1
    if both_missing.any():
        # fall back to any geometrically valid sample rather than leaving a hole
        w[both_missing & cams[0].valid] = 1.0
        w[both_missing & ~cams[0].valid] = 0.0
    return w


def blend(warped: list[np.ndarray], cams: list[CameraMap],
          band_px: float = 292.0, weights: np.ndarray | None = None) -> np.ndarray:
    """Cross-fade two projected frames across a narrow seam band."""
    w = seam_weights(cams, band_px) if weights is None else weights
    w3 = w[..., None]
    out = warped[0].astype(np.float32) * w3 + warped[1].astype(np.float32) * (1.0 - w3)
    return np.clip(out, 0, 255).astype(np.uint8)


def stitch(frame0: np.ndarray, frame1: np.ndarray,
           cams: list[CameraMap], band_px: float = 292.0,
           weights: np.ndarray | None = None) -> np.ndarray:
    """Stitch a dual-fisheye pair into one equirectangular frame.

    `weights` may be precomputed once with seam_weights() and reused across
    frames - it depends only on calibration, not on image content.
    """
    warped = [warp(frame0, cams[0]), warp(frame1, cams[1])]
    return blend(warped, cams, band_px, weights)


def coverage_report(cams: list[CameraMap]) -> dict:
    """Diagnostics for verifying a calibration set without rendering."""
    v0, v1 = cams[0].valid, cams[1].valid
    tot = cams[0].weight + cams[1].weight
    return {
        "cam0_valid_pct": 100 * float(v0.mean()),
        "cam1_valid_pct": 100 * float(v1.mean()),
        "union_pct": 100 * float((v0 | v1).mean()),
        "geometric_overlap_pct": 100 * float((v0 & v1).mean()),
        "blend_overlap_pct": 100 * float(
            ((cams[0].weight > 0.01) & (cams[1].weight > 0.01)).mean()),
        "uncovered_pct": 100 * float((tot <= 1e-6).mean()),
    }


__all__ = ["warp", "blend", "stitch", "coverage_report"]
