"""Equirectangular orientation: rotating the sphere.

The remap tables produce a panorama in the camera rig's own frame, where the
lens axis runs vertically (the stitch seam appears horizontal). Viewers expect
a level horizon with the seam vertical, so the sphere must be rotated.

Rotation is applied by resampling the equirectangular image, and can be folded
directly into the remap tables so it costs nothing per frame.
"""

from __future__ import annotations

import numpy as np
import cv2


def rot_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Rotation matrix from intrinsic yaw(Y), pitch(X), roll(Z), in degrees."""
    y, p, r = np.radians([yaw, pitch, roll])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


_DIR_CACHE: dict[tuple[int, int], np.ndarray] = {}


def directions(h: int, w: int) -> np.ndarray:
    """Unit direction vector for every pixel of an h x w equirect. -> (h,w,3)

    Cached per size: the grid is constant, and rebuilding it at 8K (0.5s, 700MB)
    every frame was a large share of the levelling cost. The cached array is
    read-only; callers must not mutate it in place.
    """
    hit = _DIR_CACHE.get((h, w))
    if hit is not None:
        return hit
    lon = (np.arange(w, dtype=np.float64) / w - 0.5) * 2 * np.pi
    lat = (0.5 - np.arange(h, dtype=np.float64) / h) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    cl = np.cos(lat)
    d = np.stack([cl * np.sin(lon), np.sin(lat), -cl * np.cos(lon)], axis=-1)
    d.flags.writeable = False
    _DIR_CACHE[(h, w)] = d
    return d


def dirs_to_uv(d: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Direction vectors -> equirect pixel coordinates."""
    lon = np.arctan2(d[..., 0], -d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    u = (lon / (2 * np.pi) + 0.5) * w
    v = (0.5 - lat / np.pi) * h
    return u.astype(np.float32), v.astype(np.float32)


def rotate_equirect(img: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate an equirectangular image by R (resampling)."""
    h, w = img.shape[:2]
    d = directions(h, w) @ R.T
    u, v = dirs_to_uv(d, w, h)
    return cv2.remap(img, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def apply_to_maps(cams, R):
    """Fold a rotation into calibration remap tables (zero per-frame cost).

    For each output pixel we look up which *source* direction it should show
    after rotation, then copy that pixel's fisheye coordinates from the
    unrotated table.
    """
    from .calibration import CameraMap

    out = []
    for c in cams:
        h, w = c.map_x.shape
        d = directions(h, w) @ R.T
        u, v = dirs_to_uv(d, w, h)
        mx = cv2.remap(c.map_x, u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        my = cv2.remap(c.map_y, u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        wt = cv2.remap(c.weight, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        vl = cv2.remap(c.valid.astype(np.uint8), u, v, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_WRAP).astype(bool)
        out.append(CameraMap(mx, my, wt, vl))
    return out


__all__ = ["rot_matrix", "rotate_equirect", "apply_to_maps", "directions", "dirs_to_uv"]
