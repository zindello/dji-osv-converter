"""Single-pass resample vs an independent numpy reference (synthetic rig).

The reference composes the levelling rotation into the lens maps in f64
(``build_map(R_extra=R)``) and samples each fisheye once with cv2.remap;
the single-pass GPU path must reproduce it to a fraction of a 10-bit LSB,
and must beat the two-pass path, whose second resample carried a (W-1)/W
stretch and a second interpolation.
"""

import unittest

import cv2
import numpy as np

try:
    import torch
except Exception:                     # pragma: no cover
    torch = None

from dji_osv_converter.constants import DEFAULT_FOV_DEG, FEATHER_FRACTION, SEAM_OVERLAP_RATIO
from dji_osv_converter.orientation import directions, dirs_to_uv, rot_matrix
from dji_osv_converter.project import build_map
from dji_osv_converter.stitch_gpu import GpuStitcher
from dji_osv_converter.stitcher import seam_weights
from tests import _rig

W, H, FW, FH = _rig.W, _rig.H, _rig.FW, _rig.FH


def _stitcher(**kw):
    return GpuStitcher(_rig.maps(), band_px=_rig.band_px(), width=W, height=H,
                       device="cpu", fisheye_size=(FW, FH), compile=False, **kw)


def _luma(packed):
    return packed[:FH].astype(np.float32)


def _reference(f0, f1, R, interp=cv2.INTER_LINEAR):
    """Blend of the two fisheyes sampled ONCE at the levelled directions."""
    cals = _rig.calibs()
    kw = dict(out_w=W, out_h=H, source="quat", order="wxyz", max_fov_deg=DEFAULT_FOV_DEG,
              feather_deg=SEAM_OVERLAP_RATIO * 360.0 * FEATHER_FRACTION)
    mr = [build_map(c, R_extra=R, **kw) for c in cals]
    ys = [cv2.remap(_luma(f), m.map_x, m.map_y, interp, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0) for f, m in zip((f0, f1), mr)]
    w = seam_weights(_rig.maps(), _rig.band_px()).astype(np.float32)
    d = directions(H, W) @ R.T
    u, v = dirs_to_uv(d, W, H)
    w_rot = cv2.remap(w, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return w_rot * ys[0] + (1.0 - w_rot) * ys[1], w_rot


def _chroma_reference(f0, f1, R):
    """U plane sampled once at correctly sited chroma coordinates."""
    cals = _rig.calibs()
    kw = dict(out_w=W, out_h=H, source="quat", order="wxyz", max_fov_deg=DEFAULT_FOV_DEG,
              feather_deg=SEAM_OVERLAP_RATIO * 360.0 * FEATHER_FRACTION)
    us = []
    for f, c in zip((f0, f1), cals):
        m = build_map(c, R_extra=R, **kw)
        U = f[FH:FH + FH // 4].reshape(FH // 2, FW // 2).astype(np.float32)
        mx, my = m.map_x[::2, ::2], m.map_y[::2, ::2]
        us.append(cv2.remap(U, mx * 0.5, (my - 0.5) * 0.5, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    w = seam_weights(_rig.maps(), _rig.band_px()).astype(np.float32)
    d = directions(H, W) @ R.T
    u, v = dirs_to_uv(d, W, H)
    w_rot = cv2.remap(w, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    wh = w_rot.reshape(H // 2, 2, W // 2, 2).mean((1, 3))
    return wh * us[0] + (1.0 - wh) * us[1]


def _lsb(Y_packed):
    return Y_packed.astype(np.float64) / 64.0


@unittest.skipIf(torch is None, "torch not installed")
class TestSinglePass(unittest.TestCase):
    def setUp(self):
        self.f0, self.f1 = _rig.fisheye_pair(seed=3)

    def test_identity_matches_two_pass(self):
        two = _stitcher(single_pass=False)
        one = _stitcher(single_pass=True)
        Y2, UV2 = two.stitch_yuv(self.f0, self.f1, np.eye(3))
        Y1, UV1 = one.stitch_yuv(self.f0, self.f1, np.eye(3))
        d = np.abs(_lsb(Y1) - _lsb(Y2))
        self.assertLessEqual(d.max(), 1.0)
        self.assertGreater((d == 0).mean(), 0.99)
        # chroma: the single-pass sites 4:2:0 chroma at luma (2i, 2j+0.5) per
        # the MPEG-2 "left" convention; the two-pass reused luma-normalised
        # coordinates on the half plane (a <=0.5 chroma-px radial scale
        # error at the rim). Judge chroma against a correctly sited
        # reference instead of against the two-pass.
        cref = _chroma_reference(self.f0, self.f1, np.eye(3))
        dc = np.abs(_lsb(UV1[:, 0::2]) - cref)
        m = (cref > 0) & (np.arange(H // 2)[:, None] > 0) & (np.arange(H // 2)[:, None] < H // 2 - 1)
        self.assertLess(dc[m].mean(), 0.75)
        self.assertLess(np.percentile(dc[m], 99), 3.0)

    def test_rotated_matches_reference_and_beats_two_pass(self):
        R = rot_matrix(37.0, 21.0, -13.0)
        ref, w_rot = _reference(self.f0, self.f1, R)
        one = _stitcher(single_pass=True)
        two = _stitcher(single_pass=False)
        Y1, _ = one.stitch_yuv(self.f0, self.f1, R)
        Y2, _ = two.stitch_yuv(self.f0, self.f1, R)
        # judge away from the coverage edge (reference's -1 fill vs ours) and
        # off the pole rows
        m = (ref > 70) & (np.arange(H)[:, None] > 1) & (np.arange(H)[:, None] < H - 2)
        e1 = np.abs(_lsb(Y1) - ref)[m]
        e2 = np.abs(_lsb(Y2) - ref)[m]
        self.assertLess(e1.mean(), 0.5, f"single-pass mean error {e1.mean():.3f} LSB")
        self.assertLess(np.percentile(e1, 99.5), 2.0)
        self.assertLess(e1.mean(), e2.mean() * 0.5,
                        f"single {e1.mean():.3f} vs two-pass {e2.mean():.3f}")

    def test_wrap_column_is_real_not_clamped(self):
        # A pure yaw by one output column must shift the panorama by exactly
        # one column (the two-pass path clamped the wrap column and stretched
        # the pano by (W-1)/W, so the last column was wrong).
        R = rot_matrix(360.0 / W, 0.0, 0.0)
        one = _stitcher(single_pass=True)
        Y0, _ = one.stitch_yuv(self.f0, self.f1, np.eye(3))
        Y1, _ = one.stitch_yuv(self.f0, self.f1, R)
        rolled = np.roll(Y0, -1, axis=1) if np.abs(_lsb(np.roll(Y0, -1, axis=1)) - _lsb(Y1)).mean() < \
            np.abs(_lsb(np.roll(Y0, 1, axis=1)) - _lsb(Y1)).mean() else np.roll(Y0, 1, axis=1)
        d = np.abs(_lsb(rolled) - _lsb(Y1))[2:-2]
        self.assertLessEqual(np.percentile(d, 99.9), 1.0)
        self.assertLessEqual(d.max(), 2.0)

    def test_seam_stack_is_band_limited(self):
        one = _stitcher(single_pass=True)
        Y_off, UV_off = one.stitch_yuv(self.f0, self.f1, np.eye(3), seam_flow=False)
        Y_on, UV_on = one.stitch_yuv(self.f0, self.f1, np.eye(3), seam_flow=True)
        self.assertTrue(np.isfinite(_lsb(Y_on)).all())
        self.assertGreaterEqual(int(Y_on.min()), 64 * 64)
        self.assertLessEqual(int(Y_on.max()), 940 * 64)
        r0, r1 = one._sf_rows
        outside = np.ones(H, bool)
        outside[max(r0 - 1, 0):min(r1 + 1, H)] = False
        np.testing.assert_array_equal(Y_on[outside], Y_off[outside])
        self.assertTrue((Y_on[r0:r1] != Y_off[r0:r1]).any())   # it did engage

    def test_pole_row_is_extrapolated_not_clamped(self):
        one = _stitcher(single_pass=True)
        one._sp_init()
        for i, uv in enumerate(one._sp["uv"]):
            self.assertEqual(tuple(uv.shape), (1, 2, H + 1, W + 1))
            # wrap column
            np.testing.assert_array_equal(uv[0, :, :, W].numpy(), uv[0, :, :, 0].numpy())
            # the pole row is one direction: constant across the row where valid
            row = uv[0, :, H].numpy()
            valid = row[0] >= 0
            if valid.any():
                self.assertLess(row[0][valid].std(), 0.05)
                self.assertLess(row[1][valid].std(), 0.05)


if __name__ == "__main__":
    unittest.main()
