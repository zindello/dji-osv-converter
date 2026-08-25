"""Unit tests for the levelling math and constants - no sample footage needed.

These encode the properties the modes were validated against (see docs/):
run with ``python -m unittest discover tests`` or ``pytest``.
"""

import unittest

import numpy as np

from dji_osv_converter.constants import (
    C_UP_FORMAT, K_PLAIN, WORLD_FWD, WORLD_UP,
)
from dji_osv_converter.pipeline import (
    Stitcher360, horizon_lock_rotation, smoothed_rotation_track,
)


def _rand_rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class TestConstants(unittest.TestCase):
    def test_k_plain_is_rotation(self):
        self.assertTrue(np.allclose(K_PLAIN @ K_PLAIN.T, np.eye(3), atol=1e-9))
        self.assertAlmostEqual(np.linalg.det(K_PLAIN), 1.0, places=9)

    def test_c_up_format_is_orthogonal_reflection(self):
        # det = -1 BY DESIGN: the equirect z-convention is left-handed vs the
        # physical IMU frame (see constants.py provenance notes).
        self.assertTrue(np.allclose(C_UP_FORMAT @ C_UP_FORMAT.T, np.eye(3), atol=1e-4))
        self.assertAlmostEqual(np.linalg.det(C_UP_FORMAT), -1.0, places=3)

    def test_world_axes_orthonormal(self):
        self.assertAlmostEqual(float(WORLD_UP @ WORLD_FWD), 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(WORLD_UP)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(WORLD_FWD)), 1.0)


class TestHorizonLock(unittest.TestCase):
    """The defining properties of the all-axis lock (validated vs DJI)."""

    def test_proper_rotation(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            R = horizon_lock_rotation(_rand_rot(rng), C_UP_FORMAT)
            self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-9))
            self.assertGreater(np.linalg.det(R), 0.999)

    def test_world_up_lands_on_output_pole(self):
        # The stitch samples output dir d at R @ d and the pole is d=[0,1,0],
        # so R's middle column must be world-up in the rig frame.
        rng = np.random.default_rng(1)
        for _ in range(50):
            raw = _rand_rot(rng)
            up_rig = C_UP_FORMAT @ (raw.T @ WORLD_UP)
            up_rig /= np.linalg.norm(up_rig)
            R = horizon_lock_rotation(raw, C_UP_FORMAT)
            self.assertTrue(np.allclose(R @ np.array([0.0, 1.0, 0.0]), up_rig,
                                        atol=1e-9))

    def test_lock_is_camera_invariant(self):
        # World -> output must not depend on the camera orientation at all:
        # O = R.T @ C_UP_FORMAT @ raw.T is one constant for every raw.
        rng = np.random.default_rng(2)
        raws = [_rand_rot(rng) for _ in range(20)]
        Os = [horizon_lock_rotation(r, C_UP_FORMAT).T @ C_UP_FORMAT @ r.T for r in raws]
        for O in Os[1:]:
            self.assertTrue(np.allclose(O, Os[0], atol=1e-6))


class TestRockSteady(unittest.TestCase):
    def test_raw_orientation_cancels(self):
        # The composition C @ raw.T @ smooth @ C.T @ K_PLAIN must reduce world->
        # output to a function of `smooth` alone - that cancellation is the fix
        # for the tumbling-axis jitter bug. Equivalently: with smooth == raw the
        # rotation is exactly K_PLAIN (a steady camera renders like plain).
        rng = np.random.default_rng(3)
        for _ in range(20):
            raw = _rand_rot(rng)
            R = C_UP_FORMAT @ raw.T @ raw @ C_UP_FORMAT.T @ K_PLAIN
            self.assertTrue(np.allclose(R, K_PLAIN, atol=1e-9))

    def test_smoothed_track_is_orthonormal_and_smoother(self):
        rng = np.random.default_rng(4)
        # synthetic slow pan + per-frame jitter
        n = 120
        quats = []
        for f in range(n):
            a = 0.01 * f + rng.normal(scale=0.01)
            quats.append([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
        raw, smooth = smoothed_rotation_track(np.array(quats), sigma=8.0)

        def jitter(track):
            d = [track[f] @ track[f - 1].T for f in range(1, n)]
            ang = [np.arccos(np.clip((np.trace(m) - 1) / 2, -1, 1)) for m in d]
            return float(np.std(np.diff(ang)))

        for m in smooth[::10]:
            self.assertTrue(np.allclose(m @ m.T, np.eye(3), atol=1e-9))
        self.assertLess(jitter(smooth), jitter(raw) * 0.5)


class TestLevellingModes(unittest.TestCase):
    def test_mode_dispatch(self):
        # A Stitcher360 with synthetic metadata dispatches every mode to a
        # proper rotation without touching any file.
        from dji_osv_converter.metadata import OSVMeta
        rng = np.random.default_rng(5)
        quats = []
        for _ in range(10):
            q = rng.normal(size=4)
            quats.append(q / np.linalg.norm(q))
        meta = OSVMeta(None, None, np.array(quats), np.zeros((10, 3)))
        for mode in ("plain", "rocksteady", "rocksteady_horizon", "none"):
            st = Stitcher360(calibs=[], meta=meta, levelling=mode)
            R = st.levelling_for(3)
            self.assertEqual(R.shape, (3, 3))
            self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-6), mode)
            self.assertGreater(np.linalg.det(R), 0.999, mode)


if __name__ == "__main__":
    unittest.main()
