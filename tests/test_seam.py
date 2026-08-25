"""Seam-repair estimator tests - pure numpy, no sample data or GPU.

Covers the three shared estimators in stitcher.py: the dynamic-seam path
(seam_path), its parallax gate (seam_gate), and the photometric mismatch
estimator (seam_luma_offsets). Each test encodes a property that was earned
from a real failure (see docs/seam.md).
"""

import unittest

import numpy as np

from dji_osv_converter.stitcher import seam_gate, seam_luma_offsets, seam_path


class TestSeamPath(unittest.TestCase):
    def test_path_follows_low_cost_valley(self):
        h, w = 60, 400
        cost = np.full((h, w), 100.0, np.float32)
        valley = (20 + 10 * np.sin(np.linspace(0, 3 * np.pi, w))).astype(int)
        cost[valley, np.arange(w)] = 0.0
        # widen the valley so the |dy|<=1 DP can track it after decimation
        for d in (-2, -1, 1, 2):
            cost[np.clip(valley + d, 0, h - 1), np.arange(w)] = 1.0
        seam = seam_path(cost, out_w=800)
        self.assertEqual(seam.shape, (800,))
        target = np.interp(np.arange(800), np.linspace(0, 799, w), valley)
        # smoothed path stays near the valley everywhere
        self.assertLess(np.abs(seam - target).max(), 6.0)

    def test_masked_region_is_avoided(self):
        h, w = 40, 200
        cost = np.full((h, w), 1e6, np.float32)   # outside-band sentinel
        cost[10:30] = 50.0                         # the real band
        cost[18] = 0.0
        seam = seam_path(cost, out_w=200)
        self.assertTrue((seam > 9).all() and (seam < 31).all())


class TestSeamGate(unittest.TestCase):
    def test_mean_statistic_ignores_single_pixel_spikes(self):
        # A max-based gate saturated on single noisy pixels (measured: 92% of
        # columns falsely flagged). One hot pixel per column must NOT open
        # the gate when the column mean stays low.
        h, w = 100, 512
        diff = np.full((h, w), 2.0, np.float32)
        diff[7, :] = 200.0                         # one outlier per column
        inner = np.ones((h, w), bool)
        g = seam_gate(diff, inner, out_w=1024)
        self.assertLess(g.max(), 0.1)

    def test_real_parallax_region_opens_the_gate(self):
        h, w = 100, 512
        diff = np.full((h, w), 2.0, np.float32)
        diff[:, 200:260] = 40.0                    # coherent disagreement
        inner = np.ones((h, w), bool)
        g = seam_gate(diff, inner, out_w=1024)
        self.assertGreater(g[400:520].max(), 0.9)  # open at the hot region
        self.assertLess(g[:200].max(), 0.2)        # closed far away


class TestSeamLumaOffsets(unittest.TestCase):
    def _recover(self, A, B, inner):
        lut, spatial = seam_luma_offsets(A, B, inner, lo=64.0, hi=940.0,
                                         n_lut=1024, sigma=30.0, agree_t=32.0)
        corr = lut[np.clip(A.astype(np.int64), 0, 1023)] + spatial
        return lut, spatial, corr

    def test_matched_lenses_are_a_noop(self):
        rng = np.random.default_rng(0)
        A = rng.uniform(100, 900, (120, 480)).astype(np.float32)
        B = A + rng.normal(0, 1.5, A.shape).astype(np.float32)
        _, _, corr = self._recover(A, B, np.ones(A.shape, bool))
        self.assertLess(np.abs(corr).max(), 3.0)

    def test_luma_dependent_offset_recovered(self):
        # The shed signature: a shadow-lift fading with luma, with the DARK
        # CONTENT SPATIALLY COHERENT (a dark ceiling region) - that is what
        # lets the pass-2 field catch the mismatch that exceeds pass 1's
        # tight agree gate (the selection bias - docs/seam.md). A random
        # dark/bright interleave is NOT recoverable by design: the field is
        # spatial, and the LUT only learns what the tight gate can see.
        rng = np.random.default_rng(1)
        A = np.empty((160, 480), np.float32)
        A[:, :240] = rng.uniform(80, 220, (160, 240))    # coherent dark half
        A[:, 240:] = rng.uniform(400, 900, (160, 240))   # bright half
        step = np.clip(60.0 - 0.08 * (A - 64.0), 0.0, None)   # up to 60 in shadows
        B = A + step + rng.normal(0, 1.0, A.shape).astype(np.float32)
        _, _, corr = self._recover(A, B, np.ones(A.shape, bool))
        resid = (B - A) - corr
        # away from the dark/bright boundary the field has converged
        dark, bright = A[:, :200] < 300, A[:, 280:] >= 300
        self.assertLess(np.abs(np.median(resid[:, :200][dark])), 8.0)
        self.assertLess(np.abs(np.median(resid[:, 280:][bright])), 4.0)

    def test_spatially_varying_offset_recovered(self):
        rng = np.random.default_rng(2)
        A = rng.uniform(200, 700, (160, 480)).astype(np.float32)
        field = 20.0 * np.sin(np.linspace(0, 2 * np.pi, 480))[None, :]
        B = A + field + rng.normal(0, 1.0, A.shape).astype(np.float32)
        _, spatial, corr = self._recover(A, B, np.ones(A.shape, bool))
        self.assertEqual(spatial.shape, A.shape)
        resid = (B - A) - corr
        self.assertLess(np.abs(resid).mean(), 4.0)


if __name__ == "__main__":
    unittest.main()
