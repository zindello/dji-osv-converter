"""GpuStitcher properties on the torch-CPU device with the synthetic rig."""

import unittest

import numpy as np

try:
    import torch
except Exception:                     # pragma: no cover
    torch = None

from dji_osv_converter.stitch_gpu import GpuStitcher
from tests import _rig


@unittest.skipIf(torch is None, "torch not installed")
class TestSampleRange(unittest.TestCase):
    def _stitcher(self, yuv_range):
        return GpuStitcher(_rig.maps(), band_px=_rig.band_px(), width=_rig.W, height=_rig.H,
                           device="cpu", fisheye_size=(_rig.FW, _rig.FH),
                           yuv_range=yuv_range, compile=False)

    def test_limited_range_clamps_to_video_levels(self):
        g = self._stitcher("limited")
        flat = np.full((_rig.FH * 3 // 2, _rig.FW), 1000, np.uint16)
        flat[_rig.FH:] = 1000
        Y, UV = g.stitch_yuv(flat, flat, np.eye(3))
        self.assertEqual(int(Y.max()), 940 * 64)
        self.assertEqual(int(UV.max()), 960 * 64)

    def test_full_range_keeps_the_whole_code_space(self):
        g = self._stitcher("full")
        flat = np.full((_rig.FH * 3 // 2, _rig.FW), 1000, np.uint16)
        Y, UV = g.stitch_yuv(flat, flat, np.eye(3))
        self.assertEqual(int(Y.max()), 1000 * 64)
        self.assertEqual(int(UV.max()), 1000 * 64)
        zero = np.zeros((_rig.FH * 3 // 2, _rig.FW), np.uint16)
        Y, _ = g.stitch_yuv(zero, zero, np.eye(3))
        self.assertEqual(int(Y.min()), 0)

    def test_rejects_unknown_range(self):
        with self.assertRaises(ValueError):
            self._stitcher("video")


if __name__ == "__main__":
    unittest.main()
