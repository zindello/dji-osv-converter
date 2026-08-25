"""Decode-format plumbing tests - no sample data, no GPU required.

Covers the pure logic added for the NVDEC paths: p010 plane views,
the planar->p010 fallback conversion, and frame-index math.
"""

import unittest

import av
import numpy as np

from dji_osv_converter.decode import _frame_index, _frame_p010, _plane_u16


def _planar_frame(fh, fw, seed=0):
    """A synthetic yuv420p10le VideoFrame plus its packed planar array."""
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 1024, size=(fh + fh // 2, fw), dtype=np.uint16)
    fr = av.VideoFrame(fw, fh, "yuv420p10le")
    planes = [packed[:fh],
              packed[fh:fh + fh // 4].reshape(fh // 2, fw // 2),
              packed[fh + fh // 4:].reshape(fh // 2, fw // 2)]
    for plane, data in zip(fr.planes, planes):
        buf = np.frombuffer(plane, np.uint16).reshape(
            plane.height, plane.line_size // 2)
        buf[:, :data.shape[1]] = data
    return fr, packed


class TestFramePlanes(unittest.TestCase):
    def test_plane_u16_full_and_strided(self):
        fr = av.VideoFrame(64, 32, "yuv420p10le")
        y = _plane_u16(fr.planes[0], 64)
        self.assertEqual(y.shape, (32, 64))
        # width narrower than the stored line exercises the stride slice
        y48 = _plane_u16(fr.planes[0], 48)
        self.assertEqual(y48.shape, (32, 48))

    def test_planar_fallback_matches_p010_layout(self):
        fh, fw = 32, 64
        fr, packed = _planar_frame(fh, fw)
        Y, UV = _frame_p010(fr)
        self.assertEqual(Y.shape, (fh, fw))
        self.assertEqual(UV.shape, (fh // 2, fw))
        # samples top-aligned: value << 6, exact
        np.testing.assert_array_equal(Y >> 6, packed[:fh])
        U = packed[fh:fh + fh // 4].reshape(fh // 2, fw // 2)
        V = packed[fh + fh // 4:].reshape(fh // 2, fw // 2)
        np.testing.assert_array_equal(UV[:, 0::2] >> 6, U)
        np.testing.assert_array_equal(UV[:, 1::2] >> 6, V)
        self.assertTrue(np.all((Y & 63) == 0))
        self.assertTrue(np.all((UV & 63) == 0))


class TestGpuEngines(unittest.TestCase):
    def test_known_cards_and_longest_match(self):
        from dji_osv_converter.constants import gpu_engines
        self.assertEqual(gpu_engines("Tesla V100-PCIE-16GB")["nvenc"], 3)
        self.assertEqual(gpu_engines("Tesla V100-PCIE-16GB")["nvdec"], 1)
        # "L40S" must match L40, not L4
        self.assertEqual(gpu_engines("NVIDIA L40S")["nvenc"], 3)
        self.assertEqual(gpu_engines("NVIDIA L4")["nvenc"], 2)

    def test_unknown_card_is_conservative(self):
        from dji_osv_converter.constants import GPU_ENGINES_DEFAULT, gpu_engines
        self.assertEqual(gpu_engines("NVIDIA FutureCard 9000"),
                         GPU_ENGINES_DEFAULT)
        self.assertEqual(gpu_engines(None), GPU_ENGINES_DEFAULT)
        self.assertEqual(GPU_ENGINES_DEFAULT["nvenc"], 1)

    def test_digit_boundary(self):
        from dji_osv_converter.constants import GPU_ENGINES_DEFAULT, gpu_engines
        # "RTX A1000" must NOT match the A100 row
        self.assertEqual(gpu_engines("NVIDIA RTX A1000"), GPU_ENGINES_DEFAULT)
        self.assertEqual(gpu_engines("NVIDIA A100-SXM4-40GB")["nvdec"], 5)


class TestFrameIndex(unittest.TestCase):
    def test_one_tick_per_frame_timeline(self):
        from fractions import Fraction
        tb, rate = Fraction(1001, 30000), 30000 / 1001
        for n in (0, 1, 29, 30, 1000, 65537):
            self.assertEqual(_frame_index(n, tb, rate), n)

    def test_rounding_is_stable_under_jitter(self):
        from fractions import Fraction
        tb, rate = Fraction(1, 90000), 30000 / 1001
        for n in range(0, 300, 7):
            pts = round(n / rate / tb)          # container-quantized pts
            self.assertEqual(_frame_index(pts, tb, rate), n)


if __name__ == "__main__":
    unittest.main()
