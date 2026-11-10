"""Setup amortisation: geometry/GPU caches keyed by the calibration hash."""

import copy
import unittest.mock
import os
import tempfile
import unittest

import numpy as np

try:
    import torch
except Exception:                     # pragma: no cover
    torch = None

from dji_osv_converter import pipeline, video
from dji_osv_converter.metadata import OSVMeta
from tests import _rig


def _stitcher360(calibs):
    meta = OSVMeta(None, None, np.zeros((0, 4)), np.zeros((0, 3)))
    return pipeline.Stitcher360(calibs=calibs, meta=meta, width=_rig.W, height=_rig.H)


class TestGeometryCache(unittest.TestCase):
    def setUp(self):
        pipeline.clear_geometry_cache()

    def test_key_tracks_calibration_and_output_size(self):
        c = _rig.calibs()
        k0 = pipeline.geometry_key(c, _rig.W, _rig.H, 193.68, 0.038)
        self.assertEqual(k0, pipeline.geometry_key(copy.deepcopy(c), _rig.W, _rig.H, 193.68, 0.038))
        c2 = copy.deepcopy(c)
        c2[1].fx *= 1.001
        self.assertNotEqual(k0, pipeline.geometry_key(c2, _rig.W, _rig.H, 193.68, 0.038))
        self.assertNotEqual(k0, pipeline.geometry_key(c, _rig.W * 2, _rig.H * 2, 193.68, 0.038))

    def test_second_file_from_the_same_unit_reuses_the_maps(self):
        a = _stitcher360(_rig.calibs())
        b = _stitcher360(copy.deepcopy(_rig.calibs()))     # "another file", same unit
        maps_a, w_a = a.maps_for()
        maps_b, w_b = b.maps_for()
        self.assertIs(maps_a, maps_b)
        self.assertIs(w_a, w_b)
        pipeline.clear_geometry_cache()
        c = _stitcher360(_rig.calibs())
        self.assertIsNot(c.maps_for()[0], maps_a)

    def test_cache_is_bounded(self):
        for i in range(pipeline.RIG_CACHE_ENTRIES + 2):
            cal = _rig.calibs()
            cal[0].fx += i
            _stitcher360(cal).maps_for()
        self.assertLessEqual(len(pipeline._RIG_CACHE), pipeline.RIG_CACHE_ENTRIES)


@unittest.skipIf(torch is None, "torch not installed")
class TestGpuStitcherCache(unittest.TestCase):
    def setUp(self):
        video.clear_gpu_cache()
        pipeline.clear_geometry_cache()

    def _get(self, st, **kw):
        maps, _ = st.maps_for()
        args = dict(lsc=False, use_rs=False, gpu_fmt="yuv420p10le", yuv_range="limited",
                    single_pass=False, lens_interp="bilinear", reuse=True)
        args.update(kw)
        with unittest.mock.patch.object(video.stitch_gpu, "GpuStitcher",
                                        lambda *a, **k: object()):
            return video._gpu_stitcher(st, maps, _rig.W, _rig.H, **args)

    def test_reused_for_the_same_geometry_and_options(self):
        import unittest.mock
        a = _stitcher360(_rig.calibs())
        b = _stitcher360(copy.deepcopy(_rig.calibs()))
        g1 = self._get(a)
        self.assertIs(self._get(b), g1)
        self.assertIsNot(self._get(b, lens_interp="bicubic"), g1)      # options differ
        self.assertIsNot(self._get(b, reuse=False), self._get(b, reuse=False))
        video.clear_gpu_cache()
        self.assertIsNot(self._get(a), g1)


class TestCompileCacheDir(unittest.TestCase):
    def test_env_is_set_and_dirs_created(self):
        td = tempfile.mkdtemp()
        old = {k: os.environ.get(k) for k in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")}
        try:
            video.set_compile_cache_dir(td)
            self.assertEqual(os.environ["TORCHINDUCTOR_CACHE_DIR"], os.path.join(td, "inductor"))
            self.assertEqual(os.environ["TRITON_CACHE_DIR"], os.path.join(td, "triton"))
            self.assertTrue(os.path.isdir(os.path.join(td, "triton")))
            video.set_compile_cache_dir(None)                  # never overrides
            self.assertEqual(os.environ["TORCHINDUCTOR_CACHE_DIR"], os.path.join(td, "inductor"))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_converter_wraps_convert_with_defaults(self):
        import unittest.mock
        calls = []
        with unittest.mock.patch.object(video, "convert", lambda i, o, **kw: calls.append((i, o, kw)) or o):
            conv = video.Converter(levelling="rocksteady_horizon", count=3)
            conv.convert("a.OSV", "a.mp4")
            conv.convert("b.OSV", "b.mp4", count=7)
        self.assertEqual(calls[0][2]["levelling"], "rocksteady_horizon")
        self.assertEqual(calls[0][2]["count"], 3)
        self.assertEqual(calls[1][2]["count"], 7)
        self.assertTrue(calls[0][2]["reuse"])


if __name__ == "__main__":
    unittest.main()
