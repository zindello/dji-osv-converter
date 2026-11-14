"""The single-pass stitch with the seam stack on the device (seam_engine="gpu").

Synthetic rigs only, torch CPU device (and MPS/CUDA when present). The
reference is the CPU seam engine - the path that shipped before the device
engine - built on the same rig:

- with the seam stack off the two engines are the same resample and must
  agree bit for bit, also after a seam frame has rewritten the band rows;
- with the reference flow injected into the device engine, the folded
  application (band rows of the lens maps and of the weight/correction map
  rewritten per frame) must reproduce the reference's sparse application;
- the band morph's direction (``seam_morph``) is pinned on a rig whose two
  lenses see one world with a known displacement: ``"corrected"`` must pull
  the lenses together, ``"legacy"`` (the shipped default) pushes them apart.
"""

import unittest

import numpy as np

try:
    import torch
except Exception:                     # pragma: no cover
    torch = None

from dji_osv_converter.constants import SEAM_OVERLAP_RATIO
from dji_osv_converter.orientation import rot_matrix
from dji_osv_converter.stitch_gpu import GpuStitcher, default_seam_engine
from dji_osv_converter.stitcher import seam_flow_small
from tests import _rig

W, H, FW, FH = _rig.W, _rig.H, _rig.FW, _rig.FH


def _accel():
    if torch is None:
        return None
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def _small(**kw):
    kw.setdefault("single_pass", True)
    return GpuStitcher(_rig.maps(), band_px=_rig.band_px(), width=W, height=H,
                       device="cpu", fisheye_size=(FW, FH), compile=False, **kw)


def _lsb(p):
    return p.astype(np.int64) // 64


def _lsc_gain():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return [(1.0 + 0.2 * np.sin(xx / 9.0) * np.cos(yy / 7.0)).astype(np.float32),
            (1.0 + 0.15 * np.cos(xx / 5.0)).astype(np.float32)]


def _inject_reference_flow(g):
    """Make the device engine use the CPU reference's DIS flow."""
    g._sp_init()

    def flow_small(ga, gb):
        a = ga[0, 0].cpu().numpy().astype(np.uint8)
        b = gb[0, 0].cpu().numpy().astype(np.uint8)
        f01, f10 = seam_flow_small(a, b)
        return torch.from_numpy(np.stack([f01, f10]).transpose(0, 3, 1, 2).copy()).to(g.dev)

    g._seam.flow_small = flow_small


@unittest.skipIf(torch is None, "torch not installed")
class TestEngineOption(unittest.TestCase):
    def test_default_engine_follows_the_device(self):
        self.assertEqual(default_seam_engine(torch.device("cpu")), "cpu")
        self.assertEqual(default_seam_engine(torch.device("cuda")), "gpu")
        self.assertEqual(default_seam_engine(torch.device("mps")), "gpu")
        self.assertEqual(_small().seam_engine, "cpu")
        self.assertEqual(_small(seam_engine="gpu").seam_engine, "gpu")

    def test_two_pass_keeps_the_cpu_engine(self):
        self.assertEqual(_small(single_pass=False, seam_engine="gpu").seam_engine, "cpu")

    def test_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            _small(seam_engine="cuda")
        with self.assertRaises(ValueError):
            _small(seam_morph="fixed")


@unittest.skipIf(torch is None, "torch not installed")
class TestEngineSmallRig(unittest.TestCase):
    def setUp(self):
        self.f0, self.f1 = _rig.fisheye_pair(seed=3)
        self.R = rot_matrix(37.0, 21.0, -13.0)

    def test_seam_off_is_bit_identical(self):
        for lsc in (None, _lsc_gain()):
            c = _small(seam_engine="cpu", lsc_gain=lsc)
            g = _small(seam_engine="gpu", lsc_gain=lsc)
            for R in (np.eye(3), self.R):
                Yc, UVc = c.stitch_yuv(self.f0, self.f1, R, seam_flow=False)
                Yg, UVg = g.stitch_yuv(self.f0, self.f1, R, seam_flow=False)
                np.testing.assert_array_equal(Yg, Yc)
                np.testing.assert_array_equal(UVg, UVc)

    def test_band_is_restored_after_a_seam_frame(self):
        g = _small(seam_engine="gpu")
        Y_first, UV_first = g.stitch_yuv(self.f0, self.f1, self.R, seam_flow=False)
        Y_on, _ = g.stitch_yuv(self.f0, self.f1, self.R, seam_flow=True)
        self.assertTrue((Y_on != Y_first).any())               # the seam frame engaged
        self.assertTrue(g._band_dirty)
        Y_off, UV_off = g.stitch_yuv(self.f0, self.f1, self.R, seam_flow=False)
        self.assertFalse(g._band_dirty)
        np.testing.assert_array_equal(Y_off, Y_first)
        np.testing.assert_array_equal(UV_off, UV_first)

    def test_seam_stats_resolve_on_access(self):
        g = _small(seam_engine="gpu")
        self.assertIsNone(g.seam_stats)
        g.stitch_yuv(self.f0, self.f1, np.eye(3), seam_flow=True)
        self.assertIn("_dev", g._seam_stats)                   # still on the device
        st = g.seam_stats
        self.assertEqual(set(st), {"pre", "photo", "flow", "post"})
        self.assertIsInstance(st["pre"], float)
        self.assertIs(st["photo"], True)
        self.assertIs(st["flow"], True)
        self.assertIs(g.seam_stats, st)                        # resolved once
        c = _small(seam_engine="cpu")
        c.stitch_yuv(self.f0, self.f1, np.eye(3), seam_flow=True)
        self.assertAlmostEqual(st["pre"], c.seam_stats["pre"], delta=0.01)

    def test_output_stays_in_video_range(self):
        g = _small(seam_engine="gpu")
        for sf in (True, "auto"):
            Y, UV = g.stitch_yuv(self.f0, self.f1, self.R, seam_flow=sf)
            self.assertGreaterEqual(int(Y.min()), 64 * 64)
            self.assertLessEqual(int(Y.max()), 940 * 64)
            self.assertLessEqual(int(UV.max()), 960 * 64)

    def test_folded_application_matches_the_reference(self):
        # same flow in both engines: what differs is only HOW the fields are
        # applied. This rig's lenses show unrelated pictures over a 3-row
        # inner band, so a float-level cost difference can move the dynamic
        # seam by a row on a few columns: judge the statistics, not the max
        # (on footage the same comparison is within 3 LSB everywhere -
        # PERFORMANCE.md).
        for lsc in (None, _lsc_gain()):
            for morph in ("legacy", "corrected"):
                c = _small(seam_engine="cpu", seam_morph=morph, lsc_gain=lsc)
                g = _small(seam_engine="gpu", seam_morph=morph, lsc_gain=lsc)
                _inject_reference_flow(g)
                for R in (np.eye(3), self.R):
                    for sf in (True, "auto"):
                        Yc, UVc = c.stitch_yuv(self.f0, self.f1, R, seam_flow=sf)
                        Yg, UVg = g.stitch_yuv(self.f0, self.f1, R, seam_flow=sf)
                        tag = f"lsc={lsc is not None} {morph} seam_flow={sf}"
                        dY = np.abs(_lsb(Yg) - _lsb(Yc))
                        dUV = np.abs(_lsb(UVg) - _lsb(UVc))
                        self.assertLess(dY.mean(), 0.5, tag)
                        self.assertLess((dY > 4).mean(), 0.01, tag)
                        self.assertLess(dUV.mean(), 0.2, tag)
                        sc, sg = c.seam_stats, g.seam_stats
                        self.assertEqual(sg["photo"], sc["photo"], tag)
                        self.assertEqual(sg["flow"], sc["flow"], tag)
                        self.assertAlmostEqual(sg["pre"], sc["pre"], delta=0.01, msg=tag)
                        self.assertAlmostEqual(sg["post"], sc["post"],
                                               delta=0.005 * sc["post"], msg=tag)


@unittest.skipIf(torch is None, "torch not installed")
class TestEngineWorldRig(unittest.TestCase):
    """One world seen by both lenses (tests/_rig.py): the seam stack has work."""

    SHIFT = 6.0                        # output px along the seam

    @classmethod
    def setUpClass(cls):
        cls.maps = _rig.maps(_rig.WORLD_W, _rig.WORLD_H, _rig.WORLD_FW)
        cls.same = _rig.world_pair(0.0)
        cls.moved = _rig.world_pair(cls.SHIFT)

    def _stitcher(self, device="cpu", **kw):
        return GpuStitcher(self.maps, band_px=SEAM_OVERLAP_RATIO * _rig.WORLD_W,
                           width=_rig.WORLD_W, height=_rig.WORLD_H, device=device,
                           fisheye_size=(_rig.WORLD_FW, _rig.WORLD_FW), compile=False,
                           single_pass=True, **kw)

    def test_morph_direction(self):
        stats = {}
        for morph in ("legacy", "corrected"):
            g = self._stitcher(seam_engine="gpu", seam_morph=morph)
            g.stitch_yuv(*self.moved, np.eye(3), seam_flow=True)
            stats[morph] = g.seam_stats
        pre = stats["legacy"]["pre"]
        self.assertGreater(pre, 30.0)                          # the shift is visible
        self.assertAlmostEqual(stats["corrected"]["pre"], pre, delta=1e-3)
        # corrected: each lens moves TOWARDS the other; legacy: away
        self.assertLess(stats["corrected"]["post"], 0.6 * pre, stats)
        self.assertGreater(stats["legacy"]["post"], 1.2 * pre, stats)

    def test_seam_stack_is_band_limited(self):
        g = self._stitcher(seam_engine="gpu")
        Y_off, UV_off = g.stitch_yuv(*self.moved, np.eye(3), seam_flow=False)
        Y_on, UV_on = g.stitch_yuv(*self.moved, np.eye(3), seam_flow=True)
        r0, r1 = g._sf_rows
        self.assertGreater(r0, 100)
        self.assertLess(r1, _rig.WORLD_H - 100)
        np.testing.assert_array_equal(Y_on[:r0], Y_off[:r0])
        np.testing.assert_array_equal(Y_on[r1:], Y_off[r1:])
        np.testing.assert_array_equal(UV_on[:r0 // 2], UV_off[:r0 // 2])
        np.testing.assert_array_equal(UV_on[(r1 + 1) // 2:], UV_off[(r1 + 1) // 2:])
        self.assertTrue((Y_on[r0:r1] != Y_off[r0:r1]).any())

    def test_auto_leaves_matched_lenses_alone(self):
        g = self._stitcher(seam_engine="gpu")
        Y_off, UV_off = g.stitch_yuv(*self.same, np.eye(3), seam_flow=False)
        Y, UV = g.stitch_yuv(*self.same, np.eye(3), seam_flow="auto")
        st = g.seam_stats
        self.assertFalse(st["photo"])
        self.assertFalse(st["flow"])
        self.assertIsNone(st["post"])
        # nothing engaged: the band rows hold the static maps again
        self.assertLessEqual(int(np.abs(_lsb(Y) - _lsb(Y_off)).max()), 1)
        self.assertLessEqual(int(np.abs(_lsb(UV) - _lsb(UV_off)).max()), 1)

    def test_rotated_frame_keeps_the_seam_in_rig_space(self):
        # the levelling rotation is applied by the same single resample: the
        # band statistics (rig space) must not depend on it
        R = rot_matrix(25.0, -12.0, 8.0)
        g = self._stitcher(seam_engine="gpu", seam_morph="corrected")
        g.stitch_yuv(*self.moved, np.eye(3), seam_flow=True)
        a = g.seam_stats
        Y, _ = g.stitch_yuv(*self.moved, R, seam_flow=True)
        b = g.seam_stats
        self.assertAlmostEqual(a["pre"], b["pre"], delta=1e-3)
        self.assertAlmostEqual(a["post"], b["post"], delta=1e-3)
        self.assertTrue(np.isfinite(Y.astype(np.float64)).all())

    @unittest.skipIf(_accel() is None, "no MPS/CUDA device")
    def test_accelerator_matches_the_cpu_device(self):
        dev = _accel()
        out = {}
        for d in ("cpu", dev):
            g = self._stitcher(device=d, seam_engine="gpu", seam_morph="corrected")
            Y, UV = g.stitch_yuv(*self.moved, np.eye(3), seam_flow="auto")
            out[d] = (Y, UV, g.seam_stats)
        (Yc, UVc, sc), (Yg, UVg, sg) = out["cpu"], out[dev]
        self.assertEqual(sg["flow"], sc["flow"])
        self.assertAlmostEqual(sg["pre"], sc["pre"], delta=0.01 * sc["pre"])
        self.assertAlmostEqual(sg["post"], sc["post"], delta=0.02 * sc["post"])
        self.assertLess(np.abs(_lsb(Yg) - _lsb(Yc)).mean(), 0.25)
        self.assertLess(np.abs(_lsb(UVg) - _lsb(UVc)).mean(), 0.25)


if __name__ == "__main__":
    unittest.main()
