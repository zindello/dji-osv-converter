"""The device seam stack (seam_gpu.py) against the CPU reference (stitcher.py).

No footage and no GPU needed: everything runs on the torch CPU device, and
again on MPS/CUDA when one is present. Each test pins one of the claims in
the module docstring: the Gaussian blurs follow OpenCV's conventions, the
photometric estimator and the dynamic-seam gate reproduce the numpy code,
the parallel dynamic programme is exactly optimal, and the Lucas-Kanade
flow recovers a known displacement with the sign the morph relies on.
"""

import unittest

import cv2
import numpy as np

try:
    import torch
except Exception:                     # pragma: no cover
    torch = None

from dji_osv_converter import seam_gpu
from dji_osv_converter.constants import (SEAM_DYN_COLS, SEAM_OFFSET_AGREE_T,
                                         SEAM_OFFSET_SIGMA)
from dji_osv_converter.stitcher import seam_gate, seam_luma_offsets, seam_path

LO, HI = 64.0, 940.0


def _devices():
    devs = ["cpu"]
    if torch is not None:
        if torch.cuda.is_available():
            devs.append("cuda")
        elif torch.backends.mps.is_available():
            devs.append("mps")
    return devs


def _inner4(bh, w4, top, bottom):
    inner = np.zeros((bh, w4), bool)
    inner[top:bottom] = True
    return inner


def _seam(bh=96, W=512, top=16, bottom=80, device="cpu", cam0_top=True):
    inner = _inner4(bh, W // 4, top, bottom)
    yy = np.arange(bh, dtype=np.float32)[:, None]
    ramp = np.clip((bottom - yy) / (bottom - top), 0.0, 1.0)
    w_band = np.repeat(ramp if cam0_top else 1.0 - ramp, W, 1)
    return seam_gpu.SeamGpu(inner, torch.from_numpy(w_band).to(device), cam0_top,
                            LO, HI, device), inner


def _reference_path_cost(cs):
    """Cost of the optimal |dy| <= 1 path (numpy, column by column)."""
    D = cs[:, 0].astype(np.float64).copy()
    for x in range(1, cs.shape[1]):
        p = np.pad(D, 1, constant_values=np.inf)
        D = cs[:, x] + np.minimum(np.minimum(p[:-2], p[1:-1]), p[2:])
    return D.min()


@unittest.skipIf(torch is None, "torch not installed")
class TestBlur(unittest.TestCase):
    def test_blur_matrix_is_cv2_gaussian_blur(self):
        rng = np.random.default_rng(0)
        for n, sigma in ((16, 1.5), (97, 3.0), (480, 12.0), (5, 4.0)):
            x = rng.normal(size=(1, n)).astype(np.float32)
            ref = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=0.1).ravel()
            got = seam_gpu.blur_matrix(n, sigma) @ x.ravel().astype(np.float64)
            np.testing.assert_allclose(got, ref, atol=2e-5)

    def test_blur_both_axes_and_long_axis(self):
        rng = np.random.default_rng(1)
        for dev in _devices():
            s, _ = _seam(device=dev)
            # 1500 columns exceeds the matrix limit: pad + conv path
            for shape, sx, sy in (((40, 200), 3.0, 3.0), ((24, 1500), 40.0, 1.5)):
                x = rng.normal(size=shape).astype(np.float32)
                ref = cv2.GaussianBlur(x, (0, 0), sigmaX=sx, sigmaY=sy)
                got = s.blur(torch.from_numpy(x).to(dev)[None, None], sx=sx, sy=sy)
                np.testing.assert_allclose(got[0, 0].cpu().numpy(), ref, atol=1e-4,
                                           err_msg=f"{dev} {shape}")


@unittest.skipIf(torch is None, "torch not installed")
class TestLumaOffsets(unittest.TestCase):
    def _pair(self, bh, w4, seed=0):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:bh, 0:w4].astype(np.float32)
        A = 500 + 300 * np.sin(xx / 17.0) * np.cos(yy / 11.0) + rng.normal(0, 2, (bh, w4))
        # a luma-dependent offset plus a slow drift along the seam
        B = A + 0.02 * (A - 500) + 8 * np.sin(xx / 60.0) + rng.normal(0, 2, (bh, w4))
        return A.astype(np.float32), B.astype(np.float32)

    def test_matches_the_numpy_estimator(self):
        bh, W = 96, 1024
        A, B = self._pair(bh, W // 4)
        for dev in _devices():
            s, inner = _seam(bh=bh, W=W, device=dev)
            lut_r, sp_r = seam_luma_offsets(A, B, inner, LO, HI, 1024,
                                            sigma=SEAM_OFFSET_SIGMA,
                                            agree_t=SEAM_OFFSET_AGREE_T)
            lut, sp = s.luma_offsets(torch.from_numpy(A).to(dev)[None, None],
                                     torch.from_numpy(B).to(dev)[None, None])
            self.assertGreater(np.abs(lut_r).max(), 1.0)       # the test has signal
            np.testing.assert_allclose(lut.cpu().numpy(), lut_r, atol=0.02, err_msg=dev)
            np.testing.assert_allclose(sp[0, 0].cpu().numpy(), sp_r, atol=0.02, err_msg=dev)

    def test_no_agreeing_pixels_is_zero(self):
        bh, W = 96, 512
        A = np.full((bh, W // 4), 300.0, np.float32)
        B = A + 400.0                                         # nothing passes the gate
        s, _ = _seam(bh=bh, W=W)
        lut, sp = s.luma_offsets(torch.from_numpy(A)[None, None],
                                 torch.from_numpy(B)[None, None])
        self.assertEqual(float(lut.abs().max()), 0.0)
        self.assertEqual(float(sp.abs().max()), 0.0)


@unittest.skipIf(torch is None, "torch not installed")
class TestSeamPath(unittest.TestCase):
    def _check(self, s, cs, dev):
        path = s.seam_path(torch.from_numpy(cs).to(dev)).cpu().numpy()
        h, n = cs.shape
        self.assertEqual(path.shape, (n,))
        self.assertTrue((path >= 0).all() and (path < h).all())
        self.assertLessEqual(np.abs(np.diff(path)).max(initial=0), 1)
        got = cs[path, np.arange(n)].astype(np.float64).sum()
        self.assertAlmostEqual(got, _reference_path_cost(cs), delta=1e-3 * max(got, 1.0))

    def test_optimal_on_random_costs(self):
        rng = np.random.default_rng(2)
        for dev in _devices():
            s, _ = _seam(device=dev)
            # widths that are and are not a multiple of the block size;
            # heights smaller than a block's reach
            for h, n in ((171, 960), (40, 100), (7, 64), (3, 5), (64, 1000), (20, 16)):
                self._check(s, rng.uniform(0, 50, (h, n)).astype(np.float32), dev)

    def test_masked_rows_are_avoided(self):
        cs = np.full((40, 200), 1e6, np.float32)
        cs[10:30] = 50.0
        cs[18] = 0.0
        for dev in _devices():
            s, _ = _seam(device=dev)
            path = s.seam_path(torch.from_numpy(cs).to(dev)).cpu().numpy()
            self.assertTrue((path == 18).all(), dev)

    def test_follows_the_cpu_path(self):
        # a unique valley: the parallel solve and the sequential one agree
        h, n = 60, SEAM_DYN_COLS
        cost = np.full((h, n), 100.0, np.float32)
        valley = (20 + 10 * np.sin(np.linspace(0, 3 * np.pi, n))).astype(int)
        for d, v in ((0, 0.0), (-1, 1.0), (1, 1.0), (-2, 2.0), (2, 2.0)):
            cost[np.clip(valley + d, 0, h - 1), np.arange(n)] = np.minimum(
                cost[np.clip(valley + d, 0, h - 1), np.arange(n)], v)
        s, _ = _seam()
        path = s.seam_path(torch.from_numpy(cost)).numpy()
        np.testing.assert_array_equal(path, valley)


@unittest.skipIf(torch is None, "torch not installed")
class TestDynamic(unittest.TestCase):
    def test_gate_and_path_match_the_cpu_code(self):
        # 3840: the band is SEAM_DYN_COLS wide at 1/4 (no resize); 7680: an
        # integer reduction (the production size); 2048 and 5760: enlarging
        # and a fractional reduction, which go through OpenCV's own matrix
        from dji_osv_converter.constants import SEAM_DYN_FEATHER_PX
        bh, top, bottom = 96, 16, 80
        for W in (3840, 7680, 2048, 5760):
            rng = np.random.default_rng(3)
            Ma = rng.uniform(300, 700, (bh, W)).astype(np.float32)
            Ma = cv2.GaussianBlur(Ma, (0, 0), 6.0)
            Mb = Ma.copy()
            # real parallax over a run of columns, with one row where the lenses agree
            c0, c1 = int(W * 0.3), int(W * 0.5)
            Mb[:, c0:c1] += 160.0
            Mb[40, c0:c1] = Ma[40, c0:c1]
            for dev in _devices():
                for cam0_top in (True, False):
                    s, inner = _seam(bh=bh, W=W, top=top, bottom=bottom, device=dev,
                                     cam0_top=cam0_top)
                    wd, post = s.dynamic(torch.from_numpy(Ma).to(dev)[None, None],
                                         torch.from_numpy(Mb).to(dev)[None, None],
                                         torch.ones((), device=dev))
                    # the CPU sequence (stitch_gpu._seam_flow_apply)
                    cost = np.abs(Ma - Mb).reshape(bh, W // 4, 4).mean(2) * 0.25
                    seam = seam_path(np.where(inner, cost, 1e6).astype(np.float32), W)
                    gate = seam_gate(cost, inner, W)
                    yy = np.arange(bh, dtype=np.float32)[:, None]
                    ref = np.clip(0.5 + (seam[None, :] - yy) / SEAM_DYN_FEATHER_PX, 0.0, 1.0)
                    if not cam0_top:
                        ref = 1.0 - ref
                    static = s.w_band[0, 0].cpu().numpy()
                    ref = static + (ref - static) * gate[None, :]
                    tag = f"{dev} W={W} cam0_top={cam0_top}"
                    self.assertGreater(gate.max(), 0.9, tag)
                    self.assertLess(gate.min(), 0.05, tag)
                    np.testing.assert_allclose(wd[0, 0].cpu().numpy(), ref, atol=2e-3,
                                               err_msg=tag)
                    self.assertAlmostEqual(float(post), float(cost[inner].mean() * 4.0),
                                           delta=0.01, msg=tag)

    def test_not_engaged_keeps_the_static_weight(self):
        bh, W = 96, 1024
        rng = np.random.default_rng(4)
        Ma = rng.uniform(300, 700, (bh, W)).astype(np.float32)
        Mb = Ma + 200.0
        s, _ = _seam(bh=bh, W=W)
        wd, _ = s.dynamic(torch.from_numpy(Ma)[None, None], torch.from_numpy(Mb)[None, None],
                          torch.zeros(()))
        np.testing.assert_array_equal(wd.numpy(), s.w_band.numpy())


@unittest.skipIf(torch is None, "torch not installed")
class TestFlow(unittest.TestCase):
    def _grays(self, sh, sw, shift, seed=5):
        """Two smooth textures with b(p + shift) = a(p)."""
        rng = np.random.default_rng(seed)
        big = cv2.GaussianBlur(rng.uniform(0, 255, (sh + 64, sw + 64)).astype(np.float32),
                               (0, 0), 5.0)
        big = (big - big.min()) / (big.max() - big.min()) * 255.0
        a = big[32:32 + sh, 32:32 + sw]
        dx, dy = shift
        b = big[32 - dy:32 - dy + sh, 32 - dx:32 - dx + sw]
        return np.ascontiguousarray(a), np.ascontiguousarray(b)

    def test_recovers_a_known_shift_in_both_directions(self):
        bh, W = 256, 2048
        for dev in _devices():
            s, _ = _seam(bh=bh, W=W, top=0, bottom=bh, device=dev)
            a, b = self._grays(s.sh, s.sw, (3, 1))
            fl = s.lk_flow(torch.from_numpy(a).to(dev)[None, None],
                           torch.from_numpy(b).to(dev)[None, None]).cpu().numpy()
            self.assertEqual(fl.shape, (2, 2, s.sh, s.sw))
            core = (slice(None), slice(None), slice(12, -12), slice(24, -24))
            f = fl[core]
            # b(p + f01) ~ a(p): f01 = +shift; a(p + f10) ~ b(p): f10 = -shift
            self.assertLess(np.abs(f[0, 0] - 3.0).mean(), 0.35, dev)
            self.assertLess(np.abs(f[0, 1] - 1.0).mean(), 0.35, dev)
            self.assertLess(np.abs(f[1, 0] + 3.0).mean(), 0.35, dev)
            self.assertLess(np.abs(f[1, 1] + 1.0).mean(), 0.35, dev)

    def test_identical_images_give_no_flow(self):
        s, _ = _seam(bh=256, W=2048, top=0, bottom=256)
        a, _ = self._grays(s.sh, s.sw, (0, 0))
        t = torch.from_numpy(a)[None, None]
        self.assertLess(float(s.flow_small(t, t).abs().max()), 1e-3)

    def test_brightness_offset_is_not_motion(self):
        # mean-normalised windows: a constant offset between the lenses
        # must not read as displacement
        s, _ = _seam(bh=256, W=2048, top=0, bottom=256)
        a, _ = self._grays(s.sh, s.sw, (0, 0))
        fl = s.lk_flow(torch.from_numpy(a)[None, None],
                       torch.from_numpy(a + 12.0)[None, None])
        self.assertLess(float(fl.abs().mean()), 0.05)


@unittest.skipIf(torch is None, "torch not installed")
class TestEstimate(unittest.TestCase):
    def test_shapes_and_auto_gating(self):
        bh, W = 256, 2048
        rng = np.random.default_rng(6)
        Y = cv2.GaussianBlur(rng.uniform(200, 800, (bh, W)).astype(np.float32), (0, 0), 8.0)
        for dev in _devices():
            s, _ = _seam(bh=bh, W=W, top=32, bottom=224, device=dev)
            t = torch.from_numpy(Y).to(dev)[None, None]
            est = s.estimate(t, t, auto=True)
            self.assertEqual(tuple(est["lut"].shape), (1024,))
            self.assertEqual(tuple(est["sp"].shape), (1, 1, bh, W // 4))
            self.assertEqual(tuple(est["flow"].shape), (2, 2, bh, W))
            # identical lenses: nothing to correct, nothing engaged
            self.assertEqual(float(est["photo"]), 0.0)
            self.assertEqual(float(est["engaged"]), 0.0)
            self.assertEqual(float(est["flow"].abs().max()), 0.0)
            self.assertAlmostEqual(float(est["pre"]), 0.0, places=5)
            # forced on: still nothing to do, but the switches are open
            est = s.estimate(t, t, auto=False)
            self.assertEqual(float(est["photo"]), 1.0)
            self.assertEqual(float(est["engaged"]), 1.0)


if __name__ == "__main__":
    unittest.main()
