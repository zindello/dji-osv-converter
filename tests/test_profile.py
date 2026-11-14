"""The per-stage profile (profile.py): accounting, warm-up, report lines.

No GPU needed: stage timing is host-side; the GPU timeline is exercised in
``sync`` mode on the torch CPU device (where synchronising is a no-op) and
the CUDA event path only on a CUDA host.
"""

import json
import logging
import threading
import time
import unittest

try:
    import torch
except Exception:                     # pragma: no cover
    torch = None

from dji_osv_converter import profile as P


def _busy(seconds: float) -> None:
    t = time.perf_counter() + seconds
    while time.perf_counter() < t:
        pass


class TestProfile(unittest.TestCase):
    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            P.Profile(mode="fast")

    def test_wall_and_cpu_per_frame_after_warmup(self):
        p = P.Profile(warmup=2)
        for i in range(6):
            p.frame_begin()
            with p.stage("work"):
                _busy(0.004)
            with p.stage("blocked"):
                time.sleep(0.004)
            if i < 2:
                with p.stage("warm_only"):
                    _busy(0.002)
            p.frame_end()
        rep = p.finish()
        self.assertEqual(rep["frames"], 6)
        self.assertEqual(rep["warmup_frames"], 2)
        st = rep["stages"]
        # a spinning stage burns its wall time; a sleeping one does not
        self.assertAlmostEqual(st["work"]["wall"], 4.0, delta=1.5)
        self.assertGreater(st["work"]["cpu"], 0.6 * st["work"]["wall"])
        self.assertGreaterEqual(st["blocked"]["wall"], 3.5)
        self.assertLess(st["blocked"]["cpu"], 0.5 * st["blocked"]["wall"])
        self.assertEqual(st["work"]["calls"], 1.0)
        # warm-up frames are excluded from the per-frame figures
        self.assertEqual(st["warm_only"]["wall"], 0.0)
        self.assertEqual(st["warm_only"]["calls"], 0.0)
        self.assertAlmostEqual(rep["ms_per_frame"], st["work"]["wall"] + st["blocked"]["wall"],
                               delta=2.0)
        self.assertAlmostEqual(rep["fps"], 1000.0 / rep["ms_per_frame"], delta=0.05 * rep["fps"])

    def test_short_run_reports_without_warmup(self):
        p = P.Profile(warmup=10)
        for _ in range(3):
            p.frame_begin()
            with p.stage("work"):
                _busy(0.002)
            p.frame_end()
        rep = p.finish()
        self.assertEqual(rep["warmup_frames"], 0)
        self.assertGreater(rep["stages"]["work"]["wall"], 1.0)
        self.assertEqual(rep["stages"]["work"]["calls"], 1.0)

    def test_shared_stage_is_counted_across_threads(self):
        p = P.Profile(warmup=0)
        p.share("dec")
        n, per = 4, 50

        def worker():
            for _ in range(per):
                with p.stage("dec"):
                    pass

        ts = [threading.Thread(target=worker) for _ in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        for _ in range(10):
            p.frame_begin()
            p.frame_end()
        rep = p.finish()
        self.assertEqual(rep["stages"]["dec"]["calls"], n * per / 10)

    def test_add_and_notes(self):
        p = P.Profile(warmup=0)
        p.note(decode_fmt="cuda", seam_engine="gpu")
        p.frame_begin()
        p.add("mux.write", 0.010, 0.002)
        p.frame_end()
        rep = p.finish()
        self.assertEqual(rep["config"], {"decode_fmt": "cuda", "seam_engine": "gpu"})
        self.assertAlmostEqual(rep["stages"]["mux.write"]["wall"], 10.0, places=3)
        self.assertAlmostEqual(rep["stages"]["mux.write"]["cpu"], 2.0, places=3)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_sync_mode_attributes_gpu_time_to_stages(self):
        p = P.Profile(mode="sync", device=torch.device("cpu"), warmup=0)
        for _ in range(3):
            p.frame_begin()
            with p.stage("st.a", gpu=True):
                _busy(0.003)
            _busy(0.003)                     # host work between GPU stages
            with p.stage("st.b", gpu=True):
                _busy(0.001)
            p.frame_end()
        st = p.finish()["stages"]
        self.assertAlmostEqual(st["st.a"]["gpu"], 3.0, delta=1.0)
        # the gap between the stages is not charged to the second one
        self.assertLess(st["st.b"]["gpu"], 2.5)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA only")
    def test_cuda_events_resolve_without_sync(self):
        dev = torch.device("cuda")
        p = P.Profile(mode="events", device=dev, warmup=0)
        x = torch.rand(2048, 2048, device=dev)
        for _ in range(4):
            p.frame_begin()
            with p.stage("st.mm", gpu=True):
                x = (x @ x).clamp(0, 1)
            p.frame_end()
        st = p.finish()["stages"]
        self.assertGreater(st["st.mm"]["gpu"], 0.0)

    def test_null_profile_is_inert(self):
        n = P.NullProfile()
        with n.stage("x", gpu=True):
            pass
        n.add("x", 1.0)
        n.frame_begin()
        n.frame_end()
        n.note(a=1)
        self.assertFalse(n.enabled)

    def test_active_profile_switch(self):
        self.assertFalse(P.active().enabled)
        p = P.Profile()
        P.activate(p)
        try:
            self.assertIs(P.active(), p)
        finally:
            P.activate(None)
        self.assertFalse(P.active().enabled)


class TestReport(unittest.TestCase):
    def test_publish_logs_two_parseable_lines(self):
        prof = {"frames": 3, "fps": 1.5, "stages": {"a": {"wall": 1.0, "cpu": 0.5, "calls": 1}}}
        mem = {"peak_rss_mib": 12.5}
        with self.assertLogs("dji_osv_converter.profile", level=logging.INFO) as cm:
            P.publish(prof, mem)
        lines = [r.getMessage() for r in cm.records]
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith(P.PROFILE_PREFIX))
        self.assertTrue(lines[1].startswith(P.MEM_PREFIX))
        self.assertEqual(json.loads(lines[0][len(P.PROFILE_PREFIX):]), prof)
        self.assertEqual(json.loads(lines[1][len(P.MEM_PREFIX):]), mem)
        self.assertNotIn("\n", lines[0])
        self.assertEqual(P.last_profile(), {"profile": prof, "memory": mem})

    def test_format_table(self):
        prof = {"stages": {"st.sample": {"wall": 1.234, "cpu": 0.5, "gpu": 20.0, "calls": 1},
                           "enc.encode": {"wall": 35.0, "cpu": 2.0, "calls": 1}}}
        t = P.format_table(prof).splitlines()
        self.assertEqual(len(t), 4)
        self.assertIn("| `st.sample` | 1.23 | 0.50 | 20.00 | 1 |", t)
        self.assertIn("| `enc.encode` | 35.00 | 2.00 |  | 1 |", t)

    def test_memory_report_has_peak_rss(self):
        m = P.memory_report()
        self.assertGreater(m["peak_rss_mib"], 10.0)
        self.assertLess(m["peak_rss_mib"], 1024.0 * 256)


if __name__ == "__main__":
    unittest.main()
