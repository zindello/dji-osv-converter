"""tools/v100_bench.py: the parts that run without an NVIDIA card.

The benchmark itself can only run on the card. What can be checked anywhere
is that it refuses to run without one - no quiet fallback - and that the
report is assembled correctly from results.
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("v100_bench", ROOT / "tools" / "v100_bench.py")
vb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vb)

NVIDIA = sys.platform == "linux" and Path("/dev/nvidiactl").exists()


def _fake_pynvml(count=1, dec=40.0, enc=30.0):
    m = types.ModuleType("pynvml")
    m.NVMLError = type("NVMLError", (Exception,), {})
    m.nvmlInit = lambda: None
    m.nvmlDeviceGetCount = lambda: count
    m.nvmlDeviceGetHandleByIndex = lambda i: i
    m.nvmlDeviceGetName = lambda h: b"Tesla V100-PCIE-16GB"
    m.nvmlDeviceGetUUID = lambda h: f"GPU-fake-{h}"
    m.nvmlSystemGetDriverVersion = lambda: "580.178.04"
    m.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=551 * 2 ** 20)
    m.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=80, memory=35)
    m.nvmlDeviceGetEncoderUtilization = lambda h: (enc, 500)
    m.nvmlDeviceGetDecoderUtilization = lambda h: (dec, 500)
    return m


def _result(fps=20.0):
    return {"profile": {"frames": 450, "warmup_frames": 30, "mode": "events", "setup_s": 9.0,
                        "total_s": 31.0, "ms_per_frame": 1000.0 / fps, "fps": fps,
                        "cpu_cores": 2.1, "config": {"device": "cuda:0", "codec": "hevc_nvenc",
                                                     "decode": "cuda"},
                        "stages": {"main.stitch": {"wall": 30.0, "cpu": 12.0, "calls": 1}}},
            "memory": {"peak_rss_mib": 5000.0, "gpu_peak_alloc_mib": 6000.0,
                       "gpu_peak_reserved_mib": 7000.0},
            "total_s": 31.0, "out_bytes": 50 * 2 ** 20,
            "table": "| stage | wall |\n|---|---:|\n| `main.stitch` | 30.00 |"}


class TestRefusals(unittest.TestCase):
    def test_no_pynvml_is_an_error(self):
        with mock.patch.dict(sys.modules, {"pynvml": None}):
            with self.assertRaisesRegex(vb.BenchError, "pynvml"):
                vb.Nvml(None)

    def test_several_cards_need_an_index(self):
        with mock.patch.dict(sys.modules, {"pynvml": _fake_pynvml(count=3)}):
            with self.assertRaisesRegex(vb.BenchError, "--gpu"):
                vb.Nvml(None)
            self.assertEqual(vb.Nvml(2).uuid, "GPU-fake-2")

    @unittest.skipIf(NVIDIA, "this machine has an NVIDIA card")
    def test_no_nvdec_is_an_error(self):
        with self.assertRaisesRegex(vb.BenchError, "NVDEC"):
            vb._decode_probe("/nonexistent/clip.OSV")

    @unittest.skipIf(NVIDIA, "this machine has an NVIDIA card")
    def test_child_reports_the_refusal(self):
        with tempfile.TemporaryDirectory() as td:
            js = os.path.join(td, "r.json")
            args = types.SimpleNamespace(child="preflight", clip="/nonexistent/clip.OSV", json=js)
            self.assertEqual(vb.child_main(args), 2)
            self.assertIn("NVDEC", json.loads(Path(js).read_text())["error"])

    @unittest.skipIf(NVIDIA, "this machine has an NVIDIA card")
    def test_run_stops_at_preflight(self):
        """The whole command, NVML faked: it must stop before any variant runs."""
        with tempfile.TemporaryDirectory() as td:
            clip = os.path.join(td, "clip.OSV")
            Path(clip).write_bytes(b"")
            report = os.path.join(td, "report.md")
            argv = ["v100_bench.py", clip, "--report", report, "--variants", "new"]
            with mock.patch.dict(sys.modules, {"pynvml": _fake_pynvml()}), \
                    mock.patch.object(sys, "argv", argv), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    vb.main()
            self.assertIn("FAILED preflight", str(cm.exception.code))
            self.assertFalse(os.path.exists(report))


class TestReport(unittest.TestCase):
    def test_stats(self):
        s = [{"sm": 10.0}, {"sm": 30.0}, {"sm": 20.0}, {"mem": 1.0}]
        st = vb._stats(s, "sm")
        self.assertEqual((st["mean"], st["p50"], st["max"], st["n"]), (20.0, 20.0, 30.0, 3))
        self.assertIsNone(vb._stats(s, "dec"))

    def test_sampler_marks_unreadable(self):
        fake = _fake_pynvml()

        def boom(h):
            raise fake.NVMLError("not supported")
        fake.nvmlDeviceGetDecoderUtilization = boom
        with mock.patch.dict(sys.modules, {"pynvml": fake}):
            n = vb.Nvml(None)
            s = n.sample()
        self.assertNotIn("dec", s)
        self.assertEqual(s["enc"], 30.0)
        self.assertEqual(n.unreadable, {"dec"})

    def test_report(self):
        samples = [{"mem": 9000.0 + i, "sm": 80.0, "enc": 30.0, "dec": 95.0, "membus": 35.0}
                   for i in range(4)]
        rows = [{"name": "baseline", "res": _result(4.5), "samples": samples, "tail": ""},
                {"name": "new", "res": {"error": "NVDEC utilisation was 0 %"}, "samples": [],
                 "tail": "Traceback ...\nRuntimeError: boom\n"}]
        head = {"gpu": "Tesla V100-PCIE-16GB", "uuid": "GPU-x", "driver": "580.178.04",
                "clip": "/clip.OSV", "frames": 450, "start": 0, "code": "`abc1234`",
                "when": "2026-11-14 10:00", "idle_mib": 551.0}
        pre = {"decode": "cuda", "capability": "7.0", "total_mib": 16384, "torch": "2.14.1+cu126",
               "torch_cuda": "12.6", "arch_list": ["sm_60", "sm_70"], "pyav": "19.0.2",
               "libavcodec": "62.11.100", "python": "3.12.15", "cpus": 40, "affinity": 6}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "report.md"
            vb.write_report(p, head, pre, rows, ["new: NVDEC utilisation was 0 %"])
            text = p.read_text()
        self.assertIn("| `baseline` | 4.50 | 222.2 | 2.10 | 9.0 | 9003 | 6000 / 7000 | 5000 | "
                      "80 / 80 | 30 / 30 | 95 / 95 | 35 / 35 |", text)
        self.assertIn("| `new` | failed |", text)
        self.assertIn("## Failures", text)
        self.assertIn("RuntimeError: boom", text)
        self.assertIn("| `main.stitch` | 30.00 |", text)

    def test_variants_are_production_plus_overrides(self):
        self.assertEqual(vb.PRODUCTION["levelling"], "rocksteady_horizon")
        self.assertFalse(vb.PRODUCTION["allow_software"])
        self.assertEqual(vb.VARIANTS["baseline"][1], {"seam_engine": "cpu"})
        self.assertEqual(vb.VARIANTS["new"][1], {})
        self.assertIs(vb.VARIANTS["new_seam_off"][1]["seam_flow"], False)
        for name in vb.DEFAULT:
            self.assertIn(name, vb.VARIANTS)


if __name__ == "__main__":
    unittest.main()
