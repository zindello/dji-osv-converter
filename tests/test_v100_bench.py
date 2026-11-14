"""tools/v100_bench.py: everything that runs without an NVIDIA card, for real.

The benchmark itself can only run on the card. Its first run there stopped
in the preflight on a PyAV call that had only ever been stubbed, so these
tests use the real thing wherever a card is not needed: real PyAV frames
and a real hardware encoder where the host has one, the real package
modules for the harness's switches and argument names, real child
processes, the real profile for the report, and the real ``nvidia-ml-py``
when it is installed. Only NVML's answers and the conversion itself are
faked, and the tests that fake them say so.
"""

import contextlib
import functools
import importlib.util
import inspect
import io
import json
import os
import sys
import tempfile
import textwrap
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import av
import numpy as np

try:
    import torch
except Exception:                     # pragma: no cover
    torch = None

from dji_osv_converter import decode, nvdec, profile, seam_gpu, video

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "v100_bench.py"
_spec = importlib.util.spec_from_file_location("v100_bench", TOOL)
vb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vb)

NVIDIA = sys.platform == "linux" and Path("/dev/nvidiactl").exists()
HAVE_PYNVML = importlib.util.find_spec("pynvml") is not None
W8K, H8K = vb.PRODUCTION["width"], vb.PRODUCTION["height"]


def _hw_hevc_encoder():
    """A hardware HEVC encoder this host really has (takes p010le), or None."""
    for codec in ("hevc_nvenc",) if NVIDIA else ("hevc_videotoolbox",):
        try:
            av.codec.Codec(codec, "w")
        except Exception:
            continue
        if codec == "hevc_videotoolbox" and sys.platform != "darwin":
            continue
        return codec
    return None


def _fake_pynvml(count=1, dec=40.0, enc=30.0):
    """NVML's answers, faked (names and return shapes as nvidia-ml-py has them;
    ``TestRealPynvml`` checks those against the real module)."""
    m = types.ModuleType("pynvml")
    m.NVMLError = type("NVMLError", (Exception,), {})
    m.nvmlInit = lambda: None
    m.nvmlDeviceGetCount = lambda: count
    m.nvmlDeviceGetHandleByIndex = lambda i: i
    m.nvmlDeviceGetName = lambda h: "Tesla V100-PCIE-16GB"
    m.nvmlDeviceGetUUID = lambda h: f"GPU-fake-{h}"
    m.nvmlSystemGetDriverVersion = lambda: "580.178.04"
    m.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=551 * 2 ** 20)
    m.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=80, memory=35)
    m.nvmlDeviceGetEncoderUtilization = lambda h: [enc, 500]
    m.nvmlDeviceGetDecoderUtilization = lambda h: [dec, 500]
    return m


def _real_result(mode="events", **notes):
    """A variant's result built by the real profile, as ``run_conversion`` builds it."""
    p = profile.Profile(mode=mode, warmup=1)
    for _ in range(4):
        p.frame_begin()
        with p.stage("decode.nvdec"):
            time.sleep(0.001)
        with p.stage("main.stitch"):
            time.sleep(0.002)
        p.frame_end()
    cfg = dict(device="cuda:0", codec="hevc_nvenc", decode="cuda", enc_sessions=3)
    cfg.update(notes)
    p.note(**cfg)
    rep = p.finish()
    return {"profile": rep, "memory": profile.memory_report(None), "total_s": 1.25,
            "table": profile.format_table(rep), "out_bytes": 50 * 2 ** 20}


def _run_main(argv, pynvml):
    """``main()`` with NVML faked; returns (exit code or SystemExit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.dict(sys.modules, {"pynvml": pynvml}), \
            mock.patch.object(sys, "argv", ["v100_bench.py", *argv]), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = vb.main()
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


class TestTestFrame(unittest.TestCase):
    """The encoder's test frame: real PyAV, no encoder needed."""

    def test_8k_p010_frame_is_built_and_filled(self):
        fr = vb.p010_test_frame(W8K, H8K, level=300)
        self.assertEqual((fr.format.name, fr.width, fr.height), ("p010le", W8K, H8K))
        y, uv = (np.frombuffer(pl, np.uint16).reshape(pl.height, pl.line_size // 2)
                 for pl in fr.planes)
        self.assertEqual(y.shape[0], H8K)
        self.assertEqual(uv.shape[0], H8K // 2)
        self.assertTrue((y[:, :W8K] == 300 << 6).all())
        self.assertTrue((uv[:, :W8K] == 512 << 6).all())

    def test_it_is_the_converters_own_builder(self):
        """The harness must build its frame the way convert() does."""
        with mock.patch.object(video, "_p010_av_frame", wraps=video._p010_av_frame) as spy:
            vb.p010_test_frame(64, 32)
        self.assertEqual(spy.call_count, 1)
        self.assertIn("return _p010_av_frame(width, height, planes)",
                      inspect.getsource(video._convert))

    def test_builder_copies_the_planes(self):
        """Odd line sizes and both plane sources (numpy, CPU tensors)."""
        rng = np.random.default_rng(0)
        for w, h in ((64, 32), (250, 130), (1918, 1078)):
            Y = (rng.integers(0, 1024, (h, w)) << 6).astype(np.uint16)
            UV = (rng.integers(0, 1024, (h // 2, w)) << 6).astype(np.uint16)
            sources = [(Y, UV)]
            if torch is not None:
                sources.append((torch.from_numpy(Y), torch.from_numpy(UV)))
            for planes in sources:
                fr = video._p010_av_frame(w, h, planes)
                got = [np.frombuffer(pl, np.uint16).reshape(pl.height, pl.line_size // 2)
                       for pl in fr.planes]
                np.testing.assert_array_equal(got[0][:, :w], Y)
                np.testing.assert_array_equal(got[1][:, :w], UV)

    def test_encode_probe_with_a_real_hardware_encoder(self):
        codec = _hw_hevc_encoder()
        if codec is None:
            self.skipTest("no hardware HEVC encoder on this host")
        self.assertGreater(vb.encode_probe(codec, W8K, H8K), 0)

    def test_encode_probe_without_the_encoder(self):
        with self.assertRaises(Exception) as cm:
            vb.encode_probe("no_such_encoder", 64, 32)
        self.assertIn("no_such_encoder", str(cm.exception))


class TestAgainstThePackage(unittest.TestCase):
    """The names and arguments the harness uses exist in the real package."""

    def test_every_variant_is_sound(self):
        self.assertEqual(vb.static_problems(list(vb.VARIANTS)), [])

    def test_a_wrong_argument_switch_or_name_is_named(self):
        bad = dict(vb.VARIANTS, typo=("", {"seam_engin": "cpu"}, {"dp_blok": 16}))
        with mock.patch.object(vb, "VARIANTS", bad), \
                mock.patch.object(vb, "PACKAGE_NAMES", vb.PACKAGE_NAMES + (("video", "nope"),)):
            got = vb.static_problems(["new", "typo"])
        self.assertEqual(len(got), 3)
        self.assertTrue(any("video.nope" in g for g in got))
        self.assertTrue(any("variant typo" in g and "seam_engin" in g for g in got))
        self.assertTrue(any("variant typo" in g and "dp_blok" in g for g in got))

    def test_variants_are_production_plus_overrides(self):
        self.assertEqual(vb.PRODUCTION["levelling"], "rocksteady_horizon")
        self.assertFalse(vb.PRODUCTION["allow_software"])
        self.assertEqual(vb.VARIANTS["baseline"][1], {"seam_engine": "cpu"})
        self.assertEqual(vb.VARIANTS["new"][1], {})
        self.assertIs(vb.VARIANTS["new_seam_off"][1]["seam_flow"], False)
        for name in vb.DEFAULT:
            self.assertIn(name, vb.VARIANTS)
        kw = vb.convert_kwargs("new", 450, 7)
        self.assertEqual(kw, dict(vb.PRODUCTION, start=7, count=450, profile=True))
        kw = vb.convert_kwargs("baseline_sync", 10, 0, dict(vb.PRODUCTION, codec="x"))
        self.assertEqual((kw["profile"], kw["seam_engine"], kw["codec"]), ("sync", "cpu", "x"))

    def test_the_profile_carries_what_the_checks_read(self):
        note = inspect.getsource(video._convert)
        note = note[note.index("prof.note("):]
        keys = {"device", "codec", "decode"}
        for _d, over, _sw in vb.VARIANTS.values():
            keys |= set(over) - {"profile"}
        for key in sorted(keys):
            self.assertIn(f"{key}=", note, key)
        self.assertIn('"decode.nvdec"', inspect.getsource(nvdec.cuda_worker))
        self.assertIn('"decode.nvdec"', inspect.getsource(nvdec.cuvid_worker))
        self.assertIn('"decode.hw"', inspect.getsource(decode))
        mem = inspect.getsource(profile.memory_report)
        report = inspect.getsource(vb.write_report)
        for key in ("peak_rss_mib", "gpu_peak_alloc_mib", "gpu_peak_reserved_mib"):
            self.assertIn(key, mem)
            self.assertIn(key, report)


class TestSwitches(unittest.TestCase):
    """The harness-only switches, set on the real modules."""

    def test_decode_switch_replaces_what_convert_reads(self):
        self.assertIn("gpu_decode_fmt", video._convert.__code__.co_names)
        applied = []
        with mock.patch.object(video, "gpu_decode_fmt", video.gpu_decode_fmt):
            vb.apply_switches({"decode": "p010le"}, applied)
            self.assertEqual(vb.switch_problems({"decode": "p010le"}, applied),
                             ["the harness switch decode='p010le' was never reached "
                              "by the conversion"])
            self.assertEqual(video.gpu_decode_fmt(True, "/some/clip.OSV"), "p010le")
            self.assertEqual(vb.switch_problems({"decode": "p010le"}, applied), [])
        self.assertIs(video.gpu_decode_fmt, decode.gpu_decode_fmt)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_dp_block_switch_on_a_real_seam_stack(self):
        from tests.test_seam_gpu import _reference_path_cost, _seam
        applied = []
        with mock.patch.object(seam_gpu.SeamGpu, "__init__", seam_gpu.SeamGpu.__init__):
            vb.apply_switches({"dp_block": 8}, applied)
            self.assertEqual(len(vb.switch_problems({"dp_block": 8}, applied)), 1)
            s, _ = _seam()
        self.assertEqual(s.dp_block, 8)
        self.assertEqual(applied, [("dp_block", 8)])
        self.assertEqual(vb.switch_problems({"dp_block": 8}, applied), [])
        # the seam it finds with that block is still the optimal one
        cs = np.random.default_rng(4).uniform(0, 50, (96, 500)).astype(np.float32)
        path = s.seam_path(torch.from_numpy(cs)).numpy()
        self.assertLessEqual(np.abs(np.diff(path)).max(), 1)
        got = cs[path, np.arange(cs.shape[1])].astype(np.float64).sum()
        self.assertAlmostEqual(got, _reference_path_cost(cs), delta=1e-3 * got)
        # and a stack built after the switch is gone has the package's block again
        self.assertNotEqual(_seam()[0].dp_block, 8)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_dp_block_switch_is_reached_through_the_stitcher(self):
        from dji_osv_converter.stitch_gpu import GpuStitcher
        from tests import _rig
        f0, f1 = _rig.fisheye_pair(seed=3)
        applied = []
        with mock.patch.object(seam_gpu.SeamGpu, "__init__", seam_gpu.SeamGpu.__init__):
            vb.apply_switches({"dp_block": 16}, applied)
            g = GpuStitcher(_rig.maps(), band_px=_rig.band_px(), width=_rig.W, height=_rig.H,
                            device="cpu", fisheye_size=(_rig.FW, _rig.FH), compile=False,
                            single_pass=True, seam_engine="gpu")
            g.stitch_yuv(f0, f1, np.eye(3), seam_flow=True)
        self.assertEqual(applied, [("dp_block", 16)])
        self.assertEqual(g._seam.dp_block, 16)


class TestCardChecks(unittest.TestCase):
    def test_a_run_on_the_card_passes(self):
        p = _real_result()["profile"]
        self.assertEqual(vb.card_problems(p, "cuda"), [])

    def test_each_departure_is_named(self):
        p = _real_result(device="mps", codec="hevc_videotoolbox", decode="yuv420p10le")["profile"]
        p["stages"]["decode.hw"] = p["stages"].pop("decode.nvdec")
        got = vb.card_problems(p, "cuda")
        self.assertEqual(len(got), 3)
        self.assertIn("'mps'", got[0])
        self.assertIn("'hevc_videotoolbox'", got[1])
        self.assertIn("decode.hw", got[2])
        # NVDEC, but not the arrangement that was asked for
        p = _real_result(decode="cuda")["profile"]
        self.assertEqual(len(vb.card_problems(p, "p010le")), 1)

    def test_a_variant_must_run_as_asked(self):
        ok = _real_result(seam_engine="cpu")["profile"]
        self.assertEqual(vb.override_problems("baseline", ok), [])
        self.assertIn("seam_engine ran as 'gpu'", vb.override_problems(
            "baseline", _real_result(seam_engine="gpu")["profile"])[0])
        self.assertIn("enc_sessions ran as 3", vb.override_problems(
            "new_enc1", _real_result(enc_sessions=3)["profile"])[0])
        self.assertIn("profile mode", vb.override_problems("new_sync", ok)[0])
        self.assertEqual(vb.override_problems(
            "new_sync", _real_result(mode="sync")["profile"]), [])


@unittest.skipIf(torch is None, "torch not installed")
class TestKernelCheck(unittest.TestCase):
    def test_a_kernel_runs_on_every_device_this_host_has(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        elif torch.backends.mps.is_available():
            devices.append("mps")
        for dev in devices:
            vb.kernel_check(torch, dev)

    @unittest.skipIf(NVIDIA, "this machine has an NVIDIA card")
    def test_a_kernel_that_cannot_run_says_why(self):
        with self.assertRaisesRegex(vb.BenchError, "a kernel did not run on cuda"):
            vb.kernel_check(torch, "cuda")
        card = {"capability": "7.0", "arch_list": ["sm_75", "sm_80"]}
        with self.assertRaisesRegex(vb.BenchError, "no kernels for sm_70; it was built for "
                                                   "sm_75 sm_80"):
            vb.kernel_check(torch, "cuda", card)
        card = {"capability": "7.0", "arch_list": ["sm_60", "sm_70"]}
        with self.assertRaises(vb.BenchError) as cm:
            vb.kernel_check(torch, "cuda", card)
        self.assertNotIn("no kernels", str(cm.exception))

    def test_a_wrong_answer_is_an_error(self):
        wrong = types.SimpleNamespace(
            float32=torch.float32,
            arange=lambda n, device, dtype: torch.arange(n - 1, device=device, dtype=dtype))
        with self.assertRaisesRegex(vb.BenchError, "wrong answer"):
            vb.kernel_check(wrong, "cpu")


@unittest.skipIf(NVIDIA, "this machine has an NVIDIA card")
class TestPreflightOffTheCard(unittest.TestCase):
    """On a host without the card every card check fails, and each is named."""

    WANT = ("NVDEC cannot decode", "hevc_cuvid to system memory", "CUDA is not available",
            "no hevc_nvenc encoder", "NVENC was not tried")

    def test_no_nvdec_is_an_error(self):
        with self.assertRaisesRegex(vb.BenchError, "NVDEC"):
            vb._decode_probe("/nonexistent/clip.OSV")
        with self.assertRaisesRegex(vb.BenchError, "hevc_cuvid"):
            vb._decode_probe("/nonexistent/clip.OSV", "p010le")

    def test_preflight_names_every_failing_check(self):
        res = vb.child_preflight("/nonexistent/clip.OSV", ["new", "new_p010"])
        for want in self.WANT:
            self.assertEqual(sum(want in e for e in res["errors"]), 1, (want, res["errors"]))
        self.assertEqual(len(res["errors"]), len(self.WANT))
        self.assertEqual(res["info"]["pyav"], av.__version__)
        # the hevc_cuvid check is made only for a variant that needs it
        res = vb.child_preflight("/nonexistent/clip.OSV", ["new"])
        self.assertFalse(any("hevc_cuvid to system memory" in e for e in res["errors"]))

    def test_a_failing_frame_or_encode_is_one_more_named_check(self):
        """What stopped the first run on the card: reported beside the others."""
        def boom(w, h, level=512):
            raise ValueError("Conversion from numpy array with format `p010le` "
                             "is not yet supported")
        with mock.patch.object(vb, "p010_test_frame", boom):
            res = vb.child_preflight("/nonexistent/clip.OSV", ["new"])
        self.assertTrue(any("p010le frame could not be built: ValueError" in e
                            for e in res["errors"]))
        self.assertTrue(any("NVDEC" in e for e in res["errors"]))
        self.assertTrue(any("CUDA" in e for e in res["errors"]))

    def test_real_preflight_process(self):
        with tempfile.TemporaryDirectory() as td:
            clip = os.path.join(td, "clip.OSV")
            Path(clip).write_bytes(b"")
            res, samples, _tail = vb.run_child(
                vb.preflight_argv(clip, list(vb.VARIANTS)), dict(os.environ), None, 120)
        self.assertEqual(samples, [])
        self.assertNotIn("error", res)
        for want in self.WANT:
            self.assertTrue(any(want in e for e in res["errors"]), (want, res["errors"]))

    def test_real_variant_processes_refuse(self):
        """Every variant's command line parses, and stops at the NVDEC probe."""
        with tempfile.TemporaryDirectory() as td:
            clip = os.path.join(td, "clip.OSV")
            Path(clip).write_bytes(b"")
            for name in vb.VARIANTS:
                out = os.path.join(td, f"{name}.mp4")
                res, _s, _t = vb.run_child(vb.variant_argv(name, clip, out, 5, 2),
                                           dict(os.environ), None, 120)
                want = ("hevc_cuvid to system memory" if name == "new_p010"
                        else "NVDEC cannot decode")
                self.assertIn(want, res["error"], name)
                self.assertFalse(os.path.exists(out))

    def test_run_stops_at_preflight_and_names_everything(self):
        """The whole command: NVML failing, no clip, no card - one message, all of it."""
        fake = _fake_pynvml()

        def no_library():
            raise fake.NVMLError("NVML Shared Library Not Found")
        fake.nvmlInit = no_library
        with tempfile.TemporaryDirectory() as td:
            report = os.path.join(td, "reports", "report.md")
            blocked = os.path.join(td, "a_file")
            Path(blocked).write_bytes(b"")
            code, _out, _err = _run_main(
                [os.path.join(td, "missing.OSV"), "--report", report, "--all",
                 "--out-dir", os.path.join(blocked, "out")], fake)
            self.assertFalse(os.path.exists(report))
        self.assertIsInstance(code, str)
        self.assertIn("FAILED preflight", code)
        for want in ("no such clip", "NVML did not initialise", "cannot be created",
                     *self.WANT):
            self.assertIn(want, code)
        self.assertIn(f"{code.count(chr(10))} check(s) failed", code)

    def test_run_stops_at_preflight_with_nvml_up(self):
        with tempfile.TemporaryDirectory() as td:
            clip = os.path.join(td, "clip.OSV")
            Path(clip).write_bytes(b"")
            report = os.path.join(td, "report.md")
            code, out, _err = _run_main([clip, "--report", report, "--variants", "new",
                                         "--out-dir", td], _fake_pynvml())
            self.assertFalse(os.path.exists(report))
        self.assertIn("FAILED preflight: 4 check(s) failed", code)
        self.assertIn("Tesla V100-PCIE-16GB (GPU-fake-0), driver 580.178.04", out)


class TestNvml(unittest.TestCase):
    def test_no_pynvml_is_an_error(self):
        with mock.patch.dict(sys.modules, {"pynvml": None}):
            with self.assertRaisesRegex(vb.BenchError, "pynvml"):
                vb.Nvml(None)

    def test_several_cards_need_an_index_in_range(self):
        with mock.patch.dict(sys.modules, {"pynvml": _fake_pynvml(count=3)}):
            with self.assertRaisesRegex(vb.BenchError, "--gpu INDEX"):
                vb.Nvml(None)
            with self.assertRaisesRegex(vb.BenchError, "--gpu 3: NVML sees 3"):
                vb.Nvml(3)
            self.assertEqual(vb.Nvml(2).uuid, "GPU-fake-2")

    def test_bytes_and_str_names(self):
        fake = _fake_pynvml()
        fake.nvmlDeviceGetName = lambda h: b"Tesla V100-PCIE-16GB"     # older pynvml
        with mock.patch.dict(sys.modules, {"pynvml": fake}):
            n = vb.Nvml(None)
        self.assertEqual((n.name, n.driver), ("Tesla V100-PCIE-16GB", "580.178.04"))

    def test_a_pynvml_without_a_call_is_named(self):
        fake = _fake_pynvml()
        del fake.nvmlDeviceGetEncoderUtilization
        with mock.patch.dict(sys.modules, {"pynvml": fake}):
            with self.assertRaisesRegex(vb.BenchError, "nvmlDeviceGetEncoderUtilization"):
                vb.Nvml(None)

    def test_an_nvml_error_at_any_step_is_a_refusal(self):
        for call in ("nvmlInit", "nvmlDeviceGetHandleByIndex", "nvmlDeviceGetUUID"):
            fake = _fake_pynvml()

            def boom(*a, _fake=fake):
                raise _fake.NVMLError("Insufficient Permissions")
            setattr(fake, call, boom)
            with mock.patch.dict(sys.modules, {"pynvml": fake}):
                with self.assertRaisesRegex(vb.BenchError, "NVML did not initialise"):
                    vb.Nvml(None)

    def test_sampler_notes_what_it_cannot_read_and_carries_on(self):
        fake = _fake_pynvml()

        def not_supported(h):
            raise fake.NVMLError("Not Supported")
        fake.nvmlDeviceGetDecoderUtilization = not_supported
        fake.nvmlDeviceGetMemoryInfo = lambda h: None           # a changed return type
        with mock.patch.dict(sys.modules, {"pynvml": fake}):
            n = vb.Nvml(None)
            s = n.sample()
            n.sample()
        self.assertEqual(s, {"sm": 80.0, "membus": 35.0, "enc": 30.0})
        self.assertEqual(sorted(n.unreadable), ["dec", "mem"])
        self.assertEqual(n.unreadable["dec"], "NVMLError: Not Supported")
        self.assertIn("AttributeError", n.unreadable["mem"])


@unittest.skipUnless(HAVE_PYNVML, "nvidia-ml-py is not installed")
class TestRealPynvml(unittest.TestCase):
    """The real nvidia-ml-py: the names, arguments and fields the sampler uses."""

    def test_names_arguments_and_fields(self):
        import pynvml
        for name in vb.NVML_CALLS:
            self.assertTrue(hasattr(pynvml, name), name)
        self.assertTrue(issubclass(pynvml.NVMLError, Exception))

        def required(fn):
            return [p.name for p in inspect.signature(fn).parameters.values()
                    if p.default is p.empty]
        self.assertEqual(required(pynvml.nvmlInit), [])
        self.assertEqual(required(pynvml.nvmlDeviceGetCount), [])
        self.assertEqual(required(pynvml.nvmlSystemGetDriverVersion), [])
        self.assertEqual(required(pynvml.nvmlDeviceGetHandleByIndex), ["index"])
        for name in ("nvmlDeviceGetName", "nvmlDeviceGetUUID", "nvmlDeviceGetMemoryInfo",
                     "nvmlDeviceGetUtilizationRates", "nvmlDeviceGetEncoderUtilization",
                     "nvmlDeviceGetDecoderUtilization"):
            self.assertEqual(required(getattr(pynvml, name)), ["handle"], name)
        self.assertIn("used", [f[0] for f in pynvml.c_nvmlMemory_t._fields_])
        fields = [f[0] for f in pynvml.c_nvmlUtilization_t._fields_]
        self.assertIn("gpu", fields)
        self.assertIn("memory", fields)

    @unittest.skipIf(NVIDIA, "this machine has an NVIDIA card")
    def test_no_driver_is_a_refusal(self):
        with self.assertRaisesRegex(vb.BenchError, "NVML did not initialise"):
            vb.Nvml(None)

    @unittest.skipUnless(NVIDIA, "needs the NVIDIA driver")
    def test_sample_on_the_card(self):
        import pynvml
        pynvml.nvmlInit()
        n = vb.Nvml(0)
        s = n.sample()
        self.assertIsInstance(n.name, str)
        self.assertGreater(s["mem"], 0.0)
        self.assertEqual(sorted(s), ["dec", "enc", "mem", "membus", "sm"], n.unreadable)


class TestRunChild(unittest.TestCase):
    """Real child processes that are not the benchmark."""

    def _script(self, td, body):
        p = Path(td) / "child.py"
        p.write_text(textwrap.dedent(body))
        return p

    def test_a_hung_child_is_killed_and_reported(self):
        ticks = []
        nvml = types.SimpleNamespace(sample=lambda: ticks.append(1) or {"mem": 1.0})
        with tempfile.TemporaryDirectory() as td, mock.patch.object(vb, "SAMPLE_S", 0.05):
            script = self._script(td, """
                import time
                print("started", flush=True)
                time.sleep(120)
            """)
            t0 = time.time()
            res, samples, tail = vb.run_child([], dict(os.environ), nvml, 1.5, script=script)
        self.assertLess(time.time() - t0, 30)
        self.assertEqual(res, {"error": "still running after 2 s (--timeout): killed"})
        self.assertGreater(len(samples), 3)
        self.assertEqual(len(samples), len(ticks))
        self.assertIn("started", tail)

    def test_a_child_that_dies_or_writes_rubbish(self):
        with tempfile.TemporaryDirectory() as td:
            script = self._script(td, """
                import sys
                print("about to die", flush=True)
                sys.exit(3)
            """)
            res, _s, tail = vb.run_child([], dict(os.environ), None, 60, script=script)
            self.assertIn("exited with code 3 before writing a result", res["error"])
            self.assertIn("about to die", tail)
            script = self._script(td, """
                import sys
                open(sys.argv[sys.argv.index("--json") + 1], "w").write("{not json")
            """)
            res, _s, _t = vb.run_child([], dict(os.environ), None, 60, script=script)
            self.assertIn("before writing a result", res["error"])
            script = self._script(td, """
                import sys
                open(sys.argv[sys.argv.index("--json") + 1], "w").write('{"profile": {}}')
                sys.exit(1)
            """)
            res, _s, _t = vb.run_child([], dict(os.environ), None, 60, script=script)
            self.assertEqual(res["error"], "the process exited with code 1")


# The children of TestWholeRun: the real harness with the three things only the
# card has (NVDEC, CUDA, the conversion) replaced. The conversion is a stand-in
# that publishes a real profile and writes a file, and - like a conversion that
# ignored them - never touches the harness switches.
_SHIM = '''
import functools, importlib.util, sys, time
from pathlib import Path
spec = importlib.util.spec_from_file_location("v100_bench", {tool!r})
vb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vb)
from dji_osv_converter import profile, video

@functools.wraps(video.convert)                  # the real signature, for the preflight
def convert(clip, out, **kw):
    mode = kw["profile"] if isinstance(kw["profile"], str) else "events"
    p = profile.Profile(mode=mode, warmup=1)
    for _ in range(4):
        p.frame_begin()
        with p.stage("decode.nvdec"):
            time.sleep(0.01)
        with p.stage("main.stitch"):
            time.sleep(0.02)
        p.frame_end()
    p.note(device="cuda:0", codec=kw["codec"], decode="cuda",
           enc_sessions=kw.get("enc_sessions") or 3, levelling=kw["levelling"],
           single_pass=kw["single_pass"], seam_flow=kw["seam_flow"],
           seam_engine=kw.get("seam_engine") or "gpu",
           seam_morph=kw.get("seam_morph", "legacy"))
    Path(out).write_bytes(b"x" * 4096)
    profile.publish(p.finish(), profile.memory_report(None))

video.convert = convert
vb._decode_probe = lambda clip, want=None: want or "cuda"
vb._need_cuda = lambda: None
vb.child_preflight = lambda clip, names: {{
    "info": {{"decode": "cuda", "torch": "2.14.1+cu126", "capability": "7.0"}},
    "errors": vb.static_problems(names)}}
raise SystemExit(vb.main())
'''


class TestWholeRun(unittest.TestCase):
    """``main()`` from the command line to the report, through real processes.

    NVML's answers are faked and the children's conversion is a stand-in
    (see ``_SHIM``); the argument handling, the processes, the sampler, the
    checks, the clean-up and the report are the real ones.
    """

    def _run(self, td, argv, pynvml):
        shim = Path(td) / "shim.py"
        shim.write_text(_SHIM.format(tool=str(TOOL)))
        clip = os.path.join(td, "clip.OSV")
        Path(clip).write_bytes(b"")
        report = os.path.join(td, "report.md")
        with mock.patch.object(vb, "run_child", functools.partial(vb.run_child, script=shim)), \
                mock.patch.object(vb, "SAMPLE_S", 0.05):
            code, out, err = _run_main([clip, "--report", report, "--frames", "4", *argv],
                                       pynvml)
        return code, out, err, Path(report).read_text() if os.path.exists(report) else None

    def test_all_variants(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = os.path.join(td, "out")
            code, out, err, report = self._run(td, ["--all", "--out-dir", out_dir],
                                               _fake_pynvml())
            left = os.listdir(out_dir)
        self.assertEqual(code, 1, err)
        self.assertEqual(left, [])                        # each output removed again
        ok = [n for n in vb.VARIANTS if n not in ("new_dp16", "new_p010")]
        for name in ok:
            self.assertRegex(out, rf"v100_bench: {name}: \d+\.\d\d fps")
            self.assertRegex(report, rf"\| `{name}` \| \d+\.\d\d \| \d+\.\d \|")
        # a switch the conversion never reached is a failure, not a quiet no-op
        self.assertIn("new_dp16: the harness switch dp_block=16 was never reached", err)
        self.assertIn("new_p010: decode did not run on NVDEC as 'p010le'", err)
        self.assertIn("the harness switch decode='p010le' was never reached", err)
        self.assertEqual(report.count("| failed |"), 2)
        self.assertIn("## Failures", report)
        self.assertIn("profile mode `sync`", report)
        self.assertIn(f"source digest `{vb.code_id()}`", report)
        self.assertIn("| card memory in use before the first variant | 551 MiB |", report)
        self.assertRegex(report, r"\| `new` \|.*\| 551 \| n/a / n/a \| \d+ \| 80 / 80 \| "
                                 r"30 / 30 \| 40 / 40 \| 35 / 35 \|")

    def test_idle_engines_and_unreadable_counters_fail_the_run(self):
        fake = _fake_pynvml(enc=0.0)

        def not_supported(h):
            raise fake.NVMLError("Not Supported")
        fake.nvmlDeviceGetDecoderUtilization = not_supported
        with tempfile.TemporaryDirectory() as td:
            code, _out, err, report = self._run(td, ["--variants", "new", "--out-dir", td], fake)
        self.assertEqual(code, 1)
        self.assertRegex(err, r"new: NVENC utilisation was 0 % in all \d+ samples")
        self.assertIn("NVML could not read 'dec' on this driver/container "
                      "(NVMLError: Not Supported)", err)
        self.assertIn("| `new` | failed |", report)

    def test_a_clean_run_exits_zero_and_keeps_files_when_asked(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = os.path.join(td, "out")
            code, out, err, report = self._run(
                td, ["--variants", "baseline", "new", "--keep", "--out-dir", out_dir],
                _fake_pynvml())
            left = sorted(os.listdir(out_dir))
        self.assertEqual(code, 0, err)
        self.assertEqual(left, ["v100_bench_baseline.mp4", "v100_bench_new.mp4"])
        self.assertNotIn("## Failures", report)
        self.assertIn("converted files kept in", out)


class TestReport(unittest.TestCase):
    HEAD = {"gpu": "Tesla V100-PCIE-16GB", "uuid": "GPU-x", "driver": "580.178.04",
            "clip": "/clip.OSV", "frames": 450, "start": 0, "code": "`abc1234`",
            "when": "2026-11-14 10:00", "idle_mib": 551.0}
    PRE = {"decode": "cuda", "capability": "7.0", "total_mib": 16384, "torch": "2.14.1+cu126",
           "torch_cuda": "12.6", "arch_list": ["sm_60", "sm_70"], "pyav": "19.0.2",
           "libavcodec": "62.11.100", "python": "3.12.15", "cpus": 40, "affinity": 6}

    def test_stats(self):
        s = [{"sm": 10.0}, {"sm": 30.0}, {"sm": 20.0}, {"mem": 1.0}]
        st = vb._stats(s, "sm")
        self.assertEqual((st["mean"], st["p50"], st["max"], st["n"]), (20.0, 20.0, 30.0, 3))
        self.assertIsNone(vb._stats(s, "dec"))

    def test_report_from_a_real_profile(self):
        res = _real_result(seam_engine="gpu")
        res["memory"].update(gpu_peak_alloc_mib=6000.0, gpu_peak_reserved_mib=7000.0)
        res["switches"] = [["dp_block", 16]]              # as it comes back through JSON
        res = json.loads(json.dumps(res))
        p = res["profile"]
        samples = [{"mem": 9000.0 + i, "sm": 80.0, "enc": 30.0, "membus": 35.0}
                   for i in range(4)]
        rows = [{"name": "new_dp16", "res": res, "samples": samples, "tail": ""},
                {"name": "new", "res": {"error": "NVDEC utilisation was 0 %"}, "samples": [],
                 "tail": "Traceback ...\nRuntimeError: boom\n"}]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "report.md"
            vb.write_report(path, dict(self.HEAD, idle_mib=None), self.PRE, rows,
                            ["new: NVDEC utilisation was 0 %"])
            text = path.read_text()
        rss = res["memory"]["peak_rss_mib"]
        self.assertIn(f"| `new_dp16` | {p['fps']:.2f} | {p['ms_per_frame']:.1f} | "
                      f"{p['cpu_cores']:.2f} | {p['setup_s']:.1f} | 9003 | 6000 / 7000 | "
                      f"{rss:.0f} | 80 / 80 | 30 / 30 | unreadable | 35 / 35 |", text)
        self.assertIn("| card memory in use before the first variant | n/a MiB |", text)
        self.assertIn("4 frames (1 warm-up left out), profile mode `events`", text)
        self.assertIn("Harness switches the conversion ran with: `dp_block=16`.", text)
        self.assertIn(f"| `main.stitch` | {p['stages']['main.stitch']['wall']:.2f} |", text)
        self.assertIn('"seam_engine": "gpu"', text)
        self.assertIn("| `new` | failed |", text)
        self.assertIn("## Failures", text)
        self.assertIn("RuntimeError: boom", text)
        self.assertIn("kernels for sm_60 sm_70", text)


class TestCommandLine(unittest.TestCase):
    def test_code_id(self):
        code, out, _err = _run_main(["--code-id"], None)
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), vb.code_id())
        self.assertRegex(vb.code_id(), r"^[0-9a-f]{12}$")

    def test_bad_arguments_are_refused_before_anything_runs(self):
        for argv in (["clip.OSV"], ["--report", "r.md"],
                     ["clip.OSV", "--report", "r.md", "--variants", "new", "newish"],
                     ["clip.OSV", "--report", "r.md", "--frames", "0"],
                     ["clip.OSV", "--report", "r.md", "--timeout", "0"]):
            code, _out, err = _run_main(argv, _fake_pynvml())
            self.assertEqual(code, 2, argv)
            self.assertIn("error:", err)

    def test_folders(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            self.assertEqual(vb.folder_problems(td / "a" / "report.md", td / "b", 450, 1), [])
            self.assertTrue((td / "a").is_dir() and (td / "b").is_dir())
            (td / "file").write_bytes(b"")
            got = vb.folder_problems(td / "file" / "report.md", td / "b", 450, 1)
            self.assertEqual(len(got), 1)
            self.assertIn("report folder", got[0])
            got = vb.folder_problems(td / "report.md", td / "b", 10 ** 9, 10)
            self.assertEqual(len(got), 1)
            self.assertIn("MiB free", got[0])


if __name__ == "__main__":
    unittest.main()
