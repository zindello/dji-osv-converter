"""convert()'s muxing/encoding policy, testable without sample footage."""

import os
import tempfile
import unittest

import numpy as np

from dji_osv_converter import video
from dji_osv_converter.spherical import inject_spherical


def _mux_frames(path: str, n: int = 4) -> None:
    out = video._open_output(path)
    vs = out.add_stream("mpeg4", rate=30)
    vs.width = vs.height = 32
    vs.pix_fmt = "yuv420p"
    import av
    for i in range(n):
        fr = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 40 * i, np.uint8), format="rgb24")
        fr.pts = i
        for pkt in vs.encode(fr):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    out.close()


class TestOutputContainer(unittest.TestCase):
    def test_mux_to_path_without_mp4_extension(self):
        """A '.partial' temp name (or no extension) still yields a real MP4."""
        import av
        td = tempfile.mkdtemp()
        for name in ("clip.mp4.partial", "noext", "weird.bin"):
            p = os.path.join(td, name)
            _mux_frames(p)
            with av.open(p) as c:
                self.assertIn("mp4", c.format.name)
                frames = [fr.to_ndarray(format="gray").mean() for fr in c.decode(video=0)]
            self.assertEqual(len(frames), 4)
            inject_spherical(p)                    # the 360 tag works on it too
            with open(p, "rb") as f:
                self.assertIn(b"sv3d", f.read())


if __name__ == "__main__":
    unittest.main()


class TestEncoderPolicy(unittest.TestCase):
    def test_no_silent_software_fallback(self):
        from unittest import mock

        def only_x265(name, mode):
            if name != "libx265":
                raise RuntimeError("unknown codec")
            return object()

        with mock.patch.object(video.av.codec, "Codec", side_effect=only_x265):
            with self.assertRaises(RuntimeError):
                video._default_codec()
            self.assertEqual(video._default_codec(allow_software=True), "libx265")

    def test_hardware_encoder_preferred(self):
        from unittest import mock
        with mock.patch.object(video.av.codec, "Codec", side_effect=lambda n, m: object()):
            self.assertEqual(video._default_codec(), "hevc_videotoolbox")

    def test_encoder_options_nvenc_mapping(self):
        o = video.encoder_options("hevc_nvenc", preset="p6", tune="hq", aq=True,
                                  lookahead=16, multipass="qres", rc="vbr",
                                  extra={"aq-strength": 8})
        self.assertEqual(o, {"preset": "p6", "tune": "hq", "spatial-aq": "1",
                             "temporal-aq": "1", "rc-lookahead": "16",
                             "multipass": "qres", "rc": "vbr", "aq-strength": "8"})
        self.assertEqual(video.encoder_options("hevc_nvenc"), {})       # defaults untouched
        # NVENC knobs never leak onto other encoders
        self.assertEqual(video.encoder_options("hevc_videotoolbox", aq=True, preset="p6"), {})


class TestColourTags(unittest.TestCase):
    class _CC:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def test_source_colour_drops_unspecified(self):
        cc = self._CC(color_range=1, color_primaries=1, color_trc=1, colorspace=1)
        self.assertEqual(video._source_colour(cc),
                         {"color_range": 1, "color_primaries": 1, "color_trc": 1, "colorspace": 1})
        cc = self._CC(color_range=0, color_primaries=2, color_trc=2, colorspace=2)
        self.assertEqual(video._source_colour(cc), {})
        self.assertEqual(video._yuv_range(cc), "limited")
        self.assertEqual(video._yuv_range(self._CC(color_range=2)), "full")

    def test_tags_survive_muxing(self):
        import av
        td = tempfile.mkdtemp()
        p = os.path.join(td, "tagged.mp4")
        out = video._open_output(p)
        vs = out.add_stream("mpeg4", rate=30)
        vs.width = vs.height = 32
        vs.pix_fmt = "yuv420p"
        video._apply_colour(vs.codec_context, {"color_range": 1, "color_primaries": 1,
                                               "color_trc": 1, "colorspace": 1})
        for i in range(2):
            fr = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")
            fr.pts = i
            for pkt in vs.encode(fr):
                out.mux(pkt)
        for pkt in vs.encode():
            out.mux(pkt)
        out.close()
        with av.open(p) as c:
            self.assertEqual(video._source_colour(c.streams.video[0].codec_context),
                             {"color_range": 1, "color_primaries": 1, "color_trc": 1,
                              "colorspace": 1})
