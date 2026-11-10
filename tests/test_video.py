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
