"""Spherical-metadata injection: streaming edit vs the whole-file reference.

The streaming path must produce byte-identical files to the original
in-memory edit for both box orders (moov after mdat - libav's default - and
moov before mdat, ``faststart``), the result must still demux and decode,
and its peak memory must not scale with the file (a long 8K master is tens
of GB; the old edit held three copies of it).
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dji_osv_converter.spherical import (_box, _boxes, _inject_spherical_inmemory,
                                         _walk_leaf, inject_spherical)


def _mux_small_mp4(path: str, faststart: bool, frames: int = 5) -> None:
    import av
    opts = {"movflags": "faststart"} if faststart else None
    out = av.open(path, "w", options=opts) if opts else av.open(path, "w")
    vs = out.add_stream("mpeg4", rate=30)
    vs.width = vs.height = 32
    vs.pix_fmt = "yuv420p"
    for i in range(frames):
        fr = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 30 * i, np.uint8), format="rgb24")
        fr.pts = i
        for pkt in vs.encode(fr):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    out.close()


def _top(path: str):
    data = Path(path).read_bytes()
    return [(off, size, k) for off, size, _, k in _boxes(data, 0, len(data))]


def _has_spherical(path: str) -> tuple[bool, bool]:
    data = Path(path).read_bytes()
    return (b"sv3d" in data, b"GSpherical:Spherical" in data)


def _decode_means(path: str) -> list[float]:
    import av
    with av.open(path) as c:
        return [float(fr.to_ndarray(format="gray").mean()) for fr in c.decode(video=0)]


def _synthetic_mp4(path: str, mdat_bytes: int, faststart: bool) -> None:
    """A minimal MP4 box tree around a SPARSE mdat (no real media)."""
    hdlr = _box(b"hdlr", struct.pack(">II4s", 0, 0, b"vide") + b"\0" * 12 + b"VideoHandler\0")
    entry = _box(b"hvc1", b"\0" * 78 + _box(b"hvcC", b"\x01" * 23))
    stsd = _box(b"stsd", struct.pack(">II", 0, 1) + entry)
    stco = _box(b"stco", struct.pack(">II", 0, 2) + struct.pack(">II", 100, 200))
    stbl = _box(b"stbl", stsd + stco)
    minf = _box(b"minf", stbl)
    mdia = _box(b"mdia", hdlr + minf)
    trak = _box(b"trak", mdia)
    moov = _box(b"moov", trak)
    ftyp = _box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomiso2mp41")
    mdat_hdr = struct.pack(">I4s", 8 + mdat_bytes, b"mdat")
    with open(path, "wb") as f:
        f.write(ftyp)
        if faststart:
            f.write(moov)
        f.write(mdat_hdr)
        f.truncate(f.tell() + mdat_bytes)       # sparse: no blocks written
        f.seek(0, os.SEEK_END)
        if not faststart:
            f.write(moov)


class TestStreamingInjection(unittest.TestCase):
    def _pair(self, faststart: bool):
        td = tempfile.mkdtemp()
        a, b = os.path.join(td, "a.mp4"), os.path.join(td, "b.mp4")
        _mux_small_mp4(a, faststart)
        Path(b).write_bytes(Path(a).read_bytes())
        return a, b

    def test_identical_to_inmemory_moov_last(self):
        a, b = self._pair(faststart=False)
        self.assertEqual([k for _, _, k in _top(a)][-1], b"moov")
        before = _decode_means(a)
        _inject_spherical_inmemory(a)
        inject_spherical(b)                      # streaming default
        self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes())
        self.assertEqual(_has_spherical(b), (True, True))
        self.assertEqual(_decode_means(b), before)

    def test_identical_to_inmemory_faststart(self):
        a, b = self._pair(faststart=True)
        kinds = [k for _, _, k in _top(a)]
        self.assertLess(kinds.index(b"moov"), kinds.index(b"mdat"))
        before = _decode_means(a)
        _inject_spherical_inmemory(a)
        inject_spherical(b)
        self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes())
        self.assertEqual(_has_spherical(b), (True, True))
        # the media moved: chunk offsets were shifted, so decode still works
        self.assertEqual(_decode_means(b), before)
        self.assertFalse(os.path.exists(b + ".spherical.tmp"))

    def test_idempotent_size_growth(self):
        a, _ = self._pair(faststart=False)
        n0 = os.path.getsize(a)
        inject_spherical(a)
        grown = os.path.getsize(a) - n0
        self.assertGreater(grown, 0)
        # moov size field accounts for exactly the growth
        top = {k: size for _, size, k in _top(a)}
        self.assertEqual(sum(size for _, size, _ in _top(a)), os.path.getsize(a))
        self.assertIn(b"moov", top)

    def test_streaming_matches_inmemory_on_synthetic_faststart(self):
        td = tempfile.mkdtemp()
        a, b = os.path.join(td, "a.mp4"), os.path.join(td, "b.mp4")
        _synthetic_mp4(a, 4096, faststart=True)
        Path(b).write_bytes(Path(a).read_bytes())
        _inject_spherical_inmemory(a)
        inject_spherical(b)
        self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes())
        data = Path(b).read_bytes()
        stco = [(off, size, hdr) for off, size, hdr, k in _walk_leaf(data, 0, len(data))
                if k == b"stco"][0]
        vals = struct.unpack_from(">II", data, stco[0] + stco[2] + 8)
        self.assertEqual(vals[1] - vals[0], 100)
        self.assertGreater(vals[0], 100)       # shifted by the moov growth

    def test_peak_memory_is_bounded_by_the_moov(self):
        """A multi-GB (sparse) file must not raise the peak RSS: O(moov) memory."""
        td = tempfile.mkdtemp()
        big = os.path.join(td, "big.mp4")
        _synthetic_mp4(big, 3 << 30, faststart=False)        # 3 GB sparse mdat
        code = (
            "import resource, sys, json\n"
            "from dji_osv_converter.spherical import inject_spherical\n"
            "r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
            f"inject_spherical({big!r})\n"
            "r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
            "print(json.dumps([r0, r1]))\n"
        )
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent.parent))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=env, check=True).stdout.strip().splitlines()[-1]
        r0, r1 = json.loads(out)
        unit = 1 if sys.platform == "darwin" else 1024           # bytes vs KB
        growth_mb = (r1 - r0) * unit / 2**20
        self.assertLess(growth_mb, 64.0, f"peak RSS grew by {growth_mb:.0f} MB")
        self.assertEqual(_has_spherical(big), (True, True))
        self.assertEqual(os.path.getsize(big), sum(s for _, s, _ in _top(big)))


if __name__ == "__main__":
    unittest.main()
