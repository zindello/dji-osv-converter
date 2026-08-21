"""Minimal pure-Python MP4/ISO-BMFF reader for OSV metadata tracks.

The OSV is a plain MP4. Everything except the HEVC/AAC media - the container
structure and the DJI metadata tracks - can be read with no external library:
this walks the box tree, resolves a track's sample table (stsz/stsc/stco/co64),
and returns the raw sample bytes. The DJI metadata is protobuf inside those
samples (see protobuf_wire / osv_calibration / metadata).

Only decoding the HEVC pixels needs a codec; the format itself does not.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path


def _boxes(buf: memoryview, start: int, end: int):
    """Yield (type, payload_start, payload_end) for boxes in [start, end)."""
    i = start
    while i + 8 <= end:
        size = struct.unpack_from(">I", buf, i)[0]
        btype = bytes(buf[i + 4:i + 8]).decode("latin1")
        header = 8
        if size == 1:                       # 64-bit largesize
            size = struct.unpack_from(">Q", buf, i + 8)[0]
            header = 16
        elif size == 0:                     # to end of file
            size = end - i
        yield btype, i + header, i + size
        i += size


def _find(buf, start, end, path):
    """Descend a slash path like 'moov/trak' returning (start,end) of the last."""
    cur = (start, end)
    for want in path.split("/"):
        found = None
        for t, s, e in _boxes(buf, cur[0], cur[1]):
            if t == want:
                found = (s, e)
                break
        if found is None:
            return None
        cur = found
    return cur


@dataclass
class Track:
    handler: str = ""
    name: str = ""
    codec: str = ""
    sample_sizes: list[int] = field(default_factory=list)
    chunk_offsets: list[int] = field(default_factory=list)
    # stsc entries: (first_chunk, samples_per_chunk, desc_index)
    stsc: list[tuple[int, int, int]] = field(default_factory=list)

    def sample_offsets(self) -> list[tuple[int, int]]:
        """Resolve (file_offset, size) for every sample via the sample table."""
        # expand stsc to samples-per-chunk for each chunk
        n_chunks = len(self.chunk_offsets)
        spc = [0] * n_chunks
        for idx, (first, count, _desc) in enumerate(self.stsc):
            last = (self.stsc[idx + 1][0] - 1) if idx + 1 < len(self.stsc) else n_chunks
            for c in range(first - 1, last):
                if 0 <= c < n_chunks:
                    spc[c] = count
        out = []
        s = 0
        for c in range(n_chunks):
            off = self.chunk_offsets[c]
            for _ in range(spc[c]):
                if s >= len(self.sample_sizes):
                    break
                sz = self.sample_sizes[s]
                out.append((off, sz))
                off += sz
                s += 1
        return out


class OSVContainer:
    """Read-only view of an OSV's tracks and their sample bytes."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._data = Path(self.path).read_bytes()
        self.buf = memoryview(self._data)
        self.tracks: list[Track] = []
        self._parse()

    def _parse(self):
        n = len(self.buf)
        moov = _find(self.buf, 0, n, "moov")
        if not moov:
            raise ValueError(f"{self.path}: no moov box (not an MP4?)")
        for t, s, e in _boxes(self.buf, *moov):
            if t == "trak":
                tr = self._parse_trak(s, e)
                if tr:
                    self.tracks.append(tr)

    def _parse_trak(self, s, e) -> Track | None:
        tr = Track()
        mdia = _find(self.buf, s, e, "mdia")
        if not mdia:
            return None
        hdlr = _find(self.buf, *mdia, "hdlr")
        if hdlr:
            hs, he = hdlr
            tr.handler = bytes(self.buf[hs + 8:hs + 12]).decode("latin1")
            name = bytes(self.buf[hs + 24:he]).split(b"\x00")[0]
            tr.name = name.decode("latin1", "replace")
        stbl = _find(self.buf, *mdia, "minf/stbl")
        if not stbl:
            return None
        # codec: first child type of stsd's entry
        stsd = _find(self.buf, *stbl, "stsd")
        if stsd:
            for t2, s2, _e2 in _boxes(self.buf, stsd[0] + 8, stsd[1]):
                tr.codec = t2
                break
        for t2, s2, e2 in _boxes(self.buf, *stbl):
            if t2 == "stsz":
                sample_size = struct.unpack_from(">I", self.buf, s2 + 4)[0]
                count = struct.unpack_from(">I", self.buf, s2 + 8)[0]
                if sample_size:
                    tr.sample_sizes = [sample_size] * count
                else:
                    tr.sample_sizes = list(struct.unpack_from(f">{count}I", self.buf, s2 + 12))
            elif t2 == "stco":
                count = struct.unpack_from(">I", self.buf, s2 + 4)[0]
                tr.chunk_offsets = list(struct.unpack_from(f">{count}I", self.buf, s2 + 8))
            elif t2 == "co64":
                count = struct.unpack_from(">I", self.buf, s2 + 4)[0]
                tr.chunk_offsets = list(struct.unpack_from(f">{count}Q", self.buf, s2 + 8))
            elif t2 == "stsc":
                count = struct.unpack_from(">I", self.buf, s2 + 4)[0]
                tr.stsc = [tuple(struct.unpack_from(">3I", self.buf, s2 + 8 + 12 * i))
                           for i in range(count)]
        return tr

    def find_track(self, name: str | None = None, codec: str | None = None) -> Track | None:
        for tr in self.tracks:
            if name is not None and name in tr.name:
                return tr
            if codec is not None and tr.codec == codec:
                return tr
        return None

    def sample_bytes(self, track: Track, index: int) -> bytes:
        off, size = track.sample_offsets()[index]
        return bytes(self.buf[off:off + size])

    def all_samples(self, track: Track):
        for off, size in track.sample_offsets():
            yield bytes(self.buf[off:off + size])


__all__ = ["OSVContainer", "Track"]
