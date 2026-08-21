"""Tag an MP4 as a 360 equirectangular video (spherical metadata).

Players (YouTube, VLC, headset apps) only treat a file as 360 if it carries
spherical-video metadata. PyAV/libav in-process cannot write it, so this
injects it after muxing - purely by editing the MP4 box tree, no re-encode.

Two schemes are written, matching what DJI Studio's own export contains:

* **V2** - an ``sv3d`` box inside the video sample entry (the current standard,
  ``sv3d > proj > prhd + equi``).
* **V1** - a ``uuid`` box of Google Spatial-Media RDF/XML in the video track
  (the legacy scheme, still what many players look for first).

Only the ``moov`` box grows; the media (``mdat``) is untouched. When ``moov``
follows ``mdat`` (libav's default) sample offsets stay valid as-is; when it
precedes ``mdat`` the chunk-offset tables (stco/co64) are shifted to match.
"""

from __future__ import annotations

import struct
from pathlib import Path

_SPHERICAL_UUID = bytes.fromhex("ffcc8263f8554a938814587a02521fdc")

_V1_XML = (
    '<?xml version="1.0"?>'
    '<rdf:SphericalVideo xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    'xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/">'
    "<GSpherical:Spherical>true</GSpherical:Spherical>"
    "<GSpherical:Stitched>true</GSpherical:Stitched>"
    "<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>"
    "<GSpherical:StitchingSoftware>dji_osv_converter</GSpherical:StitchingSoftware>"
    "</rdf:SphericalVideo>"
)

# Visual sample entry header length before its child boxes: 8 box header is
# handled separately; this is the 78-byte VisualSampleEntry body.
_VISUAL_ENTRY_BODY = 78


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def _sv3d_box() -> bytes:
    svhd = _box(b"svhd", b"dji_osv_converter\x00")
    prhd = _box(b"prhd", struct.pack(">Iiii", 0, 0, 0, 0))       # version/flags + pose 0
    equi = _box(b"equi", struct.pack(">IIIII", 0, 0, 0, 0, 0))   # full sphere, bounds 0
    return _box(b"sv3d", svhd + _box(b"proj", prhd + equi))


def _uuid_box() -> bytes:
    return _box(b"uuid", _SPHERICAL_UUID + _V1_XML.encode("utf-8"))


def _boxes(buf, start, end):
    i = start
    while i + 8 <= end:
        size = struct.unpack_from(">I", buf, i)[0]
        hdr = 8
        if size == 1:
            size = struct.unpack_from(">Q", buf, i + 8)[0]
            hdr = 16
        elif size == 0:
            size = end - i
        yield i, size, hdr, bytes(buf[i + 4:i + 8])
        i += size


def _child(buf, box_off, box_size, box_hdr, kind, body_skip=0):
    """First child box of `kind` within a container box. Returns (off,size,hdr)."""
    for off, size, hdr, k in _boxes(buf, box_off + box_hdr + body_skip, box_off + box_size):
        if k == kind:
            return off, size, hdr
    return None


def _grow_size(buf, box_off, amount):
    """Add `amount` to a 32-bit box size field (large-size boxes not expected here)."""
    old = struct.unpack_from(">I", buf, box_off)[0]
    if old == 1:
        raise ValueError("cannot grow a 64-bit-size box in place")
    struct.pack_into(">I", buf, box_off, old + amount)


def inject_spherical(path: str | Path) -> None:
    """Add V1 (uuid) and V2 (sv3d) equirectangular metadata to an MP4 in place."""
    p = Path(path)
    buf = bytearray(p.read_bytes())
    n = len(buf)

    top = {k: (off, size, hdr) for off, size, hdr, k in _boxes(buf, 0, n)}
    if b"moov" not in top:
        raise ValueError(f"{path}: no moov box")
    moov = top[b"moov"]
    mdat_off = top[b"mdat"][0] if b"mdat" in top else None

    # video trak = the trak whose mdia/hdlr handler_type is 'vide'
    trak = None
    for off, size, hdr, k in _boxes(buf, moov[0] + moov[2], moov[0] + moov[1]):
        if k != b"trak":
            continue
        mdia = _child(buf, off, size, hdr, b"mdia")
        hdlr = mdia and _child(buf, *mdia, b"hdlr")
        # hdlr: box(8) + version/flags(4) + pre_defined(4) + handler_type(4)
        if hdlr and buf[hdlr[0] + 16:hdlr[0] + 20] == b"vide":
            trak = (off, size, hdr)
            break
    if trak is None:
        raise ValueError(f"{path}: no video trak")

    mdia = _child(buf, *trak, b"mdia")
    minf = _child(buf, *mdia, b"minf")
    stbl = _child(buf, *minf, b"stbl")
    stsd = _child(buf, *stbl, b"stsd")
    if not stsd:
        raise ValueError(f"{path}: no stsd")
    # stsd body: 4 (version/flags) + 4 (entry_count), then the sample entry
    entry_off = stsd[0] + stsd[2] + 8
    entry_size = struct.unpack_from(">I", buf, entry_off)[0]

    sv3d, uuid = _sv3d_box(), _uuid_box()
    s, u = len(sv3d), len(uuid)

    # Boxes whose *size field* sits before both insertion points (so their
    # recorded offsets stay valid through the edits): grow them directly.
    _grow_size(buf, moov[0], s + u)
    _grow_size(buf, trak[0], s + u)
    for box in (mdia, minf, stbl, stsd):
        _grow_size(buf, box[0], s)
    _grow_size(buf, entry_off, s)

    # Insert higher offset first so the lower insertion offset stays valid.
    trak_end = trak[0] + trak[1]           # original end of trak
    entry_end = entry_off + entry_size     # original end of sample entry
    buf[trak_end:trak_end] = uuid          # V1, end of trak
    buf[entry_end:entry_end] = sv3d        # V2, end of sample entry

    if mdat_off is not None and moov[0] < mdat_off:
        _shift_chunk_offsets(buf, s + u)

    p.write_bytes(bytes(buf))


def _shift_chunk_offsets(buf, delta):
    """Add `delta` to every stco/co64 entry (moov precedes mdat, so it moved)."""
    n = len(buf)
    for off, size, hdr, k in _walk_leaf(buf, 0, n):
        if k == b"stco":
            cnt = struct.unpack_from(">I", buf, off + hdr + 4)[0]
            base = off + hdr + 8
            for j in range(cnt):
                v = struct.unpack_from(">I", buf, base + 4 * j)[0]
                struct.pack_into(">I", buf, base + 4 * j, v + delta)
        elif k == b"co64":
            cnt = struct.unpack_from(">I", buf, off + hdr + 4)[0]
            base = off + hdr + 8
            for j in range(cnt):
                v = struct.unpack_from(">Q", buf, base + 8 * j)[0]
                struct.pack_into(">Q", buf, base + 8 * j, v + delta)


_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def _walk_leaf(buf, start, end):
    for off, size, hdr, k in _boxes(buf, start, end):
        if k in _CONTAINERS:
            yield from _walk_leaf(buf, off + hdr, off + size)
        else:
            yield off, size, hdr, k


__all__ = ["inject_spherical"]
