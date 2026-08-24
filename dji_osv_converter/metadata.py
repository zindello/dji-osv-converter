"""Reading DJI OSV per-frame metadata (camera orientation, serial, model).

The OSV container carries two `djmd` data streams with exactly one packet per
video frame, protobuf-encoded against DJI's unpublished dvtm_oq101.proto.
Field numbers below were recovered by walking the wire format; names are ours.

Layout of the larger "CAM meta" stream (per frame):

    3.1.2               timestamp
    3.2.9  {1,2,3,4}    orientation quaternion for this frame  <-- what we want
    3.2.10 {2,3,4}      gravity / accelerometer vector
    3.3.2.1.3           repeated: same quaternion at IMU rate (~38 per frame)

The first packet of the stream additionally carries a header with the camera
serial number and model.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from .protobuf_wire import iter_fields

logger = logging.getLogger(__name__)

LEN, I32 = 2, 5


@dataclass
class OSVMeta:
    """Per-file metadata plus per-frame orientation."""

    serial: str | None
    model: str | None
    quats: np.ndarray      # (N,4) float64, raw component order as stored
    gravity: np.ndarray    # (N,3) float64

    def __len__(self) -> int:
        return len(self.quats)


def _sub(buf: bytes, *path: int) -> bytes | None:
    """Follow a chain of nested length-delimited fields."""
    cur = buf
    for want in path:
        found = None
        for fnum, wt, v in iter_fields(cur):
            if fnum == want and wt == LEN:
                found = v
                break
        if found is None:
            return None
        cur = found
    return cur


def _floats(buf: bytes, count: int) -> list[float] | None:
    """Read fixed32 float fields 1..count from a message."""
    out: dict[int, float] = {}
    try:
        for fnum, wt, v in iter_fields(buf):
            if wt == I32 and 1 <= fnum <= count:
                out[fnum] = struct.unpack("<f", v)[0]
    except Exception:
        return None
    if len(out) < count:
        return None
    return [out[i] for i in range(1, count + 1)]


def _header_strings(buf: bytes) -> tuple[str | None, str | None]:
    """Pull serial and model out of the stream header packet."""
    texts: list[str] = []

    def walk(b: bytes, depth: int = 0):
        if depth > 6:
            return
        try:
            fields = list(iter_fields(b))
        except Exception:
            return
        for _fn, wt, v in fields:
            if wt != LEN or not v:
                continue
            if all(32 <= c < 127 for c in v) and len(v) >= 4:
                texts.append(v.decode())
            else:
                walk(v, depth + 1)

    walk(buf)
    serial = next((t for t in texts if len(t) >= 12 and t[:2].isalnum()
                   and any(ch.isdigit() for ch in t) and " " not in t
                   and not t.endswith(".proto")), None)
    model = next((t for t in texts if "Osmo" in t), None)
    return serial, model


def read_metadata(path: str | Path, max_frames: int | None = None) -> OSVMeta:
    """Extract per-frame orientation quaternions from an OSV file."""
    container = av.open(str(path))
    data_streams = [s for s in container.streams if s.type == "data"]
    if not data_streams:
        raise ValueError(f"{path}: no data streams found")

    # Pick the stream by content rather than by tag or size: the right one is
    # whichever actually yields unit quaternions. Self-validating, and immune
    # to DJI renumbering streams between firmware versions.
    target = None
    for s in data_streams:
        probe = av.open(str(path))
        hits = 0
        try:
            for n, pkt in enumerate(probe.demux([probe.streams[s.index]])):
                if n > 6:
                    break
                if not pkt.size:
                    continue
                q = _sub(bytes(pkt), 3, 2, 9)
                f = _floats(q, 4) if q else None
                if f and abs(np.linalg.norm(f) - 1.0) < 1e-3:
                    hits += 1
        except Exception:
            pass
        finally:
            probe.close()
        if hits >= 2:
            target = s
            break
    if target is None:
        raise ValueError(f"{path}: no data stream contains orientation quaternions")
    logger.info("reading orientation from data stream %d", target.index)

    quats: list[list[float]] = []
    grav: list[list[float]] = []
    serial = model = None

    for i, packet in enumerate(container.demux([target])):
        if not packet.size:
            continue
        buf = bytes(packet)
        body = _sub(buf, 3)
        if body is None:
            continue
        if serial is None:
            serial, model = _header_strings(buf)

        block = _sub(body, 2)
        if block is None:
            continue
        # Append quaternion and gravity together, keeping the two tracks in
        # lockstep: a frame missing either gets NaNs in that slot rather than
        # shifting every later frame's index (which would silently mis-pair
        # orientation with the wrong frame).
        q = _sub(block, 9)
        f = _floats(q, 4) if q is not None else None
        quats.append(f if f else [np.nan] * 4)

        g = _sub(block, 10)
        gv: dict[int, float] = {}
        if g is not None:
            for fnum, wt, v in iter_fields(g):
                if wt == I32 and 2 <= fnum <= 4:
                    gv[fnum] = struct.unpack("<f", v)[0]
        grav.append([gv[2], gv[3], gv[4]] if len(gv) == 3 else [np.nan] * 3)

        if max_frames and len(quats) >= max_frames:
            break

    container.close()
    q = np.asarray(quats, dtype=np.float64)
    if len(q):
        norms = np.linalg.norm(q, axis=1)
        bad = np.abs(norms - 1.0) > 1e-3
        if bad.any():
            logger.warning("%d/%d quaternions are not unit length (max dev %.4g)",
                           bad.sum(), len(q), np.abs(norms - 1).max())
    return OSVMeta(serial, model, q, np.asarray(grav, dtype=np.float64))


def read_highrate(path: str | Path) -> list[np.ndarray | None]:
    """Per-frame high-rate IMU orientation quaternions (field 3.3.2.1.3).

    Each video frame's metadata sample carries ~33 fused orientation
    quaternions spanning the frame period (~1 kHz; sample 0 coincides with the
    per-frame quaternion 3.2.9 at frame START). Returns one (S, 4) array per
    frame, aligned with ``read_metadata``'s frame order, None where absent.
    The stream is found by content (the one whose samples carry 3.3.2.1),
    matching read_metadata's self-validating stream selection.
    """
    container = av.open(str(path))
    data_streams = [s.index for s in container.streams if s.type == "data"]
    container.close()

    target = None
    for si in data_streams:
        probe = av.open(str(path))
        try:
            for n, pkt in enumerate(probe.demux(probe.streams[si])):
                if n > 6:
                    break
                if pkt.size and _sub(bytes(pkt), 3, 3, 2, 1) is not None:
                    target = si
                    break
        except Exception:
            pass
        finally:
            probe.close()
        if target is not None:
            break
    if target is None:
        return []

    out: list[np.ndarray | None] = []
    probe = av.open(str(path))
    try:
        for pkt in probe.demux(probe.streams[target]):
            if not pkt.size:
                continue
            blk = _sub(bytes(pkt), 3, 3, 2, 1)
            hq = []
            if blk is not None:
                try:
                    for fnum, wt, v in iter_fields(blk):
                        if fnum == 3 and wt == LEN:
                            f = _floats(v, 4)
                            if f:
                                hq.append(f)
                except Exception:
                    pass
            out.append(np.asarray(hq, dtype=np.float64) if len(hq) >= 4 else None)
    finally:
        probe.close()
    return out


def quat_to_matrix(q: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """Unit quaternion -> 3x3 rotation matrix.

    `order` selects the stored component order; DJI's is determined empirically
    (see tools/, Phase 1) because the wire format does not label components.
    """
    q = np.asarray(q, dtype=np.float64)
    if order == "xyzw":
        x, y, z, w = q
    elif order == "wxyz":
        w, x, y, z = q
    else:
        raise ValueError(f"unknown quaternion order {order!r}")
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


__all__ = ["OSVMeta", "read_metadata", "read_highrate", "quat_to_matrix"]
