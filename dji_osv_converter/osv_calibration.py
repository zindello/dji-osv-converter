"""Per-unit camera calibration embedded in the OSV file.

DJI Studio does not ship calibration for every camera it supports - the 49
sets in Stitcher-Mac.bundle are DJI's own reference units. Each camera's
factory calibration travels inside its recordings instead, in the header
packet of the 'CAM meta' data stream.

Field layout (recovered from the wire format; names are ours):

    2.6.<n>         one calibration block per camera, 264 bytes
        .1  fx      focal length x, pixels
        .2  fy      focal length y, pixels
        .3  cx      principal point x
        .4  cy      principal point y
        .5-.8 k1..k4  equidistant-fisheye radial distortion
        .10 width   sensor width the calibration applies to
        .11 height
        .12 yaw     extrinsic orientation, degrees
        .13 pitch
        .14 roll
        .28 {1..4}  the same extrinsic rotation as a quaternion

Blocks 1 and 2 are the primary camera pair. Further pairs (11,12), (13,14)...
carry calibration for other recording modes; blocks are matched to the
recording by their declared width/height.

Note this is a different lens model from the JSON files DJI ships: those use
the unified/Mei model (fx~1777 plus an 'xi' term), these use the equidistant
fisheye model (fx~1044 with k1..k4). Both describe the same physical lens.
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
class CameraCalib:
    """Intrinsics and extrinsics for a single fisheye lens."""

    fx: float
    fy: float
    cx: float
    cy: float
    k: tuple[float, float, float, float]
    width: int
    height: int
    yaw: float
    pitch: float
    roll: float
    quat: tuple[float, float, float, float] | None = None

    def camera_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float64)

    def __str__(self) -> str:
        return (f"fx={self.fx:.2f} fy={self.fy:.2f} cx={self.cx:.2f} cy={self.cy:.2f} "
                f"k={[round(v, 5) for v in self.k]} {self.width}x{self.height} "
                f"ypr=({self.yaw:.3f},{self.pitch:.3f},{self.roll:.3f})")


def _fields(buf: bytes) -> dict:
    out = {}
    for fnum, wt, v in iter_fields(buf):
        out.setdefault(fnum, (wt, v))
    return out


def _f32(entry) -> float | None:
    if entry is None:
        return None
    wt, v = entry
    return struct.unpack("<f", v)[0] if wt == I32 else None


def _parse_block(buf: bytes) -> CameraCalib | None:
    f = _fields(buf)
    fx, fy = _f32(f.get(1)), _f32(f.get(2))
    cx, cy = _f32(f.get(3)), _f32(f.get(4))
    if None in (fx, fy, cx, cy):
        return None
    k = tuple(_f32(f.get(i)) or 0.0 for i in (5, 6, 7, 8))
    w = _f32(f.get(10)) or 0.0
    h = _f32(f.get(11)) or 0.0
    quat = None
    if 28 in f and f[28][0] == LEN:
        q = _fields(f[28][1])
        vals = [_f32(q.get(i)) for i in (1, 2, 3, 4)]
        if all(v is not None for v in vals):
            quat = tuple(vals)
    return CameraCalib(fx, fy, cx, cy, k, int(w), int(h),
                       _f32(f.get(12)) or 0.0, _f32(f.get(13)) or 0.0,
                       _f32(f.get(14)) or 0.0, quat)


def read_osv_calibration(path: str | Path,
                         want_size: tuple[int, int] = (3840, 3840),
                         ) -> list[CameraCalib]:
    """Read the camera pair's factory calibration from an OSV file.

    Returns [camera0, camera1] matching the requested sensor size.
    """
    container = av.open(str(path))
    header = None
    try:
        for s in container.streams:
            if s.type != "data":
                continue
            probe = av.open(str(path))
            try:
                for pkt in probe.demux([probe.streams[s.index]]):
                    if not pkt.size:
                        continue
                    buf = bytes(pkt)
                    blk = _descend(buf, 2, 6)
                    if blk is not None:
                        header = blk
                    break
            except Exception:
                pass
            finally:
                probe.close()
            if header is not None:
                break
    finally:
        container.close()

    if header is None:
        raise ValueError(f"{path}: no calibration block found in metadata header")

    blocks: list[tuple[int, CameraCalib]] = []
    for fnum, wt, v in iter_fields(header):
        if wt != LEN:
            continue
        cal = _parse_block(v)
        if cal and cal.fx > 1.0:
            blocks.append((fnum, cal))

    matching = [(n, c) for n, c in blocks if (c.width, c.height) == want_size]
    chosen = matching or blocks
    if not chosen:
        raise ValueError(f"{path}: no calibration blocks parsed")
    if len(chosen) < 2:
        raise ValueError(f"{path}: expected a camera pair, found {len(chosen)}")

    logger.info("found %d calibration blocks, %d matching %dx%d; using blocks %s",
                len(blocks), len(matching), *want_size, [chosen[0][0], chosen[1][0]])
    return [chosen[0][1], chosen[1][1]]


def _descend(buf: bytes, *path: int) -> bytes | None:
    cur = buf
    for want in path:
        nxt = None
        for fnum, wt, v in iter_fields(cur):
            if fnum == want and wt == LEN:
                nxt = v
                break
        if nxt is None:
            return None
        cur = nxt
    return cur


__all__ = ["CameraCalib", "read_osv_calibration"]
