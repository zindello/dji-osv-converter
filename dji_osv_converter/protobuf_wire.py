"""Minimal protobuf wire-format reader.

DJI's OSV metadata streams are protobuf encoded against a schema
(dvtm_oq101.proto) that is not published. The wire format is self-describing
enough to walk without it: every field carries a number and a wire type.

This decodes structure only - field *names* are not recoverable from the wire.
"""

from __future__ import annotations

import struct
from typing import Any, Iterator

VARINT, I64, LEN, SGROUP, EGROUP, I32 = 0, 1, 2, 3, 4, 5


def read_varint(buf: bytes, i: int) -> tuple[int, int]:
    val = shift = 0
    while i < len(buf):
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")
    raise ValueError("truncated varint")


def iter_fields(buf: bytes) -> Iterator[tuple[int, int, Any]]:
    """Yield (field_number, wire_type, raw_value) for a protobuf message."""
    i = 0
    while i < len(buf):
        key, i = read_varint(buf, i)
        fnum, wt = key >> 3, key & 7
        if fnum == 0:
            raise ValueError("field number 0 is invalid")
        if wt == VARINT:
            v, i = read_varint(buf, i)
        elif wt == I64:
            if i + 8 > len(buf):
                raise ValueError("truncated i64")
            v = buf[i:i + 8]; i += 8
        elif wt == LEN:
            n, i = read_varint(buf, i)
            if i + n > len(buf):
                raise ValueError("truncated length-delimited field")
            v = buf[i:i + n]; i += n
        elif wt == I32:
            if i + 4 > len(buf):
                raise ValueError("truncated i32")
            v = buf[i:i + 4]; i += 4
        elif wt in (SGROUP, EGROUP):
            v = None
        else:
            raise ValueError(f"bad wire type {wt}")
        yield fnum, wt, v


def looks_like_message(b: bytes) -> bool:
    """Heuristic: does this length-delimited blob parse cleanly as a message?

    Deliberately strict - random float payloads will often parse as *some*
    sequence of fields, so we also require plausible field numbers. Being
    wrong here only costs display fidelity, not correctness.
    """
    if len(b) < 2:
        return False
    try:
        fields = list(iter_fields(b))
    except Exception:
        return False
    if not fields:
        return False
    return all(0 < fnum < 512 for fnum, _, _ in fields)


def looks_like_text(b: bytes) -> bool:
    return bool(b) and all(32 <= c < 127 or c in (9, 10, 13) for c in b)


def dump(buf: bytes, depth: int = 0, path: str = "", max_depth: int = 8,
         out: list | None = None) -> list[str]:
    """Recursively render a protobuf message as indented text."""
    if out is None:
        out = []
    pad = "  " * depth
    try:
        fields = list(iter_fields(buf))
    except Exception as e:
        out.append(f"{pad}<unparseable: {e}>")
        return out

    for fnum, wt, v in fields:
        p = f"{path}.{fnum}" if path else str(fnum)
        if wt == VARINT:
            out.append(f"{pad}[{p}] varint = {v}")
        elif wt == I32:
            f = struct.unpack("<f", v)[0]
            u = struct.unpack("<I", v)[0]
            out.append(f"{pad}[{p}] i32 = {u}  (f32 {f:.6g})")
        elif wt == I64:
            d = struct.unpack("<d", v)[0]
            u = struct.unpack("<Q", v)[0]
            out.append(f"{pad}[{p}] i64 = {u}  (f64 {d:.6g})")
        elif wt == LEN:
            if looks_like_text(v) and len(v) > 1:
                out.append(f"{pad}[{p}] str({len(v)}) = {v.decode(errors='replace')!r}")
            elif depth < max_depth and looks_like_message(v):
                out.append(f"{pad}[{p}] msg({len(v)}) {{")
                dump(v, depth + 1, p, max_depth, out)
                out.append(f"{pad}}}")
            else:
                extra = ""
                if len(v) % 4 == 0 and 4 <= len(v) <= 128:
                    fl = struct.unpack(f"<{len(v)//4}f", v)
                    if all(abs(x) < 1e6 for x in fl):
                        extra = "  f32[] " + " ".join(f"{x:.4g}" for x in fl[:12])
                out.append(f"{pad}[{p}] bytes({len(v)}) {v[:24].hex(' ')}{extra}")
    return out


__all__ = ["iter_fields", "dump", "read_varint"]
