#!/usr/bin/env python3
"""PoC: one DJI Osmo 360 .OSV frame -> one equirectangular 360 image.

Everything about the FORMAT is read in pure Python (no ffmpeg/libav): the MP4
box tree, the DJI metadata track, the per-unit lens calibration, and the
per-frame orientation. Only the two HEVC fisheye frames are decoded with a
codec - libavcodec via PyAV, linked in-process, never the ffmpeg binary.

    python examples/poc_stitch_frame.py INPUT.OSV [frame] [out.png]

Pipeline:
    1. parse calibration (fx,fy,cx,cy, 5 radial + 2 tangential terms, extrinsic
       quaternion) from the CAM-meta track header sample
    2. parse this frame's gravity vector from its CAM-meta sample
    3. decode the two fisheye frames (libavcodec)
    4. project each through DJI's f-theta lens model into equirectangular space,
       levelled to gravity
    5. exposure-match the two lenses and multi-band blend the seam
    6. write a 7680x3840 equirectangular PNG
"""

import struct
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- format: pure Python, no libav ---
from dji_osv_converter.osv_mp4 import OSVContainer
from dji_osv_converter.protobuf_wire import iter_fields
from dji_osv_converter.osv_calibration import _parse_block, _descend
# --- geometry / blend ---
from dji_osv_converter.project import build_map
from dji_osv_converter.stitcher import warp, match_exposure, multiband_blend, seam_weights
from dji_osv_converter.pipeline import levelling_rotation
# --- pixels: libavcodec in-process (the one codec dependency) ---
from dji_osv_converter.decode import OSVDecoder

OUT_W, OUT_H = 7680, 3840
FOV_DEG = 193.68            # 180 + DJI's seam_overlap_ratio(0.038) * 360


def read_calibration(header: bytes):
    """Two CameraCalib from the CAM-meta header sample, matching 3840x3840."""
    block = _descend(header, 2, 6)
    cams = []
    for _f, w, v in iter_fields(block):
        if w != 2:
            continue
        c = _parse_block(v)
        if c and c.fx > 1 and (c.width, c.height) == (3840, 3840):
            cams.append(c)
    return cams[:2]


def _gravity_one(sample: bytes):
    block = _descend(sample, 3, 2, 10)
    if block is None:
        return None
    vals = {f: struct.unpack("<f", v)[0] for f, w, v in iter_fields(block) if w == 5}
    if {2, 3, 4} <= set(vals):
        return np.array([vals[2], vals[3], vals[4]], dtype=np.float64)
    return None


def read_gravity(container, track, frame: int, window: int = 15):
    """Gravity for `frame`, averaged over a window of neighbouring frames.

    The raw accelerometer sample carries handheld motion as well as gravity;
    on a single frame that is a couple of degrees of tilt noise, which reads as
    a horizon that rises and falls as you pan a 360. Averaging a short window
    around the frame recovers the true 'down'.
    """
    n = len(track.sample_offsets())
    lo, hi = max(0, frame - window // 2), min(n, frame + window // 2 + 1)
    samples = [_gravity_one(container.sample_bytes(track, i)) for i in range(lo, hi)]
    samples = [g for g in samples if g is not None]
    if not samples:
        return None
    return np.mean(samples, axis=0)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    osv = sys.argv[1]
    frame = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = sys.argv[3] if len(sys.argv) > 3 else "frame.png"

    # 1-2: format + metadata, pure Python
    container = OSVContainer(osv)
    meta = container.find_track(name="CAM meta")
    if meta is None:
        raise SystemExit("no CAM meta track - not a DJI 360 OSV?")
    cams = read_calibration(container.sample_bytes(meta, 0))
    gravity = read_gravity(container, meta, frame)
    print(f"calibration: fx={cams[0].fx:.1f}/{cams[1].fx:.1f}  "
          f"5 radial + 2 tangential terms per lens")
    R = levelling_rotation(gravity) if gravity is not None else np.eye(3)
    print(f"gravity {np.round(gravity, 3) if gravity is not None else None} -> horizon levelled")

    # 3: pixels, libavcodec
    with OSVDecoder(osv) as dec:
        f0, f1 = dec.get_pair(frame)
    print(f"decoded fisheye pair {f0.shape} via libavcodec (no ffmpeg binary)")

    # 4: project each lens with DJI's f-theta model, levelled to gravity
    maps = [build_map(c, OUT_W, OUT_H, source="quat", order="wxyz",
                      max_fov_deg=FOV_DEG, model="ftheta", R_extra=R) for c in cams]

    # 5: exposure match + multi-band blend
    warped = [warp(f0, maps[0]), warp(f1, maps[1])]
    warped = match_exposure(warped, maps, spatial=False)
    weights = seam_weights(maps, band_px=0.038 * OUT_W)
    pano = multiband_blend(warped, maps, weights=weights)

    # 6: write
    cv2.imwrite(out, pano)
    print(f"wrote {out}  ({pano.shape[1]}x{pano.shape[0]} equirectangular)")
    print("note: tag as 360 with spherical metadata when muxing to video "
          "(projection=equirectangular)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
