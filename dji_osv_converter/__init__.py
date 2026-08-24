"""DJI Osmo 360 ``.OSV`` -> equirectangular 360 toolkit.

A DJI Osmo 360 records two raw fisheye streams plus metadata in a proprietary
``.OSV`` container. This library reads that format and stitches it into a
single equirectangular 360 image or video, deriving *everything* from the file
itself - the per-unit lens calibration, the f-theta lens model, the per-frame
orientation, and the camera<->IMU mount rotation - so it generalises across
camera units rather than being tuned to one. No DJI code is called; only the
HEVC pixels are decoded, with libavcodec in-process (never the ffmpeg binary).

Quick start - one frame::

    from dji_osv_converter import Stitcher360, OSVDecoder

    stitcher = Stitcher360.from_file("CAM_....OSV")
    with OSVDecoder("CAM_....OSV") as dec:
        f0, f1 = dec.get_pair(0)
    pano = stitcher.stitch(f0, f1, frame=0)   # 7680x3840 equirectangular BGR

The building blocks are also public, so you can assemble your own pipeline:

    format / container   OSVContainer          (pure-Python MP4 + metadata)
    pixels               OSVDecoder            (libavcodec, in-process)
    calibration          read_osv_calibration, read_mount_rotation
    orientation          read_metadata, levelling_rotation
    projection           build_map             (fisheye -> equirect remap)
    blending             warp, blend, multiband_blend, match_exposure,
                         seam_weights

See ``examples/poc_stitch_frame.py`` for a self-contained single-frame build
and ``docs/ROADMAP.md`` for status.
"""

__version__ = "0.2.0-dev"

from .calibration import CameraMap, CalibrationError
from .osv_calibration import (
    CameraCalib,
    read_osv_calibration,
    read_mount_rotation,
    parse_mount_rotation,
    read_header_sample,
)
from .osv_mp4 import OSVContainer, Track
from .decode import OSVDecoder, read_pair
from .metadata import OSVMeta, read_metadata
from .project import build_map
from .pipeline import Stitcher360, levelling_rotation, up_track_from_meta
from .video import convert
from .spherical import inject_spherical
from .stitcher import (
    warp,
    blend,
    flow_blend,
    multiband_blend,
    match_exposure,
    seam_weights,
    stitch,
    coverage_report,
)

__all__ = [
    # high level
    "convert",
    "Stitcher360",
    "OSVDecoder",
    "OSVContainer",
    "inject_spherical",
    # calibration / metadata
    "read_osv_calibration",
    "read_mount_rotation",
    "parse_mount_rotation",
    "read_header_sample",
    "read_metadata",
    "CameraCalib",
    "OSVMeta",
    # geometry / blending
    "build_map",
    "levelling_rotation",
    "up_track_from_meta",
    "warp",
    "blend",
    "flow_blend",
    "multiband_blend",
    "match_exposure",
    "seam_weights",
    "stitch",
    "coverage_report",
    # types
    "CameraMap",
    "CalibrationError",
    "Track",
    "read_pair",
]
