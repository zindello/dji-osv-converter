"""DJI Osmo 360 ``.OSV`` -> equirectangular 360 toolkit.

A DJI Osmo 360 records two fisheye HEVC streams plus metadata in a ``.OSV``
container (a plain MP4 with protobuf metadata tracks). This library reads that
format and stitches it into an equirectangular 360 image or video, deriving
*everything possible* from the file itself - the per-unit lens calibration,
the f-theta lens model, the per-frame orientation, the camera<->IMU mount
rotation - so it generalises across camera units. The few remaining constants
(DJI's layout conventions) live in ``constants.py`` with their provenance.

No DJI code is called; only the HEVC pixels are decoded, with libavcodec
in-process (the ``ffmpeg`` binary is never invoked).

Convert a whole clip (matches DJI Studio's export, stabiliser included)::

    from dji_osv_converter import convert
    convert("CAM_....OSV", "out.mp4", levelling="rocksteady_horizon")

Or stitch a single frame::

    from dji_osv_converter import Stitcher360, OSVDecoder

    stitcher = Stitcher360.from_file("CAM_....OSV")
    with OSVDecoder("CAM_....OSV") as dec:
        f0, f1 = dec.get_pair(0)
    pano = stitcher.stitch(f0, f1, frame=0)   # equirectangular BGR

Building blocks (all public):

    calibration      read_osv_calibration, read_mount_rotation  (from the OSV)
    orientation      read_metadata, read_highrate               (from the OSV)
    projection       build_map                (fisheye -> equirect remap)
    levelling        Stitcher360.levelling_for, horizon_lock_rotation
    pixels           OSVDecoder / iter_pairs  (libavcodec, in-process)
    GPU stitch       stitch_gpu.GpuStitcher   (CUDA / Apple MPS)
    video            convert                  (decode -> stitch -> encode -> 360 tag)
"""

__version__ = "0.3.0-dev"

from .calibration import CameraMap, CalibrationError
from .constants import *  # noqa: F401,F403  - the master constants, re-exported
from .decode import OSVDecoder, read_pair
from .metadata import OSVMeta, read_metadata, read_highrate
from .osv_calibration import (
    CameraCalib,
    read_osv_calibration,
    read_mount_rotation,
    parse_mount_rotation,
    read_header_sample,
)
from .pipeline import Stitcher360, horizon_lock_rotation, smoothed_rotation_track
from .project import build_map
from .spherical import inject_spherical
from .video import convert
