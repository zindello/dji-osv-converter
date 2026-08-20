"""DJI OSV to equirectangular 360 converter.

Converts DJI Osmo 360 dual-fisheye .OSV files to equirectangular 360 MP4.

Stitching uses DJI's own shipped calibration *data* (precomputed remap lookup
tables and blend masks read from the DJI Studio application bundle). No DJI
code is called - see archive/failed_approaches/README.md for why that route
was abandoned.

    from dji_osv_converter.calibration import load_calibration
    from dji_osv_converter.stitcher import stitch

    cams = load_calibration()
    pano = stitch(fisheye0, fisheye1, cams)

Status: under active rebuild, see docs/ROADMAP.md. The legacy converter/CLI
entry points are not wired up to the new pipeline yet.
"""

__version__ = "0.2.0-dev"

from .calibration import CalibrationError, CameraMap, load_calibration
from .stitcher import blend, coverage_report, stitch, warp

__all__ = [
    "load_calibration",
    "CameraMap",
    "CalibrationError",
    "stitch",
    "warp",
    "blend",
    "coverage_report",
]
