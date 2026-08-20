"""DJI OSV to Equirectangular 360° Converter

A Python library to convert DJI Osmo 360 video files (OSV) to
equirectangular 360° MP4 format suitable for DaVinci Resolve.

Features:
- Fast conversion using DJI's stitching framework (Option A)
- Cross-platform fallback with GPU acceleration (Option C, planned)
- Python API and CLI tool
- Support for all Osmo 360 camera models

Usage:
    from dji_osv_converter import OSVConverter

    converter = OSVConverter()
    converter.convert("video.OSV", "output.mp4")

Or via CLI:
    dji-resolve input.OSV output.mp4
"""

__version__ = "0.1.0-dev"
__author__ = "DJI OSV Converter Contributors"

from .converter import OSVConverter
from .osv import OSVFile, OSVMetadata
from .stitch_dualstitcher import DualStitcher, StitcherConfig, DualStitcherError

__all__ = [
    "OSVConverter",
    "OSVFile",
    "OSVMetadata",
    "DualStitcher",
    "StitcherConfig",
    "DualStitcherError",
]


def convert(input_path: str, output_path: str, **options) -> bool:
    """
    Quick conversion function.

    Args:
        input_path: Path to input OSV file
        output_path: Path to output MP4 file
        **options: Additional conversion options

    Returns:
        True if successful, False otherwise
    """
    converter = OSVConverter()
    return converter.convert(input_path, output_path, **options)
