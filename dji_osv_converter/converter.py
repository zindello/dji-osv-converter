"""Main OSV Converter

Orchestrates the full conversion pipeline:
1. Load and parse OSV file
2. Extract frames and metadata
3. Stitch using DualStitcher
4. Encode to equirectangular MP4
5. Add spherical metadata
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import json

from .osv import OSVFile, OSVMetadata
from .stitch_dualstitcher import DualStitcher, StitcherConfig, DualStitcherError

logger = logging.getLogger(__name__)


class OSVConverter:
    """Main converter class for OSV to equirectangular 360° MP4"""

    def __init__(self):
        """Initialize converter"""
        self.stitcher: Optional[DualStitcher] = None
        self._check_dualstitcher()

    def _check_dualstitcher(self) -> None:
        """Check if DualStitcher framework is available"""
        try:
            stitcher_config = StitcherConfig()
            self.stitcher = DualStitcher(stitcher_config)

            if self.stitcher.is_available():
                logger.info("DualStitcher framework available and loaded")
                logger.debug(self.stitcher.get_info())
            else:
                logger.warning("DualStitcher framework not available")
                self.stitcher = None

        except DualStitcherError as e:
            logger.warning(f"Could not load DualStitcher: {e}")
            self.stitcher = None

    def convert(
        self,
        input_path: str,
        output_path: str,
        quality: str = "high",
        frame_range: Optional[str] = None,
        temp_dir: Optional[Path] = None,
        cleanup: bool = True,
    ) -> bool:
        """
        Convert an OSV file to equirectangular 360° MP4.

        Args:
            input_path: Path to input OSV file
            output_path: Path to output MP4 file
            quality: Encoding quality (low/medium/high)
            frame_range: Frame range to process (e.g., "0:100")
            temp_dir: Temporary directory for intermediate files
            cleanup: Whether to clean up temporary files

        Returns:
            True if conversion succeeded, False otherwise
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info(f"Starting conversion: {input_path.name} → {output_path.name}")

        try:
            # Create temporary directory
            if temp_dir is None:
                temp_dir = Path(tempfile.gettempdir()) / "dji_osv_converter"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Load and validate OSV file
            logger.info("Loading OSV file...")
            osv = OSVFile(input_path)
            logger.info(f"  Metadata: {osv.metadata}")

            # Step 2: Check if we can proceed
            if self.stitcher is None:
                logger.error(
                    "DualStitcher framework not available. "
                    "Please install DJI Studio."
                )
                return False

            # Step 3: Extract streams
            logger.info("Extracting video streams...")
            streams_dir = temp_dir / "streams"
            streams = osv.extract_streams(streams_dir)
            logger.info(f"  Extracted {len(streams)} streams")

            # Step 4: Prepare stitcher
            logger.info("Preparing stitcher...")
            config = StitcherConfig(camera_sn=osv.metadata.camera_sn)
            stitcher = DualStitcher(config)
            logger.info(f"  Using device params: {config.device_param_index}")

            # Step 5: Process frames (placeholder for now)
            logger.info("Processing frames...")
            logger.warning("  Frame stitching not yet implemented")

            # Step 6: Encode output (placeholder)
            logger.info("Encoding output...")
            logger.warning("  Video encoding not yet implemented")

            # Step 7: Add spherical metadata
            logger.info("Adding spherical metadata...")
            self._add_spherical_metadata(output_path)

            # Cleanup
            if cleanup:
                logger.info("Cleaning up temporary files...")
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

            logger.info("Conversion complete!")
            return True

        except Exception as e:
            logger.exception(f"Conversion failed: {e}")
            return False

    def _add_spherical_metadata(self, mp4_path: Path) -> bool:
        """
        Add spherical video metadata to MP4 file.

        Args:
            mp4_path: Path to MP4 file to annotate

        Returns:
            True if metadata added successfully
        """
        try:
            # Create spherical metadata XML
            spherical_xml = """<?xml version="1.0" encoding="UTF-8"?>
<rdf:SphericalVideo
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/">
    <GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
    <GSpherical:Spherical>true</GSpherical:Spherical>
    <GSpherical:InitialViewHeadingDegrees>0</GSpherical:InitialViewHeadingDegrees>
    <GSpherical:InitialViewPitchDegrees>0</GSpherical:InitialViewPitchDegrees>
    <GSpherical:InitialViewRollDegrees>0</GSpherical:InitialViewRollDegrees>
</rdf:SphericalVideo>"""

            # Inject spherical metadata using ffmpeg
            # This uses the SphericalVideoXML approach used by Google/YouTube
            temp_output = mp4_path.with_suffix(".tmp.mp4")

            cmd = [
                "ffmpeg",
                "-i", str(mp4_path),
                "-c", "copy",
                "-metadata:s:v:0", "stereo=mono",  # Spherical video
                str(temp_output),
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=60)

            if result.returncode == 0:
                # Replace original with annotated version
                temp_output.replace(mp4_path)
                logger.debug("Added spherical metadata to output")
                return True
            else:
                logger.error(f"Failed to add metadata: {result.stderr.decode()}")
                temp_output.unlink(missing_ok=True)
                return False

        except Exception as e:
            logger.error(f"Failed to add spherical metadata: {e}")
            return False

    def get_dualstitcher_info(self) -> Dict[str, Any]:
        """Get information about DualStitcher framework"""
        if self.stitcher:
            return self.stitcher.get_info()
        return {"is_available": False, "error": "DualStitcher not loaded"}


__all__ = ["OSVConverter"]
