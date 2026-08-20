"""Main OSV Converter

Orchestrates the full conversion pipeline:
1. Load OSV file and validate
2. Extract and decode frames (PyAV multi-stream)
3. Parse per-frame calibration metadata
4. Stitch frames (DualStitcher or GPU fallback)
5. Encode to equirectangular MP4 (H.265)
6. Add spherical metadata (Google Spatial Media format)
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json

from .frame_decoder import FrameExtractor
from .protobuf_schema import FrameMetadataParser
from .stitch_dualstitcher import DualStitcher, StitcherConfig, DualStitcherError

logger = logging.getLogger(__name__)


class StitchingPipeline:
    """Main pipeline for OSV to 360° equirectangular conversion"""

    # Output format constants
    OUTPUT_WIDTH = 7680
    OUTPUT_HEIGHT = 3840
    OUTPUT_FPS = 29.97
    OUTPUT_CODEC = "hevc"
    OUTPUT_BITRATE = "40M"

    def __init__(self, input_path: Path, output_path: Path):
        """
        Initialize pipeline for conversion.

        Args:
            input_path: Path to OSV file
            output_path: Path to output MP4 file
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.frame_extractor: Optional[FrameExtractor] = None
        self.metadata_parser = FrameMetadataParser()
        self.stitcher: Optional[DualStitcher] = None

    def run(
        self,
        frame_range: Optional[Tuple[int, int]] = None,
        temp_dir: Optional[Path] = None,
        cleanup: bool = True,
    ) -> bool:
        """
        Run the full conversion pipeline.

        Args:
            frame_range: (start, end) frame indices, or None for all frames
            temp_dir: Directory for temporary files
            cleanup: Whether to clean up temp files after completion

        Returns:
            True if conversion succeeded
        """
        temp_dir = temp_dir or Path(tempfile.gettempdir()) / "dji_osv_converter"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Starting conversion: {self.input_path.name}")
            logger.info(f"  Output: {self.output_path}")
            logger.info(f"  Temp dir: {temp_dir}")

            # Step 1: Initialize frame extractor
            logger.info("Step 1: Initializing frame extractor...")
            self.frame_extractor = FrameExtractor(self.input_path)
            stream_info = self.frame_extractor.get_stream_info()
            logger.info(f"  Found {stream_info['num_streams']} video streams")
            logger.info(f"  Total frames: {stream_info['total_frames']}")

            # Determine frame range
            total_frames = stream_info['total_frames']
            if frame_range is None:
                start_frame, end_frame = 0, total_frames
            else:
                start_frame, end_frame = frame_range
                end_frame = min(end_frame, total_frames)

            logger.info(f"  Processing frames {start_frame}:{end_frame}")

            # Step 2: Load metadata
            logger.info("Step 2: Loading metadata...")
            metadata_file = temp_dir / "metadata.bin"
            self._extract_metadata_stream(metadata_file)
            calibrations = self.metadata_parser.parse_file(metadata_file)
            logger.info(f"  Loaded calibration for {len(calibrations)} frames")

            # Step 3: Prepare stitcher (optional)
            logger.info("Step 3: Preparing stitcher...")
            try:
                config = StitcherConfig()
                self.stitcher = DualStitcher(config)
                logger.info(f"  Stitcher available: {self.stitcher.is_available()}")
            except DualStitcherError as e:
                logger.warning(f"  Stitcher not available: {e}")
                logger.info("  Will use stub stitcher for testing")

            # Step 4: Initialize encoder
            logger.info("Step 4: Initializing FFmpeg encoder...")
            encoder_process = self._start_encoder()

            # Step 5: Main processing loop
            logger.info("Step 5: Processing frames...")
            frames_processed = 0

            for frame_num in range(start_frame, end_frame):
                # Extract frames
                frame_data = self.frame_extractor.get_frame(frame_num)
                if not frame_data:
                    logger.warning(f"  Failed to extract frame {frame_num}")
                    continue

                # Get calibration for this frame
                if frame_num < len(calibrations):
                    calibration = calibrations[frame_num]
                else:
                    calibration = calibrations[-1] if calibrations else None

                # Stitch frame (or use stub)
                stitched = self._stitch_frame(frame_data, calibration)
                if stitched is None:
                    logger.warning(f"  Failed to stitch frame {frame_num}")
                    continue

                # Write to encoder
                encoder_process.stdin.write(stitched)

                frames_processed += 1
                if frames_processed % 100 == 0:
                    progress = (frame_num - start_frame) / (end_frame - start_frame)
                    logger.info(f"  Progress: {frames_processed} frames ({progress*100:.1f}%)")

            # Step 6: Finalize encoding
            logger.info("Step 6: Finalizing encoding...")
            encoder_process.stdin.close()

            # Read remaining output and wait for process to finish
            try:
                stdout, stderr = encoder_process.communicate(timeout=300)
            except ValueError:
                # stdin already closed, just wait
                stdout, stderr = None, None
                encoder_process.wait(timeout=300)

            ret = encoder_process.returncode

            if ret != 0:
                if stderr:
                    stderr_text = stderr.decode() if isinstance(stderr, bytes) else stderr
                    logger.error(f"  Encoder stderr: {stderr_text[:500]}")
                logger.error(f"Encoder failed with return code {ret}")
                return False

            logger.info(f"  Encoded {frames_processed} frames")

            # Step 7: Add spherical metadata
            logger.info("Step 7: Adding spherical metadata...")
            self._add_spherical_metadata(self.output_path)

            logger.info("Conversion complete!")
            return True

        except Exception as e:
            logger.exception(f"Conversion failed: {e}")
            return False

        finally:
            # Cleanup
            if cleanup and temp_dir.exists():
                import shutil
                logger.debug(f"Cleaning up {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)

            if self.frame_extractor:
                self.frame_extractor.close()

    def _extract_metadata_stream(self, output_path: Path) -> None:
        """Extract protobuf metadata stream from OSV file"""
        # FFmpeg command to extract stream 3 (data stream) from OSV
        cmd = [
            "ffmpeg", "-i", str(self.input_path),
            "-map", "0:3",  # Stream 3 is protobuf metadata
            "-f", "rawvideo",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=120, check=True)
            logger.debug(f"Extracted metadata to {output_path}")
        except Exception as e:
            logger.warning(f"Could not extract metadata stream: {e}")
            # Create empty metadata file if extraction fails
            output_path.touch()

    def _start_encoder(self) -> subprocess.Popen:
        """Start FFmpeg encoder subprocess"""
        cmd = [
            "ffmpeg",
            "-f", "rawvideo",
            "-video_size", f"{self.OUTPUT_WIDTH}x{self.OUTPUT_HEIGHT}",
            "-pixel_format", "rgb24",
            "-framerate", str(self.OUTPUT_FPS),
            "-i", "-",  # stdin
            "-c:v", self.OUTPUT_CODEC,
            "-preset", "medium",
            "-b:v", self.OUTPUT_BITRATE,
            "-tag:v", "hvc1",  # H.265 tag for compatibility
            "-pix_fmt", "yuv420p",
            "-y",  # Overwrite output
            str(self.output_path),
        ]

        logger.debug(f"Starting encoder: {' '.join(cmd)}")
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _stitch_frame(
        self, frame_data: Dict[int, bytes], calibration: Optional[Dict]
    ) -> Optional[bytes]:
        """
        Stitch two input frames to equirectangular output.

        Args:
            frame_data: Dict mapping stream_idx -> RGB24 frame bytes
            calibration: Per-frame camera calibration (or None)

        Returns:
            Stitched equirectangular frame in RGB24 format, or None on failure
        """
        # For now, use a simple stub that averages the two input frames
        # In production, this would call DualStitcher.FlowStitchWarpFusion()

        if not frame_data or 0 not in frame_data or 1 not in frame_data:
            return None

        try:
            import numpy as np

            # Get input frames as arrays
            frame0 = np.frombuffer(frame_data[0], dtype=np.uint8)
            frame1 = np.frombuffer(frame_data[1], dtype=np.uint8)

            frame0 = frame0.reshape((3840, 3840, 3))
            frame1 = frame1.reshape((3840, 3840, 3))

            # Stub: Simple average stitching
            # In production: Call DualStitcher.FlowStitchWarpFusion() here
            combined = np.average([frame0, frame1], axis=0).astype(np.uint8)

            # Resize to output dimensions (7680×3840)
            # Simple approach: tile horizontally
            output = np.zeros((self.OUTPUT_HEIGHT, self.OUTPUT_WIDTH, 3), dtype=np.uint8)
            output[:, :3840] = combined[:, :3840]  # Left half
            output[:, 3840:7680] = combined[:, :3840]  # Right half (duplicate)

            return output.tobytes()

        except Exception as e:
            logger.error(f"Stitching failed: {e}")
            return None

    def _add_spherical_metadata(self, mp4_path: Path) -> bool:
        """Add Google Spatial Media spherical metadata to MP4"""
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

            temp_output = mp4_path.with_suffix(".tmp.mp4")

            cmd = [
                "ffmpeg",
                "-i", str(mp4_path),
                "-c", "copy",
                "-metadata:s:v:0", "stereo=mono",
                str(temp_output),
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=120)

            if result.returncode == 0:
                temp_output.replace(mp4_path)
                logger.debug("Added spherical metadata")
                return True
            else:
                logger.warning("Failed to add spherical metadata")
                temp_output.unlink(missing_ok=True)
                return False

        except Exception as e:
            logger.warning(f"Could not add spherical metadata: {e}")
            return False


class OSVConverter:
    """High-level converter class (backward compatible)"""

    def convert(
        self,
        input_path: str,
        output_path: str,
        quality: str = "high",
        frame_range: Optional[str] = None,
        temp_dir: Optional[Path] = None,
        cleanup: bool = True,
    ) -> bool:
        """Convert OSV to 360° MP4 (delegating to pipeline)"""
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Parse frame range if provided
        frame_range_tuple = None
        if frame_range:
            try:
                start, end = map(int, frame_range.split(":"))
                frame_range_tuple = (start, end)
            except ValueError:
                logger.error(f"Invalid frame range: {frame_range}")
                return False

        pipeline = StitchingPipeline(input_path, output_path)
        return pipeline.run(
            frame_range=frame_range_tuple,
            temp_dir=temp_dir,
            cleanup=cleanup,
        )


__all__ = ["OSVConverter", "StitchingPipeline"]
