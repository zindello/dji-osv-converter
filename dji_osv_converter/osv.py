"""OSV File Handling

This module handles reading and parsing DJI OSV files, including:
- Extracting video streams (8 HEVC feeds for Osmo 360)
- Extracting audio track
- Parsing protobuf metadata (camera calibration)
- Coordinating frame-by-frame access
"""

import logging
import subprocess
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, BinaryIO
import struct

logger = logging.getLogger(__name__)


class OSVMetadata:
    """Metadata extracted from OSV protobuf stream"""

    def __init__(self):
        self.camera_sn = "unknown"
        self.num_cameras = 8
        self.frame_count = 0
        self.duration_seconds = 0.0
        self.fps = 29.97
        self.resolution = (3840, 3840)

        # Per-frame camera calibration
        self.camera_calibrations: List[Dict] = []

    def __repr__(self) -> str:
        return (
            f"OSVMetadata(sn={self.camera_sn}, frames={self.frame_count}, "
            f"cameras={self.num_cameras}, fps={self.fps})"
        )


class OSVFile:
    """
    Reader for DJI OSV (Osmo 360 Video) files.

    OSV files are MP4 containers with:
    - 2+ HEVC video streams (8 for Osmo 360 in 360° mode)
    - 1 AAC audio stream
    - 1+ protobuf metadata streams
    """

    def __init__(self, filepath: Path):
        """
        Initialize OSV file reader.

        Args:
            filepath: Path to .OSV file
        """
        self.filepath = Path(filepath)
        self._metadata: Optional[OSVMetadata] = None
        self._temp_dir: Optional[Path] = None

        if not self.filepath.exists():
            raise FileNotFoundError(f"OSV file not found: {self.filepath}")

    @property
    def metadata(self) -> OSVMetadata:
        """Get metadata, parsing if necessary"""
        if self._metadata is None:
            self._metadata = self._parse_metadata()
        return self._metadata

    def _parse_metadata(self) -> OSVMetadata:
        """Parse OSV metadata using ffprobe"""
        metadata = OSVMetadata()

        try:
            # Get format and stream information
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_format",
                    "-show_streams",
                    str(self.filepath),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.error(f"ffprobe failed: {result.stderr}")
                return metadata

            output = result.stdout

            # Extract duration
            for line in output.split('\n'):
                if line.startswith("duration="):
                    try:
                        metadata.duration_seconds = float(line.split('=')[1])
                    except ValueError:
                        pass

                # Count video streams
                if line.startswith("codec_type=video"):
                    metadata.num_cameras += 1

                # Extract frame rate
                if line.startswith("r_frame_rate="):
                    try:
                        parts = line.split('=')[1].split('/')
                        if len(parts) == 2:
                            num, denom = int(parts[0]), int(parts[1])
                            metadata.fps = num / denom if denom > 0 else 30.0
                    except (ValueError, IndexError):
                        pass

            # Calculate frame count
            metadata.frame_count = int(metadata.duration_seconds * metadata.fps)

            # Extract camera serial number from protobuf metadata
            metadata.camera_sn = self._extract_camera_sn()

            logger.debug(f"Parsed metadata: {metadata}")

        except subprocess.TimeoutExpired:
            logger.error("ffprobe timeout")
        except Exception as e:
            logger.error(f"Failed to parse metadata: {e}")

        return metadata

    def _extract_camera_sn(self) -> str:
        """Extract camera serial number from protobuf metadata"""
        try:
            # Extract protobuf stream (stream index 3 for Osmo 360)
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-i", str(self.filepath),
                    "-map", "0:3",
                    "-f", "rawvideo",
                    "-",
                ],
                capture_output=True,
                timeout=10,
            )

            if result.returncode != 0:
                return "unknown"

            # Search for recognizable patterns in protobuf
            data = result.stdout
            if b"Osmo 360" in data:
                return "Osmo 360"

            # Look for 95SXN pattern (DJI device serial pattern)
            for i in range(len(data) - 8):
                if data[i:i+2] == b"95":
                    # Try to extract SN (ASCII alphanumeric)
                    sn = b""
                    j = i
                    while j < len(data) and j - i < 20:
                        c = data[j]
                        if 32 <= c <= 126:  # Printable ASCII
                            sn += bytes([c])
                            j += 1
                        else:
                            break
                    if len(sn) > 8:
                        try:
                            return sn.decode('ascii')
                        except:
                            pass

            return "unknown"

        except Exception as e:
            logger.debug(f"Failed to extract camera SN: {e}")
            return "unknown"

    def extract_streams(self, output_dir: Path) -> Dict[str, Path]:
        """
        Extract individual video and audio streams from OSV.

        Args:
            output_dir: Directory to save extracted streams

        Returns:
            Dictionary mapping stream type to extracted file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {}

        try:
            # Extract video streams
            for stream_idx in range(8):
                output_file = output_dir / f"video_stream_{stream_idx}.mov"
                cmd = [
                    "ffmpeg",
                    "-i", str(self.filepath),
                    "-map", f"0:{stream_idx}",
                    "-c", "copy",
                    str(output_file),
                ]

                proc = subprocess.run(cmd, capture_output=True, timeout=300)
                if proc.returncode == 0 and output_file.exists():
                    result[f"video_{stream_idx}"] = output_file
                    logger.debug(f"Extracted stream {stream_idx}")

            # Extract audio
            audio_file = output_dir / "audio.aac"
            cmd = [
                "ffmpeg",
                "-i", str(self.filepath),
                "-map", "0:a:0",
                "-c", "aac",
                str(audio_file),
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=60)
            if proc.returncode == 0 and audio_file.exists():
                result["audio"] = audio_file

            logger.info(f"Extracted {len(result)} streams")

        except subprocess.TimeoutExpired:
            logger.error("Stream extraction timeout")
        except Exception as e:
            logger.error(f"Failed to extract streams: {e}")

        return result

    def get_frame_data(
        self, frame_index: int
    ) -> Optional[Tuple[List[bytes], Optional[bytes]]]:
        """
        Get frame data for a specific frame.

        Args:
            frame_index: Frame index (0 to num_frames-1)

        Returns:
            Tuple of (list of 8 camera images, calibration data)
            Returns None if frame cannot be extracted
        """
        raise NotImplementedError("Frame-by-frame extraction not yet implemented")

    def __repr__(self) -> str:
        return f"OSVFile({self.filepath.name})"


# Export main class
__all__ = ["OSVFile", "OSVMetadata"]
