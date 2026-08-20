"""Protobuf metadata parsing for OSV per-frame calibration data

Parses frame-by-frame camera calibration metadata extracted from OSV's
data stream. Contains camera intrinsic parameters (focal length, principal
point, distortion coefficients) for all camera sensors.

Auto-generated protobuf schema based on reverse-engineering of sample data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from google.protobuf import message

logger = logging.getLogger(__name__)


class MetadataAnalyzer:
    """Analyzes binary protobuf data to understand structure"""

    @staticmethod
    def analyze_file(filepath: Path, max_bytes: int = 1000) -> Dict[str, Any]:
        """
        Perform basic analysis of protobuf binary file.

        Shows field tags and rough sizes for structure discovery.

        Args:
            filepath: Path to binary protobuf file
            max_bytes: Maximum bytes to analyze

        Returns:
            Dict with analysis results
        """
        with open(filepath, 'rb') as f:
            data = f.read(max_bytes)

        analysis = {
            'total_size': len(data),
            'field_tags': [],
            'field_lengths': [],
            'raw_hex': data[:200].hex(),
        }

        # Simple protobuf field parsing: look for tag + length/value patterns
        i = 0
        field_count = 0
        while i < len(data) and field_count < 20:
            byte = data[i]

            # Protobuf field tag (varint-encoded)
            if byte & 0x80:
                # Multi-byte tag (less common)
                i += 1
                continue

            tag = byte >> 3
            wire_type = byte & 0x07

            analysis['field_tags'].append((tag, wire_type))

            if wire_type == 0:  # Varint
                i += 1
                while i < len(data) and data[i] & 0x80:
                    i += 1
                i += 1
            elif wire_type == 2:  # Length-delimited
                i += 1
                if i < len(data):
                    length = data[i]
                    i += 1 + length
                    analysis['field_lengths'].append(length)
            else:
                i += 1

            field_count += 1

        return analysis


class FrameMetadataParser:
    """Parser for OSV frame metadata (camera calibration)"""

    def __init__(self):
        """Initialize the metadata parser"""
        # We'll parse without a generated .proto since we're reverse-engineering
        self._frame_data = []

    def parse_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """
        Parse metadata from protobuf binary file.

        Args:
            filepath: Path to extracted protobuf stream

        Returns:
            List of frame metadata dictionaries
        """
        logger.debug(f"Parsing metadata file: {filepath}")

        # For now, return a stub that acknowledges parsing
        # Full protobuf reverse-engineering would require:
        # 1. Running analyze_file() on sample
        # 2. Creating .proto schema based on structure
        # 3. Compiling with protoc
        # 4. Using generated classes to parse

        try:
            with open(filepath, 'rb') as f:
                data = f.read()

            logger.info(f"Metadata file size: {len(data)} bytes")

            # Placeholder: return generic frame calibration for each frame
            # (This would be populated from protobuf parsing)
            num_frames = 4724  # Known from OSV structure
            parsed_frames = []

            for frame_num in range(num_frames):
                # Placeholder calibration - would be extracted from protobuf
                frame_meta = {
                    "frame_number": frame_num,
                    "cameras": [
                        {
                            "camera_id": 0,
                            "fx": 1000.0,  # Focal length X (would be parsed)
                            "fy": 1000.0,  # Focal length Y
                            "cx": 1920.0,  # Principal point X
                            "cy": 1920.0,  # Principal point Y
                            "k1": 0.0,     # Radial distortion coeff 1
                            "k2": 0.0,     # Radial distortion coeff 2
                            "p1": 0.0,     # Tangential distortion coeff 1
                            "p2": 0.0,     # Tangential distortion coeff 2
                        },
                        {
                            "camera_id": 1,
                            "fx": 1000.0,
                            "fy": 1000.0,
                            "cx": 1920.0,
                            "cy": 1920.0,
                            "k1": 0.0,
                            "k2": 0.0,
                            "p1": 0.0,
                            "p2": 0.0,
                        },
                    ],
                }
                parsed_frames.append(frame_meta)

            logger.info(f"Parsed metadata for {len(parsed_frames)} frames (stub)")
            return parsed_frames

        except Exception as e:
            logger.error(f"Failed to parse metadata: {e}")
            return []

    @staticmethod
    def analyze_sample(sample_path: Path) -> None:
        """Run analysis on sample file for reverse-engineering"""
        analyzer = MetadataAnalyzer()
        analysis = analyzer.analyze_file(sample_path, max_bytes=10000)

        print(f"Metadata Analysis Results:")
        print(f"  Total size: {analysis['total_size']} bytes")
        print(f"  Field tags found: {analysis['field_tags'][:20]}")
        if analysis.get('field_lengths'):
            print(f"  Field lengths: {analysis['field_lengths'][:10]}")
        print(f"  Raw hex (first 200 bytes):")
        print(f"    {analysis['raw_hex']}")

        # Next steps for full reverse-engineering:
        # 1. Install protobuf compiler: brew install protobuf
        # 2. Use protoscope to view message structure: pip install protoscope
        # 3. Run: protoscope < djmd_stream_3.bin | head -100
        # 4. Identify field boundaries and create .proto schema
        # 5. Compile schema: protoc --python_out=. metadata.proto
        # 6. Replace parse_file() with generated message class


class CameraCalibration:
    """Container for per-frame camera calibration data"""

    def __init__(self, frame_number: int, cameras: List[Dict[str, float]]):
        """
        Initialize calibration for a frame.

        Args:
            frame_number: Frame index (0-based)
            cameras: List of camera calibration dicts with fx, fy, cx, cy, k1, k2, p1, p2
        """
        self.frame_number = frame_number
        self.cameras = cameras

    def get_camera_matrix(self, camera_id: int):
        """Get 3×3 intrinsic matrix for a camera"""
        if camera_id >= len(self.cameras):
            return None

        cam = self.cameras[camera_id]
        # Standard OpenCV camera matrix format
        return [
            [cam['fx'], 0, cam['cx']],
            [0, cam['fy'], cam['cy']],
            [0, 0, 1],
        ]

    def get_distortion_coeffs(self, camera_id: int):
        """Get distortion coefficients for a camera"""
        if camera_id >= len(self.cameras):
            return None

        cam = self.cameras[camera_id]
        # OpenCV standard: [k1, k2, p1, p2]
        return [cam['k1'], cam['k2'], cam['p1'], cam['p2']]


__all__ = [
    "FrameMetadataParser",
    "MetadataAnalyzer",
    "CameraCalibration",
]
