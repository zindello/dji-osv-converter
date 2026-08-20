"""Abstract decoder backend interface

Defines the contract that all decoder implementations must satisfy.
Decoders handle:
- Opening OSV/MP4 files
- Discovering video streams
- Decoding frames (hardware or software)
- Synchronizing multi-stream decoding
- Converting to RGB24 bytes

Each backend returns decoded frames as RGB24 bytes (compatible with stitching pipeline).
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Any
import numpy as np


class DecoderCodec(Enum):
    """Supported video codecs"""
    HEVC = "hevc"
    H264 = "h264"


class DecoderFrame:
    """Decoded frame wrapper (backend-agnostic)"""

    def __init__(self, frame_bytes: bytes, width: int, height: int, frame_num: int):
        """
        Args:
            frame_bytes: RGB24 frame data
            width: Frame width
            height: Frame height
            frame_num: Frame number in stream
        """
        self.frame_bytes = frame_bytes
        self.width = width
        self.height = height
        self.frame_num = frame_num

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy RGB array (HxWx3)"""
        return np.frombuffer(self.frame_bytes, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )

    def __repr__(self) -> str:
        return f"DecoderFrame({self.width}x{self.height}, frame={self.frame_num})"


class DecoderBackend(ABC):
    """Abstract base class for decoder implementations"""

    def __init__(self, name: str):
        """
        Args:
            name: Backend identifier (e.g., "videotoolbox", "cpu")
        """
        self.name = name
        self.available = False

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize decoder backend.
        Returns True if successful, False if decoder unavailable.
        """
        pass

    @abstractmethod
    def open_file(self, filepath: str) -> bool:
        """
        Open an OSV/MP4 file for decoding.

        Args:
            filepath: Path to media file

        Returns:
            True if file opened successfully
        """
        pass

    @abstractmethod
    def get_stream_count(self) -> int:
        """Get number of video streams in the file"""
        pass

    @abstractmethod
    def get_stream_info(self, stream_idx: int) -> Dict[str, Any]:
        """
        Get metadata about a video stream.

        Args:
            stream_idx: Stream index (0-based)

        Returns:
            Dict with keys: width, height, fps, codec, total_frames, duration_seconds
        """
        pass

    @abstractmethod
    def decode_frame(self, stream_idx: int, frame_num: int, timeout_s: float = 10.0) -> Optional[bytes]:
        """
        Decode a specific frame from a stream.

        Args:
            stream_idx: Stream index (0-based)
            frame_num: Frame number (0-based)
            timeout_s: Maximum wait time for frame decode

        Returns:
            RGB24 frame bytes on success, None on failure/timeout
        """
        pass

    @abstractmethod
    def decode_frame_all_streams(self, frame_num: int, timeout_s: float = 10.0) -> Optional[Dict[int, bytes]]:
        """
        Decode a frame from all streams simultaneously (synchronized).

        Args:
            frame_num: Frame number (0-based)
            timeout_s: Maximum wait time

        Returns:
            Dict mapping stream_idx -> RGB24 frame bytes, or None on failure
        """
        pass

    @abstractmethod
    def get_total_frames(self) -> int:
        """Get total frame count from first video stream"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close file and clean up resources"""
        pass

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, available={self.available})"
