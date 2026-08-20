"""Abstract GPU backend interface

Defines the contract that all GPU implementations must satisfy.
Backends handle:
- HEVC decoding
- Color space conversion
- Fisheye undistortion + warping
- HEVC encoding

Each backend returns GPU-native frames (Metal textures, CUDA tensors, etc.)
Conversion between GPU and CPU memory is explicit.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Tuple, Any
import numpy as np


class GPUCodec(Enum):
    """Supported video codecs"""
    HEVC = "hevc"
    H264 = "h264"


class GPUColorSpace(Enum):
    """Supported color spaces"""
    YUV420P = "yuv420p"       # 8-bit YUV 4:2:0 (most common)
    YUV420P10 = "yuv420p10"   # 10-bit YUV 4:2:0 (high quality)
    RGB24 = "rgb24"           # RGB (for display only, avoid in pipeline)


class GPUFrame:
    """GPU-resident frame (backend-agnostic wrapper)"""

    def __init__(self, gpu_handle: Any, width: int, height: int, color_space: GPUColorSpace):
        """
        Args:
            gpu_handle: Backend-specific GPU object (Metal texture, CUDA tensor, etc.)
            width: Frame width
            height: Frame height
            color_space: Color space of the frame data
        """
        self.gpu_handle = gpu_handle
        self.width = width
        self.height = height
        self.color_space = color_space

    def to_numpy(self) -> np.ndarray:
        """Download from GPU to CPU memory as numpy array"""
        raise NotImplementedError("Must be implemented by backend")

    def __repr__(self) -> str:
        return f"GPUFrame({self.width}x{self.height}, {self.color_space.value})"


class GPUBackend(ABC):
    """Abstract base class for GPU implementations"""

    def __init__(self, name: str):
        """
        Args:
            name: Backend identifier (e.g., "metal", "cuda", "oneapi")
        """
        self.name = name
        self.available = False

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize GPU backend.
        Returns True if successful, False if GPU unavailable.
        """
        pass

    @abstractmethod
    def decode_hevc(self, hevc_data: bytes, width: int, height: int) -> Optional[GPUFrame]:
        """
        Decode HEVC stream to GPU memory.

        Args:
            hevc_data: HEVC encoded video data
            width: Expected frame width
            height: Expected frame height

        Returns:
            GPUFrame on success, None on failure
        """
        pass

    @abstractmethod
    def convert_color_space(
        self, frame: GPUFrame, target_space: GPUColorSpace
    ) -> Optional[GPUFrame]:
        """
        Convert frame between color spaces (GPU-side).

        Args:
            frame: Input GPUFrame
            target_space: Target color space

        Returns:
            Converted GPUFrame on success, None on failure
        """
        pass

    @abstractmethod
    def undistort_warp(
        self,
        frame: GPUFrame,
        calibration: Dict[str, Any],
    ) -> Optional[GPUFrame]:
        """
        Apply fisheye undistortion and warping (GPU-accelerated).

        For stitching: combines undistortion + perspective warp.

        Args:
            frame: Input GPUFrame (fisheye distorted)
            calibration: Camera calibration dict (focal_length, principal_point, distortion_coeffs, etc.)

        Returns:
            Undistorted GPUFrame on success, None on failure
        """
        pass

    @abstractmethod
    def blend_frames(
        self,
        frame_left: GPUFrame,
        frame_right: GPUFrame,
        overlap_width: int,
    ) -> Optional[GPUFrame]:
        """
        Blend two frames in overlap region (GPU-accelerated seam blending).

        Args:
            frame_left: Left input frame
            frame_right: Right input frame
            overlap_width: Width of overlap region in pixels

        Returns:
            Blended GPUFrame on success, None on failure
        """
        pass

    @abstractmethod
    def encode_hevc(
        self,
        frame: GPUFrame,
        bitrate_mbps: int = 245,
        preset: str = "medium",
    ) -> Optional[bytes]:
        """
        Encode frame to HEVC (GPU-accelerated).

        Args:
            frame: Input GPUFrame (should be YUV420p10 for quality)
            bitrate_mbps: Target bitrate in megabits/second
            preset: Encoding speed/quality tradeoff ("fast", "medium", "slow")

        Returns:
            HEVC encoded bytes on success, None on failure
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Cleanup and release GPU resources"""
        pass

    def __enter__(self):
        """Context manager support"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.shutdown()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(available={self.available})"
