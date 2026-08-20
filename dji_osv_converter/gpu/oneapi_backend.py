"""Intel oneAPI backend for Intel iGPU (Iris, Arc)"""

import logging
from typing import Any, Dict, Optional

from .backend import GPUBackend, GPUFrame, GPUColorSpace

logger = logging.getLogger(__name__)


class OneAPIBackend(GPUBackend):
    """Intel oneAPI GPU backend (Intel iGPU)"""

    def __init__(self):
        super().__init__("oneapi")

    def initialize(self) -> bool:
        """Initialize Intel GPU device"""
        logger.warning("Intel oneAPI backend not yet implemented")
        logger.info("  Would use Intel Media SDK or Quick Sync for decoding")
        logger.info("  Would use Intel Level-Zero or oneAPI for processing")
        logger.info("  Expected speedup: 50-200× vs CPU")
        return False

    def decode_hevc(self, hevc_data: bytes, width: int, height: int) -> Optional[GPUFrame]:
        """Decode HEVC using Intel Media SDK"""
        logger.error("Intel oneAPI backend not yet implemented")
        return None

    def convert_color_space(
        self, frame: GPUFrame, target_space: GPUColorSpace
    ) -> Optional[GPUFrame]:
        """Convert color space using Intel GPU compute"""
        logger.error("Intel oneAPI backend not yet implemented")
        return None

    def undistort_warp(
        self,
        frame: GPUFrame,
        calibration: Dict[str, Any],
    ) -> Optional[GPUFrame]:
        """Fisheye undistortion using Intel GPU compute"""
        logger.error("Intel oneAPI backend not yet implemented")
        return None

    def blend_frames(
        self,
        frame_left: GPUFrame,
        frame_right: GPUFrame,
        overlap_width: int,
    ) -> Optional[GPUFrame]:
        """Seam blending using Intel GPU compute"""
        logger.error("Intel oneAPI backend not yet implemented")
        return None

    def encode_hevc(
        self,
        frame: GPUFrame,
        bitrate_mbps: int = 245,
        preset: str = "medium",
    ) -> Optional[bytes]:
        """Encode to HEVC using Intel Media SDK encoder"""
        logger.error("Intel oneAPI backend not yet implemented")
        return None

    def shutdown(self) -> None:
        """Release Intel GPU resources"""
        pass
