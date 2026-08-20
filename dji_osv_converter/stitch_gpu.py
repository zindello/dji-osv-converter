"""GPU-accelerated stitcher using Metal backend

Uses GPU for fisheye undistortion and seam blending.
"""

import logging
from typing import Dict, Optional, Any
import numpy as np

from .gpu.backend import GPUFrame, GPUColorSpace

logger = logging.getLogger(__name__)


class GPUStitcher:
    """GPU-accelerated frame stitcher using Metal backend"""

    def __init__(self, gpu_backend):
        """
        Initialize GPU stitcher with a GPU backend.

        Args:
            gpu_backend: GPUBackend instance (Metal, CUDA, oneAPI, etc.)
        """
        self.gpu_backend = gpu_backend
        self.output_width = 7680
        self.output_height = 3840

    def stitch_frame(
        self, frame0: np.ndarray, frame1: np.ndarray, calibration: Optional[Dict]
    ) -> Optional[np.ndarray]:
        """
        Stitch two fisheye frames to equirectangular.

        Args:
            frame0: Left camera frame (3840×3840 RGB24)
            frame1: Right camera frame (3840×3840 RGB24)
            calibration: Camera calibration parameters

        Returns:
            Stitched equirectangular frame (3840×7680 RGB24) or None on failure
        """
        try:
            if self.gpu_backend is None:
                logger.error("GPU backend not initialized")
                return None

            if frame0.shape != (3840, 3840, 3) or frame1.shape != (3840, 3840, 3):
                logger.error(f"Invalid frame shapes: {frame0.shape}, {frame1.shape}")
                return None

            # Create GPU frames
            gpu_frame0 = GPUFrame(frame0, 3840, 3840, GPUColorSpace.RGB24)
            gpu_frame1 = GPUFrame(frame1, 3840, 3840, GPUColorSpace.RGB24)

            # Undistort using GPU (fisheye correction)
            if calibration is None:
                calibration = self._get_default_calibration()

            logger.debug("Undistorting frame 0...")
            undist0 = self.gpu_backend.undistort_warp(gpu_frame0, calibration)
            if undist0 is None:
                logger.error("Undistortion of frame 0 failed")
                return None

            logger.debug("Undistorting frame 1...")
            undist1 = self.gpu_backend.undistort_warp(gpu_frame1, calibration)
            if undist1 is None:
                logger.error("Undistortion of frame 1 failed")
                return None

            # Blend seams using GPU
            overlap_width = 200  # Typical overlap for stitching

            logger.debug(f"Blending frames with {overlap_width}px overlap...")
            blended = self.gpu_backend.blend_frames(undist0, undist1, overlap_width)
            if blended is None:
                logger.error("Blending failed")
                return None

            # Extract numpy array from GPU frame
            result = blended.gpu_handle
            if not isinstance(result, np.ndarray):
                logger.error("Failed to extract result from GPU")
                return None

            logger.debug(f"Stitched frame: {result.shape}")

            # Resize/map to output equirectangular format if needed
            # For now, return the blended result
            # TODO: Apply equirectangular projection if needed

            return result

        except Exception as e:
            logger.error(f"GPU stitching failed: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return None

    def _get_default_calibration(self) -> Dict[str, Any]:
        """Get default camera calibration parameters"""
        return {
            "focal_length": 1000.0,
            "principal_point": 1920.0,
            "distortion_coeffs": [0.15, 0.05, 0.0, 0.0],
        }
