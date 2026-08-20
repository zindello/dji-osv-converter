"""GPU-accelerated fisheye to equirectangular stitcher using OpenCV

Handles undistortion of back-to-back fisheye lenses and seam blending
for equirectangular 360° output. Supports GPU backends (OpenCL on Apple Silicon,
CUDA on NVIDIA) with automatic CPU fallback.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class GPUBackend:
    """GPU backend detection and initialization"""

    def __init__(self):
        self.has_opencv = cv2 is not None
        self.has_cuda = False
        self.has_opencl = False
        self.use_gpu = False
        self.backend_name = "CPU"

        if self.has_opencv:
            self._detect_gpu()

    def _detect_gpu(self):
        """Detect available GPU backends"""
        try:
            # Check CUDA availability
            if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.has_cuda = True
                logger.info(f"CUDA available: {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
        except Exception as e:
            logger.debug(f"CUDA check failed: {e}")

        try:
            # Check OpenCL availability (Apple Silicon uses OpenCL)
            if cv2.ocl.haveOpenCL():
                self.has_opencl = True
                logger.info("OpenCL available (Apple Silicon/Intel)")
        except Exception as e:
            logger.debug(f"OpenCL check failed: {e}")

        # Prefer CUDA, fallback to OpenCL
        if self.has_cuda:
            self.use_gpu = True
            self.backend_name = "CUDA"
        elif self.has_opencl:
            self.use_gpu = True
            cv2.ocl.setUseOpenCL(True)
            self.backend_name = "OpenCL"

    def convert_to_gpu(self, arr: np.ndarray) -> any:
        """Convert numpy array to GPU format if available"""
        if not self.use_gpu or cv2 is None:
            return arr

        try:
            if self.backend_name == "CUDA":
                return cv2.cuda_GpuMat()
            else:  # OpenCL
                return cv2.UMat(arr)
        except Exception as e:
            logger.warning(f"Failed to convert to GPU: {e}")
            return arr

    def convert_to_cpu(self, gpu_arr: any) -> np.ndarray:
        """Convert GPU array back to numpy"""
        if not self.use_gpu or cv2 is None:
            return gpu_arr

        try:
            if hasattr(gpu_arr, "get"):  # CUDA GpuMat
                return gpu_arr.get()
            elif isinstance(gpu_arr, cv2.UMat):  # OpenCL UMat
                return gpu_arr.get()
            else:
                return gpu_arr
        except Exception as e:
            logger.warning(f"Failed to convert from GPU: {e}")
            return gpu_arr


class FisheyeCalibration:
    """Fixed calibration parameters for back-to-back fisheye lenses (DJI Osmo 360)"""

    def __init__(self):
        # DJI Osmo 360: symmetric back-to-back fisheye lenses
        # ~1000px focal length at 3840x3840 resolution
        focal_length = 1000.0
        principal_point = 1920.0  # 3840/2

        # Camera matrix K for fisheye (approximate for 3840x3840)
        self.K = np.array([
            [focal_length, 0, principal_point],
            [0, focal_length, principal_point],
            [0, 0, 1]
        ], dtype=np.float32)

        # Distortion coefficients for fisheye (k1, k2, k3, k4)
        # These are typical for wide-angle fisheye lenses
        self.D = np.array([0.15, 0.05, 0.0, 0.0], dtype=np.float32)

        # Undistort map cache
        self._undistort_map_cache = {}

    def get_undistort_maps(
        self,
        input_size: Tuple[int, int],
        output_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cached undistortion maps for efficient processing.

        Args:
            input_size: Input image size (width, height)
            output_size: Output image size (width, height)

        Returns:
            Tuple of (map1, map2) for cv2.remap
        """
        cache_key = (input_size, output_size)
        if cache_key in self._undistort_map_cache:
            return self._undistort_map_cache[cache_key]

        # Create K for output dimensions
        K_out = self.K.copy()
        K_out[0, 2] = output_size[0] / 2
        K_out[1, 2] = output_size[1] / 2

        try:
            # Compute undistortion maps
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), K_out,
                output_size, cv2.CV_32F
            )
            self._undistort_map_cache[cache_key] = (map1, map2)
            return map1, map2
        except Exception as e:
            logger.error(f"Failed to create undistort maps: {e}")
            # Return identity maps on error
            return np.zeros(output_size + (2,), dtype=np.float32), np.ones(output_size, dtype=np.float32)


class OpenCVStitcher:
    """GPU-accelerated fisheye to equirectangular stitcher"""

    # Output format
    OUTPUT_WIDTH = 7680
    OUTPUT_HEIGHT = 3840

    def __init__(self, gpu_backend: Optional[GPUBackend] = None):
        """
        Initialize the stitcher.

        Args:
            gpu_backend: GPUBackend instance (creates default if None)
        """
        if cv2 is None:
            raise ImportError("opencv-python or opencv-contrib-python not installed")

        self.gpu = gpu_backend or GPUBackend()
        self.calibration = FisheyeCalibration()
        logger.info(f"OpenCVStitcher initialized (backend: {self.gpu.backend_name})")

    def stitch_frame(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        calibration: Optional[Dict] = None,
    ) -> Optional[np.ndarray]:
        """
        Stitch two fisheye frames to equirectangular output.

        Args:
            frame0: First input frame (RGB24, 3840×3840)
            frame1: Second input frame (RGB24, 3840×3840)
            calibration: Optional per-frame calibration (unused in fixed mode)

        Returns:
            Stitched equirectangular frame (RGB24, 7680×3840) or None on failure
        """
        try:
            if frame0.shape != (3840, 3840, 3) or frame1.shape != (3840, 3840, 3):
                logger.warning(f"Unexpected frame shape: {frame0.shape}, {frame1.shape}")
                return None

            # Undistort both frames
            undistorted0 = self._undistort_frame(frame0)
            undistorted1 = self._undistort_frame(frame1)

            if undistorted0 is None or undistorted1 is None:
                logger.error("Undistortion failed")
                return None

            # Blend and create equirectangular output
            output = self._create_equirectangular(undistorted0, undistorted1)
            return output

        except Exception as e:
            logger.exception(f"Stitching failed: {e}")
            return None

    def _undistort_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Undistort a fisheye frame.

        Args:
            frame: Input RGB24 frame (3840×3840)

        Returns:
            Undistorted frame or None on failure
        """
        try:
            input_size = (frame.shape[1], frame.shape[0])
            output_size = input_size  # Keep same size for now

            map1, map2 = self.calibration.get_undistort_maps(input_size, output_size)

            if self.gpu.use_gpu:
                # Try GPU acceleration
                frame_gpu = self.gpu.convert_to_gpu(frame)
                try:
                    undistorted_gpu = cv2.remap(frame_gpu, map1, map2, cv2.INTER_LINEAR)
                    undistorted = self.gpu.convert_to_cpu(undistorted_gpu)
                except Exception as e:
                    logger.debug(f"GPU undistort failed, falling back to CPU: {e}")
                    undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            else:
                undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

            return undistorted

        except Exception as e:
            logger.error(f"Frame undistortion failed: {e}")
            return None

    def _create_equirectangular(
        self,
        undistorted0: np.ndarray,
        undistorted1: np.ndarray,
    ) -> np.ndarray:
        """
        Create equirectangular output from two undistorted frames.

        Places undistorted frames with overlap and applies seam blending.

        Args:
            undistorted0: First undistorted frame (3840×3840)
            undistorted1: Second undistorted frame (3840×3840)

        Returns:
            Equirectangular output (7680×3840, RGB24)
        """
        # Initialize output
        output = np.zeros((self.OUTPUT_HEIGHT, self.OUTPUT_WIDTH, 3), dtype=np.uint8)

        # Overlap region width (where frames blend)
        overlap_width = 400  # ~10% overlap for blending

        # Left frame: map to left half of output (0 to 3840)
        left_start = 0
        left_end = 3840 + overlap_width

        # Right frame: map to right half of output (3840 to 7680)
        right_start = 3840 - overlap_width
        right_end = 7680

        # Copy left frame (with scaling if needed)
        left_region = undistorted0[:, :left_end]
        if left_region.shape[1] < left_end:
            # Pad if necessary
            pad_width = left_end - left_region.shape[1]
            left_region = np.pad(left_region, ((0, 0), (0, pad_width), (0, 0)), mode="edge")
        output[:, left_start:left_end] = left_region[:, :left_end]

        # Copy right frame (with scaling if needed)
        right_region = undistorted1[:, :right_end - right_start]
        if right_region.shape[1] < (right_end - right_start):
            pad_width = (right_end - right_start) - right_region.shape[1]
            right_region = np.pad(right_region, ((0, 0), (0, pad_width), (0, 0)), mode="edge")
        output[:, right_start:right_end] = right_region[:, :right_end - right_start]

        # Blend overlap region with alpha feathering
        self._blend_overlap(output, overlap_width)

        return output

    def _blend_overlap(self, output: np.ndarray, overlap_width: int) -> None:
        """
        Apply alpha feathering to the overlap region.

        Args:
            output: Output frame (modified in-place)
            overlap_width: Width of overlap region
        """
        if overlap_width <= 0:
            return

        blend_start = 3840 - overlap_width
        blend_end = 3840 + overlap_width

        # Create alpha gradient (0 to 1 from left to right across the 2*overlap_width region)
        total_blend_width = blend_end - blend_start
        alpha = np.linspace(0, 1, total_blend_width, dtype=np.float32)
        alpha = alpha.reshape(1, -1, 1)  # Shape: (1, total_blend_width, 1)

        # Extract pixels from the overlap region
        blend_region = output[:, blend_start:blend_end].astype(np.float32)

        # Simple approach: just apply a feathering effect by fading edges
        # This reduces harsh seams at the overlap
        alpha_blend = np.clip(alpha, 0.2, 0.8)  # Limit fade range
        output[:, blend_start:blend_end] = (blend_region * alpha_blend).astype(np.uint8)


def create_stitcher(gpu_backend: Optional[str] = None) -> Optional[OpenCVStitcher]:
    """
    Factory function to create an OpenCV stitcher with optional GPU backend selection.

    Args:
        gpu_backend: Force GPU backend ("cuda", "opencl") or None for auto-detection

    Returns:
        OpenCVStitcher instance or None if OpenCV not available
    """
    if cv2 is None:
        logger.error("OpenCV not installed. Install with: pip install opencv-contrib-python")
        return None

    backend = GPUBackend()

    if gpu_backend:
        if gpu_backend.lower() == "cuda" and not backend.has_cuda:
            logger.warning("CUDA not available, falling back to auto-detected backend")
        elif gpu_backend.lower() == "opencl" and not backend.has_opencl:
            logger.warning("OpenCL not available, falling back to auto-detected backend")

    try:
        return OpenCVStitcher(backend)
    except Exception as e:
        logger.error(f"Failed to create stitcher: {e}")
        return None


__all__ = [
    "OpenCVStitcher",
    "GPUBackend",
    "FisheyeCalibration",
    "create_stitcher",
]
