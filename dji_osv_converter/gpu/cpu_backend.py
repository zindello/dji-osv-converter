"""CPU-only backend (fallback, no GPU acceleration)"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from .backend import GPUBackend, GPUFrame, GPUColorSpace

logger = logging.getLogger(__name__)


class CPUBackend(GPUBackend):
    """CPU-based implementation (no GPU acceleration)"""

    def __init__(self):
        super().__init__("cpu")
        self.available = True  # CPU always available

    def initialize(self) -> bool:
        """CPU backend is always available"""
        logger.info("CPU backend initialized (no GPU acceleration)")
        return True

    def decode_hevc(self, hevc_data: bytes, width: int, height: int) -> Optional[GPUFrame]:
        """
        Decode HEVC using CPU (via PyAV).

        In a CPU backend, the "GPU frame" is just a numpy array wrapper.
        """
        try:
            import av
            import io

            # Decode using PyAV
            container = av.open(io.BytesIO(hevc_data))
            frame = next(container.decode(video=0))

            # Convert to RGB24 numpy array
            rgb_frame = frame.reformat(format="rgb24")
            array = rgb_frame.to_ndarray()

            container.close()

            # Wrap in GPUFrame (CPU version just holds numpy array)
            gpu_frame = GPUFrame(
                gpu_handle=array,
                width=array.shape[1],
                height=array.shape[0],
                color_space=GPUColorSpace.RGB24,
            )
            return gpu_frame

        except Exception as e:
            logger.error(f"CPU HEVC decode failed: {e}")
            return None

    def convert_color_space(
        self, frame: GPUFrame, target_space: GPUColorSpace
    ) -> Optional[GPUFrame]:
        """
        Convert color space using CPU (via OpenCV).

        For CPU backend, just do basic conversion.
        """
        try:
            import cv2

            array = frame.gpu_handle

            if frame.color_space == GPUColorSpace.RGB24 and target_space == GPUColorSpace.YUV420P:
                # RGB -> YUV420p (for encoding)
                yuv = cv2.cvtColor(array, cv2.COLOR_RGB2YCrCb)
                # TODO: proper YUV420p conversion

                return GPUFrame(
                    gpu_handle=yuv,
                    width=frame.width,
                    height=frame.height,
                    color_space=target_space,
                )

            logger.warning(f"Unsupported color conversion: {frame.color_space} -> {target_space}")
            return None

        except Exception as e:
            logger.error(f"Color conversion failed: {e}")
            return None

    def undistort_warp(
        self,
        frame: GPUFrame,
        calibration: Dict[str, Any],
    ) -> Optional[GPUFrame]:
        """
        Fisheye undistortion + warping using CPU (via OpenCV).

        This is the bottleneck - takes several seconds per frame.
        Would be 10-50× faster on GPU.
        """
        try:
            import cv2

            array = frame.gpu_handle

            # Use OpenCV fisheye undistortion (CPU-based)
            # This is where GPU acceleration would make the biggest difference

            focal_length = calibration.get("focal_length", 1000.0)
            principal_point = calibration.get("principal_point", frame.width / 2)

            K = np.array(
                [[focal_length, 0, principal_point], [0, focal_length, principal_point], [0, 0, 1]],
                dtype=np.float32,
            )
            D = np.array([0.15, 0.05, 0.0, 0.0], dtype=np.float32)

            # Undistort (CPU - SLOW)
            undistorted = cv2.fisheye.undistortImage(array, K, D)

            return GPUFrame(
                gpu_handle=undistorted,
                width=undistorted.shape[1],
                height=undistorted.shape[0],
                color_space=frame.color_space,
            )

        except Exception as e:
            logger.error(f"Undistortion failed: {e}")
            return None

    def blend_frames(
        self,
        frame_left: GPUFrame,
        frame_right: GPUFrame,
        overlap_width: int,
    ) -> Optional[GPUFrame]:
        """Simple CPU-based frame blending"""
        try:
            left = frame_left.gpu_handle
            right = frame_right.gpu_handle

            # Basic alpha blend in overlap region
            output = np.zeros(
                (frame_left.height, frame_left.width + frame_right.width, 3), dtype=np.uint8
            )

            output[:, : frame_left.width] = left
            output[:, frame_left.width :] = right

            # TODO: proper seam blending

            return GPUFrame(
                gpu_handle=output,
                width=output.shape[1],
                height=output.shape[0],
                color_space=frame_left.color_space,
            )

        except Exception as e:
            logger.error(f"Blending failed: {e}")
            return None

    def encode_hevc(
        self,
        frame: GPUFrame,
        bitrate_mbps: int = 245,
        preset: str = "medium",
    ) -> Optional[bytes]:
        """
        Encode to HEVC using CPU (via FFmpeg subprocess).

        This would be 10-100× faster on GPU.
        """
        try:
            import subprocess
            import io

            array = frame.gpu_handle

            # Use FFmpeg CLI for encoding (subprocess)
            # Much faster than pure Python, but still CPU-bound

            cmd = [
                "ffmpeg",
                "-f",
                "rawvideo",
                "-video_size",
                f"{frame.width}x{frame.height}",
                "-pixel_format",
                "rgb24",
                "-i",
                "-",
                "-c:v",
                "libx264",  # Use h264 for now (HEVC has issues on Apple Silicon)
                "-preset",
                preset,
                "-b:v",
                f"{bitrate_mbps}M",
                "-f",
                "h264",
                "-",
            ]

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            stdout, stderr = proc.communicate(input=array.tobytes(), timeout=30)

            if proc.returncode != 0:
                logger.error(f"FFmpeg encoding failed: {stderr.decode()}")
                return None

            return stdout

        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return None

    def shutdown(self) -> None:
        """No GPU resources to clean up for CPU backend"""
        pass
