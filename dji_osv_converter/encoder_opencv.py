"""OpenCV-based MP4 encoder - no FFmpeg subprocess hassles"""

import logging
import cv2
import numpy as np
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class OpenCVEncoder:
    """Simple, reliable MP4 encoder using OpenCV VideoWriter"""

    def __init__(
        self,
        output_path: str,
        width: int = 7680,
        height: int = 3840,
        fps: float = 29.97,
    ):
        """
        Initialize encoder.

        Args:
            output_path: Output MP4 file path
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.writer = None
        self.frame_count = 0

    def open(self) -> bool:
        """Open encoder and create VideoWriter"""
        try:
            # Use H.264 codec (widely compatible)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for H.264

            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (self.width, self.height),
            )

            if not self.writer.isOpened():
                logger.error("Failed to open VideoWriter")
                return False

            logger.info(f"Encoder opened: {self.output_path} ({self.width}×{self.height} @ {self.fps}fps)")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize encoder: {e}")
            return False

    def write_frame(self, frame_rgb24: np.ndarray) -> bool:
        """
        Write one frame to output.

        Args:
            frame_rgb24: RGB24 numpy array (height × width × 3, uint8)

        Returns:
            True on success
        """
        try:
            if not self.writer:
                logger.error("Writer not initialized")
                return False

            if frame_rgb24.shape != (self.height, self.width, 3):
                logger.error(f"Invalid frame shape: {frame_rgb24.shape}")
                return False

            # OpenCV uses BGR internally, convert from RGB
            frame_bgr = cv2.cvtColor(frame_rgb24, cv2.COLOR_RGB2BGR)

            # Write frame
            success = self.writer.write(frame_bgr)

            if not success:
                logger.error(f"Failed to write frame {self.frame_count}")
                return False

            self.frame_count += 1
            return True

        except Exception as e:
            logger.error(f"Write frame failed: {e}")
            return False

    def close(self) -> bool:
        """Finalize and close encoder"""
        try:
            if self.writer:
                self.writer.release()
                logger.info(f"Encoder closed: {self.frame_count} frames written")
                return True
            return False

        except Exception as e:
            logger.error(f"Close failed: {e}")
            return False

    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
