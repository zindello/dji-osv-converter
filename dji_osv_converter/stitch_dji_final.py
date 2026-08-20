"""Final integrated DJI Stitcher wrapper using Metal bridge"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import time
import cv2

logger = logging.getLogger(__name__)


class DJIStitcherFinal:
    """Integrated DJI Stitcher that uses Metal bridge for actual stitching"""

    def __init__(self):
        self.metal_bridge = None
        self._initialize()

    def _initialize(self):
        """Initialize the stitcher"""
        try:
            from dji_osv_converter.stitch_metal_bridge import MetalStitcherBridge
            self.metal_bridge = MetalStitcherBridge()
            logger.info("✓ Metal bridge initialized")
        except Exception as e:
            logger.warning(f"Metal bridge not available: {e}")

    def is_available(self) -> bool:
        """Check if stitcher is available"""
        return self.metal_bridge is not None

    def stitch_frames(self, frame_left: np.ndarray, frame_right: np.ndarray) -> Optional[np.ndarray]:
        """Stitch two fisheye frames to equirectangular"""

        if self.metal_bridge is None:
            logger.error("Metal bridge not available")
            return None

        try:
            logger.info(f"Stitching frames: {frame_left.shape} + {frame_right.shape}")
            start = time.time()

            # Convert to grayscale for optical flow
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2GRAY)

            # Compute optical flow
            logger.debug("Computing optical flow...")
            flow = cv2.calcOpticalFlowFarneback(
                gray_left, gray_right,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            logger.debug(f"Flow computed: {flow.shape}")

            # Create seam masks (7680×292 BGRA format)
            seam_height = 292
            # Create 4-channel BGRA seam masks
            seam_mask_1_bgr = np.zeros((seam_height, 7680, 3), dtype=np.uint8)
            seam_mask_2_bgr = np.zeros((seam_height, 7680, 3), dtype=np.uint8)
            
            # Fill with frame data
            seam_mask_1_bgr[:, :3840, :] = cv2.cvtColor(frame_left[:seam_height, :, :], cv2.COLOR_RGB2BGR)
            seam_mask_1_bgr[:, 3840:, :] = cv2.cvtColor(frame_right[:seam_height, :, :], cv2.COLOR_RGB2BGR)
            seam_mask_2_bgr[:, :3840, :] = cv2.cvtColor(frame_left[:seam_height, :, :], cv2.COLOR_RGB2BGR)
            seam_mask_2_bgr[:, 3840:, :] = cv2.cvtColor(frame_right[:seam_height, :, :], cv2.COLOR_RGB2BGR)
            
            # Add alpha channel
            seam_mask_1 = cv2.cvtColor(seam_mask_1_bgr, cv2.COLOR_BGR2BGRA)
            seam_mask_2 = cv2.cvtColor(seam_mask_2_bgr, cv2.COLOR_BGR2BGRA)

            # Downsample flow to 789×394
            flow_texture = np.zeros((394, 789, 4), dtype=np.float32)
            step_y, step_x = 3840 // 394, 3840 // 789
            for i in range(394):
                for j in range(789):
                    y = min(i * step_y, 3839)
                    x = min(j * step_x, 3839)
                    if y < flow.shape[0] and x < flow.shape[1]:
                        flow_texture[i, j, 0] = flow[y, x, 0]
                        flow_texture[i, j, 1] = flow[y, x, 1]

            logger.debug(f"Seam masks: {seam_mask_1.shape}, Flow: {flow_texture.shape}")

            # Call Metal bridge
            output = self.metal_bridge.stitch_frames(
                frame_left, frame_right,
                seam_mask_1, seam_mask_2,
                flow_texture
            )

            elapsed = time.time() - start

            if output is not None:
                logger.info(f"✓ Stitching succeeded in {elapsed:.1f}s, output: {output.shape}")
                return output
            else:
                logger.error("Metal bridge returned None")
                return None

        except Exception as e:
            logger.error(f"Stitching failed: {e}")
            import traceback
            traceback.print_exc()
            return None


__all__ = ["DJIStitcherFinal"]
