#!/usr/bin/env python3
"""Production-Ready DJI Stitcher Wrapper with Optical Flow and Dynamic Seam Stitching

This module implements a complete stitching pipeline that replicates DJI Studio's approach:

1. Extract fisheye frames from OSV file
2. Compute bidirectional optical flow between left and right frames
3. Detect seam line (optimal blend line in overlap region)
4. Create Metal textures for seam masks (7680×292 BGRA8Unorm)
5. Create Metal texture for optical flow data (789×394 RGBA16Float)
6. Call Stitcher.dynamicStitchSeams with proper Metal texture parameters
7. Return proper 7680×3840 equirectangular RGB panorama

Key discoveries from lldb analysis:
- Method: dynamicStitchSeams:seam1:isCompensate:enableSunGlare:sgGammaTex:flowTex:
- arg3: Metal texture 7680×292 BGRA8Unorm (seam mask 1)
- arg4: Metal texture 7680×292 BGRA8Unorm (seam mask 2)
- arg5: isCompensate = YES (true)
- arg6: enableSunGlare = NO (false)
- arg7: Flow texture 789×394 RGBA16Float
- Output: Proper 7680×3840 equirectangular 360° image

Usage:
    stitcher = DJIStitcherProper()
    output = stitcher.stitch_frames(frame_left, frame_right)
    # output is 7680×3840×3 uint8 RGB array (proper equirectangular)
"""

import logging
import json
import subprocess
import sys
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2

from .stitch_optical_flow import OpticalFlowStitcher

logger = logging.getLogger(__name__)

# Framework paths
FRAMEWORK_PATH = Path("/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework")
FRAMEWORK_RESOURCES = FRAMEWORK_PATH / "Versions/A/Resources/Stitcher-Mac.bundle/case_00"
FRAMEWORKS_DIR = Path("/Applications/DJI Studio.app/Contents/Frameworks")

# Get the path to subprocess helper
_MODULE_DIR = Path(__file__).parent
STITCHER_HELPER_SCRIPT = _MODULE_DIR / "stitcher_proper_helper.py"


class DJIStitcherProper:
    """Production-ready Stitcher using optical flow and dynamic seam stitching

    This class provides a Python interface that:
    1. Computes optical flow between fisheye frames
    2. Detects optimal seam line for blending
    3. Creates proper Metal textures for seam masks and flow data
    4. Calls dynamicStitchSeams method with correct parameters
    5. Returns proper 7680×3840 equirectangular output
    """

    # Texture dimensions (from lldb findings)
    SEAM_MASK_WIDTH = 7680
    SEAM_MASK_HEIGHT = 292
    FLOW_TEXTURE_WIDTH = 789
    FLOW_TEXTURE_HEIGHT = 394

    # Output dimensions
    OUTPUT_WIDTH = 7680
    OUTPUT_HEIGHT = 3840
    INPUT_WIDTH = 3840
    INPUT_HEIGHT = 3840

    def __init__(self, camera_sn: str = "default", enable_flow: bool = True):
        """Initialize DJI Stitcher with optical flow support.

        Args:
            camera_sn: Camera serial number for calibration
            enable_flow: Whether to use optical flow for seam detection
        """
        self.camera_sn = camera_sn
        self.enable_flow = enable_flow
        self.calibration = None
        self._checked_framework = False
        self._framework_available = False

        # Initialize optical flow stitcher for seam detection
        self.flow_stitcher = OpticalFlowStitcher(
            enable_warping=False,  # We only need seam detection
            enable_poisson_blend=False,
            seam_method="flow_guided",
            overlap_width=400,
        )

        # Load calibration
        self._load_calibration()

        # Check framework
        self._check_framework()

    def _load_calibration(self):
        """Load camera calibration from framework resources"""
        try:
            calib_file = FRAMEWORK_RESOURCES / "intrinsic_extrinsic_list" / f"{self.camera_sn}.json"

            if not calib_file.exists():
                calib_file = FRAMEWORK_RESOURCES / "intrinsic_extrinsic_list" / "custom_0.json"

            if not calib_file.exists():
                logger.warning(f"Calibration file not found: {calib_file}")
                self.calibration = None
                return

            with open(calib_file, 'r') as f:
                self.calibration = json.load(f)

            logger.info(f"Loaded calibration from {calib_file.name}")
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            self.calibration = None

    def _check_framework(self):
        """Check if framework and dependencies are available"""
        if self._checked_framework:
            return

        self._checked_framework = True

        if not FRAMEWORK_PATH.exists():
            logger.warning(f"DualStitcher framework not found at {FRAMEWORK_PATH}")
            self._framework_available = False
            return

        if not FRAMEWORK_RESOURCES.exists():
            logger.warning(f"Framework resources not found at {FRAMEWORK_RESOURCES}")
            self._framework_available = False
            return

        self._framework_available = True
        logger.info("DJI Stitcher framework is available")

    def is_available(self) -> bool:
        """Check if framework is available"""
        if not self._checked_framework:
            self._check_framework()
        return self._framework_available

    def get_info(self) -> Dict[str, Any]:
        """Get information about the stitcher"""
        return {
            "framework_path": str(FRAMEWORK_PATH),
            "framework_available": self.is_available(),
            "optical_flow_enabled": self.enable_flow,
            "camera_sn": self.camera_sn,
            "calibration_loaded": self.calibration is not None,
            "seam_mask_size": f"{self.SEAM_MASK_WIDTH}×{self.SEAM_MASK_HEIGHT}",
            "flow_texture_size": f"{self.FLOW_TEXTURE_WIDTH}×{self.FLOW_TEXTURE_HEIGHT}",
            "output_size": f"{self.OUTPUT_WIDTH}×{self.OUTPUT_HEIGHT}",
        }

    def compute_optical_flow(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute bidirectional optical flow between frames.

        Args:
            frame_left: Left fisheye frame (3840×3840 RGB uint8)
            frame_right: Right fisheye frame (3840×3840 RGB uint8)

        Returns:
            Tuple of (flow_forward, flow_backward) - dense flow fields
        """
        try:
            # Convert to grayscale
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2GRAY)

            # Compute flows
            flow_forward = self.flow_stitcher._compute_optical_flow(gray_left, gray_right)
            flow_backward = self.flow_stitcher._compute_optical_flow(gray_right, gray_left)

            logger.debug(f"Optical flow computed: forward={flow_forward.shape}, backward={flow_backward.shape}")
            return flow_forward, flow_backward
        except Exception as e:
            logger.error(f"Optical flow computation failed: {e}")
            return None, None

    def detect_seam_line(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
        flow_forward: Optional[np.ndarray] = None,
        flow_backward: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect optimal seam line for blending.

        Uses optical flow-guided energy minimization to find the best vertical path
        through the overlap region where images blend most smoothly.

        Args:
            frame_left: Left frame (3840×3840)
            frame_right: Right frame (3840×3840)
            flow_forward: Forward optical flow (optional)
            flow_backward: Backward optical flow (optional)

        Returns:
            Tuple of (seam_line_x, seam_line_y) - horizontal seam positions
        """
        try:
            h = frame_left.shape[0]

            if self.enable_flow and flow_forward is not None and flow_backward is not None:
                # Flow-guided seam detection
                logger.debug("Computing flow-guided seam line...")
                seam_line = self.flow_stitcher._detect_flow_guided_seam(frame_left, frame_right)

                # Extract x and y components
                seam_x = seam_line[:, 0]
                seam_y = np.arange(h, dtype=np.float32)

            else:
                # Simple center-based seam
                logger.debug("Using center-based seam line...")
                seam_x = np.full(h, self.INPUT_WIDTH / 2.0, dtype=np.float32)
                seam_y = np.arange(h, dtype=np.float32)

            logger.debug(f"Seam line detected: x range [{seam_x.min():.1f}, {seam_x.max():.1f}], "
                        f"y range [{seam_y.min():.1f}, {seam_y.max():.1f}]")
            return seam_x, seam_y
        except Exception as e:
            logger.error(f"Seam line detection failed: {e}")
            # Fallback to center
            h = frame_left.shape[0]
            seam_x = np.full(h, self.INPUT_WIDTH / 2.0, dtype=np.float32)
            seam_y = np.arange(h, dtype=np.float32)
            return seam_x, seam_y

    def create_seam_mask(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
        seam_x_left: np.ndarray,
        seam_x_right: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create seam mask textures for Metal.

        The seam masks are 7680×292 BGRA8Unorm images representing the blend region.
        They define where and how the left and right images should blend together.

        Args:
            frame_left: Left frame (3840×3840)
            frame_right: Right frame (3840×3840)
            seam_x_left: Seam line x-coordinates in left frame
            seam_x_right: Seam line x-coordinates in right frame (optional)

        Returns:
            Tuple of (seam_mask_1, seam_mask_2) - 7680×292 BGRA8Unorm arrays
        """
        try:
            if seam_x_right is None:
                seam_x_right = seam_x_left.copy()

            # Create blend region mask
            # The seam is the horizontal strip where blending occurs
            blend_width = self.SEAM_MASK_HEIGHT // 2

            # Create 7680×292 seam masks
            seam_mask_1 = np.zeros((self.SEAM_MASK_HEIGHT, self.SEAM_MASK_WIDTH, 4), dtype=np.uint8)
            seam_mask_2 = np.zeros((self.SEAM_MASK_HEIGHT, self.SEAM_MASK_WIDTH, 4), dtype=np.uint8)

            # Extract seam regions from input frames and place in output texture
            for row_out in range(self.SEAM_MASK_HEIGHT):
                # Map output row to input frame rows
                # The seam mask should contain the overlap region
                row_in = int((row_out / self.SEAM_MASK_HEIGHT) * self.INPUT_HEIGHT)
                row_in = min(row_in, self.INPUT_HEIGHT - 1)

                # Get seam x position for this row
                seam_x_l = int(seam_x_left[row_in]) if row_in < len(seam_x_left) else self.INPUT_WIDTH // 2
                seam_x_r = int(seam_x_right[row_in]) if row_in < len(seam_x_right) else 0

                # Extract left frame region around seam
                left_start = max(0, seam_x_l - blend_width)
                left_end = min(self.INPUT_WIDTH, seam_x_l + blend_width)
                left_region = frame_left[row_in, left_start:left_end, :]

                # Extract right frame region around seam
                right_start = max(0, seam_x_r - blend_width)
                right_end = min(self.INPUT_WIDTH, seam_x_r + blend_width)
                right_region = frame_right[row_in, right_start:right_end, :]

                # Place in seam masks (left and right sides of 7680 output)
                # Left mask: left side contains original, right side is blend region
                if left_end > left_start:
                    region_len = left_end - left_start
                    seam_mask_1[row_out, 0:region_len, :3] = left_region
                    seam_mask_1[row_out, 0:region_len, 3] = 255  # Alpha = opaque

                # Right mask: right side contains original, left side is blend region
                if right_end > right_start:
                    region_len = right_end - right_start
                    seam_mask_2[row_out, self.SEAM_MASK_WIDTH - region_len:, :3] = right_region
                    seam_mask_2[row_out, self.SEAM_MASK_WIDTH - region_len:, 3] = 255

            logger.debug(f"Created seam masks: {seam_mask_1.shape} {seam_mask_1.dtype}")
            return seam_mask_1, seam_mask_2
        except Exception as e:
            logger.error(f"Seam mask creation failed: {e}")
            # Return dummy masks
            seam_mask_1 = np.zeros((self.SEAM_MASK_HEIGHT, self.SEAM_MASK_WIDTH, 4), dtype=np.uint8)
            seam_mask_2 = np.zeros((self.SEAM_MASK_HEIGHT, self.SEAM_MASK_WIDTH, 4), dtype=np.uint8)
            return seam_mask_1, seam_mask_2

    def create_flow_texture(
        self,
        flow_forward: np.ndarray,
        flow_backward: np.ndarray,
    ) -> np.ndarray:
        """Create optical flow texture for Metal.

        The flow texture is 789×394 RGBA16Float containing the downsampled optical flow data.

        Args:
            flow_forward: Forward optical flow field (3840×3840×2)
            flow_backward: Backward optical flow field (3840×3840×2)

        Returns:
            Flow texture 789×394×4 float32 (RGBA16Float format)
        """
        try:
            if flow_forward is None or flow_backward is None:
                # Return dummy flow texture
                logger.warning("Flow data not available, using dummy flow texture")
                return np.zeros((self.FLOW_TEXTURE_HEIGHT, self.FLOW_TEXTURE_WIDTH, 4), dtype=np.float32)

            # Downsample flow to texture resolution
            # Calculate downsampling factors
            scale_y = self.INPUT_HEIGHT / self.FLOW_TEXTURE_HEIGHT
            scale_x = self.INPUT_WIDTH / self.FLOW_TEXTURE_WIDTH

            # Downsample using OpenCV
            flow_down = cv2.resize(
                flow_forward,
                (self.FLOW_TEXTURE_WIDTH, self.FLOW_TEXTURE_HEIGHT),
                interpolation=cv2.INTER_LINEAR
            )

            # Create RGBA texture (R=flow_x, G=flow_y, B=flow_magnitude, A=confidence)
            flow_texture = np.zeros((self.FLOW_TEXTURE_HEIGHT, self.FLOW_TEXTURE_WIDTH, 4), dtype=np.float32)

            # Extract flow components
            flow_x = flow_down[:, :, 0]  # Horizontal flow
            flow_y = flow_down[:, :, 1]  # Vertical flow

            # Normalize flow for texture (from pixel values to normalized range)
            # Flow is typically in pixels, normalize to [-1, 1] range
            max_flow = np.sqrt(flow_x**2 + flow_y**2).max() + 1e-6
            flow_x_norm = flow_x / max_flow
            flow_y_norm = flow_y / max_flow

            # Compute flow magnitude
            flow_mag = np.sqrt(flow_x_norm**2 + flow_y_norm**2)
            flow_mag = np.clip(flow_mag, 0, 1)

            # Fill texture channels
            flow_texture[:, :, 0] = (flow_x_norm + 1.0) / 2.0  # R: flow_x normalized to [0, 1]
            flow_texture[:, :, 1] = (flow_y_norm + 1.0) / 2.0  # G: flow_y normalized to [0, 1]
            flow_texture[:, :, 2] = flow_mag                   # B: flow magnitude
            flow_texture[:, :, 3] = 1.0                        # A: confidence (always 1)

            logger.debug(f"Created flow texture: {flow_texture.shape} {flow_texture.dtype}")
            return flow_texture
        except Exception as e:
            logger.error(f"Flow texture creation failed: {e}")
            # Return dummy texture
            return np.zeros((self.FLOW_TEXTURE_HEIGHT, self.FLOW_TEXTURE_WIDTH, 4), dtype=np.float32)

    def stitch_frames(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Stitch two fisheye frames into equirectangular panorama using optical flow.

        This is the main entry point that orchestrates the full stitching pipeline:
        1. Compute optical flow
        2. Detect seam line
        3. Create Metal textures
        4. Call dynamicStitchSeams via subprocess
        5. Return proper 7680×3840 equirectangular output

        Args:
            frame_left: Left fisheye frame (3840×3840 RGB24 uint8)
            frame_right: Right fisheye frame (3840×3840 RGB24 uint8)

        Returns:
            Stitched equirectangular frame (7680×3840 RGB24 uint8) or None on error
        """
        if not self.is_available():
            logger.error("DJI Stitcher framework not available")
            return None

        try:
            # Validate inputs
            if frame_left.shape != (self.INPUT_HEIGHT, self.INPUT_WIDTH, 3) or frame_left.dtype != np.uint8:
                raise ValueError(f"Invalid left frame: expected ({self.INPUT_HEIGHT},{self.INPUT_WIDTH},3) uint8, "
                               f"got {frame_left.shape} {frame_left.dtype}")

            if frame_right.shape != (self.INPUT_HEIGHT, self.INPUT_WIDTH, 3) or frame_right.dtype != np.uint8:
                raise ValueError(f"Invalid right frame: expected ({self.INPUT_HEIGHT},{self.INPUT_WIDTH},3) uint8, "
                               f"got {frame_right.shape} {frame_right.dtype}")

            # Ensure contiguous arrays
            frame_left = np.ascontiguousarray(frame_left)
            frame_right = np.ascontiguousarray(frame_right)

            logger.info("Starting stitching pipeline...")

            # Step 1: Compute optical flow
            logger.debug("Step 1: Computing optical flow...")
            flow_forward, flow_backward = self.compute_optical_flow(frame_left, frame_right)

            # Step 2: Detect seam line
            logger.debug("Step 2: Detecting seam line...")
            seam_x_left, seam_y = self.detect_seam_line(
                frame_left, frame_right,
                flow_forward, flow_backward
            )

            # Step 3: Create seam masks
            logger.debug("Step 3: Creating seam masks...")
            seam_mask_1, seam_mask_2 = self.create_seam_mask(
                frame_left, frame_right, seam_x_left
            )

            # Step 4: Create flow texture
            logger.debug("Step 4: Creating flow texture...")
            flow_texture = self.create_flow_texture(flow_forward, flow_backward)

            # Step 5: Call stitcher via subprocess
            logger.debug("Step 5: Calling dynamicStitchSeams via subprocess...")
            output = self._stitch_via_subprocess(
                frame_left, frame_right,
                seam_mask_1, seam_mask_2,
                flow_texture
            )

            if output is None:
                logger.error("Stitching via subprocess failed")
                return None

            logger.info(f"Stitching succeeded: output shape {output.shape}, dtype {output.dtype}")
            return output

        except Exception as e:
            logger.error(f"Stitching failed: {e}", exc_info=True)
            return None

    def _stitch_via_subprocess(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
        seam_mask_1: np.ndarray,
        seam_mask_2: np.ndarray,
        flow_texture: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Call dynamicStitchSeams via subprocess helper.

        The subprocess helper loads the DualStitcher framework, creates Metal textures,
        and calls the actual Objective-C method.

        Args:
            frame_left: Left frame
            frame_right: Right frame
            seam_mask_1: Seam mask 1 (7680×292 BGRA8)
            seam_mask_2: Seam mask 2 (7680×292 BGRA8)
            flow_texture: Flow texture (789×394 RGBA16F)

        Returns:
            Stitched output (7680×3840 RGB) or None on error
        """
        try:
            import os

            # Prepare input data
            input_data = {
                'left': frame_left,
                'right': frame_right,
                'seam_mask_1': seam_mask_1,
                'seam_mask_2': seam_mask_2,
                'flow_texture': flow_texture,
                'param_path': str(FRAMEWORK_RESOURCES),
                'camera_sn': self.camera_sn,
            }
            input_bytes = pickle.dumps(input_data)

            # Prepare environment
            env = os.environ.copy()
            env['DYLD_FRAMEWORK_PATH'] = str(FRAMEWORKS_DIR)
            env['DYLD_LIBRARY_PATH'] = str(FRAMEWORKS_DIR)
            env['DYLD_FALLBACK_FRAMEWORK_PATH'] = str(FRAMEWORKS_DIR)
            env['DYLD_FALLBACK_LIBRARY_PATH'] = str(FRAMEWORKS_DIR)

            python_exe = sys.executable

            if not STITCHER_HELPER_SCRIPT.exists():
                logger.error(f"Helper script not found: {STITCHER_HELPER_SCRIPT}")
                return None

            logger.debug(f"Running subprocess helper: {STITCHER_HELPER_SCRIPT}")

            # Run subprocess
            proc = subprocess.Popen(
                [python_exe, str(STITCHER_HELPER_SCRIPT)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            logger.debug("Subprocess started, waiting for result...")
            stdout, stderr = proc.communicate(input=input_bytes, timeout=120)

            if proc.returncode != 0:
                stderr_str = stderr.decode('utf-8', errors='replace')
                logger.error(f"Subprocess failed with code {proc.returncode}")
                logger.error(f"Stderr:\n{stderr_str}")
                return None

            # Unpack result
            try:
                output = pickle.loads(stdout)
                logger.debug(f"Unpacked result: {output.shape} {output.dtype}")
                return output
            except Exception as e:
                logger.error(f"Failed to unpack result: {e}")
                stderr_str = stderr.decode('utf-8', errors='replace')
                if stderr_str:
                    logger.error(f"Subprocess stderr:\n{stderr_str}")
                return None

        except Exception as e:
            logger.error(f"Subprocess call failed: {e}", exc_info=True)
            return None


def extract_osv_frames(osv_path: Path, frame_indices: list) -> list:
    """Extract frames from OSV file.

    Args:
        osv_path: Path to OSV file
        frame_indices: List of frame indices to extract

    Returns:
        List of (frame_left, frame_right) tuples
    """
    try:
        from .frame_decoder import FrameExtractor

        extractor = FrameExtractor(osv_path)
        frames = []

        for idx in frame_indices:
            frame_left, frame_right = extractor.get_frame_pair(idx)
            if frame_left is not None and frame_right is not None:
                frames.append((frame_left, frame_right))

        return frames
    except Exception as e:
        logger.error(f"Failed to extract OSV frames: {e}")
        return []


__all__ = [
    "DJIStitcherProper",
    "extract_osv_frames",
]
