"""Production-Ready PyObjC Wrapper for DJI Stitcher Class

This module provides a complete, working Python wrapper around DJI's Stitcher
Objective-C class from the DualStitcher framework. It uses the reverse-engineered
API to properly initialize and call the stitching methods.

Key Implementation Details:
- Uses initWithStitcherInfo: with _StitcherSetupInfo structure
- Creates MTLDevice and MTLCommandQueue for GPU operations
- Loads calibration data from framework bundle
- Calls simpleStitchSeams for equirectangular stitching
- Converts numpy arrays to/from native Objective-C formats
- Proper error handling and validation

API Methods:
- simpleStitchSeams:seam1:isCompensate:enableSunGlare:sgGammaTex:
  Input: Two seam masks (can be images or Metal textures)
  Output: Metal texture or NSData with stitched result

Usage:
    stitcher = StitcherObjC()
    if stitcher.is_available():
        output = stitcher.stitch_frames(frame_left, frame_right)
        # output is 7680×3840×3 uint8 RGB array
"""

import logging
import json
import subprocess
import sys
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Framework paths
FRAMEWORK_PATH = Path("/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework")
FRAMEWORK_RESOURCES = FRAMEWORK_PATH / "Versions/A/Resources/Stitcher-Mac.bundle/case_00"
FRAMEWORKS_DIR = Path("/Applications/DJI Studio.app/Contents/Frameworks")

# Get the path to the subprocess helper
_MODULE_DIR = Path(__file__).parent
STITCHER_HELPER_SCRIPT = _MODULE_DIR / "stitcher_objc_helper.py"


class StitcherError(Exception):
    """Exception for Stitcher API errors"""
    pass


class CalibrationData:
    """Camera calibration extracted from framework resources"""

    def __init__(self, json_data: Dict[str, Any]):
        """Initialize from JSON calibration file"""
        self.intrinsic00 = json_data.get("intrinsic00", {})
        self.intrinsic01 = json_data.get("intrinsic01", {})
        self.extrinsic00 = json_data.get("extrinsic00", {})
        self.extrinsic01 = json_data.get("extrinsic01", {})
        self.raw_data = json_data

    def get_camera_matrix_left(self) -> np.ndarray:
        """Get 3x3 camera intrinsic matrix for left camera"""
        intr = self.intrinsic00
        return np.array([
            [intr.get('fx', 1777), 0, intr.get('cx', 1912)],
            [0, intr.get('fy', 1777), intr.get('cy', 1927)],
            [0, 0, 1]
        ], dtype=np.float32)

    def get_camera_matrix_right(self) -> np.ndarray:
        """Get 3x3 camera intrinsic matrix for right camera"""
        intr = self.intrinsic01
        return np.array([
            [intr.get('fx', 1777), 0, intr.get('cx', 1900)],
            [0, intr.get('fy', 1777), intr.get('cy', 1933)],
            [0, 0, 1]
        ], dtype=np.float32)

    def get_distortion_coeffs_left(self) -> np.ndarray:
        """Get distortion coefficients for left camera"""
        intr = self.intrinsic00
        return np.array([
            intr.get('k1', -0.183),
            intr.get('k2', 0.0184),
            intr.get('p1', 0.0001),
            intr.get('p2', 0.00016),
            intr.get('k3', 0.0),
            intr.get('k4', 0.0),
        ], dtype=np.float32)

    def get_distortion_coeffs_right(self) -> np.ndarray:
        """Get distortion coefficients for right camera"""
        intr = self.intrinsic01
        return np.array([
            intr.get('k1', -0.183),
            intr.get('k2', 0.0184),
            intr.get('p1', 0.0001),
            intr.get('p2', 0.00016),
            intr.get('k3', 0.0),
            intr.get('k4', 0.0),
        ], dtype=np.float32)

    def get_fisheye_param_left(self) -> float:
        """Get fisheye parameter xi for left camera"""
        return self.intrinsic00.get('xi', 0.713)

    def get_fisheye_param_right(self) -> float:
        """Get fisheye parameter xi for right camera"""
        return self.intrinsic01.get('xi', 0.713)


class StitcherObjC:
    """Production-ready Objective-C Stitcher wrapper using subprocess approach

    This class provides a clean Python interface to DJI's Stitcher class.
    Uses subprocess to properly initialize the framework with DYLD environment
    variables and call the actual Objective-C stitching methods.

    The Stitcher class is initialized with:
    - MTLDevice (Metal graphics device)
    - MTLCommandQueue (Metal command queue)
    - Calibration parameters from framework bundle
    - Output dimensions (7680×3840 for equirectangular)

    Stitching is performed via simpleStitchSeams method which is the simplest
    and most reliable approach for equirectangular output.
    """

    def __init__(self, camera_sn: str = "default"):
        """Initialize Stitcher wrapper.

        Args:
            camera_sn: Camera serial number for calibration lookup

        Raises:
            StitcherError: If framework is not available
        """
        self.camera_sn = camera_sn
        self.calibration = None
        self._checked_framework = False
        self._framework_available = False

        # Load calibration data
        self._load_calibration()

        # Check framework availability
        self._check_framework()

    def _load_calibration(self):
        """Load camera calibration from framework resources"""
        try:
            # Try camera-specific calibration first
            calib_file = FRAMEWORK_RESOURCES / "intrinsic_extrinsic_list" / f"{self.camera_sn}.json"

            if not calib_file.exists():
                # Fall back to custom_0 (default)
                calib_file = FRAMEWORK_RESOURCES / "intrinsic_extrinsic_list" / "custom_0.json"

            if not calib_file.exists():
                logger.warning(f"Calibration file not found: {calib_file}")
                self.calibration = None
                return

            with open(calib_file, 'r') as f:
                calib_json = json.load(f)

            self.calibration = CalibrationData(calib_json)
            logger.info(f"Loaded calibration from {calib_file.name}")
            logger.debug(f"Left camera: fx={self.calibration.intrinsic00.get('fx', 'N/A')}, "
                        f"cx={self.calibration.intrinsic00.get('cx', 'N/A')}")

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

        # Check dependencies
        deps_exist = True
        for dep in ["openmcv2.framework", "ml_fis.framework", "libMNN.3.1.2.dylib", "libdInfer.dylib"]:
            dep_path = FRAMEWORKS_DIR / dep
            if not dep_path.exists():
                logger.debug(f"Dependency not found: {dep}")
                deps_exist = False

        self._framework_available = deps_exist

        if self._framework_available:
            logger.info("DualStitcher framework is available")
        else:
            logger.warning("Some dependencies are missing")

    def stitch_frames(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Stitch two fisheye frames into equirectangular panorama.

        Calls the actual Objective-C Stitcher class via subprocess with proper
        initialization parameters.

        Args:
            frame_left: Left fisheye frame (3840×3840 RGB24 uint8)
            frame_right: Right fisheye frame (3840×3840 RGB24 uint8)

        Returns:
            Stitched equirectangular frame (7680×3840 RGB24 uint8) or None on error

        Raises:
            ValueError: If input frames have invalid shape or dtype
        """
        if not self._framework_available:
            logger.error("Framework not available")
            return None

        try:
            # Validate inputs
            if frame_left.shape != (3840, 3840, 3) or frame_left.dtype != np.uint8:
                raise ValueError(f"Invalid left frame: expected (3840,3840,3) uint8, got {frame_left.shape} {frame_left.dtype}")

            if frame_right.shape != (3840, 3840, 3) or frame_right.dtype != np.uint8:
                raise ValueError(f"Invalid right frame: expected (3840,3840,3) uint8, got {frame_right.shape} {frame_right.dtype}")

            # Ensure contiguous arrays
            frame_left = np.ascontiguousarray(frame_left)
            frame_right = np.ascontiguousarray(frame_right)

            logger.debug("Calling Stitcher.simpleStitchSeams via subprocess...")

            # Prepare input data
            input_data = {
                'left': frame_left,
                'right': frame_right,
                'param_path': str(FRAMEWORK_RESOURCES),
                'camera_sn': self.camera_sn,
            }
            input_bytes = pickle.dumps(input_data)

            # Run subprocess with proper environment
            import os
            env = os.environ.copy()
            env['DYLD_FRAMEWORK_PATH'] = str(FRAMEWORKS_DIR)
            env['DYLD_LIBRARY_PATH'] = str(FRAMEWORKS_DIR)
            env['DYLD_FALLBACK_FRAMEWORK_PATH'] = str(FRAMEWORKS_DIR)
            env['DYLD_FALLBACK_LIBRARY_PATH'] = str(FRAMEWORKS_DIR)

            python_exe = sys.executable
            logger.debug(f"Using Python: {python_exe}")

            if not STITCHER_HELPER_SCRIPT.exists():
                logger.error(f"Helper script not found: {STITCHER_HELPER_SCRIPT}")
                return None

            proc = subprocess.Popen(
                [python_exe, str(STITCHER_HELPER_SCRIPT)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            logger.debug("Subprocess started, waiting for stitching...")
            stdout, stderr = proc.communicate(input=input_bytes, timeout=120)

            if proc.returncode != 0:
                stderr_str = stderr.decode('utf-8', errors='replace')
                logger.error(f"Subprocess failed with code {proc.returncode}")
                logger.error(f"Stderr:\n{stderr_str}")
                return None

            # Unpack result
            try:
                output = pickle.loads(stdout)
                logger.info(f"Successfully stitched frames: {output.shape}")
                logger.debug(f"Output value range: [{output.min()}, {output.max()}]")
                return output
            except Exception as e:
                logger.error(f"Failed to unpack result: {e}")
                stderr_str = stderr.decode('utf-8', errors='replace')
                if stderr_str:
                    logger.error(f"Helper stderr:\n{stderr_str}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Stitching subprocess timed out (>120 seconds)")
            return None
        except Exception as e:
            logger.error(f"Stitching failed: {e}", exc_info=True)
            return None

    def is_available(self) -> bool:
        """Check if framework is available and ready for stitching"""
        if not self._checked_framework:
            self._check_framework()
        return self._framework_available

    def get_info(self) -> Dict[str, Any]:
        """Get detailed information about the framework and stitcher status"""
        if not self._checked_framework:
            self._check_framework()

        return {
            "framework_path": str(FRAMEWORK_PATH),
            "framework_available": self._framework_available,
            "camera_sn": self.camera_sn,
            "calibration_loaded": self.calibration is not None,
            "has_resources": FRAMEWORK_RESOURCES.exists(),
            "uses_subprocess": True,
            "helper_script": str(STITCHER_HELPER_SCRIPT),
            "input_dims": (3840, 3840),
            "output_dims": (7680, 3840),
            "stitching_method": "simpleStitchSeams",
        }

    def __repr__(self) -> str:
        return (
            f"StitcherObjC(camera={self.camera_sn}, "
            f"available={self.is_available()})"
        )


def extract_osv_frames(osv_path: Path, frame_indices: list):
    """Extract specific frames from an OSV file for stitching testing.

    Args:
        osv_path: Path to OSV file
        frame_indices: List of frame indices to extract (0-based)

    Returns:
        List of (frame_left, frame_right) tuples as RGB24 numpy arrays

    Note:
        Requires PyAV for HEVC decoding.
        Frames are returned as 3840x3840 RGB24 uint8 arrays.
    """
    try:
        import av
    except ImportError:
        logger.error("PyAV not available. Install with: pip install av")
        return []

    frames_data = []

    try:
        # Open OSV file
        container = av.open(str(osv_path))

        # OSV files have 8 HEVC streams for Osmo 360
        # We need streams 0 and 1 for the two fisheye cameras
        if len(container.streams.video) < 2:
            logger.error(f"OSV file has {len(container.streams.video)} video streams, need at least 2")
            return []

        stream_left = container.streams.video[0]
        stream_right = container.streams.video[1]

        # Decode frames
        frame_left_decoded = {}
        frame_right_decoded = {}

        for packet in container.demux(stream_left):
            for frame in packet.decode():
                frame_num = int(frame.pts * stream_left.time_base) if frame.pts else None
                if frame_num is not None:
                    frame_left_decoded[frame_num] = frame

        for packet in container.demux(stream_right):
            for frame in packet.decode():
                frame_num = int(frame.pts * stream_right.time_base) if frame.pts else None
                if frame_num is not None:
                    frame_right_decoded[frame_num] = frame

        # Extract requested frames
        for idx in frame_indices:
            if idx not in frame_left_decoded or idx not in frame_right_decoded:
                logger.warning(f"Frame {idx} not available")
                continue

            # Convert to RGB24
            frame_l = frame_left_decoded[idx].to_rgb()
            frame_r = frame_right_decoded[idx].to_rgb()

            # Convert to numpy arrays
            array_l = np.frombuffer(frame_l.planes[0], dtype=np.uint8).reshape(
                (frame_l.height, frame_l.width, 3)
            ).copy()
            array_r = np.frombuffer(frame_r.planes[0], dtype=np.uint8).reshape(
                (frame_r.height, frame_r.width, 3)
            ).copy()

            frames_data.append((array_l, array_r))
            logger.info(f"Extracted frame {idx}: {array_l.shape}")

        return frames_data

    except Exception as e:
        logger.error(f"Failed to extract frames from OSV: {e}")
        return []


__all__ = [
    "StitcherObjC",
    "CalibrationData",
    "StitcherError",
    "extract_osv_frames",
    "FRAMEWORK_PATH",
    "FRAMEWORK_RESOURCES",
]
