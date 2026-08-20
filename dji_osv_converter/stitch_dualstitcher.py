"""DualStitcher Framework Wrapper

This module provides Python bindings to DJI's DualStitcher framework
for optical flow-based stitching of multi-camera 360° video.

DualStitcher is a proprietary framework included with DJI Studio.
This wrapper enables programmatic access to its stitching capabilities.
"""

import ctypes
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
import json

logger = logging.getLogger(__name__)

# Framework paths
FRAMEWORK_PATH = Path("/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework")
FRAMEWORK_BINARY = FRAMEWORK_PATH / "Versions/A/DualStitcher"
FRAMEWORK_RESOURCES = FRAMEWORK_PATH / "Versions/A/Resources/Stitcher-Mac.bundle"

# Device types
DEVICE_TYPE_APPLE_GPU = 0  # Metal
DEVICE_TYPE_NVIDIA_GPU = 1  # CUDA


class DualStitcherError(Exception):
    """Exception raised for DualStitcher-related errors"""
    pass


class StitcherConfig:
    """Configuration for stitcher"""

    def __init__(self, camera_sn: str = "default"):
        """
        Initialize stitcher configuration.

        Args:
            camera_sn: Camera serial number (matches device-specific parameters)
        """
        self.camera_sn = camera_sn
        self.device_type = DEVICE_TYPE_APPLE_GPU  # Metal on macOS
        self.enable_sun_glare_correction = True
        self.enable_lsc = False  # Lens shading correction
        self.enable_ai = True  # Use AI-based improvements
        self.dynamic_stitching = True  # Handle moving scenes

        # Load device-specific parameters
        self._load_device_params()

    def _load_device_params(self):
        """Load device-specific parameters from framework resources"""
        params_file = FRAMEWORK_RESOURCES / "case_00/param_list.json"
        if not params_file.exists():
            logger.warning(f"Parameter list not found: {params_file}")
            self.device_param_index = 0
            return

        try:
            with open(params_file, 'r') as f:
                params = json.load(f)

            # Find this device's parameter index
            for entry in params.get("param_list", []):
                if entry["sn_name"] == self.camera_sn:
                    self.device_param_index = entry["input"]
                    logger.debug(f"Loaded parameters for {self.camera_sn}: index {self.device_param_index}")
                    return

            # Default to index 0
            logger.warning(f"Camera {self.camera_sn} not found in param_list, using default")
            self.device_param_index = 0

        except Exception as e:
            logger.error(f"Failed to load device parameters: {e}")
            self.device_param_index = 0


class DualStitcher:
    """
    Wrapper for DJI's DualStitcher optical flow stitching framework.

    This provides programmatic access to DualStitcher's Objective-C API
    for stitching multi-camera 360° video feeds.
    """

    def __init__(self, config: Optional[StitcherConfig] = None):
        """
        Initialize the DualStitcher wrapper.

        Args:
            config: StitcherConfig object (uses defaults if None)

        Raises:
            DualStitcherError: If framework cannot be loaded
        """
        if config is None:
            config = StitcherConfig()

        self.config = config
        self._framework = None
        self._stitcher = None

        self._load_framework()

    def _load_framework(self):
        """Load the DualStitcher framework"""
        if not FRAMEWORK_BINARY.exists():
            raise DualStitcherError(
                f"DualStitcher framework not found. "
                f"Please install DJI Studio at {FRAMEWORK_PATH.parent.parent}"
            )

        try:
            # DualStitcher depends on openmcv2 and other frameworks
            # Load dependencies first
            frameworks_dir = FRAMEWORK_PATH.parent
            dependencies = [
                frameworks_dir / "openmcv2.framework/Versions/A/openmcv2",
            ]

            for dep_path in dependencies:
                if dep_path.exists():
                    try:
                        ctypes.CDLL(str(dep_path), use_errno=True)
                        logger.debug(f"Loaded dependency: {dep_path.name}")
                    except OSError as e:
                        logger.warning(f"Could not load dependency {dep_path.name}: {e}")

            # Now load the main DualStitcher framework
            self._framework = ctypes.CDLL(str(FRAMEWORK_BINARY), use_errno=True)
            logger.debug(f"Loaded DualStitcher framework from {FRAMEWORK_BINARY}")

        except OSError as e:
            raise DualStitcherError(
                f"Failed to load DualStitcher framework: {e}. "
                "Make sure DJI Studio is installed and dependencies are available."
            ) from e

    def is_available(self) -> bool:
        """
        Check if DualStitcher framework is available.

        Returns:
            True if framework is loaded and accessible
        """
        return self._framework is not None and FRAMEWORK_BINARY.exists()

    def stitch_frame(
        self,
        img0: bytes,
        img1: bytes,
        img_width: int = 3840,
        img_height: int = 3840,
        output_width: int = 7680,
        output_height: int = 3840,
    ) -> bytes:
        """
        Stitch two 360° camera images into an equirectangular frame.

        Note: This is a simplified API for testing. The actual DualStitcher
        framework handles 8 cameras, not just 2.

        Args:
            img0: First image bytes (HEVC-encoded or raw)
            img1: Second image bytes (HEVC-encoded or raw)
            img_width: Input image width
            img_height: Input image height
            output_width: Output equirectangular width (typically 7680)
            output_height: Output equirectangular height (typically 3840)

        Returns:
            Stitched image bytes

        Raises:
            DualStitcherError: If stitching fails
        """
        raise NotImplementedError(
            "Direct stitching API not yet implemented. "
            "Use OSVConverter for full pipeline."
        )

    def get_info(self) -> dict:
        """Get information about the loaded DualStitcher framework"""
        return {
            "framework_path": str(FRAMEWORK_BINARY),
            "is_available": self.is_available(),
            "camera_sn": self.config.camera_sn,
            "device_param_index": self.config.device_param_index,
            "device_type": "Metal" if self.config.device_type == DEVICE_TYPE_APPLE_GPU else "CUDA",
            "config": {
                "enable_sun_glare_correction": self.config.enable_sun_glare_correction,
                "enable_lsc": self.config.enable_lsc,
                "enable_ai": self.config.enable_ai,
                "dynamic_stitching": self.config.dynamic_stitching,
            },
        }

    def __repr__(self) -> str:
        return (
            f"DualStitcher(camera={self.config.camera_sn}, "
            f"available={self.is_available()})"
        )


# Export main classes
__all__ = ["DualStitcher", "StitcherConfig", "DualStitcherError"]
