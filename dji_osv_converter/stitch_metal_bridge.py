#!/usr/bin/env python3
"""Python wrapper for Metal Stitcher Bridge

This module provides a Python interface to the native Objective-C Metal bridge
that creates actual Metal textures and calls DJI's dynamicStitchSeams method.

The bridge is implemented in metal_stitcher_bridge.m and is compiled into a
shared library that can be called from Python via ctypes.

Usage:
    bridge = MetalStitcherBridge()
    output = bridge.stitch_frames(left_frame, right_frame, seam_mask_1, seam_mask_2, flow_texture)
    # output is 7680×3840×3 uint8 RGB array
"""

import logging
import subprocess
import sys
import os
import pickle
import tempfile
import struct
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import ctypes
from ctypes import CDLL, c_void_p, c_char_p, c_int, c_float, POINTER, Structure

logger = logging.getLogger(__name__)

# Get module directory
_MODULE_DIR = Path(__file__).parent
BRIDGE_SOURCE = _MODULE_DIR / "metal_stitcher_bridge.m"
BRIDGE_BUILD_DIR = _MODULE_DIR / ".metal_bridge_build"
BRIDGE_DYLIB = BRIDGE_BUILD_DIR / "libmetalstitcher.dylib"


class FrameDataC(Structure):
    """C structure matching FrameData in metal_stitcher_bridge.m"""
    _fields_ = [
        ("left_frame", POINTER(c_char_p)),
        ("right_frame", POINTER(c_char_p)),
        ("seam_mask_1", POINTER(c_char_p)),
        ("seam_mask_2", POINTER(c_char_p)),
        ("flow_texture", POINTER(c_float)),
        ("left_height", c_int),
        ("left_width", c_int),
        ("left_channels", c_int),
        ("right_height", c_int),
        ("right_width", c_int),
        ("right_channels", c_int),
        ("seam1_height", c_int),
        ("seam1_width", c_int),
        ("seam1_channels", c_int),
        ("seam2_height", c_int),
        ("seam2_width", c_int),
        ("seam2_channels", c_int),
        ("flow_height", c_int),
        ("flow_width", c_int),
        ("flow_channels", c_int),
        ("camera_sn", c_char_p),
        ("param_path", c_char_p),
    ]


class MetalStitcherBridge:
    """Native Metal bridge for DJI Stitcher

    This class provides Python bindings to the native Objective-C bridge
    that creates real Metal textures and calls dynamicStitchSeams.
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

    def __init__(self, rebuild: bool = False):
        """Initialize Metal Stitcher Bridge

        Args:
            rebuild: Force rebuild of the bridge dylib
        """
        self.rebuild = rebuild
        self.dylib = None
        self._ensure_bridge_built()

    @staticmethod
    def compile_bridge() -> bool:
        """Compile the Objective-C bridge into a dylib

        Returns:
            True if compilation successful, False otherwise
        """
        if not BRIDGE_SOURCE.exists():
            logger.error(f"Bridge source not found: {BRIDGE_SOURCE}")
            return False

        # Create build directory
        BRIDGE_BUILD_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"Compiling Metal bridge from {BRIDGE_SOURCE}...")

        # Compile with clang
        compile_cmd = [
            "clang",
            "-fPIC",
            "-shared",
            "-framework", "Foundation",
            "-framework", "Metal",
            "-framework", "Cocoa",
            "-fobjc-arc",
            str(BRIDGE_SOURCE),
            "-o", str(BRIDGE_DYLIB),
        ]

        logger.debug(f"Compile command: {' '.join(compile_cmd)}")

        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Bridge compiled successfully: {BRIDGE_DYLIB}")
            if result.stderr:
                logger.debug(f"Compiler warnings:\n{result.stderr}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compile bridge: {e}")
            logger.error(f"Stdout:\n{e.stdout}")
            logger.error(f"Stderr:\n{e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Compilation error: {e}")
            return False

    def _ensure_bridge_built(self) -> bool:
        """Ensure the bridge dylib exists and is built

        Returns:
            True if bridge is available, False otherwise
        """
        if not self.rebuild and BRIDGE_DYLIB.exists():
            logger.debug(f"Using existing bridge: {BRIDGE_DYLIB}")
            self._load_bridge()
            return self.dylib is not None

        logger.info("Building Metal bridge...")
        if not self.compile_bridge():
            logger.error("Failed to build Metal bridge")
            return False

        self._load_bridge()
        return self.dylib is not None

    def _load_bridge(self):
        """Load the compiled dylib"""
        if not BRIDGE_DYLIB.exists():
            logger.error(f"Bridge dylib not found: {BRIDGE_DYLIB}")
            return

        try:
            self.dylib = CDLL(str(BRIDGE_DYLIB))
            logger.info(f"Bridge dylib loaded: {BRIDGE_DYLIB}")
        except Exception as e:
            logger.error(f"Failed to load bridge dylib: {e}")
            self.dylib = None

    def stitch_frames(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
        seam_mask_1: np.ndarray,
        seam_mask_2: np.ndarray,
        flow_texture: np.ndarray,
        camera_sn: str = "default",
        param_path: str = "/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework/Versions/A/Resources/Stitcher-Mac.bundle/case_00"
    ) -> Optional[np.ndarray]:
        """Perform stitching with Metal bridge

        Args:
            frame_left: Left frame (3840×3840×3 uint8 RGB)
            frame_right: Right frame (3840×3840×3 uint8 RGB)
            seam_mask_1: Seam mask 1 (7680×292×4 uint8 BGRA)
            seam_mask_2: Seam mask 2 (7680×292×4 uint8 BGRA)
            flow_texture: Flow texture data (789×394×4 float32 RGBA)
            camera_sn: Camera serial number
            param_path: Path to DJI framework resources

        Returns:
            Stitched frame (7680×3840×3 uint8 RGB) or None on error
        """
        if not self.dylib:
            logger.error("Bridge dylib not loaded")
            return None

        # Validate inputs
        frame_left = np.ascontiguousarray(frame_left, dtype=np.uint8)
        frame_right = np.ascontiguousarray(frame_right, dtype=np.uint8)
        seam_mask_1 = np.ascontiguousarray(seam_mask_1, dtype=np.uint8)
        seam_mask_2 = np.ascontiguousarray(seam_mask_2, dtype=np.uint8)
        flow_texture = np.ascontiguousarray(flow_texture, dtype=np.float32)

        assert frame_left.shape == (3840, 3840, 3), f"Invalid left frame shape: {frame_left.shape}"
        assert frame_right.shape == (3840, 3840, 3), f"Invalid right frame shape: {frame_right.shape}"
        assert seam_mask_1.shape == (292, 7680, 4), f"Invalid seam mask 1: {seam_mask_1.shape}"
        assert seam_mask_2.shape == (292, 7680, 4), f"Invalid seam mask 2: {seam_mask_2.shape}"
        assert flow_texture.shape == (394, 789, 4), f"Invalid flow texture: {flow_texture.shape}"

        logger.info("Performing stitching with Metal bridge...")
        logger.debug(f"Left: {frame_left.shape} {frame_left.dtype}")
        logger.debug(f"Right: {frame_right.shape} {frame_right.dtype}")
        logger.debug(f"Seam 1: {seam_mask_1.shape} {seam_mask_1.dtype}")
        logger.debug(f"Seam 2: {seam_mask_2.shape} {seam_mask_2.dtype}")
        logger.debug(f"Flow: {flow_texture.shape} {flow_texture.dtype}")

        # For now, we'll use the subprocess approach since the full C bridge
        # needs more complex pickle/data marshalling
        return self._stitch_via_subprocess(
            frame_left, frame_right,
            seam_mask_1, seam_mask_2,
            flow_texture,
            camera_sn, param_path
        )

    def _stitch_via_subprocess(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
        seam_mask_1: np.ndarray,
        seam_mask_2: np.ndarray,
        flow_texture: np.ndarray,
        camera_sn: str,
        param_path: str
    ) -> Optional[np.ndarray]:
        """Call stitcher via subprocess (helper bridge approach)

        This is a bridge to the existing stitcher_proper_helper.py which uses
        PyObjC to create Metal textures and call dynamicStitchSeams.

        The full C/Objective-C bridge would eliminate the PyObjC dependency.

        Args:
            Same as stitch_frames

        Returns:
            Stitched frame or None on error
        """
        from .stitcher_proper_helper import main as subprocess_main
        import sys as sys_orig

        # Import the helper module to use its functions
        from .stitcher_proper_helper import (
            load_stitcher_framework,
            create_stitcher_instance,
            stitch_frames_dynamic,
            convert_objc_result_to_numpy
        )

        try:
            # Load framework using PyObjC (until full C bridge is complete)
            stitcher_class, metal_device, pool = load_stitcher_framework()
            if not stitcher_class:
                logger.error("Failed to load Stitcher class")
                return None

            # Create instance
            stitcher = create_stitcher_instance(stitcher_class, metal_device)
            if not stitcher:
                logger.error("Failed to create Stitcher instance")
                return None

            # Perform stitching
            output = stitch_frames_dynamic(
                stitcher, frame_left, frame_right,
                seam_mask_1, seam_mask_2, flow_texture,
                metal_device
            )

            if pool:
                pool.release()

            return output

        except Exception as e:
            logger.error(f"Subprocess stitching failed: {e}", exc_info=True)
            return None


def stitch_frames_metal_bridge(
    frame_left: np.ndarray,
    frame_right: np.ndarray,
    seam_mask_1: np.ndarray,
    seam_mask_2: np.ndarray,
    flow_texture: np.ndarray,
    camera_sn: str = "default",
    rebuild: bool = False
) -> Optional[np.ndarray]:
    """Convenience function for stitching with Metal bridge

    Args:
        frame_left: Left frame (3840×3840×3 uint8)
        frame_right: Right frame (3840×3840×3 uint8)
        seam_mask_1: Seam mask 1 (292×7680×4 uint8)
        seam_mask_2: Seam mask 2 (292×7680×4 uint8)
        flow_texture: Flow texture (394×789×4 float32)
        camera_sn: Camera serial number
        rebuild: Force rebuild of bridge

    Returns:
        Stitched frame (7680×3840×3 uint8) or None
    """
    bridge = MetalStitcherBridge(rebuild=rebuild)
    return bridge.stitch_frames(
        frame_left, frame_right,
        seam_mask_1, seam_mask_2,
        flow_texture,
        camera_sn
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    bridge = MetalStitcherBridge()
    print(f"Bridge ready: {bridge.dylib}")
