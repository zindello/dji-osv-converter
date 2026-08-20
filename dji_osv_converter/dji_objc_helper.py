#!/usr/bin/env python3
"""DJI DualStitcher Objective-C Helper - Subprocess wrapper for framework loading

This script runs in a subprocess with proper DYLD_* environment variables.
It loads the DualStitcher framework via Objective-C API using PyObjC.

Stdin: pickle-encoded dict with:
  - 'left': numpy array (3840x3840x3 uint8)
  - 'right': numpy array (3840x3840x3 uint8)
  - 'param_path': string path to framework resources

Stdout: pickle-encoded numpy array (7680x3840x3 uint8)
"""

import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Setup logging to stderr so it doesn't interfere with pickle output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def load_framework_objc():
    """Load DualStitcher framework via PyObjC and return Stitcher class

    Returns:
        Stitcher class instance or None on error
    """
    try:
        import objc
        from Foundation import NSBundle, NSAutoreleasePool
        from Metal import MTLCreateSystemDefaultDevice
        logger.debug("PyObjC imports successful")
    except ImportError as e:
        logger.error(f"PyObjC not available: {e}")
        return None, None

    framework_path = Path("/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework")

    try:
        # Create autorelease pool
        pool = NSAutoreleasePool.alloc().init()

        # Load framework via NSBundle
        logger.debug(f"Loading framework from {framework_path}")
        bundle = NSBundle.bundleWithPath_(str(framework_path))
        if not bundle:
            logger.error("Failed to load framework via NSBundle")
            return None, None

        logger.debug("Framework loaded via NSBundle")

        # Get the Stitcher class
        try:
            from objc import lookUpClass
            stitcher_class = lookUpClass("Stitcher")
            logger.debug("Stitcher class found")
        except Exception as e:
            logger.error(f"Could not find Stitcher class: {e}")
            return None, None

        # Get Metal device
        try:
            metal_device = MTLCreateSystemDefaultDevice()
            if metal_device:
                logger.debug("Metal device created")
            else:
                logger.warning("No Metal device available")
        except Exception as e:
            logger.warning(f"Failed to create Metal device: {e}")
            metal_device = None

        return stitcher_class, pool

    except Exception as e:
        logger.error(f"Framework loading failed: {e}", exc_info=True)
        return None, None


def stitch_frames_objc(stitcher_class, frame_left: np.ndarray, frame_right: np.ndarray) -> Optional[np.ndarray]:
    """Perform stitching using Objective-C Stitcher class

    Args:
        stitcher_class: Stitcher ObjC class
        frame_left: Left frame (3840×3840×3 uint8)
        frame_right: Right frame (3840×3840×3 uint8)

    Returns:
        Stitched equirectangular frame (7680×3840×3 uint8) or None
    """
    try:
        from Foundation import NSData
        import objc

        # Create Stitcher instance
        logger.debug("Creating Stitcher instance")
        stitcher = stitcher_class.alloc().init()
        if not stitcher:
            logger.error("Failed to create Stitcher instance")
            return None

        logger.debug("Stitcher instance created")

        # Ensure frames are contiguous
        frame_left = np.ascontiguousarray(frame_left)
        frame_right = np.ascontiguousarray(frame_right)

        # Try different stitching methods
        result = None

        # Try method 1: aiStitchSeams:seam1:isCompensate:enableSunGlare:sgGammaTex:
        try:
            logger.debug("Trying aiStitchSeams (5-param)...")
            result = stitcher.aiStitchSeams_seam1_isCompensate_enableSunGlare_sgGammaTex_(
                frame_left,  # seam0
                frame_right,  # seam1
                True,  # isCompensate
                False,  # enableSunGlare
                None   # sgGammaTex
            )
            if result is not None:
                logger.debug("aiStitchSeams (5-param) succeeded")
                return result
        except Exception as e:
            logger.debug(f"aiStitchSeams (5-param) failed: {e}")

        # Try method 2: aiStitchSeams:seam1:isCompensate:useLSC:enableSunGlare:sgGammaTex:
        try:
            logger.debug("Trying aiStitchSeams (6-param)...")
            result = stitcher.aiStitchSeams_seam1_isCompensate_useLSC_enableSunGlare_sgGammaTex_(
                frame_left,  # seam0
                frame_right,  # seam1
                True,  # isCompensate
                False,  # useLSC
                False,  # enableSunGlare
                None   # sgGammaTex
            )
            if result is not None:
                logger.debug("aiStitchSeams (6-param) succeeded")
                return result
        except Exception as e:
            logger.debug(f"aiStitchSeams (6-param) failed: {e}")

        # Try method 3: Simple process
        try:
            logger.debug("Trying processImage...")
            result = stitcher.processImage_(frame_left, frame_right)
            if result is not None:
                logger.debug("processImage succeeded")
                return result
        except Exception as e:
            logger.debug(f"processImage failed: {e}")

        logger.error("No stitching method succeeded")
        return None

    except Exception as e:
        logger.error(f"Stitching failed: {e}", exc_info=True)
        return None


def main():
    """Main entry point for subprocess stitcher"""
    try:
        # Read input from stdin
        logger.debug("Reading input from stdin...")
        input_data = pickle.load(sys.stdin.buffer)

        frame_left = input_data['left']
        frame_right = input_data['right']
        param_path = Path(input_data['param_path'])

        logger.debug(f"Loaded frames: left={frame_left.shape}, right={frame_right.shape}")
        logger.debug(f"Param path: {param_path}")

        # Verify inputs
        assert frame_left.shape == (3840, 3840, 3), f"Invalid left shape: {frame_left.shape}"
        assert frame_right.shape == (3840, 3840, 3), f"Invalid right shape: {frame_right.shape}"
        assert frame_left.dtype == np.uint8, f"Invalid left dtype: {frame_left.dtype}"
        assert frame_right.dtype == np.uint8, f"Invalid right dtype: {frame_right.dtype}"

        # Ensure contiguous arrays
        frame_left = np.ascontiguousarray(frame_left)
        frame_right = np.ascontiguousarray(frame_right)

        # Load calibration (for reference, not used in ObjC stitching)
        calib_file = param_path / "intrinsic_extrinsic.json"
        logger.debug(f"Loading calibration from {calib_file}")
        if calib_file.exists():
            with open(calib_file, 'r') as f:
                calib_data = json.load(f)
            logger.debug(f"Calibration loaded: {list(calib_data.keys())}")

        # Load framework and get Stitcher class
        logger.debug("Loading DualStitcher framework via Objective-C...")
        stitcher_class, pool = load_framework_objc()

        if not stitcher_class:
            logger.error("Failed to load Stitcher class")
            sys.exit(1)

        logger.info("Framework loaded successfully")

        # Perform stitching
        logger.info("Performing Objective-C stitching...")
        output_buffer = stitch_frames_objc(stitcher_class, frame_left, frame_right)

        if output_buffer is None:
            logger.error("Stitching failed")
            sys.exit(1)

        # Verify output
        logger.debug(f"Stitching output shape: {output_buffer.shape}")
        logger.debug(f"Output value range: [{output_buffer.min()}, {output_buffer.max()}]")
        logger.debug(f"Non-zero pixels: {np.count_nonzero(output_buffer)} / {output_buffer.size}")

        # Handle different output shapes
        if output_buffer.shape != (3840, 7680, 3):
            if output_buffer.shape == (7680, 3840, 3):
                logger.warning("Output is rotated (7680×3840 instead of 3840×7680)")
            else:
                logger.warning(f"Unexpected output shape: {output_buffer.shape}")
                # Try to reshape
                expected_size = 7680 * 3840 * 3
                if output_buffer.size == expected_size:
                    output_buffer = output_buffer.reshape((3840, 7680, 3))
                    logger.info("Reshaped output to (3840, 7680, 3)")
                else:
                    logger.error(f"Output size doesn't match expected: {output_buffer.size} != {expected_size}")
                    sys.exit(1)

        # Write output to stdout
        logger.debug(f"Writing output to stdout: {output_buffer.shape}, {output_buffer.dtype}")
        pickle.dump(output_buffer, sys.stdout.buffer)
        logger.info("Stitching completed successfully")

    except Exception as e:
        logger.error(f"Error in stitcher: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
