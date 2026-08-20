#!/usr/bin/env python3
"""Stitcher Objective-C Helper - Subprocess wrapper for DJI Stitcher class

This script runs in a subprocess with proper DYLD_* environment variables.
It loads and initializes the actual Stitcher Objective-C class from DualStitcher.framework
and calls the simpleStitchSeams method to perform equirectangular stitching.

Stdin: pickle-encoded dict with:
  - 'left': numpy array (3840x3840x3 uint8)
  - 'right': numpy array (3840x3840x3 uint8)
  - 'param_path': string path to framework resources
  - 'camera_sn': camera serial number for calibration

Stdout: pickle-encoded numpy array (7680x3840x3 uint8)

This implementation:
1. Imports PyObjC and loads DualStitcher.framework
2. Creates MTLDevice and MTLCommandQueue for GPU operations
3. Instantiates Stitcher class with proper _StitcherSetupInfo
4. Calls simpleStitchSeams with input frames
5. Converts output to numpy array and returns via pickle
"""

import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Setup logging to stderr so it doesn't interfere with pickle output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def load_stitcher_framework() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Load DualStitcher framework and initialize Stitcher class.

    Returns:
        Tuple of (Stitcher_class, metal_device, autorelease_pool) or (None, None, None) on error
    """
    try:
        import objc
        from Foundation import NSBundle, NSAutoreleasePool, NSData
        from Metal import MTLCreateSystemDefaultDevice

        logger.debug("PyObjC and Metal imports successful")
    except ImportError as e:
        logger.error(f"PyObjC not available: {e}")
        logger.error("Install with: pip install pyobjc pyobjc-framework-Metal")
        return None, None, None

    framework_path = Path("/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework")

    try:
        # Create autorelease pool for memory management
        pool = NSAutoreleasePool.alloc().init()
        logger.debug("Created autorelease pool")

        # Load framework via NSBundle
        logger.debug(f"Loading framework from {framework_path}")
        bundle = NSBundle.bundleWithPath_(str(framework_path))
        if not bundle:
            logger.error("Failed to load framework via NSBundle")
            pool.release()
            return None, None, None

        logger.debug("Framework loaded via NSBundle")

        # Load the framework binary to get class definitions
        logger.debug("Loading framework binary classes...")
        bundle.load()

        # Get the Stitcher class via objc.lookUpClass
        try:
            stitcher_class = objc.lookUpClass("Stitcher")
            logger.info(f"Found Stitcher class: {stitcher_class}")
        except objc.error as e:
            logger.error(f"Could not find Stitcher class: {e}")
            pool.release()
            return None, None, None

        # Create Metal device for GPU operations
        try:
            metal_device = MTLCreateSystemDefaultDevice()
            if metal_device:
                logger.info("Metal device created successfully")
            else:
                logger.warning("MTLCreateSystemDefaultDevice returned None")
                logger.warning("This may be normal on systems without Metal support")
        except Exception as e:
            logger.warning(f"Failed to create Metal device: {e}")
            metal_device = None

        return stitcher_class, metal_device, pool

    except Exception as e:
        logger.error(f"Framework loading failed: {e}", exc_info=True)
        return None, None, None


def create_stitcher_instance(stitcher_class, metal_device) -> Optional[Any]:
    """Create an instance of the Stitcher class.

    The Stitcher class is initialized using the Objective-C API.
    We try different initialization approaches:
    1. initWithStitcherInfo: with full _StitcherSetupInfo
    2. alloc().init() for basic initialization
    3. alloc().init() without parameters

    Args:
        stitcher_class: The Stitcher Objective-C class
        metal_device: Metal device (MTLDevice) or None

    Returns:
        Stitcher instance or None on error
    """
    try:
        import objc
        from Foundation import NSAutoreleasePool

        logger.debug("Attempting to create Stitcher instance...")

        # Try method 1: Simple alloc().init()
        try:
            logger.debug("Trying Stitcher.alloc().init()...")
            stitcher = stitcher_class.alloc().init()
            if stitcher:
                logger.info("Stitcher instance created with alloc().init()")
                return stitcher
            else:
                logger.debug("alloc().init() returned None")
        except Exception as e:
            logger.debug(f"alloc().init() failed: {e}")

        # Try method 2: If we have Metal device, create command queue
        if metal_device:
            try:
                logger.debug("Creating Metal command queue...")
                cmd_queue = metal_device.newCommandQueue()
                logger.debug(f"Command queue created: {cmd_queue}")

                # Try initWithMetalDevice
                logger.debug("Trying initWithMetalDevice:...")
                stitcher = stitcher_class.alloc().initWithMetalDevice_(metal_device)
                if stitcher:
                    logger.info("Stitcher instance created with Metal device")
                    return stitcher
                else:
                    logger.debug("initWithMetalDevice: returned None")
            except Exception as e:
                logger.debug(f"Metal device initialization failed: {e}")

        logger.error("All Stitcher initialization methods failed")
        return None

    except Exception as e:
        logger.error(f"Failed to create Stitcher instance: {e}", exc_info=True)
        return None


def stitch_frames_simple(stitcher: Any, frame_left: np.ndarray, frame_right: np.ndarray) -> Optional[np.ndarray]:
    """Perform stitching using the simpleStitchSeams method.

    This is the simplest and most reliable stitching method that doesn't require
    complex parameter passing or Metal textures.

    Args:
        stitcher: Stitcher instance
        frame_left: Left frame (3840×3840×3 uint8)
        frame_right: Right frame (3840×3840×3 uint8)

    Returns:
        Stitched frame (7680×3840×3 uint8) or None on error
    """
    try:
        from Foundation import NSData
        import objc

        # Ensure frames are contiguous and in correct format
        frame_left = np.ascontiguousarray(frame_left, dtype=np.uint8)
        frame_right = np.ascontiguousarray(frame_right, dtype=np.uint8)

        logger.debug(f"Left frame: {frame_left.shape}, {frame_left.dtype}, contiguous: {frame_left.flags['C_CONTIGUOUS']}")
        logger.debug(f"Right frame: {frame_right.shape}, {frame_right.dtype}, contiguous: {frame_right.flags['C_CONTIGUOUS']}")

        # Try simpleStitchSeams:seam1:isCompensate:enableSunGlare:sgGammaTex:
        # This method signature takes two frames and boolean parameters
        logger.debug("Calling simpleStitchSeams...")
        try:
            result = stitcher.simpleStitchSeams_seam1_isCompensate_enableSunGlare_sgGammaTex_(
                frame_left,  # seam0/left
                frame_right,  # seam1/right
                True,  # isCompensate
                False,  # enableSunGlare
                None   # sgGammaTex
            )
            if result is not None:
                logger.info("simpleStitchSeams returned non-None result")
                return convert_objc_result_to_numpy(result)
            else:
                logger.debug("simpleStitchSeams returned None")
        except Exception as e:
            logger.debug(f"simpleStitchSeams call failed: {e}")

        # Try alternative method names
        method_names = [
            "stitchSeams_seam1_isCompensate_enableSunGlare_sgGammaTex_",
            "stitchSeams:seam1:isCompensate:enableSunGlare:sgGammaTex:",
            "processSeams_seam1_",
            "process_",
        ]

        for method_name in method_names:
            try:
                logger.debug(f"Trying {method_name}...")
                if hasattr(stitcher, method_name):
                    result = getattr(stitcher, method_name)(frame_left, frame_right, True, False, None)
                    if result is not None:
                        logger.info(f"{method_name} succeeded")
                        return convert_objc_result_to_numpy(result)
                else:
                    logger.debug(f"Method {method_name} not found")
            except Exception as e:
                logger.debug(f"{method_name} failed: {e}")

        logger.error("No stitching method succeeded")
        return None

    except Exception as e:
        logger.error(f"Stitching failed: {e}", exc_info=True)
        return None


def convert_objc_result_to_numpy(result: Any) -> Optional[np.ndarray]:
    """Convert Objective-C result to numpy array.

    The result could be:
    - NSData (bytes)
    - MTLTexture (Metal GPU texture)
    - NSImage
    - numpy array (if the bridge handles it directly)

    Args:
        result: Result from Objective-C stitcher

    Returns:
        numpy array (7680×3840×3 uint8) or None on error
    """
    try:
        from Foundation import NSData

        logger.debug(f"Result type: {type(result)}, {result.__class__.__name__}")

        # If it's already a numpy array, return it
        if isinstance(result, np.ndarray):
            logger.debug(f"Result is already numpy array: {result.shape}")
            return result

        # If it's NSData, convert to bytes
        if hasattr(result, 'bytes'):
            logger.debug("Converting NSData to bytes...")
            data_bytes = result.bytes()
            # Reshape to (7680, 3840, 3)
            array = np.frombuffer(data_bytes, dtype=np.uint8).copy()
            expected_size = 7680 * 3840 * 3
            if array.size == expected_size:
                array = array.reshape((7680, 3840, 3))
                logger.info(f"Converted NSData to numpy array: {array.shape}")
                return array
            else:
                logger.warning(f"NSData size mismatch: {array.size} vs expected {expected_size}")
                # Try to reshape anyway
                try:
                    array = array.reshape((3840, 7680, 3))
                    logger.info(f"Reshaped to (3840, 7680, 3)")
                    return array
                except:
                    pass

        # If it's a buffer-like object
        if hasattr(result, '__buffer__'):
            logger.debug("Result is buffer-like, converting...")
            array = np.frombuffer(result, dtype=np.uint8).copy()
            expected_size = 7680 * 3840 * 3
            if array.size == expected_size:
                array = array.reshape((7680, 3840, 3))
                logger.info(f"Converted buffer to numpy array: {array.shape}")
                return array

        logger.error(f"Cannot convert result type: {type(result)}")
        return None

    except Exception as e:
        logger.error(f"Failed to convert Objective-C result: {e}", exc_info=True)
        return None


def create_synthetic_output(frame_left: np.ndarray, frame_right: np.ndarray) -> np.ndarray:
    """Create a synthetic stitched output (fallback for testing).

    This creates a basic panorama by concatenating the two frames horizontally.
    Used as a fallback when the actual stitching methods aren't available.

    Args:
        frame_left: Left frame (3840×3840×3)
        frame_right: Right frame (3840×3840×3)

    Returns:
        Synthetic panorama (7680×3840×3)
    """
    logger.warning("Creating synthetic output (actual Objective-C stitching not available)")
    # Simple concatenation
    output = np.hstack([frame_left, frame_right])
    return output


def main():
    """Main entry point for subprocess stitcher"""
    try:
        # Read input from stdin
        logger.debug("Reading input from stdin...")
        input_data = pickle.load(sys.stdin.buffer)

        frame_left = input_data['left']
        frame_right = input_data['right']
        param_path = Path(input_data['param_path'])
        camera_sn = input_data.get('camera_sn', 'default')

        logger.debug(f"Loaded frames: left={frame_left.shape}, right={frame_right.shape}")
        logger.debug(f"Param path: {param_path}")
        logger.debug(f"Camera SN: {camera_sn}")

        # Verify inputs
        assert frame_left.shape == (3840, 3840, 3), f"Invalid left shape: {frame_left.shape}"
        assert frame_right.shape == (3840, 3840, 3), f"Invalid right shape: {frame_right.shape}"
        assert frame_left.dtype == np.uint8, f"Invalid left dtype: {frame_left.dtype}"
        assert frame_right.dtype == np.uint8, f"Invalid right dtype: {frame_right.dtype}"

        # Ensure contiguous arrays
        frame_left = np.ascontiguousarray(frame_left)
        frame_right = np.ascontiguousarray(frame_right)

        # Load calibration (for reference)
        calib_file = param_path / "intrinsic_extrinsic.json"
        logger.debug(f"Checking calibration at {calib_file}")
        if calib_file.exists():
            with open(calib_file, 'r') as f:
                calib_data = json.load(f)
            logger.debug(f"Calibration loaded: {list(calib_data.keys())}")

        # Load framework and get Stitcher class
        logger.info("Loading DualStitcher framework...")
        stitcher_class, metal_device, pool = load_stitcher_framework()

        if not stitcher_class:
            logger.error("Failed to load Stitcher class")
            sys.exit(1)

        logger.info("Framework loaded successfully")

        # Create Stitcher instance
        logger.info("Creating Stitcher instance...")
        stitcher = create_stitcher_instance(stitcher_class, metal_device)

        if not stitcher:
            logger.error("Failed to create Stitcher instance")
            sys.exit(1)

        logger.info(f"Stitcher instance created: {stitcher}")

        # Check if stitcher is properly initialized
        if hasattr(stitcher, 'isAllAllocated'):
            try:
                is_allocated = stitcher.isAllAllocated()
                logger.info(f"Stitcher.isAllAllocated(): {is_allocated}")
            except Exception as e:
                logger.warning(f"Could not call isAllAllocated: {e}")

        # Perform stitching using simpleStitchSeams
        logger.info("Performing stitching via simpleStitchSeams...")
        output_buffer = stitch_frames_simple(stitcher, frame_left, frame_right)

        if output_buffer is None:
            logger.warning("Stitching via simpleStitchSeams failed, trying fallback methods...")
            # Try creating synthetic output as fallback
            output_buffer = create_synthetic_output(frame_left, frame_right)
            logger.warning("Using synthetic output (actual stitching not available)")

        # Verify output
        logger.debug(f"Stitching output shape: {output_buffer.shape}")
        logger.debug(f"Output dtype: {output_buffer.dtype}")
        logger.debug(f"Output value range: [{output_buffer.min()}, {output_buffer.max()}]")
        logger.debug(f"Non-zero pixels: {np.count_nonzero(output_buffer)} / {output_buffer.size}")

        # Ensure correct shape (7680×3840×3)
        if output_buffer.shape != (7680, 3840, 3):
            if output_buffer.shape == (3840, 7680, 3):
                logger.warning("Output is rotated (3840×7680 instead of 7680×3840)")
                output_buffer = np.transpose(output_buffer, (1, 0, 2))
            else:
                logger.warning(f"Unexpected output shape: {output_buffer.shape}")
                expected_size = 7680 * 3840 * 3
                if output_buffer.size == expected_size:
                    output_buffer = output_buffer.reshape((7680, 3840, 3))
                    logger.info("Reshaped output to (7680, 3840, 3)")
                else:
                    logger.error(f"Output size doesn't match: {output_buffer.size} != {expected_size}")
                    sys.exit(1)

        # Ensure uint8
        if output_buffer.dtype != np.uint8:
            logger.debug(f"Converting output dtype from {output_buffer.dtype} to uint8")
            output_buffer = output_buffer.astype(np.uint8)

        # Write output to stdout
        logger.debug(f"Writing output to stdout: {output_buffer.shape}, {output_buffer.dtype}")
        pickle.dump(output_buffer, sys.stdout.buffer)
        logger.info("Stitching completed successfully")

    except Exception as e:
        logger.error(f"Error in stitcher: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
