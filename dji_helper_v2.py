#!/usr/bin/env python3
"""DJI DualStitcher Helper v2 - Improved initialization with better diagnostics"""

import sys
import json
import pickle
import ctypes
from pathlib import Path
from typing import Dict, Any
import numpy as np
import logging
import signal

# Setup logging to stderr
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def load_dependencies():
    """Pre-load dependencies in correct order"""
    frameworks_dir = Path("/Applications/DJI Studio.app/Contents/Frameworks")

    deps = [
        "libdInfer.dylib",
        "libMNN.3.1.2.dylib",
        "ml_fis.framework/ml_fis",
        "openmcv2.framework/openmcv2",
    ]

    logger.debug("Pre-loading dependencies...")
    for dep_name in deps:
        dep_path = frameworks_dir / dep_name
        if dep_path.exists():
            try:
                lib = ctypes.CDLL(str(dep_path), mode=0x1)  # RTLD_LAZY
                logger.debug(f"Loaded: {dep_name}")
            except OSError as e:
                logger.warning(f"Failed to pre-load {dep_name}: {e}")
        else:
            logger.warning(f"Dependency not found: {dep_path}")


def try_alternative_init(lib, param_path: str):
    """Try alternative ways to call the init function"""

    logger.debug("Trying init with different approaches...")

    # Approach 1: NULL-style initialization (see if init can work without path)
    logger.debug("Approach 1: Init with NULL pointers")
    try:
        # Try with NULL pointers instead of strings
        lib.ML_ImageStitcherInit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.ML_ImageStitcherInit.restype = ctypes.c_int
        result = lib.ML_ImageStitcherInit(None, None, None)
        logger.info(f"Init with NULL pointers returned: {result}")
        if result == 0:
            return True
    except Exception as e:
        logger.debug(f"Approach 1 failed: {e}")

    # Approach 2: Try with c_char_p but no path
    logger.debug("Approach 2: Init with empty c_char_p")
    try:
        lib.ML_ImageStitcherInit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.ML_ImageStitcherInit.restype = ctypes.c_int
        result = lib.ML_ImageStitcherInit(b"", b"", b"")
        logger.info(f"Init with empty strings returned: {result}")
        if result == 0:
            return True
    except Exception as e:
        logger.debug(f"Approach 2 failed: {e}")

    # Approach 3: Try with c_char_p and param path
    logger.debug("Approach 3: Init with param_path in 2nd arg")
    try:
        lib.ML_ImageStitcherInit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.ML_ImageStitcherInit.restype = ctypes.c_int
        param_path_bytes = param_path.encode('utf-8')
        result = lib.ML_ImageStitcherInit(b"", param_path_bytes, b"")
        logger.info(f"Init with param_path returned: {result}")
        if result == 0:
            return True
    except Exception as e:
        logger.debug(f"Approach 3 failed: {e}")

    return False


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

        # Verify inputs
        assert frame_left.shape == (3840, 3840, 3), f"Invalid left shape: {frame_left.shape}"
        assert frame_right.shape == (3840, 3840, 3), f"Invalid right shape: {frame_right.shape}"

        # Ensure contiguous arrays
        frame_left = np.ascontiguousarray(frame_left)
        frame_right = np.ascontiguousarray(frame_right)

        # Load calibration
        calib_file = param_path / "intrinsic_extrinsic.json"
        if not calib_file.exists():
            logger.error(f"Calibration file not found: {calib_file}")
            sys.exit(1)

        # Pre-load dependencies
        load_dependencies()

        # Get framework path
        framework_path = Path(input_data.get('framework_path',
            "/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework/Versions/A/DualStitcher"))

        if not framework_path.exists():
            logger.error(f"Framework binary not found: {framework_path}")
            sys.exit(1)

        # Load framework via ctypes
        logger.debug("Loading framework via ctypes...")
        try:
            lib = ctypes.CDLL(str(framework_path), mode=0x1)  # Try RTLD_LAZY instead
            logger.info(f"Successfully loaded framework")
        except OSError as e:
            logger.error(f"Failed to load framework: {e}")
            sys.exit(1)

        # Define function signatures
        logger.debug("Defining C function signatures...")

        lib.ML_ImageStitcherInit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.ML_ImageStitcherInit.restype = ctypes.c_int

        lib.ML_ImageStitcherDeInit.argtypes = []
        lib.ML_ImageStitcherDeInit.restype = ctypes.c_int

        lib.ML_ImageStitcherProcessStepA.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.ML_ImageStitcherProcessStepA.restype = ctypes.c_int

        lib.ML_ImageStitcherProcessStepB.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.ML_ImageStitcherProcessStepB.restype = ctypes.c_int

        # Try initializing with alternative approaches
        logger.info("Attempting initialization...")
        if not try_alternative_init(lib, str(param_path)):
            logger.warning("All init approaches failed, trying standard init anyway...")

        # Try calling the standard init
        logger.info("Calling ML_ImageStitcherInit...")
        init_result = lib.ML_ImageStitcherInit(b"", str(param_path).encode('utf-8'), b"")
        logger.info(f"ML_ImageStitcherInit returned: {init_result}")

        # Call ML_ImageStitcherProcessStepA
        logger.info("Calling ML_ImageStitcherProcessStepA...")
        width, height = 3840, 3840
        stride = width * 3

        left_ptr = frame_left.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        right_ptr = frame_right.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        step_a_result = lib.ML_ImageStitcherProcessStepA(left_ptr, right_ptr, width, height, stride)
        logger.info(f"ML_ImageStitcherProcessStepA returned: {step_a_result}")

        # Call ML_ImageStitcherProcessStepB
        logger.info("Calling ML_ImageStitcherProcessStepB...")
        output_width, output_height = 7680, 3840

        output_buffer = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        output_buffer = np.ascontiguousarray(output_buffer)
        output_ptr = output_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        step_b_result = lib.ML_ImageStitcherProcessStepB(output_ptr, output_width, output_height)
        logger.info(f"ML_ImageStitcherProcessStepB returned: {step_b_result}")

        # Cleanup
        logger.info("Calling ML_ImageStitcherDeInit...")
        lib.ML_ImageStitcherDeInit()

        # Write output
        logger.info(f"Stitching completed, output shape: {output_buffer.shape}")
        pickle.dump(output_buffer, sys.stdout.buffer)

    except Exception as e:
        logger.error(f"Error in stitcher: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
