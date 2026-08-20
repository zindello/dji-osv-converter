#!/usr/bin/env python3
"""DJI DualStitcher Helper - Subprocess wrapper for framework loading

This script runs in a subprocess with proper DYLD_* environment variables.
It loads the DualStitcher framework and performs the actual stitching via ctypes.

Stdin: pickle-encoded dict with:
  - 'left': numpy array (3840x3840x3 uint8)
  - 'right': numpy array (3840x3840x3 uint8)
  - 'param_path': string path to framework resources

Stdout: pickle-encoded numpy array (7680x3840x3 uint8)
"""

import sys
import json
import pickle
import ctypes
from pathlib import Path
from typing import Dict, Any
import numpy as np
import logging

# Setup logging to stderr so it doesn't interfere with pickle output
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
                lib = ctypes.CDLL(str(dep_path), mode=0x9, use_errno=True)
                logger.debug(f"Loaded: {dep_name}")
            except OSError as e:
                logger.warning(f"Failed to pre-load {dep_name}: {e}")
        else:
            logger.warning(f"Dependency not found: {dep_path}")


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

        # Load calibration
        calib_file = param_path / "intrinsic_extrinsic.json"
        logger.debug(f"Loading calibration from {calib_file}")

        if not calib_file.exists():
            logger.error(f"Calibration file not found: {calib_file}")
            sys.exit(1)

        with open(calib_file, 'r') as f:
            calib_data = json.load(f)

        logger.debug(f"Calibration loaded: {list(calib_data.keys())}")

        # Pre-load dependencies
        load_dependencies()

        # Get framework path from environment or derive it
        framework_path = Path(input_data.get('framework_path',
            "/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework/Versions/A/DualStitcher"))

        logger.debug(f"Framework path: {framework_path}")

        if not framework_path.exists():
            logger.error(f"Framework binary not found: {framework_path}")
            sys.exit(1)

        # Load framework via ctypes
        logger.debug("Loading framework via ctypes...")
        try:
            lib = ctypes.CDLL(str(framework_path), mode=0x9, use_errno=True)
            logger.info(f"Successfully loaded framework: {framework_path}")
        except OSError as e:
            logger.error(f"Failed to load framework: {e}")
            logger.error(f"Error code: {ctypes.get_errno()}")
            sys.exit(1)

        # Define C function signatures
        logger.debug("Defining C function signatures...")

        lib.ML_ImageStitcherInit.argtypes = [
            ctypes.c_char_p,  # reserved (empty)
            ctypes.c_char_p,  # param_path
            ctypes.c_char_p   # reserved (empty)
        ]
        lib.ML_ImageStitcherInit.restype = ctypes.c_int

        lib.ML_ImageStitcherDeInit.argtypes = []
        lib.ML_ImageStitcherDeInit.restype = ctypes.c_int

        lib.ML_ImageStitcherProcessStepA.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),  # left
            ctypes.POINTER(ctypes.c_uint8),  # right
            ctypes.c_int,                     # width
            ctypes.c_int,                     # height
            ctypes.c_int,                     # stride
        ]
        lib.ML_ImageStitcherProcessStepA.restype = ctypes.c_int

        lib.ML_ImageStitcherProcessStepB.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),  # output
            ctypes.c_int,                     # width
            ctypes.c_int,                     # height
        ]
        lib.ML_ImageStitcherProcessStepB.restype = ctypes.c_int

        # Call ML_ImageStitcherInit
        logger.info("Calling ML_ImageStitcherInit...")
        init_result = lib.ML_ImageStitcherInit(
            b"",  # reserved
            str(param_path).encode('utf-8'),
            b""   # reserved
        )

        if init_result != 0:
            logger.error(f"ML_ImageStitcherInit failed with code {init_result}")
            sys.exit(1)

        logger.info("ML_ImageStitcherInit succeeded")

        # Call ML_ImageStitcherProcessStepA
        logger.info("Calling ML_ImageStitcherProcessStepA...")
        width, height = 3840, 3840
        stride = width * 3  # RGB24 = 3 bytes per pixel

        left_ptr = frame_left.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        right_ptr = frame_right.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        step_a_result = lib.ML_ImageStitcherProcessStepA(left_ptr, right_ptr, width, height, stride)

        if step_a_result != 0:
            logger.error(f"ML_ImageStitcherProcessStepA failed with code {step_a_result}")
            lib.ML_ImageStitcherDeInit()
            sys.exit(1)

        logger.info("ML_ImageStitcherProcessStepA succeeded")

        # Call ML_ImageStitcherProcessStepB
        logger.info("Calling ML_ImageStitcherProcessStepB...")
        output_width, output_height = 7680, 3840

        # Allocate output buffer
        output_buffer = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        output_buffer = np.ascontiguousarray(output_buffer)
        output_ptr = output_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        step_b_result = lib.ML_ImageStitcherProcessStepB(output_ptr, output_width, output_height)

        if step_b_result != 0:
            logger.error(f"ML_ImageStitcherProcessStepB failed with code {step_b_result}")
            lib.ML_ImageStitcherDeInit()
            sys.exit(1)

        logger.info("ML_ImageStitcherProcessStepB succeeded")

        # Cleanup
        logger.info("Calling ML_ImageStitcherDeInit...")
        lib.ML_ImageStitcherDeInit()

        # Write output to stdout
        logger.debug(f"Writing output to stdout: {output_buffer.shape}, {output_buffer.dtype}")
        logger.debug(f"Output value range: [{output_buffer.min()}, {output_buffer.max()}]")
        logger.debug(f"Non-zero pixels: {np.count_nonzero(output_buffer)} / {output_buffer.size}")

        pickle.dump(output_buffer, sys.stdout.buffer)
        logger.info("Stitching completed successfully")

    except Exception as e:
        logger.error(f"Error in stitcher: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
