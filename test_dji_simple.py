#!/usr/bin/env python3
"""Simple test of DJI DualStitcher framework loading"""

import sys
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from dji_osv_converter.stitch_dji_wrapper import (
    DJIDualStitcher,
    FRAMEWORK_BINARY,
    FRAMEWORKS_DIR,
)

def test_framework_loading():
    """Test if framework can be loaded"""
    print("\n" + "="*70)
    print("TEST: Framework Loading and Initialization")
    print("="*70)

    print(f"Framework binary: {FRAMEWORK_BINARY}")
    print(f"  Exists: {FRAMEWORK_BINARY.exists()}")
    print(f"Frameworks dir: {FRAMEWORKS_DIR}")
    print(f"  Exists: {FRAMEWORKS_DIR.exists()}")

    # Check dependencies
    deps = ["openmcv2.framework/openmcv2", "ml_fis.framework/ml_fis",
            "libMNN.3.1.2.dylib", "libdInfer.dylib"]
    print(f"\nDependencies:")
    for dep in deps:
        dep_path = FRAMEWORKS_DIR / dep
        exists = dep_path.exists()
        print(f"  {dep}: {'EXISTS' if exists else 'MISSING'}")

    try:
        print("\nLoading DJIDualStitcher...")
        stitcher = DJIDualStitcher(camera_sn="default")
        print(f"Success: {stitcher}")

        info = stitcher.get_info()
        print(f"\nFramework info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        print("SUCCESS: Framework loaded and available!")
        print("\nNote: The new implementation uses subprocess for actual stitching")
        print("to properly handle DYLD_* environment variables at process startup.")
        return True

    except Exception as e:
        logger.exception(f"Failed to load framework: {e}")
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_framework_loading()
    sys.exit(0 if success else 1)
