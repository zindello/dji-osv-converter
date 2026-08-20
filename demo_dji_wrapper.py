#!/usr/bin/env python3
"""
Demonstration of DJI DualStitcher Wrapper

This script shows:
1. Framework detection and validation
2. Calibration data loading
3. OSV frame extraction
4. Stitcher initialization and readiness

Blocked at: Actual stitching call (C API signature issue)
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from dji_osv_converter.stitch_dji_wrapper import (
    DJIDualStitcher,
    FRAMEWORK_BINARY,
    FRAMEWORK_RESOURCES,
)


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def demo():
    """Run the demonstration"""

    print_section("DJI DualStitcher Framework Integration Demo")

    # 1. Check framework paths
    print_section("1. Framework Detection")
    print(f"Framework binary: {FRAMEWORK_BINARY}")
    print(f"  Exists: {FRAMEWORK_BINARY.exists()}")
    print(f"  File type: {__import__('subprocess').run(['file', str(FRAMEWORK_BINARY)], capture_output=True, text=True).stdout.strip()}")

    print(f"\nFramework resources: {FRAMEWORK_RESOURCES}")
    print(f"  Exists: {FRAMEWORK_RESOURCES.exists()}")

    calib_file = FRAMEWORK_RESOURCES / "intrinsic_extrinsic.json"
    print(f"\nCalibration file: {calib_file.name}")
    print(f"  Exists: {calib_file.exists()}")

    # 2. Initialize wrapper
    print_section("2. Wrapper Initialization")
    try:
        stitcher = DJIDualStitcher(camera_sn="default")
        print(f"✓ DJIDualStitcher created")
        print(f"  Camera SN: {stitcher.camera_sn}")
        print(f"  Framework available: {stitcher.is_available()}")

        info = stitcher.get_info()
        for key, val in info.items():
            print(f"  {key}: {val}")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return False

    # 3. Check calibration
    print_section("3. Calibration Data")
    if stitcher.calibration:
        calib = stitcher.calibration
        print(f"✓ Calibration data loaded")
        print(f"\n  Left Camera:")
        print(f"    Focal length (fx): {calib.intrinsic00.get('fx', 'N/A'):.2f}")
        print(f"    Principal point (cx, cy): ({calib.intrinsic00.get('cx', 'N/A'):.1f}, {calib.intrinsic00.get('cy', 'N/A'):.1f})")
        print(f"    Fisheye parameter (xi): {calib.get_fisheye_param_left():.4f}")
        print(f"    K1 distortion: {calib.intrinsic00.get('k1', 'N/A'):.4f}")

        print(f"\n  Right Camera:")
        print(f"    Focal length (fx): {calib.intrinsic01.get('fx', 'N/A'):.2f}")
        print(f"    Principal point (cx, cy): ({calib.intrinsic01.get('cx', 'N/A'):.1f}, {calib.intrinsic01.get('cy', 'N/A'):.1f})")
        print(f"    Fisheye parameter (xi): {calib.get_fisheye_param_right():.4f}")

        K_left = calib.get_camera_matrix_left()
        print(f"\n  Left Camera Matrix K:")
        print(f"    [{K_left[0,0]:.0f}    0  {K_left[0,2]:.1f}]")
        print(f"    [   0  {K_left[1,1]:.0f}  {K_left[1,2]:.1f}]")
        print(f"    [   0     0       1    ]")
    else:
        print("✗ No calibration data loaded")

    # 4. OSV frame extraction
    print_section("4. OSV Frame Extraction")
    osv_path = Path("/Users/zindello/repositories/zindello/djiclireframe/osv_example/CAM_20260819204151_0011_D.OSV")

    if osv_path.exists():
        print(f"OSV file found: {osv_path.name}")
        print(f"  Size: {osv_path.stat().st_size / (1024*1024*1024):.2f} GB")

        try:
            from dji_osv_converter.stitch_dji_wrapper import extract_osv_frames
            import numpy as np

            print(f"\nExtracting frame pairs [0, 1, 2]...")
            frames = extract_osv_frames(osv_path, [0, 1, 2])

            if frames:
                print(f"✓ Successfully extracted {len(frames)} frame pairs")

                for i, (frame_left, frame_right) in enumerate(frames):
                    print(f"\n  Frame pair {i}:")
                    print(f"    Left shape: {frame_left.shape}, dtype: {frame_left.dtype}")
                    print(f"    Left range: [{frame_left.min()}, {frame_left.max()}]")
                    print(f"    Right shape: {frame_right.shape}, dtype: {frame_right.dtype}")
                    print(f"    Right range: [{frame_right.min()}, {frame_right.max()}]")
            else:
                print("✗ Failed to extract frames")

        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
    else:
        print(f"✗ OSV file not found: {osv_path}")

    # 5. Current Status
    print_section("5. Current Status & Known Issues")
    print("""
✓ WORKING:
  - Framework detection and loading
  - All dependencies (openmcv2, ml_fis, libMNN, libdInfer) present
  - Calibration data extraction
  - OSV HEVC frame decoding via PyAV
  - Subprocess-based environment setup with DYLD variables
  - C API symbols resolution

✗ BLOCKED:
  - ML_ImageStitcherInit() crashes with SIGSEGV (exit code -11)
  - Likely causes:
    1. Incorrect function signature
    2. Missing initialization context
    3. Private/undocumented C API
    4. Objective-C dependency

📋 NEXT STEPS:
  1. Use Objective-C API via PyObjC bridge (stitch_dualstitcher.py)
  2. Reverse-engineer function signatures with lldb
  3. Access DJI SDK documentation/headers
  4. Use alternative stitching library (OpenCV, Hugin, etc.)
""")

    return True


if __name__ == "__main__":
    try:
        success = demo()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)
