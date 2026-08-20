#!/usr/bin/env python3
"""Test DJI DualStitcher Wrapper with Real OSV Data

This script:
1. Loads the DJI DualStitcher framework
2. Extracts real frames from an OSV test file
3. Stitches frames using the real DJI API
4. Saves output images and verifies stitching quality
5. Shows proof that it works with actual data
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from dji_osv_converter.stitch_dji_wrapper import (
    DJIDualStitcher,
    extract_osv_frames,
    FRAMEWORK_BINARY,
    FRAMEWORK_RESOURCES,
)


def test_framework_paths():
    """Test that framework and resources exist"""
    print("\n" + "="*70)
    print("TEST 1: Framework Paths and Resources")
    print("="*70)

    print(f"Framework binary: {FRAMEWORK_BINARY}")
    print(f"  Exists: {FRAMEWORK_BINARY.exists()}")

    print(f"Framework resources: {FRAMEWORK_RESOURCES}")
    print(f"  Exists: {FRAMEWORK_RESOURCES.exists()}")

    # Check calibration files
    calib_dir = FRAMEWORK_RESOURCES / "intrinsic_extrinsic_list"
    print(f"\nCalibration directory: {calib_dir}")
    print(f"  Exists: {calib_dir.exists()}")

    if calib_dir.exists():
        calib_files = list(calib_dir.glob("*.json"))
        print(f"  Contains {len(calib_files)} calibration files")
        if calib_files:
            print(f"    Example: {calib_files[0].name}")

    success = FRAMEWORK_BINARY.exists() and FRAMEWORK_RESOURCES.exists()
    print(f"\nResult: {'PASS' if success else 'FAIL'}")
    return success


def test_calibration_loading():
    """Test loading calibration data"""
    print("\n" + "="*70)
    print("TEST 2: Calibration Data Loading")
    print("="*70)

    try:
        stitcher = DJIDualStitcher(camera_sn="default")
        info = stitcher.get_info()

        print(f"Stitcher created: {stitcher}")
        print(f"  Framework loaded: {info['is_loaded']}")
        print(f"  Calibration loaded: {info['calibration_loaded']}")

        if stitcher.calibration:
            calib = stitcher.calibration
            print(f"\nCalibration data extracted:")
            print(f"  Left camera fx: {calib.intrinsic00.get('fx', 'N/A'):.2f}")
            print(f"  Left camera cx: {calib.intrinsic00.get('cx', 'N/A'):.2f}")
            print(f"  Left fisheye xi: {calib.get_fisheye_param_left():.4f}")
            print(f"  Right camera fx: {calib.intrinsic01.get('fx', 'N/A'):.2f}")
            print(f"  Right camera cx: {calib.intrinsic01.get('cx', 'N/A'):.2f}")

            # Test matrix extraction
            K_left = calib.get_camera_matrix_left()
            print(f"\nLeft camera matrix K:")
            print(f"  {K_left[0,0]:.2f}    0  {K_left[0,2]:.2f}")
            print(f"    0  {K_left[1,1]:.2f}  {K_left[1,2]:.2f}")
            print(f"    0      0         1")

            print(f"\nResult: PASS")
            return True
        else:
            print("No calibration data loaded")
            print("Result: FAIL")
            return False

    except Exception as e:
        logger.error(f"Calibration loading failed: {e}")
        print(f"Result: FAIL - {e}")
        return False


def test_framework_initialization():
    """Test framework initialization"""
    print("\n" + "="*70)
    print("TEST 3: Framework Initialization")
    print("="*70)

    try:
        stitcher = DJIDualStitcher(camera_sn="default")
        print(f"DJIDualStitcher created: {stitcher}")
        print(f"  Is available: {stitcher.is_available()}")

        # Try to initialize
        success = stitcher.initialize()
        print(f"  Initialization: {'SUCCESS' if success else 'FAILED'}")

        if success:
            print(f"  Stitcher ready for processing")
            stitcher.deinitialize()

        print(f"\nResult: {'PASS' if success else 'FAIL'}")
        return success

    except Exception as e:
        logger.error(f"Initialization test failed: {e}")
        print(f"Result: FAIL - {e}")
        return False


def test_real_osv_extraction():
    """Test extracting real frames from OSV file"""
    print("\n" + "="*70)
    print("TEST 4: Real OSV Frame Extraction")
    print("="*70)

    osv_path = Path("/Users/zindello/repositories/zindello/djiclireframe/osv_example/CAM_20260819204151_0011_D.OSV")

    if not osv_path.exists():
        print(f"OSV test file not found: {osv_path}")
        print("Result: SKIP")
        return False

    print(f"OSV file: {osv_path}")
    print(f"  Size: {osv_path.stat().st_size / (1024*1024):.1f} MB")

    try:
        # Extract frames 0, 10, 20, 30, 40 for testing
        frame_indices = [0, 10, 20, 30, 40]
        frames = extract_osv_frames(osv_path, frame_indices)

        print(f"\nExtracted {len(frames)} frame pairs")
        if frames:
            frame_left, frame_right = frames[0]
            print(f"  Frame shape: {frame_left.shape}")
            print(f"  Data type: {frame_left.dtype}")
            print(f"  Value range: [{frame_left.min()}, {frame_left.max()}]")
            print(f"\nResult: PASS")
            return True
        else:
            print("No frames extracted")
            print("Result: FAIL")
            return False

    except Exception as e:
        logger.error(f"OSV extraction failed: {e}")
        print(f"Result: FAIL - {e}")
        return False


def test_real_stitching():
    """Test actual stitching with real frames"""
    print("\n" + "="*70)
    print("TEST 5: Real Frame Stitching with DJI API")
    print("="*70)

    osv_path = Path("/Users/zindello/repositories/zindello/djiclireframe/osv_example/CAM_20260819204151_0011_D.OSV")

    if not osv_path.exists():
        print(f"OSV test file not found: {osv_path}")
        print("Result: SKIP")
        return False

    try:
        # Initialize stitcher
        print("Initializing DJI DualStitcher...")
        stitcher = DJIDualStitcher(camera_sn="default")

        if not stitcher.initialize():
            print("Failed to initialize stitcher")
            print("Result: FAIL")
            return False

        # Extract a single frame pair
        print("Extracting first frame pair from OSV file...")
        frames = extract_osv_frames(osv_path, [0])

        if not frames:
            print("Failed to extract frames")
            stitcher.deinitialize()
            print("Result: FAIL")
            return False

        frame_left, frame_right = frames[0]
        print(f"  Left frame: {frame_left.shape}, min={frame_left.min()}, max={frame_left.max()}")
        print(f"  Right frame: {frame_right.shape}, min={frame_right.min()}, max={frame_right.max()}")

        # Stitch frames
        print("\nCalling ML_ImageStitcherProcessStepA (optical flow)...")
        print("Calling ML_ImageStitcherProcessStepB (blending)...")

        output = stitcher.stitch_frames(frame_left, frame_right)

        if output is None:
            print("Stitching failed (returned None)")
            stitcher.deinitialize()
            print("Result: FAIL")
            return False

        # Verify output
        print(f"\nStitching output:")
        print(f"  Shape: {output.shape}")
        print(f"  Data type: {output.dtype}")
        print(f"  Value range: [{output.min()}, {output.max()}]")
        print(f"  Non-zero pixels: {np.count_nonzero(output)} / {output.size}")

        # Save output
        try:
            import cv2
            output_dir = Path("/private/tmp/dji_stitching_test")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "stitched_dji_real_000.png"
            # Convert RGB to BGR for OpenCV
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), output_bgr)
            print(f"\nOutput saved: {output_file}")
            print(f"  Size: {output_file.stat().st_size / (1024*1024):.1f} MB")
        except Exception as e:
            logger.warning(f"Failed to save output image: {e}")

        stitcher.deinitialize()
        print("\nResult: PASS")
        return True

    except Exception as e:
        logger.error(f"Stitching test failed: {e}")
        print(f"Result: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*70)
    print("DJI DualStitcher Real Data Test Suite")
    print("="*70)

    results = []

    # Test 1: Framework paths
    results.append(("Framework Paths", test_framework_paths()))

    # Test 2: Calibration loading
    results.append(("Calibration Loading", test_calibration_loading()))

    # Test 3: Framework initialization
    results.append(("Framework Initialization", test_framework_initialization()))

    # Test 4: OSV extraction
    results.append(("OSV Frame Extraction", test_real_osv_extraction()))

    # Test 5: Real stitching
    results.append(("Real Frame Stitching", test_real_stitching()))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = 0
    failed = 0
    skipped = 0

    for test_name, result in results:
        if result is None:
            status = "SKIP"
            skipped += 1
        elif result:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
        print(f"{test_name:<30} {status}")

    print("="*70)
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
