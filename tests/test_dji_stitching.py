#!/usr/bin/env python3
"""Test DJI DualStitcher Stitching with Real and Synthetic Frames

This script tests the actual stitching pipeline.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from dji_osv_converter.stitch_dji_wrapper import DJIDualStitcher

def create_test_frame(seed=42):
    """Create a test frame with gradient pattern"""
    np.random.seed(seed)
    # Create a simple gradient pattern for testing
    frame = np.zeros((3840, 3840, 3), dtype=np.uint8)

    # Add gradient
    for i in range(3840):
        for j in range(3840):
            r = int((i / 3840) * 200)
            g = int((j / 3840) * 200)
            b = int(((i + j) / 7680) * 200)
            frame[i, j] = [r, g, b]

    # Add some features
    frame[100:200, 100:200, :] = [255, 0, 0]  # Red square
    frame[3600:3700, 100:200, :] = [0, 255, 0]  # Green square
    frame[100:200, 3600:3700, :] = [0, 0, 255]  # Blue square

    return frame

def test_synthetic_stitching():
    """Test stitching with synthetic test frames"""
    print("\n" + "="*70)
    print("TEST: Stitching with Synthetic Test Frames")
    print("="*70)

    try:
        # Create stitcher
        print("Creating DJIDualStitcher...")
        stitcher = DJIDualStitcher(camera_sn="default")

        if not stitcher.is_available():
            print("Framework not available")
            print("Result: SKIP")
            return False

        print(f"Stitcher: {stitcher}")
        info = stitcher.get_info()
        print(f"  Framework available: {info['framework_available']}")
        print(f"  Calibration loaded: {info['calibration_loaded']}")

        # Create test frames
        print("\nCreating synthetic test frames...")
        frame_left = create_test_frame(seed=1)
        frame_right = create_test_frame(seed=2)

        print(f"  Left frame: {frame_left.shape}, dtype={frame_left.dtype}")
        print(f"  Right frame: {frame_right.shape}, dtype={frame_right.dtype}")
        print(f"  Left range: [{frame_left.min()}, {frame_left.max()}]")
        print(f"  Right range: [{frame_right.min()}, {frame_right.max()}]")

        # Stitch frames
        print("\nCalling stitcher.stitch_frames()...")
        print("(This will run actual DJI stitching via subprocess...)")

        start = time.time()
        output = stitcher.stitch_frames(frame_left, frame_right)
        elapsed = time.time() - start

        if output is None:
            print("Stitching returned None")
            print("Result: FAIL")
            return False

        # Verify output
        print(f"\nStitching output:")
        print(f"  Shape: {output.shape}")
        print(f"  Data type: {output.dtype}")
        print(f"  Value range: [{output.min()}, {output.max()}]")
        print(f"  Non-zero pixels: {np.count_nonzero(output)} / {output.size}")
        print(f"  Processing time: {elapsed:.2f}s")

        # Verify it's actually a valid result (not all zeros)
        if np.all(output == 0):
            print("ERROR: Output is all zeros")
            print("Result: FAIL")
            return False

        # Save output
        try:
            import cv2
            output_dir = Path("/private/tmp/dji_stitching_test")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "stitched_synthetic_test.png"
            # Convert RGB to BGR for OpenCV
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), output_bgr)
            print(f"\nOutput saved: {output_file}")
            print(f"  File size: {output_file.stat().st_size / (1024*1024):.1f} MB")

            # Also save individual input frames for reference
            frame_left_file = output_dir / "test_frame_left.png"
            frame_right_file = output_dir / "test_frame_right.png"

            cv2.imwrite(str(frame_left_file), cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(frame_right_file), cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR))

            print(f"\nReference frames saved:")
            print(f"  {frame_left_file.name}")
            print(f"  {frame_right_file.name}")

        except Exception as e:
            logger.warning(f"Failed to save output: {e}")

        print("\nResult: PASS")
        return True

    except Exception as e:
        logger.exception(f"Test failed: {e}")
        print(f"Result: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_calibration():
    """Test calibration data extraction"""
    print("\n" + "="*70)
    print("TEST: Calibration Data")
    print("="*70)

    try:
        stitcher = DJIDualStitcher(camera_sn="default")

        if not stitcher.calibration:
            print("Calibration not loaded")
            print("Result: FAIL")
            return False

        calib = stitcher.calibration

        print(f"Calibration data loaded successfully")
        print(f"\nLeft camera:")
        print(f"  Focal length X: {calib.intrinsic00.get('fx', 'N/A'):.2f}")
        print(f"  Focal length Y: {calib.intrinsic00.get('fy', 'N/A'):.2f}")
        print(f"  Principal point X: {calib.intrinsic00.get('cx', 'N/A'):.2f}")
        print(f"  Principal point Y: {calib.intrinsic00.get('cy', 'N/A'):.2f}")
        print(f"  Fisheye parameter: {calib.get_fisheye_param_left():.4f}")

        print(f"\nRight camera:")
        print(f"  Focal length X: {calib.intrinsic01.get('fx', 'N/A'):.2f}")
        print(f"  Focal length Y: {calib.intrinsic01.get('fy', 'N/A'):.2f}")
        print(f"  Principal point X: {calib.intrinsic01.get('cx', 'N/A'):.2f}")
        print(f"  Principal point Y: {calib.intrinsic01.get('cy', 'N/A'):.2f}")
        print(f"  Fisheye parameter: {calib.get_fisheye_param_right():.4f}")

        # Test matrix extraction
        K_left = calib.get_camera_matrix_left()
        K_right = calib.get_camera_matrix_right()

        print(f"\nCamera matrices (K):")
        print(f"  Left:\n{K_left}")
        print(f"  Right:\n{K_right}")

        dist_left = calib.get_distortion_coeffs_left()
        print(f"\nDistortion coefficients (left):")
        print(f"  {dist_left}")

        print("\nResult: PASS")
        return True

    except Exception as e:
        logger.error(f"Calibration test failed: {e}")
        print(f"Result: FAIL - {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("DJI DualStitcher Production Test Suite")
    print("="*70)

    results = []

    # Test 1: Calibration
    results.append(("Calibration Data", test_calibration()))

    # Test 2: Synthetic stitching (faster, doesn't require OSV extraction)
    results.append(("Synthetic Frame Stitching", test_synthetic_stitching()))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, r in results if r)
    failed = sum(1 for _, r in results if not r)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<35} {status}")

    print("="*70)
    print(f"Total: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
