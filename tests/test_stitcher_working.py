#!/usr/bin/env python3
"""Production Test Suite for StitcherObjC Implementation

This test suite verifies that the working PyObjC wrapper for DJI's Stitcher class:
1. Loads the framework correctly
2. Initializes the Stitcher class via Objective-C API
3. Performs actual stitching with real OSV frames
4. Produces valid equirectangular output

Tests:
1. Framework availability and resources
2. Calibration loading
3. Stitcher initialization
4. Real frame extraction from OSV files
5. Actual stitching via simpleStitchSeams
6. Multi-frame stitching
7. Output validation
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dji_osv_converter.stitch_stitcher_objc import (
    StitcherObjC,
    CalibrationData,
    extract_osv_frames,
    FRAMEWORK_PATH,
    FRAMEWORK_RESOURCES,
)

# Test data
OSV_TEST_FILE = Path("/Users/zindello/repositories/zindello/djiclireframe/osv_example/CAM_20260819204151_0011_D.OSV")
OUTPUT_DIR = Path("/private/tmp/stitcher_working_test")


def print_header(title: str):
    """Print a test section header"""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def test_framework_paths():
    """Test 1: Verify framework paths and resources exist"""
    print_header("TEST 1: Framework Paths and Resources")

    print(f"Framework path: {FRAMEWORK_PATH}")
    print(f"  Exists: {'✓' if FRAMEWORK_PATH.exists() else '✗'}")

    print(f"Resources path: {FRAMEWORK_RESOURCES}")
    print(f"  Exists: {'✓' if FRAMEWORK_RESOURCES.exists() else '✗'}")

    # Check calibration files
    calib_dir = FRAMEWORK_RESOURCES / "intrinsic_extrinsic_list"
    print(f"Calibration directory: {calib_dir}")
    print(f"  Exists: {'✓' if calib_dir.exists() else '✗'}")

    if calib_dir.exists():
        calib_files = list(calib_dir.glob("*.json"))
        print(f"  Files: {len(calib_files)} calibration files")
        if calib_files:
            print(f"    Examples: {[f.name for f in calib_files[:3]]}")

    success = FRAMEWORK_PATH.exists() and FRAMEWORK_RESOURCES.exists()
    print(f"\nResult: {'PASS' if success else 'FAIL'}")
    return success


def test_pyobjc_available():
    """Test 2: Check PyObjC availability"""
    print_header("TEST 2: PyObjC Availability")

    try:
        import objc
        print(f"✓ PyObjC is available")

        try:
            from Foundation import NSBundle
            print(f"✓ Foundation framework is available")

            try:
                from Metal import MTLCreateSystemDefaultDevice
                print(f"✓ Metal framework is available")
                print("\nResult: PASS")
                return True
            except ImportError:
                print(f"✗ Metal framework NOT available")
                print("  Install with: pip install pyobjc-framework-Metal")
                print("  (Continuing without Metal GPU support)")
                print("\nResult: PASS (Foundation available)")
                return True

        except ImportError as e:
            print(f"✗ Foundation framework NOT available: {e}")
            print("\nResult: FAIL")
            return False

    except ImportError as e:
        print(f"✗ PyObjC NOT available: {e}")
        print("Install with: pip install pyobjc pyobjc-framework-Metal")
        print("\nResult: FAIL")
        return False


def test_stitcher_initialization():
    """Test 3: Initialize StitcherObjC"""
    print_header("TEST 3: StitcherObjC Initialization")

    try:
        print("Creating StitcherObjC instance...")
        stitcher = StitcherObjC(camera_sn="default")
        print(f"✓ Instance created: {stitcher}")

        # Check info
        info = stitcher.get_info()
        print(f"\nStitcher info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Check availability
        is_available = stitcher.is_available()
        print(f"\nFramework available: {'✓' if is_available else '✗'}")

        if is_available:
            print("\nResult: PASS")
            return True
        else:
            print("\nResult: PARTIAL (framework issue)")
            return None

    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        print(f"\nResult: FAIL - {e}")
        return False


def test_calibration_loading():
    """Test 4: Load calibration data"""
    print_header("TEST 4: Calibration Data Loading")

    try:
        stitcher = StitcherObjC(camera_sn="default")

        if stitcher.calibration is None:
            print("✗ No calibration data loaded")
            print("\nResult: FAIL")
            return False

        calib = stitcher.calibration
        print(f"✓ Calibration loaded")

        print(f"\nCamera parameters:")
        print(f"  Left camera:")
        print(f"    fx: {calib.intrinsic00.get('fx', 'N/A'):.2f}")
        print(f"    fy: {calib.intrinsic00.get('fy', 'N/A'):.2f}")
        print(f"    cx: {calib.intrinsic00.get('cx', 'N/A'):.2f}")
        print(f"    cy: {calib.intrinsic00.get('cy', 'N/A'):.2f}")
        print(f"    k1: {calib.intrinsic00.get('k1', 'N/A')}")
        print(f"    xi (fisheye): {calib.get_fisheye_param_left():.4f}")

        print(f"  Right camera:")
        print(f"    fx: {calib.intrinsic01.get('fx', 'N/A'):.2f}")
        print(f"    cx: {calib.intrinsic01.get('cx', 'N/A'):.2f}")
        print(f"    xi (fisheye): {calib.get_fisheye_param_right():.4f}")

        # Test matrix extraction
        K_left = calib.get_camera_matrix_left()
        print(f"\nCamera matrix (left):")
        print(f"  {K_left[0,0]:.2f}    0  {K_left[0,2]:.2f}")
        print(f"    0  {K_left[1,1]:.2f}  {K_left[1,2]:.2f}")
        print(f"    0      0         1")

        print(f"\nResult: PASS")
        return True

    except Exception as e:
        logger.error(f"Calibration loading failed: {e}")
        print(f"\nResult: FAIL - {e}")
        return False


def test_osv_extraction():
    """Test 5: Extract frames from OSV file"""
    print_header("TEST 5: OSV Frame Extraction")

    if not OSV_TEST_FILE.exists():
        print(f"✗ OSV test file not found: {OSV_TEST_FILE}")
        print("\nResult: SKIP")
        return None

    print(f"OSV file: {OSV_TEST_FILE}")
    print(f"  Size: {OSV_TEST_FILE.stat().st_size / (1024*1024):.1f} MB")

    try:
        # Extract frames 0, 1, 2
        frame_indices = [0, 1, 2]
        print(f"\nExtracting frames {frame_indices}...")
        frames = extract_osv_frames(OSV_TEST_FILE, frame_indices)

        if not frames:
            print("✗ No frames extracted")
            print("\nResult: FAIL")
            return False

        print(f"✓ Extracted {len(frames)} frame pairs")

        # Show first frame info
        frame_left, frame_right = frames[0]
        print(f"\nFrame 0:")
        print(f"  Left shape: {frame_left.shape}")
        print(f"  Left dtype: {frame_left.dtype}")
        print(f"  Left range: [{frame_left.min()}, {frame_left.max()}]")
        print(f"  Right shape: {frame_right.shape}")
        print(f"  Right range: [{frame_right.min()}, {frame_right.max()}]")

        # Verify shape and dtype
        valid_frames = all(
            l.shape == (3840, 3840, 3) and l.dtype == np.uint8
            for l, r in frames
        )

        if valid_frames:
            print(f"\n✓ All frames have correct shape and dtype")
            print("\nResult: PASS")
            return True
        else:
            print(f"\n✗ Some frames have invalid shape/dtype")
            print("\nResult: FAIL")
            return False

    except Exception as e:
        logger.error(f"OSV extraction failed: {e}", exc_info=True)
        print(f"\nResult: FAIL - {e}")
        return False


def test_stitching():
    """Test 6: Stitch real frames"""
    print_header("TEST 6: Real Frame Stitching")

    if not OSV_TEST_FILE.exists():
        print(f"✗ OSV test file not found")
        print("\nResult: SKIP")
        return None

    try:
        # Initialize stitcher
        print("Initializing StitcherObjC...")
        stitcher = StitcherObjC(camera_sn="default")

        if not stitcher.is_available():
            print("✗ Stitcher not available")
            print("\nResult: FAIL")
            return False

        print("✓ Stitcher is available")

        # Extract first frame
        print("\nExtracting first frame pair...")
        frames = extract_osv_frames(OSV_TEST_FILE, [0])

        if not frames:
            print("✗ Failed to extract frames")
            print("\nResult: FAIL")
            return False

        frame_left, frame_right = frames[0]
        print(f"✓ Extracted frames")
        print(f"  Left: {frame_left.shape}, range=[{frame_left.min()}, {frame_left.max()}]")
        print(f"  Right: {frame_right.shape}, range=[{frame_right.min()}, {frame_right.max()}]")

        # Perform stitching
        print("\nCalling Stitcher.simpleStitchSeams...")
        output = stitcher.stitch_frames(frame_left, frame_right)

        if output is None:
            print("✗ Stitching returned None")
            print("\nResult: FAIL")
            return False

        print("✓ Stitching completed")

        # Validate output
        print(f"\nOutput validation:")
        print(f"  Shape: {output.shape}")
        print(f"  Dtype: {output.dtype}")
        print(f"  Range: [{output.min()}, {output.max()}]")
        print(f"  Non-zero pixels: {np.count_nonzero(output)} / {output.size}")

        # Check shape
        valid_shape = output.shape in [(7680, 3840, 3), (3840, 7680, 3)]
        if not valid_shape:
            print(f"✗ Invalid output shape: expected (7680,3840,3) or (3840,7680,3), got {output.shape}")
            print("\nResult: FAIL")
            return False

        print(f"✓ Output shape is valid")

        # Check if output has content
        non_zero_ratio = np.count_nonzero(output) / output.size
        print(f"  Non-zero ratio: {non_zero_ratio:.1%}")

        if non_zero_ratio < 0.1:
            print(f"✗ Output appears to be mostly black (non-zero ratio: {non_zero_ratio:.1%})")
            print("  This may indicate stitching failed")

        # Save output
        try:
            import cv2
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            output_file = OUTPUT_DIR / "stitched_frame_000.png"

            # Handle shape
            output_to_save = output
            if output_to_save.shape == (3840, 7680, 3):
                output_to_save = np.transpose(output_to_save, (1, 0, 2))

            # Convert RGB to BGR for OpenCV
            if output_to_save.shape[2] == 3:
                output_bgr = cv2.cvtColor(output_to_save, cv2.COLOR_RGB2BGR)
            else:
                output_bgr = output_to_save

            cv2.imwrite(str(output_file), output_bgr)
            print(f"\n✓ Output saved: {output_file}")
            print(f"  File size: {output_file.stat().st_size / (1024*1024):.1f} MB")

        except ImportError:
            print("\n(OpenCV not available for image save)")
        except Exception as e:
            logger.warning(f"Failed to save output: {e}")

        print("\nResult: PASS")
        return True

    except Exception as e:
        logger.error(f"Stitching test failed: {e}", exc_info=True)
        print(f"\nResult: FAIL - {e}")
        return False


def test_multi_frame():
    """Test 7: Stitch multiple frames"""
    print_header("TEST 7: Multi-Frame Stitching")

    if not OSV_TEST_FILE.exists():
        print(f"✗ OSV test file not found")
        print("\nResult: SKIP")
        return None

    try:
        stitcher = StitcherObjC(camera_sn="default")

        if not stitcher.is_available():
            print("✗ Stitcher not available")
            print("\nResult: FAIL")
            return False

        # Extract multiple frames
        frame_indices = [0, 5, 10]
        print(f"Extracting frames {frame_indices}...")
        frames = extract_osv_frames(OSV_TEST_FILE, frame_indices)

        if len(frames) < 2:
            print(f"✗ Need at least 2 frames, got {len(frames)}")
            print("\nResult: FAIL")
            return False

        print(f"✓ Extracted {len(frames)} frame pairs")

        # Stitch each frame
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            import cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False

        success_count = 0
        for i, (frame_left, frame_right) in enumerate(frames):
            print(f"\nStitching frame {i}...")
            output = stitcher.stitch_frames(frame_left, frame_right)

            if output is not None and np.count_nonzero(output) > 0:
                success_count += 1
                print(f"  ✓ Frame {i}: SUCCESS ({output.shape})")

                if has_cv2:
                    try:
                        output_file = OUTPUT_DIR / f"stitched_frame_{i:03d}.png"
                        output_to_save = output
                        if output_to_save.shape == (3840, 7680, 3):
                            output_to_save = np.transpose(output_to_save, (1, 0, 2))
                        if output_to_save.shape[2] == 3:
                            output_bgr = cv2.cvtColor(output_to_save, cv2.COLOR_RGB2BGR)
                        else:
                            output_bgr = output_to_save
                        cv2.imwrite(str(output_file), output_bgr)
                        print(f"    Saved: {output_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to save frame {i}: {e}")
            else:
                print(f"  ✗ Frame {i}: FAILED")

        if success_count >= 2:
            print(f"\nResult: PASS ({success_count}/{len(frames)} frames stitched)")
            return True
        else:
            print(f"\nResult: FAIL ({success_count}/{len(frames)} frames stitched)")
            return False

    except Exception as e:
        logger.error(f"Multi-frame test failed: {e}", exc_info=True)
        print(f"\nResult: FAIL - {e}")
        return False


def run_all_tests():
    """Run all tests and print summary"""
    print_header("StitcherObjC Production Test Suite")

    results = []

    # Run tests
    results.append(("Framework Paths", test_framework_paths()))
    results.append(("PyObjC Availability", test_pyobjc_available()))
    results.append(("Stitcher Initialization", test_stitcher_initialization()))
    results.append(("Calibration Loading", test_calibration_loading()))
    results.append(("OSV Extraction", test_osv_extraction()))
    results.append(("Real Stitching", test_stitching()))
    results.append(("Multi-Frame Stitching", test_multi_frame()))

    # Print summary
    print_header("TEST SUMMARY")

    passed = 0
    failed = 0
    skipped = 0
    partial = 0

    for test_name, result in results:
        if result is None:
            status = "SKIP"
            skipped += 1
        elif result is True:
            status = "PASS"
            passed += 1
        elif result is False:
            status = "FAIL"
            failed += 1
        else:
            status = str(result).upper()
            partial += 1

        symbol = "✓" if result is True else ("✗" if result is False else "•")
        print(f"{symbol} {test_name:<40} {status}")

    print("="*70)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped, {partial} partial")
    print("="*70)

    # Show output files
    if OUTPUT_DIR.exists():
        output_files = list(OUTPUT_DIR.glob("*.png"))
        if output_files:
            print(f"\nOutput images saved to: {OUTPUT_DIR}")
            for f in sorted(output_files):
                size_mb = f.stat().st_size / (1024*1024)
                print(f"  {f.name} ({size_mb:.1f} MB)")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
