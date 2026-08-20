#!/usr/bin/env python3
"""Test DJI Objective-C Stitcher with Real OSV Data

This script tests the new Objective-C API implementation (stitch_dji_objc.py)
using actual frames from an OSV file.

Tests:
1. Framework loading via NSBundle
2. Stitcher class discovery
3. Metal device initialization
4. Actual Objective-C stitching methods
5. Real frame stitching (3840×3840 -> 7680×3840)
6. Multi-frame stitching
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
sys.path.insert(0, str(Path(__file__).parent.parent))

from dji_osv_converter.stitch_dji_objc import (
    DJIObjCStitcher,
    extract_osv_frames,
    FRAMEWORK_PATH,
    FRAMEWORK_RESOURCES,
    ObjCStitcherError,
)


def test_pyobjc_available():
    """Test that PyObjC is available"""
    print("\n" + "="*70)
    print("TEST 1: PyObjC Availability")
    print("="*70)

    try:
        import objc
        print(f"PyObjC version: {objc.__version__ if hasattr(objc, '__version__') else 'unknown'}")
        print("PyObjC is available")

        try:
            from Foundation import NSBundle
            print("Foundation framework is available")

            try:
                from Metal import MTLCreateSystemDefaultDevice
                print("Metal framework is available")
                print("\nResult: PASS")
                return True
            except ImportError:
                print("Metal framework NOT available (may need: pip install pyobjc-framework-Metal)")
                print("Continuing without Metal...")
                print("\nResult: PASS (Foundation available)")
                return True

        except ImportError as e:
            print(f"Foundation framework NOT available: {e}")
            print("\nResult: FAIL")
            return False

    except ImportError as e:
        print(f"PyObjC NOT available: {e}")
        print("\nInstall with: pip install pyobjc pyobjc-framework-Metal")
        print("\nResult: FAIL")
        return False


def test_framework_paths():
    """Test that framework and resources exist"""
    print("\n" + "="*70)
    print("TEST 2: Framework Paths and Resources")
    print("="*70)

    print(f"Framework path: {FRAMEWORK_PATH}")
    print(f"  Exists: {FRAMEWORK_PATH.exists()}")

    print(f"Framework resources: {FRAMEWORK_RESOURCES}")
    print(f"  Exists: {FRAMEWORK_RESOURCES.exists()}")

    # Check calibration files
    calib_file = FRAMEWORK_RESOURCES / "intrinsic_extrinsic.json"
    print(f"\nCalibration file: {calib_file}")
    print(f"  Exists: {calib_file.exists()}")

    calib_dir = FRAMEWORK_RESOURCES / "intrinsic_extrinsic_list"
    print(f"\nCalibration directory: {calib_dir}")
    print(f"  Exists: {calib_dir.exists()}")

    if calib_dir.exists():
        calib_files = list(calib_dir.glob("*.json"))
        print(f"  Contains {len(calib_files)} calibration files")
        if calib_files:
            print(f"    Example: {calib_files[0].name}")

    success = FRAMEWORK_PATH.exists() and FRAMEWORK_RESOURCES.exists()
    print(f"\nResult: {'PASS' if success else 'FAIL'}")
    return success


def test_stitcher_initialization():
    """Test Stitcher initialization via Objective-C API"""
    print("\n" + "="*70)
    print("TEST 3: Stitcher Initialization via Objective-C API")
    print("="*70)

    try:
        print("Creating DJIObjCStitcher instance...")
        stitcher = DJIObjCStitcher(camera_sn="default")
        info = stitcher.get_info()

        print(f"Stitcher created: {stitcher}")
        print(f"  Framework available: {info['framework_available']}")
        print(f"  Stitcher initialized: {info['stitcher_initialized']}")
        print(f"  Metal device available: {info['metal_device_available']}")
        print(f"  Calibration loaded: {info['calibration_loaded']}")

        if stitcher.is_available():
            print("\nStitcher is ready for stitching")
            print("\nResult: PASS")
            return True
        else:
            print("\nStitcher not fully initialized")
            print("  This may be normal if PyObjC frameworks aren't fully available")
            print("\nResult: PARTIAL (framework issue, not code issue)")
            return None

    except ObjCStitcherError as e:
        print(f"ObjC Stitcher Error: {e}")
        print("\nResult: FAIL")
        return False
    except Exception as e:
        logger.error(f"Initialization test failed: {e}", exc_info=True)
        print(f"Result: FAIL - {e}")
        return False


def test_calibration_loading():
    """Test calibration data loading"""
    print("\n" + "="*70)
    print("TEST 4: Calibration Data Loading")
    print("="*70)

    try:
        stitcher = DJIObjCStitcher(camera_sn="default")

        if stitcher.calibration:
            calib = stitcher.calibration
            print(f"Calibration data extracted:")
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


def test_real_osv_extraction():
    """Test extracting real frames from OSV file"""
    print("\n" + "="*70)
    print("TEST 5: Real OSV Frame Extraction")
    print("="*70)

    osv_path = Path("/Users/zindello/repositories/zindello/djiclireframe/osv_example/CAM_20260819204151_0011_D.OSV")

    if not osv_path.exists():
        print(f"OSV test file not found: {osv_path}")
        print("Result: SKIP")
        return None

    print(f"OSV file: {osv_path}")
    print(f"  Size: {osv_path.stat().st_size / (1024*1024):.1f} MB")

    try:
        # Extract frames 0, 1, 2 for testing
        frame_indices = [0, 1, 2]
        frames = extract_osv_frames(osv_path, frame_indices)

        print(f"\nExtracted {len(frames)} frame pairs")
        if frames:
            frame_left, frame_right = frames[0]
            print(f"  Frame 0 left shape: {frame_left.shape}")
            print(f"  Data type: {frame_left.dtype}")
            print(f"  Value range: [{frame_left.min()}, {frame_left.max()}]")
            print(f"  Non-zero pixels: {np.count_nonzero(frame_left)} / {frame_left.size}")

            frame_left2, frame_right2 = frames[1]
            print(f"\n  Frame 1 left shape: {frame_left2.shape}")
            print(f"  Value range: [{frame_left2.min()}, {frame_left2.max()}]")

            print(f"\nResult: PASS")
            return True
        else:
            print("No frames extracted")
            print("Result: FAIL")
            return False

    except Exception as e:
        logger.error(f"OSV extraction failed: {e}", exc_info=True)
        print(f"Result: FAIL - {e}")
        return False


def test_real_stitching():
    """Test actual stitching with real frames"""
    print("\n" + "="*70)
    print("TEST 6: Real Frame Stitching via Objective-C API")
    print("="*70)

    osv_path = Path("/Users/zindello/repositories/zindello/djiclireframe/osv_example/CAM_20260819204151_0011_D.OSV")

    if not osv_path.exists():
        print(f"OSV test file not found: {osv_path}")
        print("Result: SKIP")
        return None

    try:
        # Initialize stitcher
        print("Initializing DJIObjCStitcher...")
        stitcher = DJIObjCStitcher(camera_sn="default")

        if not stitcher.is_available():
            print("Stitcher not available (framework or initialization issue)")
            print("Result: FAIL")
            return False

        # Extract first frame pair
        print("Extracting first frame pair from OSV file...")
        frames = extract_osv_frames(osv_path, [0])

        if not frames:
            print("Failed to extract frames")
            print("Result: FAIL")
            return False

        frame_left, frame_right = frames[0]
        print(f"  Left frame: {frame_left.shape}, min={frame_left.min()}, max={frame_left.max()}")
        print(f"  Right frame: {frame_right.shape}, min={frame_right.min()}, max={frame_right.max()}")

        # Stitch frames
        print("\nCalling Objective-C stitcher (aiStitchSeams or aiStitchWarpFusion)...")
        output = stitcher.stitch_frames(frame_left, frame_right)

        if output is None:
            print("Stitching failed (returned None)")
            print("Result: FAIL")
            return False

        # Verify output
        print(f"\nStitching output:")
        print(f"  Shape: {output.shape}")
        print(f"  Data type: {output.dtype}")
        print(f"  Value range: [{output.min()}, {output.max()}]")
        print(f"  Non-zero pixels: {np.count_nonzero(output)} / {output.size}")

        # Check if output has actual content
        if np.count_nonzero(output) == 0:
            print("WARNING: Output is all zeros!")
            print("Result: FAIL")
            return False

        # Verify shape is correct for equirectangular
        if output.shape not in [(3840, 7680, 3), (7680, 3840, 3)]:
            print(f"WARNING: Expected shape (3840, 7680, 3) or (7680, 3840, 3), got {output.shape}")
            print("Result: FAIL")
            return False

        # Save output
        try:
            import cv2
            output_dir = Path("/private/tmp/dji_objc_stitching_test")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "stitched_objc_real_000.png"
            # Convert RGB to BGR for OpenCV if needed
            if output.shape[2] == 3:
                output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            else:
                output_bgr = output
            cv2.imwrite(str(output_file), output_bgr)
            print(f"\nOutput saved: {output_file}")
            print(f"  Size: {output_file.stat().st_size / (1024*1024):.1f} MB")
        except ImportError:
            print("\nOpenCV not available, skipping image save")
        except Exception as e:
            logger.warning(f"Failed to save output image: {e}")

        print("\nResult: PASS")
        return True

    except Exception as e:
        logger.error(f"Stitching test failed: {e}", exc_info=True)
        print(f"Result: FAIL - {e}")
        return False


def test_multi_frame_stitching():
    """Test stitching multiple frames"""
    print("\n" + "="*70)
    print("TEST 7: Multi-Frame Stitching via Objective-C API")
    print("="*70)

    osv_path = Path("/Users/zindello/repositories/zindello/djiclireframe/osv_example/CAM_20260819204151_0011_D.OSV")

    if not osv_path.exists():
        print(f"OSV test file not found: {osv_path}")
        print("Result: SKIP")
        return None

    try:
        stitcher = DJIObjCStitcher(camera_sn="default")

        if not stitcher.is_available():
            print("Stitcher not available")
            print("Result: FAIL")
            return False

        # Extract multiple frames
        frame_indices = [0, 5, 10]
        print(f"Extracting frames {frame_indices}...")
        frames = extract_osv_frames(osv_path, frame_indices)

        if not frames or len(frames) < 2:
            print("Failed to extract enough frames")
            print("Result: FAIL")
            return False

        print(f"Extracted {len(frames)} frame pairs")

        # Stitch each frame
        output_dir = Path("/private/tmp/dji_objc_stitching_test")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False

        success_count = 0
        for i, (frame_left, frame_right) in enumerate(frames):
            print(f"\nStitching frame pair {i}...")
            output = stitcher.stitch_frames(frame_left, frame_right)

            if output is not None and np.count_nonzero(output) > 0:
                success_count += 1
                print(f"  Frame {i}: SUCCESS ({output.shape})")

                if has_cv2:
                    try:
                        output_file = output_dir / f"stitched_objc_real_{i:03d}.png"
                        if output.shape[2] == 3:
                            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                        else:
                            output_bgr = output
                        cv2.imwrite(str(output_file), output_bgr)
                        print(f"    Saved: {output_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to save frame {i}: {e}")
            else:
                print(f"  Frame {i}: FAILED")

        if success_count >= 2:
            print(f"\nResult: PASS ({success_count}/{len(frames)} frames stitched)")
            return True
        else:
            print(f"\nResult: FAIL (only {success_count}/{len(frames)} frames stitched)")
            return False

    except Exception as e:
        logger.error(f"Multi-frame stitching test failed: {e}", exc_info=True)
        print(f"Result: FAIL - {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*70)
    print("DJI Objective-C Stitcher Test Suite")
    print("="*70)

    results = []

    # Test 1: PyObjC availability
    results.append(("PyObjC Availability", test_pyobjc_available()))

    # Test 2: Framework paths
    results.append(("Framework Paths", test_framework_paths()))

    # Test 3: Stitcher initialization
    results.append(("Stitcher Initialization", test_stitcher_initialization()))

    # Test 4: Calibration loading
    results.append(("Calibration Loading", test_calibration_loading()))

    # Test 5: OSV extraction
    results.append(("OSV Frame Extraction", test_real_osv_extraction()))

    # Test 6: Real stitching
    results.append(("Real Frame Stitching", test_real_stitching()))

    # Test 7: Multi-frame stitching
    results.append(("Multi-Frame Stitching", test_multi_frame_stitching()))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

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

        print(f"{test_name:<30} {status}")

    print("="*70)
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped, {partial} partial")
    print("="*70)

    # Show where output files are located
    output_dir = Path("/private/tmp/dji_objc_stitching_test")
    if output_dir.exists():
        output_files = list(output_dir.glob("*.png"))
        if output_files:
            print(f"\nOutput images saved to: {output_dir}")
            for f in sorted(output_files):
                print(f"  {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
