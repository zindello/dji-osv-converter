#!/usr/bin/env python3
"""Test DJI DualStitcher with Synthetic Data

This script tests the stitcher with generated frames instead of OSV extraction.
It's much faster and validates that the framework loading and stitching logic works.
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

from dji_osv_converter.stitch_dji_wrapper import DJIDualStitcher


def generate_test_frame(frame_id: int) -> np.ndarray:
    """Generate a test fisheye frame

    Creates a 3840x3840 RGB fisheye pattern with:
    - Gradient background based on frame_id
    - Circular pattern to simulate fisheye
    - Deterministic so each frame_id produces same result
    """
    height, width = 3840, 3840
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Create gradient based on frame_id
    base_level = 50 + (frame_id * 10) % 150

    # Fill with gradient
    for y in range(height):
        for x in range(width):
            # Distance from center (normalized)
            cy, cx = height / 2, width / 2
            dist_sq = ((x - cx) ** 2 + (y - cy) ** 2) / ((cx ** 2 + cy ** 2))

            # Create fisheye pattern: brighter at edges
            level = int(base_level + (dist_sq * 100))
            level = min(255, level)

            # Add frame_id to R channel so each frame is different
            r_val = (level + frame_id * 5) % 256
            frame[y, x] = [r_val, level, level]

    return frame


def test_framework_availability():
    """Quick test that framework is available"""
    print("\n" + "="*70)
    print("TEST: Framework Availability")
    print("="*70)

    try:
        stitcher = DJIDualStitcher(camera_sn="default")
        is_available = stitcher.is_available()

        print(f"DJIDualStitcher: {stitcher}")
        print(f"Is available: {is_available}")

        if not is_available:
            print("WARNING: Framework not available")
            print("Possible causes:")
            print("  1. DJI Studio not installed at /Applications/DJI Studio.app")
            print("  2. DualStitcher.framework missing")
            print("  3. Dependencies (openmcv2, ml_fis, libMNN, libdInfer) missing")
            return False

        print("Result: PASS")
        return True

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"Result: FAIL - {e}")
        return False


def test_synthetic_stitching():
    """Test stitching with synthetic frames"""
    print("\n" + "="*70)
    print("TEST: Synthetic Frame Stitching")
    print("="*70)

    try:
        # Initialize stitcher
        print("Initializing DJI DualStitcher...")
        stitcher = DJIDualStitcher(camera_sn="default")

        if not stitcher.is_available():
            print("Framework not available, skipping test")
            print("Result: SKIP")
            return None

        # Generate test frames
        print("Generating synthetic test frames...")
        frame_left = generate_test_frame(0)
        frame_right = generate_test_frame(1)

        print(f"  Left frame: {frame_left.shape}, dtype={frame_left.dtype}")
        print(f"    Value range: [{frame_left.min()}, {frame_left.max()}]")
        print(f"  Right frame: {frame_right.shape}, dtype={frame_right.dtype}")
        print(f"    Value range: [{frame_right.min()}, {frame_right.max()}]")

        # Stitch frames
        print("\nCalling DJI stitcher...")
        output = stitcher.stitch_frames(frame_left, frame_right)

        if output is None:
            print("Stitching returned None")
            print("Result: FAIL")
            return False

        # Verify output
        print(f"\nStitched output:")
        print(f"  Shape: {output.shape}")
        print(f"  Data type: {output.dtype}")
        print(f"  Value range: [{output.min()}, {output.max()}]")
        print(f"  Non-zero pixels: {np.count_nonzero(output)} / {output.size}")

        # Verify shape is equirectangular output
        expected_shapes = [(3840, 7680, 3), (7680, 3840, 3)]
        if output.shape not in expected_shapes:
            print(f"WARNING: Expected shape {expected_shapes}, got {output.shape}")
            print("Result: FAIL")
            return False

        # Check if output has actual content
        if np.count_nonzero(output) == 0:
            print("WARNING: Output is all zeros!")
            print("Result: FAIL")
            return False

        # Save output image if OpenCV available
        try:
            import cv2
            output_dir = Path("/private/tmp/dji_stitching_test")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "stitched_dji_synthetic.png"

            # Handle both possible output shapes
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
        logger.error(f"Stitching failed: {e}", exc_info=True)
        print(f"Result: FAIL - {e}")
        return False


def test_multiple_frames():
    """Test stitching multiple synthetic frames"""
    print("\n" + "="*70)
    print("TEST: Multiple Frame Stitching")
    print("="*70)

    try:
        stitcher = DJIDualStitcher(camera_sn="default")

        if not stitcher.is_available():
            print("Framework not available, skipping test")
            print("Result: SKIP")
            return None

        print("Stitching 3 synthetic frame pairs...")

        success_count = 0
        for frame_id in range(3):
            frame_left = generate_test_frame(frame_id * 2)
            frame_right = generate_test_frame(frame_id * 2 + 1)

            print(f"\n  Frame pair {frame_id}...")
            output = stitcher.stitch_frames(frame_left, frame_right)

            if output is not None and np.count_nonzero(output) > 0:
                success_count += 1
                print(f"    Result: SUCCESS ({output.shape})")

                # Optionally save
                try:
                    import cv2
                    output_dir = Path("/private/tmp/dji_stitching_test")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = output_dir / f"stitched_dji_synthetic_{frame_id}.png"

                    if output.shape[2] == 3:
                        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                    else:
                        output_bgr = output

                    cv2.imwrite(str(output_file), output_bgr)
                except Exception as e:
                    pass
            else:
                print(f"    Result: FAILED")

        if success_count >= 2:
            print(f"\nResult: PASS ({success_count}/3 frames stitched)")
            return True
        else:
            print(f"\nResult: FAIL ({success_count}/3 frames stitched)")
            return False

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"Result: FAIL - {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("DJI DualStitcher Synthetic Data Test Suite")
    print("="*70)

    results = []

    # Test framework availability
    results.append(("Framework Availability", test_framework_availability()))

    # Test synthetic stitching
    results.append(("Synthetic Stitching", test_synthetic_stitching()))

    # Test multiple frames
    results.append(("Multiple Frames", test_multiple_frames()))

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

    # Show output directory
    output_dir = Path("/private/tmp/dji_stitching_test")
    if output_dir.exists():
        output_files = sorted(output_dir.glob("*.png"))
        if output_files:
            print(f"\nOutput images in: {output_dir}")
            for f in output_files:
                print(f"  {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
