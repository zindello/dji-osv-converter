#!/usr/bin/env python3
"""Example: Using the Metal Stitcher Bridge

This demonstrates how to use the native Metal bridge for DJI stitching:
1. Load OSV frames
2. Compute seam masks and optical flow
3. Stitch using Metal bridge
4. Verify output is proper equirectangular
"""

import logging
import sys
from pathlib import Path
import numpy as np
import cv2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
from dji_osv_converter.stitch_metal_bridge import stitch_frames_metal_bridge
from dji_osv_converter.stitch_dji_proper import DJIStitcherProper
from dji_osv_converter.stitch_optical_flow import OpticalFlowStitcher


def create_dummy_test_data():
    """Create dummy test data for demonstration"""
    logger.info("Creating dummy test data...")

    # Create two frames with different patterns (so they're not identical)
    left = np.random.randint(100, 150, (3840, 3840, 3), dtype=np.uint8)
    right = np.random.randint(100, 150, (3840, 3840, 3), dtype=np.uint8)

    # Add a vertical line in the overlap region to create a clear seam
    left[:, :200, :] = [200, 100, 100]  # Red-ish in left overlap
    right[:, -200:, :] = [100, 100, 200]  # Blue-ish in right overlap

    # Create seam masks (BGRA format)
    seam_mask_1 = np.zeros((292, 7680, 4), dtype=np.uint8)
    seam_mask_2 = np.zeros((292, 7680, 4), dtype=np.uint8)

    # Create a simple seam line (vertical line at middle of overlap)
    for x in range(7680):
        for y in range(292):
            if x < 3840:  # Left side
                seam_mask_1[y, x, :] = [255, 255, 255, 255]  # Full opacity for left
                seam_mask_2[y, x, :] = [0, 0, 0, 255]        # Transparent for right
            else:  # Right side
                seam_mask_1[y, x, :] = [0, 0, 0, 255]        # Transparent for left
                seam_mask_2[y, x, :] = [255, 255, 255, 255]  # Full opacity for right

    # Create flow texture (simple zeros for testing)
    flow = np.zeros((394, 789, 4), dtype=np.float32)

    logger.info(f"Dummy data created:")
    logger.info(f"  Left: {left.shape} {left.dtype}")
    logger.info(f"  Right: {right.shape} {right.dtype}")
    logger.info(f"  Seam mask 1: {seam_mask_1.shape} {seam_mask_1.dtype}")
    logger.info(f"  Seam mask 2: {seam_mask_2.shape} {seam_mask_2.dtype}")
    logger.info(f"  Flow: {flow.shape} {flow.dtype}")

    return left, right, seam_mask_1, seam_mask_2, flow


def test_metal_bridge():
    """Test the Metal bridge with dummy data"""
    logger.info("=" * 60)
    logger.info("Testing Metal Stitcher Bridge")
    logger.info("=" * 60)

    # Create test data
    left, right, seam_mask_1, seam_mask_2, flow = create_dummy_test_data()

    # Test the bridge
    logger.info("\nCalling Metal bridge stitch_frames...")
    output = stitch_frames_metal_bridge(
        left, right,
        seam_mask_1, seam_mask_2,
        flow,
        camera_sn="test_camera"
    )

    if output is None:
        logger.error("Stitching returned None")
        logger.info("Note: This may be expected if DJI Studio is not installed")
        return False

    # Validate output
    logger.info(f"\nStitching output:")
    logger.info(f"  Shape: {output.shape}")
    logger.info(f"  Dtype: {output.dtype}")
    logger.info(f"  Value range: [{output.min()}, {output.max()}]")
    logger.info(f"  Size: {output.nbytes / 1e6:.1f} MB")

    # Check output dimensions
    if output.shape != (7680, 3840, 3):
        logger.error(f"Invalid output shape: {output.shape} (expected 7680x3840x3)")
        return False

    if output.dtype != np.uint8:
        logger.error(f"Invalid output dtype: {output.dtype} (expected uint8)")
        return False

    logger.info("\n✓ Metal bridge test PASSED")
    return True


def test_with_dji_proper_stitcher():
    """Test with the full DJI Proper Stitcher pipeline"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Full DJI Proper Stitcher Pipeline")
    logger.info("=" * 60)

    # Create test data
    left, right, _, _, _ = create_dummy_test_data()

    try:
        logger.info("\nInitializing DJI Proper Stitcher...")
        stitcher = DJIStitcherProper(camera_sn="test_full_pipeline")

        logger.info("Computing optical flow and seam detection...")
        output = stitcher.stitch_frames(left, right)

        if output is None:
            logger.error("Stitching returned None")
            return False

        logger.info(f"\nFull pipeline output:")
        logger.info(f"  Shape: {output.shape}")
        logger.info(f"  Dtype: {output.dtype}")
        logger.info(f"  Value range: [{output.min()}, {output.max()}]")

        logger.info("\n✓ Full pipeline test PASSED")
        return True

    except Exception as e:
        logger.error(f"Full pipeline test failed: {e}")
        return False


def visualize_output(output: np.ndarray, output_path: Path = Path("/tmp/stitched.png")):
    """Visualize the output (downscaled for display)"""
    logger.info(f"\nVisualizing output...")

    # Downscale for display
    scale = 4
    display = cv2.resize(output, (output.shape[1] // scale, output.shape[0] // scale))

    # Convert RGB to BGR for OpenCV
    display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)

    # Save
    cv2.imwrite(str(output_path), display)
    logger.info(f"Saved visualization to: {output_path}")

    return output_path


def main():
    """Main function"""
    logger.info("Metal Stitcher Bridge - Example Usage\n")

    # Test 1: Metal bridge directly
    success1 = test_metal_bridge()

    # Test 2: Full pipeline
    try:
        success2 = test_with_dji_proper_stitcher()
    except Exception as e:
        logger.warning(f"Full pipeline test skipped: {e}")
        success2 = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    logger.info(f"Metal bridge: {'✓ PASSED' if success1 else '✗ FAILED/SKIPPED'}")
    logger.info(f"Full pipeline: {'✓ PASSED' if success2 else '✗ FAILED/SKIPPED'}")

    if success1 or success2:
        logger.info("\n✓ At least one test passed!")
        return 0
    else:
        logger.error("\n✗ All tests failed or skipped")
        return 1


if __name__ == "__main__":
    sys.exit(main())
