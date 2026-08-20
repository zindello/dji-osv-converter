#!/usr/bin/env python3
"""Demo: DJI Stitcher with Optical Flow and Dynamic Seams

This script demonstrates the production-ready DJI stitcher that uses:
1. Optical flow to detect optimal seam lines
2. Metal textures for seam masks and flow data
3. dynamicStitchSeams method for proper equirectangular stitching

Input: Two 3840×3840 uint8 RGB fisheye frames from OSV file
Output: 7680×3840 proper equirectangular panorama (pannable 360°)

Usage:
    python3 demo_stitch_dji_proper.py
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from dji_osv_converter.stitch_dji_proper import (
    DJIStitcherProper,
    extract_osv_frames,
)


def test_optical_flow():
    """Test optical flow computation"""
    logger.info("\n" + "="*70)
    logger.info("Testing Optical Flow Computation")
    logger.info("="*70)

    # Create dummy frames for testing
    logger.info("\nCreating test frames...")
    frame_left = np.random.randint(0, 256, (3840, 3840, 3), dtype=np.uint8)
    frame_right = np.random.randint(0, 256, (3840, 3840, 3), dtype=np.uint8)

    logger.info(f"Left frame:  {frame_left.shape} {frame_left.dtype}")
    logger.info(f"Right frame: {frame_right.shape} {frame_right.dtype}")

    # Create stitcher
    logger.info("\nInitializing stitcher...")
    stitcher = DJIStitcherProper(enable_flow=True)

    # Compute optical flow
    logger.info("Computing optical flow...")
    flow_fwd, flow_bwd = stitcher.compute_optical_flow(frame_left, frame_right)

    if flow_fwd is not None and flow_bwd is not None:
        logger.info(f"✓ Forward flow:  {flow_fwd.shape} {flow_fwd.dtype}")
        logger.info(f"✓ Backward flow: {flow_bwd.shape} {flow_bwd.dtype}")
        logger.info(f"  Flow magnitude: [{np.linalg.norm(flow_fwd, axis=2).min():.2f}, "
                   f"{np.linalg.norm(flow_fwd, axis=2).max():.2f}] pixels")
    else:
        logger.warning("✗ Optical flow computation failed")

    return stitcher


def test_seam_detection(stitcher):
    """Test seam line detection"""
    logger.info("\n" + "="*70)
    logger.info("Testing Seam Line Detection")
    logger.info("="*70)

    # Create dummy frames
    logger.info("\nCreating test frames...")
    frame_left = np.random.randint(0, 256, (3840, 3840, 3), dtype=np.uint8)
    frame_right = np.random.randint(0, 256, (3840, 3840, 3), dtype=np.uint8)

    # Detect seam
    logger.info("Detecting seam line...")
    seam_x, seam_y = stitcher.detect_seam_line(frame_left, frame_right)

    if seam_x is not None and seam_y is not None:
        logger.info(f"✓ Seam X: {seam_x.shape} [{seam_x.min():.1f}, {seam_x.max():.1f}]")
        logger.info(f"✓ Seam Y: {seam_y.shape} [{seam_y.min():.1f}, {seam_y.max():.1f}]")
    else:
        logger.warning("✗ Seam detection failed")


def test_texture_creation(stitcher):
    """Test Metal texture creation"""
    logger.info("\n" + "="*70)
    logger.info("Testing Metal Texture Creation")
    logger.info("="*70)

    # Create dummy frames
    logger.info("\nCreating test frames...")
    frame_left = np.random.randint(0, 256, (3840, 3840, 3), dtype=np.uint8)
    frame_right = np.random.randint(0, 256, (3840, 3840, 3), dtype=np.uint8)

    # Detect seam
    seam_x, seam_y = stitcher.detect_seam_line(frame_left, frame_right)

    # Create seam masks
    logger.info("Creating seam masks...")
    seam_mask_1, seam_mask_2 = stitcher.create_seam_mask(frame_left, frame_right, seam_x)

    if seam_mask_1 is not None and seam_mask_2 is not None:
        logger.info(f"✓ Seam mask 1: {seam_mask_1.shape} {seam_mask_1.dtype}")
        logger.info(f"✓ Seam mask 2: {seam_mask_2.shape} {seam_mask_2.dtype}")
        logger.info(f"  Expected: ({stitcher.SEAM_MASK_HEIGHT}, {stitcher.SEAM_MASK_WIDTH}, 4) uint8")
    else:
        logger.warning("✗ Seam mask creation failed")

    # Create flow texture
    logger.info("Creating flow texture...")
    flow_fwd, flow_bwd = stitcher.compute_optical_flow(frame_left, frame_right)
    flow_tex = stitcher.create_flow_texture(flow_fwd, flow_bwd)

    if flow_tex is not None:
        logger.info(f"✓ Flow texture: {flow_tex.shape} {flow_tex.dtype}")
        logger.info(f"  Expected: ({stitcher.FLOW_TEXTURE_HEIGHT}, {stitcher.FLOW_TEXTURE_WIDTH}, 4) float32")
    else:
        logger.warning("✗ Flow texture creation failed")


def test_osv_stitching():
    """Test stitching with real OSV data"""
    logger.info("\n" + "="*70)
    logger.info("Testing Real OSV Data Stitching")
    logger.info("="*70)

    osv_path = Path("/Users/zindello/repositories/zindello/djiclireframe/osv_example/CAM_20260819204151_0011_D.OSV")

    if not osv_path.exists():
        logger.warning(f"OSV file not found: {osv_path}")
        return

    logger.info(f"\nOSV file: {osv_path.name}")
    logger.info(f"Size: {osv_path.stat().st_size / (1024**3):.2f} GB")

    # Extract frames
    logger.info("Extracting frames from OSV...")
    try:
        frames = extract_osv_frames(osv_path, [0, 1])
        if not frames:
            logger.warning("Failed to extract frames from OSV")
            return
        logger.info(f"✓ Extracted {len(frames)} frame pairs")
    except Exception as e:
        logger.warning(f"✗ Failed to extract OSV frames: {e}")
        return

    # Initialize stitcher
    logger.info("\nInitializing DJI Stitcher (proper)...")
    stitcher = DJIStitcherProper(enable_flow=True)

    if not stitcher.is_available():
        logger.warning("✗ DJI Stitcher framework not available")
        return

    logger.info("✓ Framework available")

    # Show info
    logger.info("\nStitcher info:")
    for key, value in stitcher.get_info().items():
        logger.info(f"  {key}: {value}")

    # Stitch frames
    for i, (frame_left, frame_right) in enumerate(frames):
        logger.info(f"\nStitching frame pair {i}...")
        logger.info(f"  Input: Left {frame_left.shape} {frame_left.dtype}, "
                   f"Right {frame_right.shape} {frame_right.dtype}")

        output = stitcher.stitch_frames(frame_left, frame_right)

        if output is None:
            logger.error(f"  ✗ Stitching failed for frame {i}")
            continue

        logger.info(f"  ✓ Stitched: {output.shape} {output.dtype}")
        logger.info(f"    Value range: [{output.min()}, {output.max()}]")
        logger.info(f"    Non-zero: {np.count_nonzero(output)} / {output.size}")

        # Save output
        try:
            import cv2
            output_dir = Path("/private/tmp/dji_proper_demo")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"stitch_proper_{i:03d}.png"

            # Convert RGB to BGR for OpenCV
            if output.shape == (7680, 3840, 3):
                output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            else:
                output_bgr = output

            cv2.imwrite(str(output_file), output_bgr)
            logger.info(f"  Saved: {output_file}")
            logger.info(f"  Size: {output_file.stat().st_size / (1024**2):.1f} MB")

        except Exception as e:
            logger.warning(f"  (Could not save output: {e})")


def main():
    """Main demo"""
    logger.info("\n" + "="*70)
    logger.info("DJI Stitcher with Optical Flow - Production Demo")
    logger.info("="*70)

    try:
        # Test 1: Optical flow
        stitcher = test_optical_flow()

        # Test 2: Seam detection
        test_seam_detection(stitcher)

        # Test 3: Texture creation
        test_texture_creation(stitcher)

        # Test 4: Real OSV data
        test_osv_stitching()

        logger.info("\n" + "="*70)
        logger.info("Demo completed!")
        logger.info("="*70)
        return True

    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
