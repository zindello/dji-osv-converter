#!/usr/bin/env python3
"""Demo: Using the Working StitcherObjC Wrapper

This script demonstrates how to use the production-ready PyObjC wrapper
for DJI's Stitcher class to stitch 360° panoramas from OSV files.

Usage:
    python3 demo_stitcher_working.py
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

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from dji_osv_converter.stitch_stitcher_objc import (
    StitcherObjC,
    extract_osv_frames,
)


def main():
    """Main demo"""
    print("\n" + "="*70)
    print("DJI Stitcher PyObjC Wrapper - Demo")
    print("="*70)

    # Step 1: Create stitcher instance
    print("\n1. Initializing Stitcher...")
    stitcher = StitcherObjC(camera_sn="default")
    print(f"   ✓ Created: {stitcher}")

    # Step 2: Check availability
    print("\n2. Checking framework...")
    if not stitcher.is_available():
        print("   ✗ Framework not available")
        print("   Make sure DJI Studio is installed")
        return False

    print("   ✓ Framework is available")

    # Step 3: Show info
    print("\n3. Framework information:")
    info = stitcher.get_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    # Step 4: Load calibration
    if stitcher.calibration:
        print("\n4. Camera calibration:")
        calib = stitcher.calibration
        print(f"   Left camera:")
        print(f"     fx={calib.intrinsic00.get('fx', 'N/A'):.2f}, "
              f"fy={calib.intrinsic00.get('fy', 'N/A'):.2f}")
        print(f"     cx={calib.intrinsic00.get('cx', 'N/A'):.2f}, "
              f"cy={calib.intrinsic00.get('cy', 'N/A'):.2f}")
        print(f"   Right camera:")
        print(f"     fx={calib.intrinsic01.get('fx', 'N/A'):.2f}, "
              f"fy={calib.intrinsic01.get('fy', 'N/A'):.2f}")
    else:
        print("\n4. Calibration: Not loaded")

    # Step 5: Try to find OSV file
    osv_path = Path("/Users/zindello/repositories/zindello/djiclireframe/osv_example/CAM_20260819204151_0011_D.OSV")

    if not osv_path.exists():
        print(f"\n5. OSV file: Not found at {osv_path}")
        print("   Provide an OSV file to test stitching")
        return True

    # Step 6: Extract frames
    print(f"\n5. Extracting frames from OSV...")
    print(f"   File: {osv_path.name}")
    frames = extract_osv_frames(osv_path, [0, 1])

    if not frames:
        print("   ✗ Failed to extract frames")
        return False

    print(f"   ✓ Extracted {len(frames)} frame pairs")

    # Step 7: Stitch frames
    for i, (frame_left, frame_right) in enumerate(frames):
        print(f"\n6.{i+1}. Stitching frame pair {i}...")
        print(f"   Input: Left {frame_left.shape} uint8, Right {frame_right.shape} uint8")

        output = stitcher.stitch_frames(frame_left, frame_right)

        if output is None:
            print("   ✗ Stitching failed")
            continue

        print(f"   ✓ Stitched: {output.shape} uint8")
        print(f"     Value range: [{output.min()}, {output.max()}]")
        print(f"     Non-zero pixels: {np.count_nonzero(output)} / {output.size}")

        # Try to save
        try:
            import cv2
            output_dir = Path("/private/tmp/stitcher_demo")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"demo_output_{i:03d}.png"

            # Convert to correct shape if needed
            output_to_save = output
            if output_to_save.shape == (3840, 7680, 3):
                output_to_save = np.transpose(output_to_save, (1, 0, 2))

            # Convert RGB to BGR for OpenCV
            if output_to_save.shape[2] == 3:
                output_bgr = cv2.cvtColor(output_to_save, cv2.COLOR_RGB2BGR)
            else:
                output_bgr = output_to_save

            cv2.imwrite(str(output_file), output_bgr)
            print(f"   Saved: {output_file}")
            print(f"   Size: {output_file.stat().st_size / (1024*1024):.1f} MB")

        except Exception as e:
            print(f"   (Could not save output: {e})")

    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
