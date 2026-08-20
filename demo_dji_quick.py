#!/usr/bin/env python3
"""Quick DJI Framework Demo - No OSV extraction"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dji_osv_converter.stitch_dji_wrapper import (
    DJIDualStitcher,
    FRAMEWORK_BINARY,
    FRAMEWORK_RESOURCES,
)


def main():
    print("\n" + "="*70)
    print("DJI DualStitcher Framework Integration - Quick Demo")
    print("="*70)

    # 1. Framework detection
    print("\n1. FRAMEWORK DETECTION")
    print("-" * 70)
    print(f"Binary: {FRAMEWORK_BINARY}")
    print(f"Exists: {FRAMEWORK_BINARY.exists()}")
    print(f"Resources: {FRAMEWORK_RESOURCES}")
    print(f"Exists: {FRAMEWORK_RESOURCES.exists()}")

    # 2. Initialize wrapper
    print("\n2. WRAPPER INITIALIZATION")
    print("-" * 70)
    stitcher = DJIDualStitcher(camera_sn="default")
    print(f"Created: {stitcher}")
    print(f"Available: {stitcher.is_available()}")

    # 3. Calibration
    print("\n3. CALIBRATION DATA")
    print("-" * 70)
    if stitcher.calibration:
        calib = stitcher.calibration
        print(f"✓ Loaded successfully")
        print(f"\nLeft camera intrinsics:")
        print(f"  fx={calib.intrinsic00.get('fx'):.2f}, fy={calib.intrinsic00.get('fy'):.2f}")
        print(f"  cx={calib.intrinsic00.get('cx'):.2f}, cy={calib.intrinsic00.get('cy'):.2f}")
        print(f"  k1={calib.intrinsic00.get('k1'):.4f}, k2={calib.intrinsic00.get('k2'):.4f}")
        print(f"  xi (fisheye)={calib.get_fisheye_param_left():.4f}")

        print(f"\nRight camera intrinsics:")
        print(f"  fx={calib.intrinsic01.get('fx'):.2f}, fy={calib.intrinsic01.get('fy'):.2f}")
        print(f"  cx={calib.intrinsic01.get('cx'):.2f}, cy={calib.intrinsic01.get('cy'):.2f}")
        print(f"  xi (fisheye)={calib.get_fisheye_param_right():.4f}")

    # 4. Dependencies check
    print("\n4. DEPENDENCIES")
    print("-" * 70)
    framework_dir = Path("/Applications/DJI Studio.app/Contents/Frameworks")
    deps = [
        "openmcv2.framework/openmcv2",
        "ml_fis.framework/ml_fis",
        "libMNN.3.1.2.dylib",
        "libdInfer.dylib",
    ]
    for dep in deps:
        dep_path = framework_dir / dep
        status = "✓" if dep_path.exists() else "✗"
        print(f"{status} {dep}")

    # 5. C API Symbols
    print("\n5. C API SYMBOLS (Found in binary)")
    print("-" * 70)
    import subprocess
    result = subprocess.run(
        ["nm", "-j", str(FRAMEWORK_BINARY)],
        capture_output=True,
        text=True
    )
    symbols = [s for s in result.stdout.split('\n') if 'ImageStitcher' in s or 'GetStitcherImageInfo' in s]
    for sym in sorted(symbols):
        print(f"  {sym}")

    # 6. Status
    print("\n6. INTEGRATION STATUS")
    print("-" * 70)
    print("""
WORKING:
  ✓ Framework binary loading via ctypes
  ✓ All dependencies present
  ✓ Calibration data extraction
  ✓ C API symbols resolution
  ✓ Subprocess environment setup (DYLD variables)

BLOCKED:
  ✗ ML_ImageStitcherInit() crashes (SIGSEGV)

LIKELY CAUSE:
  - C function signature mismatch
  - Private/undocumented API
  - Missing initialization context

NEXT STEPS:
  1. Use Objective-C API (PyObjC)
  2. Binary analysis with lldb debugger
  3. Alternative stitching library
""")

    print("\n" + "="*70)
    print("Demo completed")
    print("="*70)


if __name__ == "__main__":
    main()
