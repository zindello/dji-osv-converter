#!/usr/bin/env python3
"""Debug DJI DualStitcher Framework Loading

This script helps diagnose framework loading issues by examining
the symbols available and testing the C API step by step.
"""

import sys
import logging
import ctypes
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)

# Framework path
FRAMEWORK_BINARY = Path("/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework/Versions/A/DualStitcher")
FRAMEWORKS_DIR = Path("/Applications/DJI Studio.app/Contents/Frameworks")
FRAMEWORK_RESOURCES = Path("/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework/Versions/A/Resources/Stitcher-Mac.bundle/case_00")


def test_framework_binary_exists():
    """Check if binary exists"""
    print("\n" + "="*70)
    print("TEST: Binary Exists")
    print("="*70)
    exists = FRAMEWORK_BINARY.exists()
    print(f"Framework binary: {FRAMEWORK_BINARY}")
    print(f"Exists: {exists}")
    return exists


def test_load_with_ctypes():
    """Try loading the framework with ctypes"""
    print("\n" + "="*70)
    print("TEST: Load with ctypes (in process)")
    print("="*70)

    try:
        lib = ctypes.CDLL(str(FRAMEWORK_BINARY), mode=0x9, use_errno=True)
        print(f"Successfully loaded: {FRAMEWORK_BINARY}")

        # Try to find available symbols
        print("\nTrying to access symbols:")
        symbols = [
            "ML_ImageStitcherInit",
            "ML_ImageStitcherDeInit",
            "ML_ImageStitcherProcessStepA",
            "ML_ImageStitcherProcessStepB",
        ]

        for sym in symbols:
            try:
                func = getattr(lib, sym)
                print(f"  {sym}: OK")
            except AttributeError as e:
                print(f"  {sym}: NOT FOUND - {e}")

        return True

    except OSError as e:
        print(f"Failed to load: {e}")
        print(f"Error code: {ctypes.get_errno()}")
        return False


def test_subprocess_loading():
    """Test loading in subprocess with DYLD vars"""
    print("\n" + "="*70)
    print("TEST: Subprocess Loading with DYLD")
    print("="*70)

    script = f"""
import ctypes
import sys
import os

print("PID:", os.getpid())
print("DYLD_FRAMEWORK_PATH:", os.environ.get('DYLD_FRAMEWORK_PATH'))

try:
    lib = ctypes.CDLL(r"{FRAMEWORK_BINARY}", mode=0x9, use_errno=True)
    print("SUCCESS: Loaded framework")

    # Try calling init with NULL pointers
    lib.ML_ImageStitcherInit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.ML_ImageStitcherInit.restype = ctypes.c_int

    print("SUCCESS: Found ML_ImageStitcherInit")
    print("Calling ML_ImageStitcherInit with empty strings...")
    result = lib.ML_ImageStitcherInit(b"", b"{FRAMEWORK_RESOURCES}", b"")
    print(f"Result: {{result}}")
    sys.exit(0 if result == 0 else 1)

except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""

    env = dict(os.environ)
    env['DYLD_FRAMEWORK_PATH'] = str(FRAMEWORKS_DIR)
    env['DYLD_LIBRARY_PATH'] = str(FRAMEWORKS_DIR)

    print("Running in subprocess...")
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=10
        )

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        print(f"Exit code: {result.returncode}")
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("ERROR: Subprocess timed out (likely hung or crashed)")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_dependencies():
    """Check if dependencies are available"""
    print("\n" + "="*70)
    print("TEST: Dependencies")
    print("="*70)

    deps = [
        "openmcv2.framework/openmcv2",
        "ml_fis.framework/ml_fis",
        "libMNN.3.1.2.dylib",
        "libdInfer.dylib",
    ]

    all_exist = True
    for dep in deps:
        dep_path = FRAMEWORKS_DIR / dep
        exists = dep_path.exists()
        print(f"{dep}: {'OK' if exists else 'MISSING'}")
        if not exists:
            all_exist = False

    return all_exist


def test_resources():
    """Check if resources are available"""
    print("\n" + "="*70)
    print("TEST: Resources")
    print("="*70)

    print(f"Resources dir: {FRAMEWORK_RESOURCES}")
    print(f"Exists: {FRAMEWORK_RESOURCES.exists()}")

    if FRAMEWORK_RESOURCES.exists():
        calib_file = FRAMEWORK_RESOURCES / "intrinsic_extrinsic.json"
        print(f"Calibration file: {calib_file.name} - {'OK' if calib_file.exists() else 'MISSING'}")

        # List some files
        files = list(FRAMEWORK_RESOURCES.glob("*.json"))
        print(f"JSON files: {len(files)}")
        if files:
            print(f"  First: {files[0].name}")

        return calib_file.exists()

    return False


def main():
    print("\n" + "="*70)
    print("DJI DualStitcher Framework Debug")
    print("="*70)

    results = []

    # Test binary exists
    results.append(("Binary Exists", test_framework_binary_exists()))

    # Test dependencies
    results.append(("Dependencies", test_dependencies()))

    # Test resources
    results.append(("Resources", test_resources()))

    # Test in-process loading
    results.append(("In-process Loading", test_load_with_ctypes()))

    # Test subprocess loading
    results.append(("Subprocess Loading", test_subprocess_loading()))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25} {status}")

    print("="*70)

    all_pass = all(r for _, r in results)
    return 0 if all_pass else 1


if __name__ == "__main__":
    import os
    sys.exit(main())
