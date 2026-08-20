#!/usr/bin/env python3
"""Test different ML_ImageStitcherInit call variations"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
import sys

FRAMEWORKS_DIR = "/Applications/DJI Studio.app/Contents/Frameworks"
FRAMEWORK_BINARY = "/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework/Versions/A/DualStitcher"
PARAM_PATH = "/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework/Versions/A/Resources/Stitcher-Mac.bundle/case_00"

test_script = f'''
import ctypes
import sys
from pathlib import Path

frameworks_dir = Path("{FRAMEWORKS_DIR}")

# Pre-load dependencies
deps = ["libdInfer.dylib", "libMNN.3.1.2.dylib", "ml_fis.framework/ml_fis", "openmcv2.framework/openmcv2"]
for dep_name in deps:
    dep_path = frameworks_dir / dep_name
    if dep_path.exists():
        ctypes.CDLL(str(dep_path), mode=0x9)

# Load main library
lib = ctypes.CDLL("{FRAMEWORK_BINARY}", mode=0x9)

print("Testing different ML_ImageStitcherInit signatures...")

# Define the function
lib.ML_ImageStitcherInit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
lib.ML_ImageStitcherInit.restype = ctypes.c_int

param_path = "{PARAM_PATH}".encode('utf-8')

# Test 1: All empty strings
print("\\nTest 1: All empty strings")
try:
    result = lib.ML_ImageStitcherInit(b"", b"", b"")
    print(f"  Result: {{result}}")
except Exception as e:
    print(f"  Error: {{e}}")

# Test 2: param_path in 2nd arg
print("\\nTest 2: param_path in 2nd arg")
try:
    result = lib.ML_ImageStitcherInit(b"", param_path, b"")
    print(f"  Result: {{result}}")
except Exception as e:
    print(f"  Error: {{e}}")

# Test 3: param_path in 1st arg
print("\\nTest 3: param_path in 1st arg")
try:
    result = lib.ML_ImageStitcherInit(param_path, b"", b"")
    print(f"  Result: {{result}}")
except Exception as e:
    print(f"  Error: {{e}}")

# Test 4: param_path in 3rd arg
print("\\nTest 4: param_path in 3rd arg")
try:
    result = lib.ML_ImageStitcherInit(b"", b"", param_path)
    print(f"  Result: {{result}}")
except Exception as e:
    print(f"  Error: {{e}}")

print("\\nAll tests completed")
'''

print("="*70)
print("Running init variation tests...")
print("="*70)

env = dict(__import__('os').environ)
env['DYLD_FRAMEWORK_PATH'] = FRAMEWORKS_DIR
env['DYLD_LIBRARY_PATH'] = FRAMEWORKS_DIR

try:
    result = subprocess.run(
        [sys.executable, "-c", test_script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60
    )

    print("\nSTDOUT:")
    print(result.stdout)

    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)

    print(f"\nExit code: {result.returncode}")

except subprocess.TimeoutExpired:
    print("ERROR: Subprocess timed out (likely crashed)")
except Exception as e:
    print(f"ERROR: {e}")
