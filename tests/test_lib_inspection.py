#!/usr/bin/env python3
"""Inspect DualStitcher library in detail"""

import sys
import subprocess
from pathlib import Path

FRAMEWORKS_DIR = "/Applications/DJI Studio.app/Contents/Frameworks"
FRAMEWORK_BINARY = "/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework/Versions/A/DualStitcher"
PARAM_PATH = "/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework/Versions/A/Resources/Stitcher-Mac.bundle/case_00"

# Use nm to list symbols
print("="*70)
print("Symbols in DualStitcher binary:")
print("="*70)

result = subprocess.run(
    ["nm", "-j", FRAMEWORK_BINARY],
    capture_output=True,
    text=True
)

symbols = [s for s in result.stdout.split('\n') if 'ImageStitcher' in s or 'ML_' in s]
for sym in sorted(symbols)[:40]:
    print(f"  {sym}")

# Try calling with different RTLD modes
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
        try:
            ctypes.CDLL(str(dep_path), mode=0x1)  # RTLD_LAZY
        except:
            pass

print("Testing different RTLD modes...")

for mode_name, mode_val in [("RTLD_LAZY", 0x1), ("RTLD_NOW", 0x2), ("RTLD_GLOBAL", 0x100)]:
    try:
        print(f"\\nTrying mode {{mode_name}} ({{mode_val}}):...")
        lib = ctypes.CDLL("{FRAMEWORK_BINARY}", mode=mode_val)
        print(f"  Loaded successfully")

        # Try to get the function
        try:
            func = lib.ML_ImageStitcherInit
            print(f"  Found ML_ImageStitcherInit at {{func}}")
        except AttributeError as e:
            print(f"  ML_ImageStitcherInit not found: {{e}}")

    except OSError as e:
        print(f"  Failed to load: {{e}}")

print("\\nDone")
'''

print("\n" + "="*70)
print("Testing library loading with different modes:")
print("="*70)

env = dict(__import__('os').environ)
env['DYLD_FRAMEWORK_PATH'] = FRAMEWORKS_DIR

try:
    result = subprocess.run(
        [sys.executable, "-c", test_script],
        env=env,
        capture_output=True,
        text=True,
        timeout=10
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])

except subprocess.TimeoutExpired:
    print("ERROR: Timed out")
except Exception as e:
    print(f"ERROR: {e}")

# Check if there's Objective-C support
print("\n" + "="*70)
print("Checking Objective-C support:")
print("="*70)

objc_script = f'''
try:
    import objc
    print("PyObjC is available")

    from Foundation import NSBundle
    fw_path = "{FRAMEWORK_BINARY.replace('/Versions/A/DualStitcher', '')}"
    bundle = NSBundle.bundleWithPath_(fw_path)
    print(f"Bundle loaded: {{bundle}}")

    if bundle:
        print("Bundle classes available:")
        # Try to access Stitcher class
        try:
            stitcher_class = objc.lookUpClass("Stitcher")
            print(f"  Stitcher class: {{stitcher_class}}")
        except Exception as e:
            print(f"  Could not find Stitcher class: {{e}}")

except ImportError:
    print("PyObjC not available")
except Exception as e:
    print(f"Error: {{e}}")
    import traceback
    traceback.print_exc()
'''

result = subprocess.run(
    [sys.executable, "-c", objc_script],
    capture_output=True,
    text=True,
    timeout=10
)

print(result.stdout)
if result.stderr:
    print("Errors:", result.stderr[:300])
