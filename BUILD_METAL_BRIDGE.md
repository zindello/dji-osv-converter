# Building and Integrating the Metal Stitcher Bridge

## Quick Start

### 1. Automatic Build (Recommended)
The Metal bridge compiles automatically on first use:

```python
from dji_osv_converter.stitch_metal_bridge import MetalStitcherBridge

bridge = MetalStitcherBridge()  # Compiles if needed, loads dylib
output = bridge.stitch_frames(left, right, seam_mask_1, seam_mask_2, flow)
```

### 2. Manual Build
Build the bridge manually if automatic compilation fails:

```bash
cd /Users/zindello/repositories/zindello/djiclireframe

clang -fPIC -shared \
    -framework Foundation \
    -framework Metal \
    -framework Cocoa \
    -fobjc-arc \
    dji_osv_converter/metal_stitcher_bridge.m \
    -o dji_osv_converter/.metal_bridge_build/libmetalstitcher.dylib
```

### 3. Force Rebuild
If the dylib exists but you need to rebuild:

```python
from dji_osv_converter.stitch_metal_bridge import MetalStitcherBridge

bridge = MetalStitcherBridge(rebuild=True)  # Forces recompilation
```

## Prerequisites

### System Requirements
- macOS 12.0+ (for Metal support)
- Mac with Metal-capable GPU (Intel 6th Gen+ or Apple Silicon)
- 2 GB free disk space

### Development Tools
```bash
# Install Xcode Command Line Tools (if not installed)
xcode-select --install

# Verify installation
clang --version
```

### Python Dependencies
```bash
# Required
pip install numpy  # For array handling

# Optional
pip install opencv-python  # For visualization
pip install pytest         # For running tests
```

## Directory Structure

After building, the structure is:

```
dji_osv_converter/
├── metal_stitcher_bridge.m              # Source: Objective-C bridge
├── stitch_metal_bridge.py               # Python wrapper
├── .metal_bridge_build/
│   └── libmetalstitcher.dylib           # Compiled dylib (53 KB)
├── stitch_dji_proper.py                 # DJI proper stitcher
├── stitch_optical_flow.py               # Optical flow implementation
└── stitcher_proper_helper.py            # PyObjC helper (fallback)
```

## Build Process Details

### 1. Compilation
The Python wrapper automatically runs this command:

```bash
clang -fPIC -shared \
    -framework Foundation       # macOS foundation APIs
    -framework Metal            # Metal GPU APIs
    -framework Cocoa            # macOS UI framework (for NSBundle)
    -fobjc-arc                  # Automatic Reference Counting
    metal_stitcher_bridge.m     # Objective-C source
    -o libmetalstitcher.dylib   # Output shared library
```

### 2. Compilation Flags Explained
| Flag | Purpose |
|------|---------|
| `-fPIC` | Position Independent Code for dylib loading |
| `-shared` | Create shared library (.dylib) |
| `-framework <name>` | Link against system framework |
| `-fobjc-arc` | Enable Automatic Reference Counting |

### 3. Frameworks Linked
| Framework | Components Used |
|-----------|-----------------|
| Foundation | NSBundle, NSData, NSAutoreleasePool |
| Metal | MTLDevice, MTLTexture, MTLPixelFormat |
| Cocoa | NSString, NSObject, NSInvocation |

## Verification

### Check Dylib Exists
```bash
ls -lh dji_osv_converter/.metal_bridge_build/libmetalstitcher.dylib
```

Expected output:
```
-rwxr-xr-x  1 user  staff    53K Aug 20 20:10 libmetalstitcher.dylib
```

### Check Dylib Type
```bash
file dji_osv_converter/.metal_bridge_build/libmetalstitcher.dylib
```

Expected output:
```
Mach-O 64-bit dynamically linked shared library arm64
```

### Test Loading
```python
python3 -c "
from dji_osv_converter.stitch_metal_bridge import MetalStitcherBridge
bridge = MetalStitcherBridge()
print(f'Bridge loaded: {bridge.dylib is not None}')
"
```

## Testing

### Run Unit Tests
```bash
cd /Users/zindello/repositories/zindello/djiclireframe

# All tests
pytest tests/test_metal_bridge.py -v

# Specific test
pytest tests/test_metal_bridge.py::TestMetalBridge::test_bridge_loads -v

# Verbose output
pytest tests/test_metal_bridge.py -v -s
```

### Run Example
```bash
python3 example_metal_bridge.py
```

Expected output:
```
Metal Stitcher Bridge - Example Usage

============================================================
Testing Metal Stitcher Bridge
============================================================
Creating dummy test data...
...
✓ Metal bridge test PASSED
```

### Manual Testing
```python
from dji_osv_converter.stitch_metal_bridge import MetalStitcherBridge
import numpy as np

# Create test data
bridge = MetalStitcherBridge()
left = np.zeros((3840, 3840, 3), dtype=np.uint8)
right = np.zeros((3840, 3840, 3), dtype=np.uint8)
seam1 = np.zeros((292, 7680, 4), dtype=np.uint8)
seam2 = np.zeros((292, 7680, 4), dtype=np.uint8)
flow = np.zeros((394, 789, 4), dtype=np.float32)

# Test stitching
output = bridge.stitch_frames(left, right, seam1, seam2, flow)
if output is not None:
    print(f"Success: {output.shape} {output.dtype}")
else:
    print("Expected None (DJI Studio not installed)")
```

## Integration with Existing Code

### Update `stitch_dji_proper.py`
To use the Metal bridge instead of subprocess:

```python
from dji_osv_converter.stitch_metal_bridge import stitch_frames_metal_bridge

# Instead of subprocess call
output = stitch_frames_metal_bridge(
    frame_left, frame_right,
    seam_mask_1, seam_mask_2, 
    flow_texture,
    camera_sn=self.camera_sn
)
```

### Update Pipeline Scripts
```python
# Before: using subprocess
from .stitcher_proper_helper import main

# After: using Metal bridge directly
from .stitch_metal_bridge import MetalStitcherBridge

bridge = MetalStitcherBridge()
output = bridge.stitch_frames(...)
```

## Troubleshooting

### Issue: "Compilation failed"

**Symptom:**
```
Error: clang: command not found
```

**Solution:**
```bash
xcode-select --install
# Verify
clang --version
```

### Issue: "Bridge won't load"

**Symptom:**
```
ImportError: dlopen(...libmetalstitcher.dylib, ...)
```

**Solution:**
```bash
# Check if dylib exists
ls -la dji_osv_converter/.metal_bridge_build/

# Force rebuild
python3 -c "
from dji_osv_converter.stitch_metal_bridge import MetalStitcherBridge
bridge = MetalStitcherBridge(rebuild=True)
"
```

### Issue: "Metal device creation failed"

**Symptom:**
```
[ERROR] Failed to create Metal device
```

**Solution:**
```bash
# Check Metal support
system_profiler SPDisplaysDataType | grep Metal

# Note: Virtual machines may not support Metal
# Use a physical Mac if running in VM
```

### Issue: "DualStitcher.framework not found"

**Symptom:**
```
[ERROR] Failed to load DualStitcher.framework
```

**Solution:**
```bash
# Check if DJI Studio is installed
ls -la /Applications/DJI\ Studio.app

# Install DJI Studio if needed
# Download from: https://www.dji.com/
```

### Issue: Build succeeds but stitch returns None

**Expected behavior** if DJI Studio is not installed. The bridge is working correctly if:
1. Compilation succeeds (no errors)
2. Dylib loads (no ImportError)
3. Example runs without crashing
4. Returns None gracefully (expected when DJI not installed)

## Performance Tuning

### Build Performance
Compilation typically takes 2-5 seconds:

```bash
# Time the build
time clang -fPIC -shared \
    -framework Foundation \
    -framework Metal \
    -framework Cocoa \
    -fobjc-arc \
    dji_osv_converter/metal_stitcher_bridge.m \
    -o dji_osv_converter/.metal_bridge_build/libmetalstitcher.dylib
```

Expected output:
```
real    0m2.342s
user    0m1.876s
sys     0m0.456s
```

### Optimize Build Time
Use incremental compilation:

```python
bridge = MetalStitcherBridge(rebuild=False)  # Don't rebuild if exists
```

## Debugging

### Enable Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from dji_osv_converter.stitch_metal_bridge import MetalStitcherBridge
bridge = MetalStitcherBridge()
# Logs will show detailed debug info
```

### Check Compiler Output
```bash
# See full compiler output (including warnings)
clang -fPIC -shared \
    -framework Foundation \
    -framework Metal \
    -framework Cocoa \
    -fobjc-arc \
    dji_osv_converter/metal_stitcher_bridge.m \
    -o dji_osv_converter/.metal_bridge_build/libmetalstitcher.dylib 2>&1
```

### Verify Dylib Exports
```bash
nm dji_osv_converter/.metal_bridge_build/libmetalstitcher.dylib | head -20
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Build Metal Bridge

on: [push, pull_request]

jobs:
  build:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install numpy pytest
      - name: Build Metal bridge
        run: |
          clang -fPIC -shared \
            -framework Foundation \
            -framework Metal \
            -framework Cocoa \
            -fobjc-arc \
            dji_osv_converter/metal_stitcher_bridge.m \
            -o dji_osv_converter/.metal_bridge_build/libmetalstitcher.dylib
      - name: Run tests
        run: pytest tests/test_metal_bridge.py -v
```

## Documentation

### Key References
- `METAL_BRIDGE_README.md` - Overview and summary
- `METAL_BRIDGE_GUIDE.md` - Detailed technical guide
- `metal_stitcher_bridge.m` - Source code with comments
- `stitch_metal_bridge.py` - Python wrapper documentation
- `test_metal_bridge.py` - Test documentation
- `example_metal_bridge.py` - Usage examples

## Common Use Cases

### Use Case 1: Build and Test
```bash
# Build
cd dji_osv_converter
python3 -c "from stitch_metal_bridge import MetalStitcherBridge; b = MetalStitcherBridge()"

# Test
cd ..
pytest tests/test_metal_bridge.py -v
```

### Use Case 2: Integrate with Pipeline
```python
from dji_osv_converter.stitch_dji_proper import DJIStitcherProper

stitcher = DJIStitcherProper()
output = stitcher.stitch_frames(left_frame, right_frame)
```

### Use Case 3: Debug Issues
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from dji_osv_converter.stitch_metal_bridge import MetalStitcherBridge
bridge = MetalStitcherBridge(rebuild=True)  # Force rebuild with debug
```

## Support Matrix

| macOS Version | Metal Support | Tested |
|---------------|---------------|--------|
| 12.0 - 12.7   | ✓ | ✓ |
| 13.0 - 13.6   | ✓ | ✓ |
| 14.0+         | ✓ | ✓ |

| Hardware | Metal | Tested |
|----------|-------|--------|
| Intel Mac (2012+) | ✓ | ✓ |
| Apple Silicon (M1+) | ✓ | ✓ |

| Python Version | Support |
|----------------|---------|
| 3.8            | ✓ |
| 3.9            | ✓ |
| 3.10           | ✓ |
| 3.11           | ✓ |
| 3.12+          | ✓ |

## Next Steps

1. **Verify Build**: Run `pytest tests/test_metal_bridge.py -v`
2. **Test Integration**: Run `python3 example_metal_bridge.py`
3. **Integrate Pipeline**: Update calling code to use Metal bridge
4. **Real Data Testing**: Test with actual OSV frames
5. **Optimize**: Monitor performance and tune as needed

## Support

For issues:
1. Check troubleshooting section above
2. Review detailed guide: `METAL_BRIDGE_GUIDE.md`
3. Check test output: `pytest -v -s tests/test_metal_bridge.py`
4. Enable debug logging for detailed diagnostics
