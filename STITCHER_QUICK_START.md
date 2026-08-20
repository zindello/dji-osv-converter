# StitcherObjC Quick Start Guide

## Installation

### Prerequisites

```bash
# PyObjC and Metal framework support
pip install pyobjc pyobjc-framework-Metal

# For frame extraction from OSV files
pip install av opencv-python

# NumPy (usually already installed)
pip install numpy
```

### Verify Installation

```bash
# Check DJI Studio is installed
ls -la "/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework"

# Verify DualStitcher.framework exists - output should show framework binary
```

## Basic Usage (5 minutes)

### Step 1: Import

```python
from dji_osv_converter.stitch_stitcher_objc import StitcherObjC, extract_osv_frames
import numpy as np
from pathlib import Path
```

### Step 2: Create Stitcher

```python
# Initialize stitcher
stitcher = StitcherObjC(camera_sn="default")

# Check if framework is available
if not stitcher.is_available():
    print("DJI Studio not installed")
    exit(1)

print("✓ Stitcher ready")
```

### Step 3: Prepare Frames

**Option A: Extract from OSV file**

```python
osv_path = Path("CAM_20260819204151_0011_D.OSV")

# Extract first 3 frames
frames = extract_osv_frames(osv_path, [0, 1, 2])

for frame_left, frame_right in frames:
    # Both are 3840×3840×3 uint8 numpy arrays
    print(f"Frame pair: {frame_left.shape}")
```

**Option B: Load your own frames**

```python
# Frames must be:
# - Shape: (3840, 3840, 3)
# - Dtype: uint8
# - Color: RGB (not BGR)

frame_left = np.zeros((3840, 3840, 3), dtype=np.uint8)
frame_right = np.zeros((3840, 3840, 3), dtype=np.uint8)
# ... load actual data
```

### Step 4: Stitch Frames

```python
# Stitch a frame pair
output = stitcher.stitch_frames(frame_left, frame_right)

if output is not None:
    print(f"✓ Stitched: {output.shape}")
    print(f"  Range: [{output.min()}, {output.max()}]")
else:
    print("✗ Stitching failed")
```

### Step 5: Save Output

```python
import cv2

# Convert RGB to BGR for OpenCV
output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

# Save
cv2.imwrite("panorama.png", output_bgr)
print("✓ Saved panorama.png")
```

## Complete Example

```python
#!/usr/bin/env python3
import sys
from pathlib import Path
from dji_osv_converter.stitch_stitcher_objc import StitcherObjC, extract_osv_frames
import cv2

# Initialize
stitcher = StitcherObjC()
if not stitcher.is_available():
    print("Framework not available")
    sys.exit(1)

# Extract frames
osv_path = Path("CAM_20260819204151_0011_D.OSV")
frames = extract_osv_frames(osv_path, [0, 1, 2])

# Process each frame
for i, (left, right) in enumerate(frames):
    print(f"Stitching frame {i}...", end=" ", flush=True)
    
    output = stitcher.stitch_frames(left, right)
    
    if output is not None:
        print("✓")
        # Save
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"panorama_{i:03d}.png", output_bgr)
    else:
        print("✗")

print("\n✓ Done!")
```

## Common Tasks

### Get Framework Information

```python
stitcher = StitcherObjC()

info = stitcher.get_info()
for key, value in info.items():
    print(f"{key}: {value}")

# Output:
# framework_path: /Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework
# framework_available: True
# stitching_method: simpleStitchSeams
# input_dims: (3840, 3840)
# output_dims: (7680, 3840)
```

### Get Camera Calibration

```python
if stitcher.calibration:
    calib = stitcher.calibration
    
    # Get intrinsic parameters
    print("Left camera:")
    print(f"  fx={calib.intrinsic00.get('fx'):.2f}")
    print(f"  fy={calib.intrinsic00.get('fy'):.2f}")
    print(f"  cx={calib.intrinsic00.get('cx'):.2f}")
    print(f"  cy={calib.intrinsic00.get('cy'):.2f}")
    
    # Get transformation matrices
    K_left = calib.get_camera_matrix_left()      # 3×3
    dist_left = calib.get_distortion_coeffs_left()  # 6 values
```

### Process Multiple OSV Files

```python
from pathlib import Path

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

for osv_path in Path(".").glob("*.OSV"):
    print(f"Processing {osv_path.name}...")
    
    frames = extract_osv_frames(osv_path, range(10))
    
    for i, (left, right) in enumerate(frames):
        output = stitcher.stitch_frames(left, right)
        if output is not None:
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            filename = output_dir / f"{osv_path.stem}_{i:03d}.png"
            cv2.imwrite(str(filename), output_bgr)
            print(f"  Frame {i}: ✓")
```

### Validate Input Frames

```python
import numpy as np

def validate_frame(frame, name="frame"):
    """Check if frame is valid for stitching"""
    
    if not isinstance(frame, np.ndarray):
        print(f"✗ {name}: Not a numpy array")
        return False
    
    if frame.shape != (3840, 3840, 3):
        print(f"✗ {name}: Invalid shape {frame.shape}, need (3840, 3840, 3)")
        return False
    
    if frame.dtype != np.uint8:
        print(f"✗ {name}: Invalid dtype {frame.dtype}, need uint8")
        return False
    
    if frame.min() < 0 or frame.max() > 255:
        print(f"✗ {name}: Value range [{frame.min()}, {frame.max()}], need [0, 255]")
        return False
    
    print(f"✓ {name}: Valid")
    return True

# Use it
validate_frame(frame_left, "Left")
validate_frame(frame_right, "Right")
```

## Troubleshooting

### "Framework not available"

Check if DJI Studio is installed:

```bash
ls -la "/Applications/DJI Studio.app"
```

If not installed, download from DJI website and install.

### "PyObjC not available"

Install PyObjC:

```bash
pip install pyobjc pyobjc-framework-Metal
```

### Output is all black

Possible causes:
1. Input frames are wrong (not actual fisheye camera frames)
2. Frames are misaligned (from different timestamps)
3. Stitching failed silently

Try:
```python
# Check input frames visually
cv2.imwrite("debug_left.png", frame_left)
cv2.imwrite("debug_right.png", frame_right)

# Check output
if np.count_nonzero(output) < 100:
    print("WARNING: Output is mostly black")
```

### Subprocess timeout

If stitching takes >120 seconds:

1. Check system resources (CPU, GPU, memory)
2. Close other applications
3. Try with simpler frames
4. Check if GPU is available: `MTLCreateSystemDefaultDevice()` returns None

### ImportError: cannot import name 'MTLCreateSystemDefaultDevice'

Metal framework not installed:

```bash
pip install pyobjc-framework-Metal
```

## Performance Tips

### Faster Processing

1. **Disable unused frameworks**: Only load Metal if GPU available
2. **Process in parallel**: Use multiprocessing for multiple files
3. **Cache calibration**: Reuse same stitcher instance
4. **Batch frames**: Extract multiple frames at once

### Example: Parallel Processing

```python
from multiprocessing import Pool

def stitch_frame(args):
    frame_left, frame_right, i = args
    stitcher = StitcherObjC()
    output = stitcher.stitch_frames(frame_left, frame_right)
    return i, output

# Process 4 frame pairs in parallel
osv_path = Path("test.OSV")
frames = extract_osv_frames(osv_path, range(4))
frame_args = [(l, r, i) for i, (l, r) in enumerate(frames)]

with Pool(4) as p:
    results = p.map(stitch_frame, frame_args)

for i, output in results:
    if output is not None:
        print(f"Frame {i}: ✓")
```

## API Reference (Quick)

### StitcherObjC

```python
# Create instance
stitcher = StitcherObjC(camera_sn="default")

# Methods
stitcher.is_available() → bool
stitcher.get_info() → dict
stitcher.stitch_frames(left, right) → np.ndarray or None
stitcher.calibration → CalibrationData or None
```

### CalibrationData

```python
# Access parameters
calib.intrinsic00 → dict (left camera)
calib.intrinsic01 → dict (right camera)
calib.extrinsic00 → dict
calib.extrinsic01 → dict

# Get matrices
calib.get_camera_matrix_left() → np.ndarray (3×3)
calib.get_camera_matrix_right() → np.ndarray (3×3)
calib.get_distortion_coeffs_left() → np.ndarray (6,)
calib.get_distortion_coeffs_right() → np.ndarray (6,)
calib.get_fisheye_param_left() → float (xi)
calib.get_fisheye_param_right() → float (xi)
```

### extract_osv_frames()

```python
# Extract frames from OSV file
frames = extract_osv_frames(
    Path("test.OSV"),           # Path to OSV file
    [0, 1, 2, 5, 10]            # Frame indices
)

# Result: list of (left_frame, right_frame) tuples
# Each frame is 3840×3840×3 uint8 RGB
```

## Next Steps

- See `STITCHER_WORKING_IMPLEMENTATION.md` for detailed documentation
- Run `tests/test_stitcher_working.py` for full test suite
- See `demo_stitcher_working.py` for complete example

## Support

If you encounter issues:

1. Check that DJI Studio is installed
2. Verify PyObjC installation: `python3 -c "import objc; print(objc.__version__)"`
3. Run test suite: `python3 tests/test_stitcher_working.py`
4. Check logs in stderr for detailed error messages

---

**Status**: ✅ Production-ready, tested with real OSV data
