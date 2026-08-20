# Metal Stitcher Bridge - Implementation Guide

## Overview

The Metal Stitcher Bridge is a native Objective-C/C bridge that creates actual Metal textures and calls DJI's `dynamicStitchSeams` method for hardware-accelerated image stitching.

This eliminates the PyObjC dependency and provides direct access to Metal APIs for texture creation and GPU-accelerated stitching.

## Architecture

### Components

1. **metal_stitcher_bridge.m** - Native Objective-C/C implementation
   - Metal device creation
   - Metal texture creation (BGRA8Unorm, RGBA16Float)
   - DualStitcher framework loading
   - Stitcher instance creation
   - dynamicStitchSeams method invocation
   - Output data extraction

2. **stitch_metal_bridge.py** - Python wrapper
   - Compiles the Objective-C bridge to a dylib
   - Provides Python API for calling the bridge
   - Data marshalling between Python/NumPy and Objective-C
   - Error handling and logging

3. **test_metal_bridge.py** - Comprehensive test suite
   - Bridge compilation tests
   - Metal texture creation validation
   - Stitching with dummy and real data
   - Output validation (equirectangular verification)

## Metal Texture Formats

The bridge creates Metal textures with specific formats required by DJI's stitcher:

### Seam Masks
- **Format**: `MTLPixelFormatBGRA8Unorm`
- **Dimensions**: 7680×292 (width×height)
- **Bytes per pixel**: 4 (B, G, R, A)
- **Total size**: 7680 × 292 × 4 = 8,957,440 bytes per mask
- **NumPy shape**: (292, 7680, 4) with dtype uint8

### Flow Texture
- **Format**: `MTLPixelFormatRGBA16Float`
- **Dimensions**: 789×394 (width×height)
- **Bytes per pixel**: 8 (4 channels × 2 bytes per channel)
- **Total size**: 789 × 394 × 8 = 2,490,792 bytes
- **NumPy shape**: (394, 789, 4) with dtype float32

### Output
- **Dimensions**: 7680×3840 (width×height)
- **Channels**: 3 (RGB)
- **Format**: uint8
- **Total size**: 7680 × 3840 × 3 = 88,473,600 bytes
- **NumPy shape**: (7680, 3840, 3) with dtype uint8

## Method Signature

The `dynamicStitchSeams` method signature from lldb analysis:

```objc
- (id)dynamicStitchSeams:(id<MTLTexture>)seam0
                   seam1:(id<MTLTexture>)seam1
            isCompensate:(BOOL)isCompensate
         enableSunGlare:(BOOL)enableSunGlare
            sgGammaTex:(id<MTLTexture>)sgGammaTex
              flowTex:(id<MTLTexture>)flowTex
```

**Parameters:**
- `seam0`: Left seam mask (7680×292 BGRA8Unorm)
- `seam1`: Right seam mask (7680×292 BGRA8Unorm)
- `isCompensate`: Enable exposure compensation (YES)
- `enableSunGlare`: Enable sun glare correction (NO)
- `sgGammaTex`: Sun glare gamma texture (nil)
- `flowTex`: Optical flow data (789×394 RGBA16Float)

**Returns:** Image data (NSData or similar) containing 7680×3840×3 uint8 RGB

## Compilation

### Prerequisites

```bash
# macOS Xcode Command Line Tools
xcode-select --install

# Python with numpy
pip install numpy
```

### Automatic Compilation

The Python wrapper automatically compiles the bridge on first use:

```python
from dji_osv_converter.stitch_metal_bridge import MetalStitcherBridge

bridge = MetalStitcherBridge()  # Compiles if needed
```

### Manual Compilation

```bash
cd dji_osv_converter

clang -fPIC -shared \
    -framework Foundation \
    -framework Metal \
    -framework Cocoa \
    -fobjc-arc \
    metal_stitcher_bridge.m \
    -o .metal_bridge_build/libmetalstitcher.dylib
```

### Compilation Details

The bridge is compiled with:
- **Language**: Objective-C with ARC (Automatic Reference Counting)
- **Frameworks**: Foundation, Metal, Cocoa
- **Position Independent Code**: -fPIC for dylib compatibility
- **Output**: Shared library (.dylib) in `.metal_bridge_build/` directory

## Usage

### Python API

```python
from dji_osv_converter.stitch_metal_bridge import MetalStitcherBridge
import numpy as np

# Create bridge (compiles if needed)
bridge = MetalStitcherBridge()

# Prepare input data
left_frame = np.zeros((3840, 3840, 3), dtype=np.uint8)      # Your left frame
right_frame = np.zeros((3840, 3840, 3), dtype=np.uint8)     # Your right frame
seam_mask_1 = np.zeros((292, 7680, 4), dtype=np.uint8)      # Seam mask 1
seam_mask_2 = np.zeros((292, 7680, 4), dtype=np.uint8)      # Seam mask 2
flow_texture = np.zeros((394, 789, 4), dtype=np.float32)    # Flow data

# Perform stitching
output = bridge.stitch_frames(
    left_frame, right_frame,
    seam_mask_1, seam_mask_2,
    flow_texture,
    camera_sn="camera_001"
)

# Output is 7680×3840×3 uint8 RGB
if output is not None:
    print(f"Stitched: {output.shape} {output.dtype}")
```

### With DJI Proper Stitcher

```python
from dji_osv_converter.stitch_dji_proper import DJIStitcherProper

stitcher = DJIStitcherProper()
output = stitcher.stitch_frames(left_frame, right_frame)
# Returns 7680×3840×3 uint8 RGB equirectangular
```

## Implementation Details

### Metal Texture Creation

```c
id<MTLTexture> create_metal_texture(
    id<MTLDevice> device,
    const void *data,
    size_t data_size,
    NSUInteger width,
    NSUInteger height,
    MTLPixelFormat format)
```

**Steps:**
1. Create MTLTextureDescriptor with specified format and dimensions
2. Set resource options to MTLResourceStorageModeManaged
3. Create texture from descriptor
4. Calculate bytes per row based on format and width
5. Copy pixel data to texture using replaceRegion
6. Return texture handle

### Framework Loading

```c
Class load_stitcher_class(NSBundle **out_bundle)
```

**Steps:**
1. Load DualStitcher.framework from `/Applications/DJI Studio.app`
2. Get NSBundle for the framework
3. Load the framework binary
4. Look up Stitcher class using NSClassFromString
5. Return class handle

### Method Invocation

```c
id call_dynamic_stitch_seams(
    id stitcher,
    id<MTLTexture> seam_tex_1,
    id<MTLTexture> seam_tex_2,
    BOOL is_compensate,
    BOOL enable_sun_glare,
    id<MTLTexture> sg_gamma_tex,
    id<MTLTexture> flow_tex)
```

**Steps:**
1. Get method signature using NSMethodSignature
2. Create NSInvocation with the signature
3. Set target and selector
4. Set arguments at correct indices
5. Invoke the method
6. Extract and return result

## Data Flow

```
Python NumPy Arrays
    ↓
[Input Validation & Format Check]
    ↓
[Create MTLDevice]
    ↓
[Create Metal Textures]
    - seam_mask_1 → MTLPixelFormatBGRA8Unorm
    - seam_mask_2 → MTLPixelFormatBGRA8Unorm
    - flow_texture → MTLPixelFormatRGBA16Float
    ↓
[Load DualStitcher.framework]
    ↓
[Create Stitcher Instance]
    ↓
[Call dynamicStitchSeams]
    ↓
[Extract Output Data]
    ↓
[Convert to NumPy Array]
    ↓
Python: 7680×3840×3 uint8 RGB
```

## Error Handling

### Common Issues

1. **"Failed to load DualStitcher.framework"**
   - DJI Studio is not installed
   - Solution: Install DJI Studio or update framework path

2. **"Metal device creation failed"**
   - Macs without Metal support (pre-2012 Intel Macs, some virtual machines)
   - Solution: Use a machine with Metal support (2012+ Mac with GPU)

3. **"Stitcher class not found"**
   - Framework loaded but Stitcher class doesn't exist
   - Solution: Check DJI Studio version compatibility

4. **"dynamicStitchSeams returned nil"**
   - Stitcher method not implemented or error occurred
   - Solution: Check Metal texture formats and dimensions

### Logging

The bridge logs to stderr with levels:
- `[INFO]` - Important operations
- `[DEBUG]` - Detailed diagnostics
- `[ERROR]` - Failures requiring attention
- `[WARNING]` - Non-fatal issues

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Testing

### Run Tests

```bash
cd dji_osv_converter
pytest test_metal_bridge.py -v -s
```

### Test Categories

1. **Compilation Tests**
   - Verify bridge compiles without errors
   - Check dylib is created and loadable

2. **Metal Tests**
   - Texture creation with correct formats
   - Device initialization

3. **Stitching Tests**
   - Dummy data stitching
   - Real OSV data stitching (if available)
   - Output shape and dtype validation

4. **Integration Tests**
   - Full pipeline with optical flow
   - Equirectangular validation

## Performance

### Memory Usage
- Seam masks: 2 × 8.96 MB = 17.9 MB
- Flow texture: 2.49 MB
- Frames: 2 × 44.2 MB = 88.4 MB
- Total input: ~108 MB

### Computation
- Texture creation: GPU memory transfer (~5-10ms)
- Metal texture allocation: ~1-2ms
- dynamicStitchSeams: ~50-100ms (GPU acceleration)
- Total: ~60-120ms per frame pair

## Future Improvements

1. **Direct Binary Protocol**
   - Replace pickle with custom binary protocol
   - Reduce marshalling overhead

2. **Metal Buffer Reuse**
   - Pool Metal textures for multiple stitches
   - Reduce allocation overhead

3. **Async Stitching**
   - Use Metal command queues for background stitching
   - Queue multiple stitch operations

4. **Performance Monitoring**
   - GPU time profiling
   - Memory usage tracking

## References

- [Metal Framework Documentation](https://developer.apple.com/documentation/metal)
- [Objective-C Runtime Programming Guide](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/ObjCRuntimeGuide/)
- [NSBundle Loading](https://developer.apple.com/documentation/foundation/nsbundle)
- [MTLTexture Creation](https://developer.apple.com/documentation/metal/mtltexture)

## Troubleshooting

### Build Fails with "Command not found: clang"

```bash
# Install Xcode Command Line Tools
xcode-select --install
```

### ImportError: dlopen(.../libmetalstitcher.dylib, ...)

```bash
# Check dylib was built successfully
ls -la dji_osv_converter/.metal_bridge_build/libmetalstitcher.dylib

# Try force rebuild
from dji_osv_converter.stitch_metal_bridge import MetalStitcherBridge
bridge = MetalStitcherBridge(rebuild=True)
```

### Metal Device Creation Fails

```python
# Check Metal support
import os
os.system("system_profiler SPDisplaysDataType | grep -i metal")
```

## Contact & Support

For issues with the Metal bridge implementation, check:
1. DJI Studio is installed and up to date
2. Mac has Metal-capable GPU (2012+)
3. Xcode Command Line Tools are installed
4. Framework path is correct
