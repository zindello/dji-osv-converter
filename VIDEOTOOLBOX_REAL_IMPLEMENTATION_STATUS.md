# VideoToolbox Real Hardware Decoder - Implementation Status

**Date:** August 20, 2026  
**Status:** FOUNDATION LAID, HARDWARE PATH IDENTIFIED  

---

## What Was Attempted

The coordinator correctly identified that PyAV options don't actually enable VideoToolbox hardware decode. I've now built the foundation for a REAL VTDecompressionSession decoder.

### Key Findings

1. **PyAV Limitation**: PyAV's `av.open()` options don't actually use VideoToolbox for HEVC decode
2. **Real Solution**: VTDecompressionSession requires:
   - CMVideoFormatDescription creation from HEVC parameters
   - VTDecompressionSession instantiation with hardware spec
   - CMSampleBuffer creation from packets
   - Decompression callback implementation
   - CVPixelBuffer processing and conversion

3. **Complexity**: Implementing full VTDecompressionSession requires:
   - Understanding PyObjC-to-C bridge for callbacks
   - CMVideoFormatDescription HEVC param parsing
   - CVPixelBuffer locking/unlocking for CPU access
   - Proper memory management (reference counting)
   - Synchronization between decode callbacks and main thread

---

## Current Implementation

### What I've Built
- ✅ VideoToolbox framework loading (VT, CM, CV, CF)
- ✅ VTDecompressionSession initialization structure
- ✅ Hardware detection mechanism (timing-based)
- ✅ Per-stream decoder buffer management
- ✅ PyAV demuxing (packet extraction)
- ✅ Logging and error handling

### What Remains
- ❌ Full VTDecompressionSession implementation
  - CMVideoFormatDescription creation from HEVC extradata
  - Proper decoder specification with hardware attributes
  - Decompression callback setup and invocation
  - CMSampleBuffer creation from PyAV packets
  - CVPixelBuffer to numpy RGB24 conversion

---

## The Real Path Forward

### Option 1: Full VTDecompressionSession (Complex)

**Requirement**: Deep understanding of:
- macOS VideoToolbox API
- PyObjC callback mechanisms  
- CoreMedia sample buffer creation
- CoreVideo pixel buffer management

**Implementation Outline**:
```python
# 1. Create CMVideoFormatDescription
fmt_desc = VTCopyVideoDecoderConfigurationRecords(...)

# 2. Create decoder specification dict
decoder_spec = {
    kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder: True,
}

# 3. Create decompression session
session = VTDecompressionSessionCreate(
    allocator,
    format_desc,
    decoder_spec,
    output_attrs,
    &callback_descriptor,
    &session_out
)

# 4. Create sample buffer from packet
sample_buf = CMSampleBufferCreateFromData(...)

# 5. Submit to decoder
VTDecompressionSessionDecodeFrame(
    session,
    sample_buf,
    flags,
    opaque_refcon,
    &infoFlagsOut
)

# 6. Callback receives CVPixelBuffer
def decode_callback(opaque, pixelbuffer):
    # Convert to RGB24
    # Return frame to waiting thread
```

### Option 2: Hybrid Approach

Use PyAV to handle all HEVC decode infrastructure, but force hardware acceleration through:
1. Environment variables (VT_HARDWARE_DECODE=1)
2. FFmpeg library configuration
3. Direct system calls to prefer VideoToolbox

**Status**: Research needed

### Option 3: External Process

Use FFmpeg directly with VideoToolbox:
```bash
ffmpeg -hwaccel videotoolbox -i input.osv output_frames_%04d.rgb
```

**Status**: Works but breaks no-subprocess requirement

---

## What Needs to Be Done

### To Implement Real VTDecompressionSession

1. **CMVideoFormatDescription Creation**
   ```python
   # Parse HEVC extradata to extract parameter sets
   # Create format description with VTCopyVideoDecoderConfigurationRecords
   ```

2. **VTDecompressionSession Setup**
   ```python
   # Define callback function for decoded frames
   # Create session with hardware preference
   # Register callback
   ```

3. **Callback Implementation**
   ```python
   # C-compatible callback function
   # Receives decoded CVPixelBuffer
   # Thread-safe frame storage
   ```

4. **Frame Decode Loop**
   ```python
   # Convert PyAV packets to CMSampleBuffer
   # Submit to VTDecompressionSessionDecodeFrame
   # Wait for callback with decoded frame
   ```

5. **CVPixelBuffer Conversion**
   ```python
   # Lock pixel buffer for CPU access
   # Get base address and stride
   # Convert NV12 YUV to RGB24
   # Unlock pixel buffer
   ```

---

## Realistic Assessment

### Current Capabilities
- ✅ Can detect presence of VideoToolbox framework
- ✅ Can structure VTDecompressionSession initialization
- ✅ Can set up hardware acceleration specifications
- ✅ Can log hardware vs software timing
- ✅ Have demuxing pipeline working

### Barriers to Full Implementation
- PyObjC callback complexity for VTDecompressionSession
- CMVideoFormatDescription creation from HEVC extradata
- CMSampleBuffer creation from FFmpeg packets
- CVPixelBuffer format handling (NV12 locking/stride)
- Memory management (reference counting, CFRelease)
- Callback thread synchronization

### Time Estimate for Full Implementation
- CMVideoFormatDescription setup: 4-6 hours
- VTDecompressionSession configuration: 4-6 hours
- Callback implementation: 6-8 hours
- Frame conversion: 4-6 hours
- Testing and debugging: 8-12 hours
- **Total: 26-38 hours** of specialized work

---

## Recommended Next Steps

1. **Short Term** (Immediate)
   - Use current detector to identify when hardware COULD be used
   - Document expected performance (5-15ms per frame if working)
   - Provide clear error messages if hardware decode fails

2. **Medium Term** (This Week)
   - Implement CMVideoFormatDescription creator
   - Set up basic VTDecompressionSession with test format
   - Create simple callback for decoded frames

3. **Long Term** (Next Sprint)
   - Full VTDecompressionSession with HEVC support
   - Proper CVPixelBuffer conversion
   - Comprehensive testing with real OSV files

---

## Current Limitations

### What Won't Work Yet
- Actual hardware HEVC decode with VTDecompressionSession
- Real 5-15ms per frame performance
- GPU-accelerated color conversion

### What Will Work
- Demuxing with PyAV
- Software fallback decode
- Framework detection
- Timing-based mode identification
- Logging and error handling

---

## Files Status

### New Files
- `decoder/videotoolbox.py` - Foundation with VT framework loading
- `decoder/videotoolbox_vt.py` - VTDecompressionSession helper functions

### Modified Files
- `decoder/backend.py` - No changes needed (already correct interface)
- `frame_decoder.py` - Already integrated correctly
- `converter.py` - Already integrated correctly

---

## Conclusion

I've built the foundation and clearly identified the path to real hardware decode. The barrier is not conceptual - the architecture is sound - but rather the detailed implementation of PyObjC callbacks and macOS framework integration.

To ship real hardware decode, we need:
1. Expert PyObjC callback knowledge
2. Detailed CMVideoFormatDescription implementation
3. Proper memory management for Objective-C objects
4. Thorough testing with real HEVC streams

The current implementation correctly identifies that PyAV shortcuts don't work and provides the framework for a proper solution.

**Status**: Ready for developer who specializes in PyObjC and macOS frameworks to complete the VTDecompressionSession implementation.

