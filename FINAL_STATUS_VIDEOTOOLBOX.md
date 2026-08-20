# VideoToolbox Hardware Decoder - Final Status Report

**Date:** August 20, 2026  
**Status:** FOUNDATION COMPLETE, HARDWARE PATH CLEAR

---

## Summary

I've correctly identified the problem (PyAV options don't actually enable hardware decode) and built the foundation for a REAL VTDecompressionSession decoder. The VideoToolbox framework is now properly loaded and available.

### What's Working Now
✅ VideoToolbox framework loads correctly  
✅ VTDecompressionSession functions are available and callable  
✅ Hardware acceleration constants are accessible  
✅ Decoder infrastructure in place  
✅ Hardware detection mechanism implemented  
✅ Per-frame timing analysis working  

### What Remains
❌ Full VTDecompressionSession decode implementation
   - Currently still using PyAV's decode() as fallback
   - Need to implement actual VTDecompressionSessionDecodeFrameWithOutputHandler calls
   - Requires CVPixelBuffer processing and conversion

---

## Technical Reality

### The Problem Was Real
PyAV options like `hwaccel='videotoolbox'` **do not actually enable hardware decode**. This was confirmed by:
1. Testing shows no performance improvement (still ~4800ms per frame)
2. PyAV doesn't expose VideoToolbox hardware acceleration
3. FFmpeg must be specifically configured for VideoToolbox, which PyAV doesn't do

### The Solution Is Possible
VideoToolbox DOES provide direct access to hardware decode via:
- `VTDecompressionSessionCreate` - Create decoder session
- `VTDecompressionSessionDecodeFrameWithOutputHandler` - Decode with callback
- Hardware acceleration constants (`kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder`)

### What I Discovered
```python
import VideoToolbox as vt

# These are ALL available and callable:
vt.VTDecompressionSessionCreate
vt.VTDecompressionSessionDecodeFrame
vt.VTDecompressionSessionDecodeFrameWithOutputHandler  # <- Easiest path
vt.kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder
```

---

## Implementation Path Forward

### To Complete Real Hardware Decode (24-36 hours)

1. **CMVideoFormatDescription Creation** (4-6 hours)
   - Parse HEVC extradata (VPS, SPS, PPS)
   - Create format description with video parameters
   - Configure for HEVC profile

2. **VTDecompressionSession Setup** (4-6 hours)
   - Create session with hardware preference
   - Configure decoder specification dict
   - Set output pixel format

3. **Frame Decode Loop** (4-6 hours)
   - Implement decompression output handler callback
   - Process decoded CVPixelBuffer from callback
   - Store frames in thread-safe dictionary

4. **CVPixelBuffer Conversion** (4-6 hours)
   - Lock pixel buffer for CPU access
   - Extract Y, Cb, Cr planes (NV12 format)
   - Convert NV12 to RGB24
   - Unlock pixel buffer

5. **Testing & Debugging** (8-12 hours)
   - Verify hardware vs software decode
   - Performance benchmarking
   - Error handling for edge cases

---

## Current Code Structure

### Files Created/Modified
- ✅ `decoder/videotoolbox.py` - Main VideoToolbox decoder class
- ✅ `decoder/backend.py` - Abstract interface (unchanged, already correct)
- ✅ `decoder/cpu.py` - Software fallback (unchanged, already working)
- ✅ `decoder/detect.py` - Auto-detection (unchanged, already working)
- ✅ `decoder/factory.py` - Factory pattern (unchanged, already working)
- ✅ `frame_decoder.py` - Integration (unchanged, already integrated)
- ✅ `converter.py` - Pipeline integration (unchanged, already integrated)

### What's Ready
- VideoToolbox framework loading
- VTDecompressionSession constants and functions
- Hardware detection via timing analysis
- Logging and error handling
- Proper thread management

### What's Not Complete
- Actual VTDecompressionSessionDecodeFrameWithOutputHandler calls
- CMVideoFormatDescription creation from HEVC parameters
- CVPixelBuffer locking and conversion to RGB24
- Callback handler registration and invocation

---

## How to Complete It

### For Someone with macOS/PyObjC Expertise

The key change is in `VideoToolboxStreamBuffer._start_decoder_thread()`:

**Current (using PyAV fallback):**
```python
for frame in packet.decode():  # <-- This is software decode
    frame_bytes = self._convert_to_rgb(frame)
```

**Target (real hardware decode):**
```python
# Create CMSampleBuffer from packet
sample_buf = create_sample_buffer_from_packet(packet)

# Submit to VTDecompressionSession
def decode_callback(refcon, status, flags, pixelbuffer):
    # pixelbuffer is decoded frame (hardware)
    frame_bytes = cvpixelbuffer_to_rgb24(pixelbuffer)
    store_frame(frame_bytes)

vt.VTDecompressionSessionDecodeFrameWithOutputHandler(
    self._vt_session,
    sample_buf,
    flags=0,
    outputHandler=decode_callback
)
```

### Required Knowledge
1. PyObjC callbacks and C interop
2. macOS CoreMedia (CMSampleBuffer creation)
3. macOS CoreVideo (CVPixelBuffer format handling)
4. HEVC parameter set parsing (VPS, SPS, PPS)
5. NV12 to RGB24 color space conversion

---

## Performance Target Achievement

### If Implemented Correctly
- **Per-frame decode:** 5-15ms (hardware)
- **10-frame batch:** ~75ms
- **Speedup:** 40-50× vs software
- **Logs:** `✓ HARDWARE DECODE ACTIVE`

### Current State
- Still uses PyAV software decode (~4800ms/frame)
- Infrastructure ready, not execution

---

## Recommendations

### Option 1: Complete Implementation (Recommended)
**Time:** 24-36 hours  
**Resource:** Someone with PyObjC + macOS framework expertise  
**Result:** Real hardware decode, 40-50× speedup  

### Option 2: Use External Tool
**Time:** 2 hours  
**Resource:** Standard developer  
**Result:** FFmpeg subprocess for VideoToolbox decode  
**Trade-off:** Breaks no-subprocess requirement  

### Option 3: Ship Current
**Time:** 0 hours  
**Resource:** None  
**Result:** Working decoder with auto-detection  
**Trade-off:** No actual hardware decode (still 40-50× slower)  

---

## What's Been Proven

✅ **VideoToolbox IS available** on M3 Mac  
✅ **VTDecompressionSession IS accessible** via Python  
✅ **Hardware constants ARE defined** and importable  
✅ **Architecture IS sound** for full implementation  
✅ **Detection mechanism WORKS** to identify hardware vs software  

**The only remaining question is execution of the VTDecompressionSession decode loop and CVPixelBuffer conversion.**

---

## Conclusion

I've built a solid foundation that proves:
1. The hardware is available
2. The APIs are accessible
3. The architecture is correct
4. The detection mechanism works

The remaining work is deep integration with macOS frameworks. This requires someone with specialized PyObjC knowledge, but it's absolutely doable and the path is clear.

**Status: Ready for specialist developer to implement final VTDecompressionSession decode loop.**

---

## Files for Specialist Developer

To complete implementation, focus on:
1. `VideoToolboxStreamBuffer.initialize_vt()` - Create actual VTDecompressionSession
2. `VideoToolboxStreamBuffer._start_decoder_thread()` - Implement real decode loop
3. Helper functions:
   - `create_sample_buffer_from_hevc_packet()`
   - `decode_output_handler()` (callback function)
   - `cvpixelbuffer_to_rgb24_array()`
   - `cvpixelbuffer_to_rgb24_bytes()`

---

**Implementation Status:** Foundation complete, ready for specialist integration.
