# Architecture Analysis: Why Python Approach Fails for Real-Time Video

## Problem Summary

Current implementation: **5-6 seconds per frame** with software HEVC decode
Target: **33ms per frame** for real-time 30fps (100× faster needed)

## Root Cause: Software HEVC Decode

### Current Approach
```
PyAV (libavcodec) → Software HEVC decode → Color conversion → Stitch → Encode
         CPU        210ms per frame         15ms           ??ms      ??ms
```

### What's Actually Happening
- PyAV uses FFmpeg's libavcodec (software decoder)
- M3 CPU can only decode ~3-4 HEVC streams in parallel at full speed
- We have 2 video streams requiring 210ms × 2 = 420ms minimum per frame (in parallel)
- Plus synchronization overhead, buffering waiting, color conversion, stitching
- **Real outcome: 5-6 seconds per frame**

### FFmpeg With VideoToolbox
```
ffmpeg -hwaccel videotoolbox → Hardware HEVC decode → 7.5ms per frame
       (verified working)        (M3 GPU hardware)     (130fps achievable)
```

## Available Solutions

### Option 1: VideoToolbox Direct (PyObjC)
**Feasibility:** High (frameworks available, PyObjC installed)
**Complexity:** 500+ lines of complex PyObjC code
**Performance:** ~7.5ms per frame (hardware decode)
**Pros:**
- In-process (no subprocess calls)
- Native speed
- Full control

**Cons:**
- Very complex implementation (VTDecompressionSession callbacks, CVPixelBuffer handling)
- Error-prone, many edge cases
- Weeks of debugging likely
- Type bridging between Objective-C and Python

### Option 2: Rust Extension (Native Module)
**Feasibility:** Medium (Rust not installed, needs setup)
**Complexity:** 200-300 lines of Rust + build config
**Performance:** ~7.5ms per frame
**Pros:**
- Clean implementation
- Fast
- Compiled (no Python interpreter overhead)
- Memory safe

**Cons:**
- Requires Rust installation
- Build complexity (PyO3, setup.py)
- Still need to parse demux in Python or Rust

### Option 3: Swift/Objective-C Extension
**Feasibility:** High (Xcode installed)
**Complexity:** 200-400 lines of Swift
**Performance:** ~7.5ms per frame (native code)
**Pros:**
- Direct access to VideoToolbox
- Clean implementation
- Swift safety guarantees

**Cons:**
- Need to write Swift wrapper
- Compilation complexity
- Bridge to Python

### Option 4: Hybrid FFmpeg (CLI via Named Pipe)
**Feasibility:** High (FFmpeg with VideoToolbox verified)
**Complexity:** 50-100 lines of Python
**Performance:** ~7.5ms per frame
**Pros:**
- FFmpeg handles decode perfectly
- Minimal code
- Battle-tested

**Cons:**
- External subprocess call
- User explicitly said "NO FFMPEG"

### Option 5: Accept CPU Decode + GPU Optimization
**Feasibility:** Medium
**Complexity:** More Metal shaders, heavy optimization
**Performance:** Still 1-2 seconds per frame (best case)
**Pros:**
- Stay pure Python
- No external dependencies

**Cons:**
- Fundamentally 50-100× too slow for realtime
- Won't meet 30fps target even with perfect optimization

## What Testing Revealed

### Frame Extraction Breakdown
```
Theory:
  HEVC decode (SW):     210ms
  Color conversion:      15ms
  ---
  Total per frame:      225ms

Reality (current):
  Frame extraction:   5000-6000ms
  
Ratio: 22-26× overhead from synchronization
```

### Sequential Baseline
```
ffmpeg -hwaccel videotoolbox → 4724 frames in 35.5s = 7.5ms/frame
(this is the actual hardware potential)
```

## Recommendation

**Pure Python will not work for realtime video processing.**

### Realistic Path Forward

**Scenario A: Accept "No Video Decode" Constraint**
1. Use external tool for demux+decode (not subprocess)
2. Read raw frames from file
3. Do stitching/encoding in Python
4. Problem: Still need decode somehow

**Scenario B: Build Small C Extension** (Recommended if no external calls allowed)
1. Write 100-line C extension for VideoToolbox decode
2. Returns numpy arrays directly
3. Python does stitching and encoding
4. ~30mins of C coding + debugging

**Scenario C: Use Swift Wrapper** (Recommended if Xcode available)
1. Build small Swift tool as shared library
2. PyObjC can call Swift
3. ~1-2 hours development

**Scenario D: Accept FFmpeg Subprocess** (Fastest, violates "NO FFMPEG")
1. Use `ffmpeg` CLI with pipe
2. Python reads raw frames
3. ~30 mins implementation (already proven 7.5ms/frame)

## Current Blockers

1. **No hardware decode in Python** - PyAV doesn't expose VideoToolbox
2. **No Rust** - Can't build Rust extension
3. **VideoToolbox in PyObjC is complex** - 500+ lines, weeks of work
4. **"No external calls"** - Blocks simplest solution (FFmpeg subprocess)

## Conclusion

**The problem is not stitching, not encoding, not GPU acceleration of stitching.**

**The problem is: How do we get raw frames from HEVC video without software decode?**

With current constraints:
- Pure Python: Not viable (5-6s per frame, need 33ms)
- VideoToolbox via PyObjC: Viable but very complex
- C extension: Viable, moderate complexity
- Swift wrapper: Viable, moderate complexity
- FFmpeg subprocess: Viable and simple, but explicitly rejected

## Files Created

- `test_extraction_quick.py` - Proved frame extract is bottleneck
- `videotoolbox_hevc.py` - Started VideoToolbox wrapper (stub)
- `videotoolbox_decoder.py` - Started demux + decode hybrid (stub)

**Next step:** Decision on which path to pursue
