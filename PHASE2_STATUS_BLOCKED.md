# Phase 2 Status: Framework Access Blocked - Pivot Strategy

## Problem Summary

DJI's DualStitcher.framework is a **private framework** within the DJI Studio.app bundle:

1. **Framework binary loads successfully** when dependencies are preloaded:
   - libMNN.3.1.2.dylib ✓
   - openmcv2.framework ✓
   - ml_fis.framework ✓
   - DualStitcher.framework ✓

2. **But Objective-C class symbols are private** (marked lowercase 't' in `nm` output):
   ```
   000000000004be2c t -[Stitcher FlowStitchSeams:seam1:...]
   00000000000489d0 t -[Stitcher initWithStitcherInfo:]
   ```
   This means:
   - NSBundle loads the framework binary ✓
   - But `objc.lookUpClass("Stitcher")` fails
   - PyObjC cannot access private class symbols

3. **No public CLI tool** for stitching in DJI Studio.app

## Root Cause

DualStitcher is an internal-only framework designed for use by DJI Studio's GUI. It does not expose a public API, either via:
- Objective-C class exports (private)
- C function bindings (mangled/private)
- Command-line tools

## Available Paths Forward

### Option A: Reverse-Engineer Private Framework (High Risk, Uncertain ROI)

**Approach:**
- Use LLDB to instrument DJI Studio and trace Stitcher calls
- Dynamically instantiate private Objective-C class via runtime manipulation
- Hook FFmpeg/Metal calls to capture stitching parameters

**Pros:**
- Potentially reuse DJI's optimized optical flow
- No need to implement stitching algorithm

**Cons:**
- Extremely fragile (breaks on app updates)
- May violate terms of service
- Requires deep Objective-C runtime knowledge
- High maintenance burden
- No guarantees of success

### Option B: GPU-Accelerated OpenCV (Recommended - Phase 2 Backlog)

**Approach:**
- Implement optical flow stitching using OpenCV + GPU acceleration
- Support Metal (Apple Silicon M1/M2/M3), NVIDIA CUDA, Intel GPU
- Create frame extraction → optical flow → warping → blending pipeline

**Pros:**
- Cross-platform (macOS, Linux, Windows)
- Open-source, maintainable
- Full control over algorithm tuning
- Clear timeline and known complexity
- Commercial libraries available if needed

**Cons:**
- Requires implementing/tuning stitching algorithm
- 40-60 hours estimated for full implementation
- GPU optimization per-platform

## Recommendation: Hybrid Approach

**Immediate (Next 4-8 hours):**
1. ✅ Complete Phase 1: Dependencies (DONE)
2. Create Phase 3: Frame Extraction & Metadata Parsing
   - Implement `frame_decoder.py` (PyAV multi-stream HEVC decoding)
   - Implement `protobuf_schema.py` (metadata parser)
   - Validate with sample OSV file
   - This is 100% feasible and needed regardless of stitching approach

3. Document DualStitcher API surface (for future reference)
   - Export method signatures via class-dump
   - Create reverse-engineered `.h` file

**Medium term (backlog):**
- Evaluate commercial stitching libraries (e.g., DJI SDK if available)
- Prototype OpenCV optical flow stitching
- Benchmark against reference output

**Long term (Phase 5+):**
- If DJI ever releases public SDK → pivot to official API
- Fall back to GPU-accelerated OpenCV as production stitcher

## Discovered Stitcher Methods

For future reference, the Stitcher class has these instance methods:

**Optical Flow:**
- `FlowStitchCalcFlow:seam1:enableSG:sgGammaTex:bidir_flow:fence:deviceType:`
- `FlowStitchSeams:seam1:isDynamic:isCompensate:useLSC:deviceType:enableSunGlare:sgGammaTex:`
- `FlowStitchWarpFusion:seam1:bidir_flow:blended_out:fence:isDynamic:isCompensate:useLSC:deviceType:`

**AI-Based:**
- `aiStitchCalcFlow:seam1:enableSG:sgGammaTex:flowLr:fence:`
- `aiStitchSeams:seam1:isCompensate:enableSunGlare:sgGammaTex:`
- `aiStitchWarpFusion:seam1:flowLr:blended_out:fence:isCompensate:useLSC:`

**Dynamic (motion-aware):**
- `dynamicStitchCalcFlow:seam1:enableSG:sgGammaTex:bidir_flow:fence:`
- `dynamicStitchSeams:seam1:[variations]`
- `dynamicStitchWarpFusion:seam1:bidir_flow:[variations]`

**Utility:**
- `initWithStitcherInfo:`
- `initWithRawStitcherInfo:`
- `initWithStitcherInfoSep:`
- `createTextureFromFile:device:width:height:`

## Next Actions

1. **Decision Point:** Choose pivot strategy (reverse-engineer vs. GPU OpenCV)
2. **Proceed with Phase 3:** Frame extraction and metadata parsing (independent of stitcher choice)
3. **Implement minimal stitcher stub:** Placeholder that allows full pipeline testing
4. **Test end-to-end pipeline:** With stub stitcher, validate frame→metadata→encode flow

## Timeline Impact

- If reverting to OpenCV: Adds 40-60 hours to Phase 2, but solves cross-platform problem
- If waiting for DJI SDK: Risk of indefinite delay
- Recommended: Start Phase 3 immediately while deciding on stitcher approach
