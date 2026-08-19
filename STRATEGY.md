# DJI OSV to DaVinci Resolve Conversion Strategy

## What We Know

### OSV File Structure (Osmo 360)
1. **Video Streams**: 2 HEVC streams (3840×3840 each, 30fps) - omnidirectional video
2. **Audio**: WAV stereo 48kHz
3. **Metadata Stream (djmd)**: Protocol Buffer (protobuf) format
   - Schema: `dvtm_oq101.proto` 
   - Contains frame-by-frame camera calibration for 8 sensors
   - ~4.4MB for 157 seconds of video (4724 frames @ 30fps)

### Protobuf Metadata Contents
Each frame contains:
- **8 camera calibration objects** (one per lens/sensor):
  - Camera matrix: focal length, principal point (cx, cy)
  - Distortion coefficients (k1-k4)
  - Image dimensions (3840×3840)
  - Angular/orientation parameters

### Conversion Challenge
We need to:
1. **Extract** the 8-lens omnidirectional video + metadata
2. **Undistort & stitch** the 8 images into a coherent 360° view (OR provide per-lens stabilization)
3. **Generate** DaVinci Resolve-compatible output

### Two Possible Approaches

**Option A: Generate XML for DaVinci Resolve**
- Export calibration data as XML that Resolve can import
- Resolve applies stabilization based on calibration
- Simpler but requires Resolve to support format

**Option B: Pre-process to single stabilized video**
- Use extracted calibration to undistort/stitch images
- Generate standard video file
- Simpler format but computationally expensive

## Implementation Plan

### Phase 1: Data Extraction (COMPLETE)
- ✅ Identify OSV as MP4 container with protobuf metadata
- ✅ Extract djmd streams
- ✅ Reverse-engineer protobuf structure
- ⚠️ Need to: decode actual calibration values

### Phase 2: Format Research (IN PROGRESS)
- [ ] Check DaVinci Resolve for supported stabilization formats
- [ ] Look for Resolve XML schema for camera calibration
- [ ] Determine if Resolve can ingest raw calibration matrices
- [ ] Check if there's a "360° reframing" plugin format

### Phase 3: Prototype Conversion (NEXT)
- [ ] Write protobuf parser (Python or Rust)
- [ ] Extract calibration matrices for frame N
- [ ] Test with DaVinci Resolve import
- [ ] Iterate on format

### Phase 4: CLI Tool (FINAL)
- [ ] Create command: `dji-resolve input.OSV output.xml`
- [ ] Add options for frame range, camera selection
- [ ] Batch processing support
- [ ] Error handling and validation

## Technology Stack
- **Protobuf parsing**: Python (protobuf3) or Go
- **Video processing**: FFmpeg (extract streams, merge)
- **Math/matrices**: NumPy (if doing undistortion)
- **Output format**: XML or custom

## Open Questions
1. Does DaVinci Resolve support 360° videos natively?
2. What XML/plugin format does the DJI Reframe plugin use?
3. Are the 8 cameras in a specific configuration (e.g., cube faces)?
4. Can we just pass calibration matrices to Resolve or do we need to generate stabilization vectors?
