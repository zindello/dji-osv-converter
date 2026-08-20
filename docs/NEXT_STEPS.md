# Next Steps: Bridge to Implementation

## Current Status
✅ **Research Complete**: We understand the OSV format (MP4 + protobuf metadata)
✅ **Tools Created**: Protobuf analyzer to inspect metadata structure
✅ **Binary Analysis**: Found references to stitching, stabilization, and DaVinci export in DJI Studio
⚠️ **Missing**: Exact DaVinci Resolve import format

## Critical Path to First Working Version

### Phase A: Understand the Export Format (2-4 hours)
**Goal**: Figure out what format DJI Studio actually exports for DaVinci Resolve.

**Option A1: Trace DJI Studio's Export** (Recommended first)
1. Open a test OSV file in DJI Studio
2. Attempt to export for "DaVinci Resolve" (if that option exists)
3. Monitor with `lsof` or `fs_usage` to see what files are created:
   ```bash
   fs_usage -w -f filesystem | grep -i "resolve\|xml\|json\|export"
   ```
4. Examine the output files (likely `.xml`, `.json`, or Resolve `.drp` project format)

**Option A2: Check DaVinci Resolve Documentation**
1. Search Blackmagic Design docs for "360" or "calibration" support
2. Check if Resolve has a plugin API or effects format
3. Look for "Fusion" (Resolve's compositing engine) plugin examples

**Option A3: Reverse Engineering via Hex Dump**
1. If DJI Studio's export is a binary format, extract and analyze it
2. Look for patterns: headers, magic numbers, embedded metadata
3. Try to match against known video project formats

### Phase B: Build the Metadata Parser (4-6 hours)
**Goal**: Extract calibration data from protobuf in a usable format.

**Tasks**:
1. Find or write a protobuf message definition for `dvtm_oq101.proto`
   - Search DJI SDK or GitHub for the `.proto` file
   - If not found: use our analysis to hand-write the schema
   
2. Create a Python script to:
   ```python
   # Pseudocode
   osv_file = open("video.OSV")
   metadata_stream = extract_stream(osv_file, stream_index=3)  # djmd stream
   
   for frame_num in range(num_frames):
       calibration = parse_protobuf_frame(metadata_stream)
       # Extract per-camera calibration matrices
       for camera_id in 0..7:
           fx, fy = focal_lengths[camera_id]
           cx, cy = principal_points[camera_id]
           distortion = distortion_coefficients[camera_id]
           export_to_json(camera_id, frame_num, calibration_data)
   ```

### Phase C: Build the Export Module (4-8 hours)
**Goal**: Convert our extracted calibration to Resolve-compatible format.

**Paths**:
1. **If Resolve accepts XML** (most likely):
   - Generate `.xml` with frame-by-frame camera parameters
   - Include focal length, principal point, distortion
   - Test import into Resolve

2. **If Resolve needs JSON**:
   - Similar to above but JSON format
   - Follow Resolve's expected schema

3. **If Resolve needs Fusion plugin format**:
   - May need to generate `.lua` or Python script
   - More complex but doable

### Phase D: CLI Tool (2-3 hours)
```bash
dji-resolve input.OSV [output-format]
# Outputs: XML, JSON, or Resolve-importable format
```

Options:
- `--format xml|json|resolve`
- `--fps 30` (default from file)
- `--frame-range 0:100` (optional)
- `--cameras 0,1,2` (specific cameras only)

## Blockers & Unknowns

| Question | Impact | Investigation |
|----------|--------|-----------------|
| What's the DaVinci export format? | CRITICAL | Try Option A1 first |
| Can Resolve even import 360 calibration? | HIGH | Check docs + try export |
| Do we need to stitch/undistort the video? | MEDIUM | Depends on Resolve's capability |
| Where is `dvtm_oq101.proto` schema? | MEDIUM | GitHub, DJI SDK, or reverse-engineer |
| What's in the other data streams (dbgi)? | LOW | Likely debug info, can ignore |

## Quick Wins (While Researching)

### 1. Extract the .proto Schema
```bash
# Search GitHub for DJI SDKs or open-source projects using this format
gh search repos "dvtm_oq" --language proto

# Check if DJI released any SDKs
curl https://api.github.com/search/repositories?q=user:DJI+dvtm
```

### 2. Parse Full Metadata
Enhance `analyze_protobuf.py` to:
- Extract all 4724 frames worth of data
- Parse into a clean JSON structure
- Group by camera (8 cameras × 4724 frames)

### 3. Test Resolve Import
- Create a minimal XML with one frame's calibration data
- Try importing into DaVinci Resolve
- See what errors/results we get

## Success Criteria

### MVP (Minimum Viable Product)
- [ ] Extract protobuf metadata from OSV
- [ ] Parse calibration matrices (focal length, principal point, distortion)
- [ ] Export to JSON format
- [ ] CLI tool: `dji-resolve input.OSV output.json`

### v1.0 (Production Ready)
- [ ] Support all 3 file types (OSV, LRF, WAV)
- [ ] Generate DaVinci Resolve-compatible format
- [ ] Test with actual DaVinci Resolve import
- [ ] Batch processing support
- [ ] Comprehensive error handling

### Future (Nice-to-Have)
- [ ] GUI drag-and-drop interface
- [ ] Real-time preview of stitching/stabilization
- [ ] Direct Resolve plugin integration
- [ ] Optical flow computation for enhanced stabilization

## Recommended Starting Point

**Priority #1**: Trace DJI Studio's export → understand the actual format needed

**Estimated time**: 30 minutes to 2 hours depending on how obvious the output is

**Tools needed**:
- Terminal (fs_usage, lsof, or Activity Monitor)
- Hex editor (to inspect binary output if needed)
- DJI Studio app (you already have)

**Next meeting should start with**: "Here's the export format DJI Studio generates..."
