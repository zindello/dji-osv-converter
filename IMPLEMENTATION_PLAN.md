# Implementation Plan: OSV → Equirectangular 360°

## Format Confirmed ✅

The DJI Reframe plugin for DaVinci Resolve expects:
- **Equirectangular 360° video** (7680×3840)
- **H.265/HEVC Main 10 codec**
- **Spherical metadata atom** marking it as equirectangular
- **Spatial orientation data** (yaw/pitch/roll)

## Conversion Pipeline

```
input.OSV
├─ Extract 8 HEVC streams (3840×3840 each)
├─ Extract protobuf metadata (camera calibration)
├─ Extract audio track (WAV/AAC)
└─ CONVERT:
   ├─ Undistort 8 images using calibration (each frame)
   ├─ Stitch 8 undistorted views → equirectangular sphere
   ├─ Encode output as H.265/HEVC
   ├─ Add spherical metadata
   └─ Mux with audio
output.mp4 (7680×3840 equirectangular)
```

## Phase 1: Metadata Extraction (6-8 hours)

### Task 1.1: Full Protobuf Parser
**Goal**: Extract camera calibration for all 8 lenses, all 4724 frames

**Deliverable**: Python module with:
```python
osv = OSVFile("input.OSV")
frames = osv.get_metadata_frames()  # 4724 frames
frame_0 = frames[0]
calibration = frame_0.camera_calibration  # All 8 cameras
# calibration[i] = {
#   'focal_length_x': float,
#   'focal_length_y': float,
#   'principal_point_x': float,
#   'principal_point_y': float,
#   'distortion': [k1, k2, p1, p2],
#   'image_size': (3840, 3840),
#   'position_index': i,  # 0-7
# }
```

**Tools**: 
- protobuf library (Python)
- FFmpeg (extract stream)
- Binary analysis of protobuf wire format

**Subtasks**:
- [ ] Create/find `dvtm_oq101.proto` schema
- [ ] Write protobuf parser
- [ ] Extract all frames
- [ ] Validate calibration data
- [ ] Save to JSON for testing

### Task 1.2: Video Stream Extraction
**Goal**: Extract individual camera streams for processing

**Deliverable**: 
```bash
# Extract each of 8 streams
for i in {0,1}; do
  ffmpeg -i input.OSV -map 0:$i -c copy stream_$i.mov
done
```

## Phase 2: Stitching Algorithm (20-30 hours)

This is the core work. Two approaches:

### Approach A: Use OpenCV (Recommended)
**Pros**: Fast, proven, well-documented  
**Cons**: Need to understand camera calibration → OpenCV format

**Steps**:
1. Parse camera calibration into OpenCV camera matrix format
2. Load each frame's 8 camera images
3. **Undistort** using `cv2.undistort()` with calibration
4. **Compute homographies** (perspective transforms) to align views
5. **Blend** overlapping regions (feathering, gradient masks)
6. **Project** stitched result to equirectangular
7. Encode to H.265

**Libraries**: `opencv-python`, `numpy`, `scipy`

### Approach B: Use Raw Processing (More Control)
**Pros**: Full control, can optimize  
**Cons**: More code, slower, harder to debug

**Implementation**:
```python
def process_frame(camera_images, calibrations, frame_index):
    # 1. Undistort each image
    undistorted = []
    for i, img in enumerate(camera_images):
        K = calibrations[i]['camera_matrix']
        dist = calibrations[i]['distortion']
        undist = cv2.undistort(img, K, dist)
        undistorted.append(undist)
    
    # 2. Compute stitching transforms
    # This is the hard part - need to figure out camera positions
    # Likely from camera indices 0-7 arranged in specific layout
    
    # 3. Warp to common view
    # Probably a cube or sphere
    
    # 4. Project to equirectangular
    # Map 3D sphere → 2D equirectangular
    
    # 5. Return 7680×3840 image
    return equirectangular_image
```

**Critical unknown**: **What's the spatial arrangement of the 8 cameras?**
- Are they arranged as a cube? (front, back, left, right, top, bottom + 2 diagonals?)
- Are they arranged as an octahedron? (8 faces)
- What's the camera look direction for each?

**Research needed**: Look at protobuf data for position/orientation hints

## Phase 3: Video Encoding & Metadata (6-8 hours)

### Task 3.1: H.265 Encoding
```python
# Use ffmpeg with specific codec settings
# Match original quality: H.265 Main 10, similar bitrate
subprocess.run([
    'ffmpeg',
    '-r', '30',  # Frame rate
    '-i', 'frame_%05d.png',  # Input frames
    '-c:v', 'libx265',
    '-preset', 'medium',  # Balance speed/quality
    '-crf', '23',  # Quality (lower = better)
    '-pix_fmt', 'yuv420p10le',  # 10-bit
    '-tag:v', 'hvc1',  # H.265 tag
    'video.mp4'
])
```

### Task 3.2: Spherical Metadata Injection
```bash
# Use ffmpeg with spatial media parameters
ffmpeg -i video.mp4 \
  -c:v copy \
  -c:a copy \
  -metadata:s:v:0 "stereo=mono" \
  -metadata:s:v:0 "rotation=0" \
  output_with_metadata.mp4

# Or use custom atoms if needed
# ffmpeg-python or direct MP4 atom manipulation
```

## Phase 4: CLI Tool & Testing (4-6 hours)

### Task 4.1: Python CLI
```bash
# Usage
dji-resolve input.OSV output.mp4 [options]

# Options
--fps 30                    # Output frame rate
--quality high|medium|low   # Encoding quality
--frame-range 0:100         # Process subset of frames
--temp-dir /tmp/dji         # Working directory
--no-cleanup                # Keep intermediate files
--verbose                   # Debug output
```

### Task 4.2: Validation
- [ ] Test with sample OSV
- [ ] Verify output plays in DaVinci Resolve
- [ ] Check spherical metadata is preserved
- [ ] Performance benchmark
- [ ] Error handling for corrupted frames

## Implementation Order (Recommended)

### Week 1: Foundation
1. **Protobuf parser** (Task 1.1) - 6-8h
2. **Video extraction** (Task 1.2) - 2-3h
3. **Test with sample data** - 2-3h
4. **Commit & document** - 1h

### Week 2: Stitching
1. **Determine camera layout** (research) - 4-6h
2. **Implement undistortion** (Task 2) - 4-6h
3. **Implement stitching** (Task 2) - 12-18h
4. **Test single frame** - 2-3h

### Week 3: Polish
1. **Add encoding** (Task 3.1) - 3-4h
2. **Add metadata** (Task 3.2) - 2-3h
3. **CLI wrapper** (Task 4.1) - 3-4h
4. **Full testing** (Task 4.2) - 4-6h
5. **Documentation** - 2-3h

**Total: ~60-75 hours**

## Open Questions Requiring Research

1. **Camera arrangement**: How are the 8 Osmo 360 cameras positioned relative to each other?
   - Need to find in protobuf data or DJI documentation
   - Or empirically test with known calibration points

2. **Stitching approach**: Does DJI use optical flow, feature matching, or other?
   - We found "OPTICAL_FLOW_STITCHING" in binary
   - Might need feature-based matching for robustness

3. **Blending strategy**: How are overlapping regions blended?
   - Feathering (smooth transition)?
   - Gradient-based?
   - Simple crossfade?

4. **Equirectangular mapping**: Standard formula or custom?
   - Likely standard spherical projection
   - Standard code exists in most computer vision libraries

## Success Criteria

### MVP (Minimum Viable Product)
- ✅ Parse OSV protobuf metadata
- ✅ Extract camera calibration
- ✅ Undistort 8 images  
- ✅ Stitch to equirectangular
- ✅ Encode to H.265
- ✅ Add spherical metadata
- ✅ Import into DaVinci Resolve
- ✅ CLI tool works end-to-end

### Production Ready
- Performance: Process video in reasonable time (1-2 hours for 2.7GB OSV)
- Quality: Seamless stitching, no visible artifacts
- Reliability: Handle all Osmo 360 recordings
- Documentation: Complete usage guide

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Camera layout unknown | Schedule slip 8-10h | Check protobuf, ask DJI community, empirical testing |
| Stitching artifacts | Quality issue | Test extensively, implement fallback blending methods |
| Performance too slow | Usability | Optimize with OpenCV GPU acceleration, multi-threading |
| Metadata injection fails | Critical | Test with ffmpeg, write custom MP4 atom handler |

## Dependencies

```
Required Libraries:
- ffmpeg (system)
- Python 3.8+
- opencv-python (cv2)
- numpy
- protobuf
- pydantic (validation)

Optional:
- tqdm (progress bars)
- click (better CLI)
- pytest (testing)
```

## Next Steps

1. Start Phase 1.1 - Protobuf parser
2. Test with osv_example data
3. Research camera layout in protobuf
4. Begin stitching implementation

Ready to start coding? 🚀
