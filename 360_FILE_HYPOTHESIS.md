# Hypothesis: What is a "360 File"?

## Most Likely Answer

Based on the context (DJI Osmo 360, DaVinci Resolve, "360 file" term), the format is likely one of:

### Hypothesis 1: MPEG-I Omnidirectional Video (HIGH CONFIDENCE)
**What it is**: MPEG standardized format for 360° video
- Stores video in **equirectangular projection** (maps sphere to 2D rectangle)
- Typical resolution: 4096×2048 or 8192×4096 (2:1 aspect ratio)
- Can include spatial metadata (ambisonic audio, view orientation)
- Wrapped in MP4 container with special metadata tags

**Why this fits**:
- Industry standard for 360° content
- DaVinci Resolve likely has built-in support
- DJI would adopt standards-based approach
- Plugin could interpret the metadata

**How conversion would work**:
```
OSV (8 separate 3840×3840 views)
  ↓ stitch/warp the 8 views together
  ↓ project onto sphere → equirectangular
Equirectangular MP4 (4096×2048 or 8192×4096)
  ↓ add spatial metadata
"360 file" → import to Resolve
```

**Evidence**:
- Google/YouTube use equirectangular for 360° videos
- Facebook uses equirectangular standard
- MPEG-I is published standard (ISO/IEC 23090-2)

### Hypothesis 2: Cubemap Format (MEDIUM CONFIDENCE)
**What it is**: Six cube faces (or 8 for octahedron) arranged in specific layout
- Natural fit for Osmo 360's 8-camera design
- Could be 6 or 8 square images arranged in grid
- Less common than equirectangular but used in VR

**How it would work**:
```
OSV (8 camera streams)
  ↓ keep as-is or arrange in grid
8-face cubemap image sequence or video
  ↓ embed in container with layout metadata
"360 file" → Resolve can interpret layout
```

**Evidence**:
- Osmo 360 has 8 cameras (not 6), suggesting 8-face arrangement
- Cubemaps are common in VR/3D graphics
- Less re-processing needed (no stitching)

**Against**:
- Less standardized than equirectangular
- Harder for consumer software to handle

### Hypothesis 3: Spatial Media Format (MEDIUM CONFIDENCE)
**What it is**: Google/YouTube standard for 360° metadata
- MP4 file with SphericalVideoXML metadata atom
- Describes video projection type (equirectangular, cubic, etc.)
- Used for YouTube/VR180 distribution

**How it works**:
```
Standard MP4 video
  ↓ add SphericalVideoXML metadata
  ↓ specify projection type and camera orientation
MP4 with spatial metadata
  ↓ Resolve interprets the metadata
```

**Evidence**:
- Widely adopted standard
- Easy to implement (just add XML metadata)
- DaVinci Resolve is professional tool (would support standards)

**Against**:
- Needs actual 360° video first (equirectangular or cubemap)

## The Key Unknown

**Can Resolve handle cubemap video natively, or does it need equirectangular?**

This determines the complexity:
- If cubemap OK: We keep 8 streams, add metadata → ~4-6 hours work
- If equirectangular needed: We need stitching algorithm → ~40+ hours work

## What DJI Studio Likely Does

Based on binary analysis finding "OPTICAL_FLOW_STITCHING" and "StitcherCorrectionJsonStorage":

1. **Reads** the 8 camera streams + protobuf calibration
2. **Computes** optical flow between camera boundaries
3. **Stitches** the 8 views into a single 360° video
4. **Projects** to equirectangular (or leaves as cubemap)
5. **Adds metadata** about camera calibration/rotation
6. **Exports** as "360 file"

## How to Determine Which One

### Test 1: Examine DJI Studio's Export
```bash
# If you export an OSV for Resolve in DJI Studio:
# - What's the output file extension?
# - What's the video resolution?
# - How many video streams?

# Then compare against known formats:
# Equirectangular: 4096×2048 or 8192×4096, single stream, 16:9-ish ratio
# Cubemap (grid): 8192×8192 or similar, single stream, square
# Cubemap (video): 8 separate streams or tiles, metadata referencing layout
```

### Test 2: Check Resolve Documentation
- Search Resolve manual for "360", "spherical", "omnidirectional"
- Look for supported input formats for spatial video
- Check if there's Fusion plugin documentation

### Test 3: Inspect Sample Output
- Create a test project in DJI Studio
- Export for Resolve
- Hex dump the output to find magic numbers
- Compare file structure against known video formats

## Technical Details by Format

### Equirectangular MP4 + Metadata
```
File: 360_video.mp4

Video Stream:
  Codec: H.265/HEVC or H.264
  Resolution: 4096×2048 (or 8192×4096)
  Duration: 157.6 seconds
  Frame Rate: 29.97 fps

Metadata Atoms:
  - sphericalVideoXML (if spatial media format)
  - Camera calibration data
  - Initial viewing direction
  - Projection type (equirectangular)
```

### Cubemap MP4 + Metadata
```
File: 360_cubemap.mp4

Either:
Option A - Single stream with grid layout
  Resolution: 8192×8192 (8 faces in 2x4 grid)
  Metadata describes which face is which

Option B - Multiple streams
  8 video streams, each 3840×3840
  Metadata describes camera positions/orientations
  (More like OSV but with extra metadata)
```

## What We Should Do Now

1. **Confirm by testing** (if you can):
   - Export a video from DJI Studio for Resolve
   - Check the output file with ffprobe
   - Note resolution, codec, stream count
   
2. **Search for documentation**:
   - DJI developer forum
   - DJI GitHub repositories
   - Blackmagic Design Resolve documentation
   - VR/360° video production forums

3. **Prepare for each scenario**:
   - Have equirectangular stitching approach ready
   - Have cubemap approach ready
   - Have metadata injection ready

## Impact on Our Tool

### If Equirectangular Needed
- Much more complex
- Requires optical flow / image stitching library
- Estimated 40-60 hours to implement from scratch
- Could use existing library: OpenCV, OpenVDB, or custom

### If Cubemap OK
- Keep 8 streams as-is
- Add metadata referencing layout
- Estimated 4-8 hours to implement
- Much simpler CLI tool

### If Just Metadata Needed
- Extract protobuf, convert to XML/JSON
- Wrap OSV with metadata
- Estimated 2-4 hours
- Simplest approach

**This distinction will determine our entire implementation strategy.**
