# Technical Summary: DJI OSV to Resolve Conversion

## What We've Discovered

### The OSV File Format
OSV is **not** a proprietary format — it's a standard MP4 container (ISO 14496-12) with:
- **8 video streams** (HEVC/H.265, 10-bit, 3840×3840 each)
- **1 audio stream** (PCM WAV, 48kHz stereo)
- **4 metadata streams** (DJI-specific protobuf data)

### Osmo 360 Camera Configuration
The 8 video streams represent 8 separate camera sensors capturing different faces of a cube (or octahedron arrangement). The 360° effect comes from stitching these 8 views together.

### Metadata Format (CRITICAL)
- **Codec**: `djmd` (DJI MetaData)
- **Format**: Protocol Buffer (protobuf) binary serialization
- **Schema**: `dvtm_oq101.proto` (version 0.1.0.1)
- **Size**: ~4.2MB per 2.7GB video (1MB per 700MB roughly)
- **Content**: Per-frame camera calibration data for all 8 lenses

### What's Inside the Metadata
For each frame (4724 frames @ 30fps = 157.6 seconds):
1. **Camera matrix** (per lens):
   - Focal length fx, fy
   - Principal point cx, cy
   - Image size (3840×3840)

2. **Distortion coefficients** (radial & tangential):
   - k1, k2, p1, p2 (4 parameters per lens)

3. **Orientation parameters**:
   - Angular offsets (pitch, yaw, roll)
   - Field-of-view information
   - Stitching/blending boundaries (likely in Fields 20-23)

4. **Other data**:
   - dbgi streams (fields 5-6) - possibly debug/diagnostic info

## What DaVinci Resolve Expects

### Key Unknowns (to research next)
1. **Does Resolve support 360° videos natively?**
   - If yes: format for stitching parameters?
   - If no: we need to pre-stitch to 2D format

2. **For the "DJI Reframe plugin"**:
   - Is this a built-in Resolve feature or a separate plugin?
   - What's its input format? (XML, binary, structured data?)
   - Can it ingest raw camera calibration matrices?

3. **Stabilization vs. Reframing**:
   - Does Resolve just remove motion blur?
   - Does it provide viewpoint reframing (pan/tilt/zoom)?
   - Do we need to compute optical flow or image warping?

## The Reverse-Engineering Challenge

### Why the DJI App Exists
The DJI Reframe app likely:
1. Reads OSV (extracts protobuf metadata)
2. Parses the 8 camera calibrations
3. **Either**:
   - Converts to Resolve's internal stabilization format, OR
   - Pre-processes the video (stitching, stabilization) and exports a standard format

### What We Need to Figure Out
- The **exact XML or binary format** Resolve expects
- Whether we can bypass the app's processing or need to replicate it
- If Resolve can accept raw camera matrices or needs processed transforms

## Next Research Steps

### Immediate (1-2 hours)
1. **Inspect DJI Studio binary** for Resolve export logic:
   ```bash
   # Search for file paths, function names
   strings /Applications/DJI\ Studio.app/Contents/MacOS/DJIStudio | grep -i "resolve\|reframe\|xml\|export"
   
   # Check for embedded .proto files
   find /Applications/DJI\ Studio.app -name "*.proto" -o -name "*resolve*"
   
   # Look for plugin interfaces
   find /Applications/DJI\ Studio.app/Contents/PlugIns -type f -exec file {} \;
   ```

2. **Check DaVinci Resolve documentation**:
   - Look for "Fusion" plugin API (Resolve's compositing engine)
   - Check if there are publicly documented stabilization formats
   - Search for 360° video support

3. **Trace DJI Studio's file I/O**:
   - Create a test project in DJI Studio
   - Use `dtrace` to monitor what files it reads/writes during export
   - Capture the output format

### If We Get Lucky
DJI Studio might export to a simple format like:
- **XML** with frame-by-frame camera matrix data
- **JSON** with calibration parameters
- **MP4 with XML sidecar** containing metadata

### If Not
We might need to:
- Parse protobuf + implement our own stitching algorithm
- Use a library like OpenCV or libav to handle image warping
- Generate a standard video file + stabilization vectors

## Feasibility Assessment

### CLI Tool Feasibility: HIGH ✅
The data format is well-understood. We have:
- Clear protobuf structure
- Extractable metadata
- Standard video codecs
- Well-documented math (camera calibration is 100-year-old science)

### Complexity Estimate
- **Easy**: Extract & parse protobuf metadata → **2-4 hours**
- **Medium**: Understand Resolve's format → **4-8 hours** (depends on docs)
- **Hard**: Implement stitching/stabilization from scratch → **40+ hours**

### Most Likely Outcome
We can build a CLI that:
1. Extracts OSV metadata
2. Generates Resolve-compatible XML (if docs exist)
3. Possibly falls back to "export video + calibration file" format

The hardest part is figuring out what format Resolve actually wants — but that's researchable without implementation.
