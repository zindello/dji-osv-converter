# Current Status & Next Action

## What We Know

### OSV File (SOLID ✅)
- **Format**: MP4 container (ISO 14496-12)
- **Content**: 8 HEVC video streams (3840×3840 each) + audio + protobuf metadata
- **Camera**: DJI Osmo 360 (8-lens omnidirectional camera)
- **Metadata Schema**: `dvtm_oq101.proto` (Protocol Buffer format)
- **Extracted**: 4.2MB of metadata containing frame-by-frame camera calibration

### DaVinci Resolve Plugin (KEY UNKNOWN ⚠️)
- **Plugin name**: "DJI Reframe"
- **Input format**: "360 file" (format unclear)
- **Possible formats** (in order of likelihood):
  1. **Equirectangular MP4** (4096×2048 or 8192×4096)
     - Standard 360° video format used by YouTube/VR industry
     - Requires stitching of 8 camera views → single sphere projection
     - Complexity: **HIGH** (40-60 hours for stitching algorithm)

  2. **Cubemap format** (8192×8192 grid or 8 separate streams)
     - Uses natural camera layout from Osmo 360
     - Less standard but simpler
     - Complexity: **MEDIUM** (8-12 hours for layout management)

  3. **Spatial Media MP4** (metadata wrapper)
     - Standard Google/YouTube format with SphericalVideoXML
     - Could work with any projection type
     - Complexity: **LOW-MEDIUM** (4-8 hours for metadata injection)

  4. **Multi-stream with metadata** (like OSV but with Resolve-specific metadata)
     - Keeps 8 streams, adds interpretation layer
     - Complexity: **MEDIUM** (6-10 hours)

## The Critical Question

**What exactly is a "360 file" for the DJI Reframe plugin?**

This one question determines everything:
- Equirectangular: 40-60 hours of work
- Cubemap: 8-12 hours of work  
- Metadata wrapper: 2-4 hours of work

## How to Find Out (Estimated: 15-30 minutes)

### Option A: Export from DJI Studio (FASTEST)
1. Open an OSV file in DJI Studio
2. Look for "Export for DaVinci Resolve" option
3. If found: Export to test directory
4. Run the detection script: `DEBUG_360_FORMAT.md`
5. Report back with results

### Option B: Check DJI/Blackmagic Documentation
1. Search DJI developer site for "360 file" specs
2. Search Resolve documentation for "360° video" format
3. Look for plugin API documentation
4. Check community forums

### Option C: Trace DJI Studio Internals
If no export option visible:
1. Monitor file system during DJI Studio operations
2. Look for hidden export/temp files
3. Reverse engineer output format

## Documents for Each Scenario

| Scenario | Primary Doc | Time | Next Step |
|----------|------------|------|-----------|
| Find export option | [DEBUG_360_FORMAT.md](DEBUG_360_FORMAT.md) | 15m | Run detection script |
| Equirectangular needed | [360_FILE_HYPOTHESIS.md](360_FILE_HYPOTHESIS.md#hypothesis-1-mpeg-i-omnidirectional-video-high-confidence) | 20m | Plan stitching approach |
| Cubemap format | [360_FILE_HYPOTHESIS.md](360_FILE_HYPOTHESIS.md#hypothesis-2-cubemap-format-medium-confidence) | 20m | Plan layout packing |
| Metadata wrapper | [360_FILE_HYPOTHESIS.md](360_FILE_HYPOTHESIS.md#hypothesis-3-spatial-media-format-medium-confidence) | 20m | Plan metadata injection |

## Repository Structure

```
djiclireframe/
├── README.md                      # Project overview
├── CURRENT_STATUS.md             # ← YOU ARE HERE
├── NEXT_STEPS.md                 # Original implementation roadmap
│
├── TECHNICAL_SUMMARY.md          # Deep dive into OSV format
├── STRATEGY.md                   # Conversion strategy
│
├── 360_FILE_HYPOTHESIS.md        # What "360 file" likely is
├── RESEARCH_360_FORMAT.md        # Research questions
├── DEBUG_360_FORMAT.md           # Practical debugging guide
│
├── analyze_protobuf.py           # Tool to inspect metadata
│
└── osv_example/                  # Sample files & documentation
    ├── CAM_*.OSV / .LRF / .WAV   # Sample media files
    ├── djmd_stream_3.bin         # Extracted metadata
    └── README.md                 # Usage guide
```

## Immediate Action Items (In Priority Order)

### 1️⃣ CRITICAL: Determine the "360 File" Format
**Time**: 15-30 minutes  
**Method**: Run detection script from [DEBUG_360_FORMAT.md](DEBUG_360_FORMAT.md)  
**Deliverable**: "The 360 file is [FORMAT], with resolution [X], and properties [Y]"

### 2️⃣ HIGH: Extract Full Protobuf Metadata
**Time**: 1-2 hours  
**Depends on**: Completing #1 (to understand what metadata is needed)  
**Tools**: Python script to parse all 4724 frames of calibration data  

### 3️⃣ MEDIUM: Choose Implementation Approach
**Time**: 30 minutes (given #1 and #2)  
**Decision tree**:
- If equirectangular: Plan stitching algorithm or find OpenCV-based library
- If cubemap: Plan grid layout logic
- If metadata wrapper: Plan protobuf-to-XML conversion

### 4️⃣ Build the Converter
**Time**: 4-60 hours depending on #3  
**Output**: Python/Rust tool to transform OSV → 360 file format

### 5️⃣ Create CLI Wrapper
**Time**: 2-3 hours  
**Output**: Command like `dji-resolve input.OSV output.mov`

## Success Metrics

### Minimum Viable (Sprint 1)
- [ ] Determine "360 file" format
- [ ] Extract and parse all protobuf metadata
- [ ] CLI tool works for basic conversion
- [ ] Output imports into DaVinci Resolve

### Production Ready (Sprint 2)
- [ ] Handle all edge cases
- [ ] Batch processing
- [ ] Progress reporting
- [ ] Error handling & validation
- [ ] Documentation complete

### Future Nice-to-Have
- [ ] GUI drag-and-drop interface
- [ ] Real-time preview
- [ ] Direct Resolve plugin integration
- [ ] Performance optimization

## Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| "360 file" requires stitching | Schedule slip 40+ hours | Determine format immediately (action #1) |
| Protobuf schema complex | Slow parsing | Use existing protobuf libraries, not custom code |
| DaVinci Resolve version differences | Format changes | Document version used, add version detection |
| No public documentation | Need reverse engineering | Community forums, DJI SDK repos, hex analysis |

## Communication Plan

When you've run the format detection:
- Report: "360 file is [FORMAT] with [PROPERTIES]"
- Share: Any sample output files (if size permits)
- Confirm: The actual resolution and stream structure

From there, I can:
- Calculate exact implementation time
- Design the converter architecture
- Begin implementation immediately

---

**⏱️ Next session should start with**: "Here's what the 360 file format is..."
