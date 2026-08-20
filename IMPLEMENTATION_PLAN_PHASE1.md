# DJI OSV Converter: Phase 1 Implementation Plan

## Context

This project converts DJI Osmo 360 camera recordings (`.OSV` files) to equirectangular 360° MP4 format compatible with DaVinci Resolve's DJI Reframe plugin. The approach wraps DJI Studio's bundled `DualStitcher.framework` rather than reimplementing optical-flow stitching from scratch.

**Investigation findings:** The current ctypes approach is fundamentally incompatible with Objective-C. DualStitcher exports only Objective-C instance methods (e.g., `-[Stitcher initWithStitcherInfo:]`), which require PyObjC's proper message dispatch, not ctypes C function calls.

**Timeline:** ~32-41 hours for Phase 1 (DualStitcher Option A); Phase 2 (OpenCV GPU fallback) deferred to backlog.

---

## High-Level Architecture

```
OSV Input (2.7GB)
  ↓ FFmpeg stream extraction
  ↓ Frame-by-frame HEVC decode (PyAV)
  ↓ Protobuf metadata parsing (per-frame camera calibration)
  ↓ DualStitcher PyObjC bridge → FlowStitchWarpFusion() × frame_count
  ↓ H.265 encoding + spherical metadata injection
Equirectangular MP4 Output (4.5GB, 7680×3840)
```

---

## Implementation Plan

### Phase 1: Dependencies (2-3 hours)

**Deliverable:** All required packages installed and verified.

**Steps:**
1. Install PyObjC and frameworks:
   ```bash
   pip install "pyobjc-framework-CoreFoundation>=10.0" "pyobjc-core>=10.0"
   pip install "pyobjc-framework-Metal>=10.0" "pyobjc-framework-AppKit>=10.0"
   ```

2. Install protobuf and video libraries:
   ```bash
   pip install "protobuf>=3.20.0" "PyAV>=10.0.0"
   ```

3. Verify installations:
   ```bash
   python -c "import objc; from google.protobuf import message; import av; print('OK')"
   ```

4. Update `setup.py` and `requirements.txt`:
   - Add to `install_requires`: pyobjc-framework-CoreFoundation, pyobjc-core, protobuf, PyAV
   - Add to `extras_require['gpu']`: opencv-python, numpy (for Phase 2)

**Verification:** All imports work; no errors.

---

### Phase 2: PyObjC Bridge (7-9 hours)

#### Phase 2.1: Framework Loading (3-4 hours)

**File:** `/dji_osv_converter/_objc_bridge.py` (NEW)

**Deliverable:** 
- Loads DualStitcher.framework via PyObjC
- Handles dependency frameworks (openmcv2, ml_fis)
- Provides `get_objc_bridge()` singleton
- Raises clear errors if framework unavailable

**Pseudo-code structure:**
```python
class ObjCBridge:
  def _load_framework(self):
    # Load dependencies first (openmcv2, ml_fis)
    for dep in DEPENDENCY_FRAMEWORKS:
      objc.loadBundle(dep.stem, bundle_path=dep.parent.parent)
    
    # Load DualStitcher framework
    self._bundle = objc.loadBundle("DualStitcher", bundle_path=FRAMEWORK_PATH)
    
    # Verify Stitcher class accessible
    self._stitcher_class = objc.objc_getClass("Stitcher")
```

**Tests:**
- `tests/test_pyobjc_bridge.py::TestFrameworkAvailability::test_framework_file_exists`
- `tests/test_pyobjc_bridge.py::TestFrameworkAvailability::test_bridge_creation`

#### Phase 2.2: Stitcher Instantiation (4-5 hours)

**File:** `/dji_osv_converter/stitch_dualstitcher.py` (REWRITE - replace ctypes version)

**Deliverable:**
- `DualStitcher` class via PyObjC
- Proper `StitcherInfo` struct for Objective-C interop
- `StitcherConfig` loading device parameters from `param_list.json`
- Placeholder `stitch_frame()` for Phase 4 integration

**Key changes:**
- Replace `ctypes.CDLL()` with `objc.loadBundle()` and `objc.objc_getClass()`
- Convert Objective-C method names: `initWithStitcherInfo:` → `initWithStitcherInfo_` (PyObjC convention)
- Use PyObjC's `Structure` for C struct marshaling
- Load device parameters (0-56) from `/Stitcher-Mac.bundle/case_00/param_list.json`

**Critical code path:**
```python
# PyObjC instantiation pattern
stitcher = self._stitcher_class.alloc()
stitcher = stitcher.initWithStitcherInfo_(stitcher_info)
```

**Tests:**
- `tests/test_dualstitcher.py::TestDualStitcher::test_framework_loading` (updated)
- `tests/test_dualstitcher.py::TestDualStitcher::test_stitcher_instantiation` (updated)

---

### Phase 3: Frame Extraction & Decoding (9-11 hours)

#### Phase 3.1: Multi-Stream HEVC Decoder (5-6 hours)

**File:** `/dji_osv_converter/frame_decoder.py` (NEW)

**Deliverable:**
- `HEVCDecoder`: Opens OSV file, synchronizes 8 HEVC streams frame-by-frame
- `FrameExtractor`: High-level interface to get synchronized 8-frame tuples
- `FrameIndex`: Thread-safe frame coordination

**Key requirements:**
- Use PyAV (`av` library) for HEVC decoding
- Handle all 8 video streams (indices 0-7)
- Ensure frame N from all streams are decoded before returning
- Convert decoded frames to RGB24 bytes for stitcher input

**Pseudo-code:**
```python
class HEVCDecoder:
  def decode_all_frames(self):
    for packet in container.demux(video_streams):
      for frame in packet.decode():
        stream_idx = self._streams.index(packet.stream)
        self._frame_index.add_frame(stream_idx, frame_num, frame)

class FrameExtractor:
  def get_frame(self, frame_num):
    frames = self._decoder.get_frame_tuple(frame_num)
    return {i: f.reformat('rgb24').to_ndarray().tobytes() for i, f in frames.items()}
```

**Integration with `osv.py`:**
- Modify `OSVFile.get_frame_data()` to use `FrameExtractor`
- Return tuple of (frames_dict, calibration_metadata)

**Tests:**
- `tests/test_frame_extraction.py::TestFrameExtraction::test_open_sample_osv`
- `tests/test_frame_extraction.py::TestFrameExtraction::test_frame_consistency`

#### Phase 3.2: Protobuf Metadata Parser (4-5 hours)

**File:** `/dji_osv_converter/protobuf_schema.py` (NEW)

**Deliverable:**
- `ProtobufMetadataParser`: Parses per-frame camera calibration
- `SchemaReverseEngineer`: Tools to analyze protobuf binary
- Auto-generated `.proto` file (compiled with protoc)

**Key steps:**
1. Extract protobuf stream from OSV via FFmpeg:
   ```bash
   ffmpeg -i input.OSV -map 0:3 -f rawvideo djmd_metadata.bin
   ```

2. Analyze structure using `analyze_protobuf.py`:
   ```bash
   python analyze_protobuf.py djmd_stream_3.bin
   ```

3. Reverse-engineer field structure → create `metadata.proto`:
   ```protobuf
   syntax = "proto3";
   message FrameMetadata {
     uint32 frame_number = 1;
     repeated CameraCalibration calibrations = 2;
   }
   message CameraCalibration {
     float fx = 1; float fy = 2; float cx = 3; float cy = 4;
     DistortionCoeff distortion = 5;
   }
   ```

4. Compile: `protoc --python_out=dji_osv_converter/ metadata.proto`

5. Use generated classes to parse `/djmd_stream_3.bin`

**Critical unknowns to resolve:**
- Exact field numbers and types (via reverse engineering)
- Whether metadata repeats per frame or contains all frames
- Handling variable-length arrays and nested messages

**Tests:**
- Integration test that parses sample `djmd_stream_3.bin` without error
- Verify extracted calibration has reasonable values (fx/fy > 0, cx/cy within image bounds)

---

### Phase 4: Main Orchestration (5-6 hours)

**File:** `/dji_osv_converter/converter.py` (REWRITE)

**Deliverable:**
- `StitchingPipeline`: 8-step conversion pipeline
- Frame-by-frame stitching loop with progress bar
- FFmpeg encoding subprocess coordination
- Spherical metadata injection

**Steps:**
1. Load OSV file and validate
2. Initialize stitcher (DualStitcher via PyObjC)
3. Open frame extractor
4. Parse protobuf metadata
5. Start FFmpeg encoder (stdin→stdout pipe)
6. **Main loop**: for each frame:
   - Extract 8 frames from all cameras
   - Get per-frame calibration
   - Call `stitcher.stitch_frame(frame_data, calibration)`
   - Write stitched output to encoder stdin
7. Finalize encoding (close stdin, wait for encoder)
8. Inject spherical metadata (Google Spatial Media format)

**Pseudo-code for stitch_frame():**
```python
def stitch_frame(self, frame_data: Dict[int, bytes], calibration: Dict) -> bytes:
  # Convert frame bytes → Metal textures (Objective-C handles this)
  # Call: [stitcher FlowStitchWarpFusion:in_img0 seam1:... bidir_flow:... ...]
  # Return stitched equirectangular frame bytes
  # [IMPLEMENTATION DEFERRED - requires Metal texture marshaling research]
```

**Encoder command:**
```bash
ffmpeg -f rawvideo -video_size 7680x3840 -pixel_format rgb24 -framerate 29.97 \
  -i - -c:v libx265 -preset medium -b:v 40M -tag:v hvc1 -pix_fmt yuv420p \
  output.mp4
```

**Tests:**
- `tests/test_integration.py::TestFullConversion::test_conversion_with_frame_range`
- `tests/test_integration.py::TestOutputValidation::test_output_format`
- Output validation: dimensions 7680×3840, codec hevc, frame count correct

---

### Phase 5: Testing & Validation (9-12 hours)

#### Phase 5.1: Unit Tests (3-4 hours)

**Files to update/create:**
- `tests/test_pyobjc_bridge.py` (NEW)
- `tests/test_dualstitcher.py` (UPDATE for PyObjC)
- `tests/test_frame_extraction.py` (NEW)

**Coverage:**
- Framework loading without full stitching
- Configuration parameter loading
- Frame extraction synchronization
- Protobuf parsing

#### Phase 5.2: Integration Tests (4-5 hours)

**File:** `tests/test_integration.py` (NEW)

**Key tests:**
- End-to-end conversion with frame range (single frame)
- Output format validation (7680×3840, HEVC)
- Comparison against reference `exported_360.mp4`
- Performance benchmarking

#### Phase 5.3: Validation Script (2-3 hours)

**File:** `validate_output.py` (NEW)

**Checks:**
- Video dimensions, codec, frame count vs. reference
- Spherical metadata presence
- File size reasonableness
- Frame-by-frame SSIM comparison (optional)

---

## Critical Files to Create/Modify

| File | Type | Purpose |
|------|------|---------|
| `dji_osv_converter/_objc_bridge.py` | NEW | PyObjC framework loading & singleton |
| `dji_osv_converter/stitch_dualstitcher.py` | REWRITE | Objective-C Stitcher instantiation via PyObjC |
| `dji_osv_converter/frame_decoder.py` | NEW | Multi-stream HEVC decoder (PyAV) |
| `dji_osv_converter/protobuf_schema.py` | NEW | Metadata parser & reverse-engineering tools |
| `dji_osv_converter/converter.py` | REWRITE | Main orchestration pipeline |
| `dji_osv_converter/osv.py` | UPDATE | Integrate FrameExtractor into OSVFile |
| `dji_osv_converter/metadata.proto` | NEW | Auto-generated from reverse engineering |
| `setup.py` | UPDATE | Add PyObjC, protobuf, PyAV dependencies |
| `requirements.txt` | UPDATE | Document dependencies |
| `tests/test_pyobjc_bridge.py` | NEW | PyObjC bridge tests |
| `tests/test_integration.py` | NEW | End-to-end conversion tests |
| `validate_output.py` | NEW | Output validation against reference |

---

## Verification Plan

**Per-phase checks:**

1. **Phase 1 Done**: All imports work, no version conflicts
   ```bash
   python -c "import objc; from google.protobuf import message; import av; print('OK')"
   ```

2. **Phase 2 Done**: Framework loads, Stitcher class accessible
   ```bash
   python -c "from dji_osv_converter._objc_bridge import get_objc_bridge; b = get_objc_bridge(); print('Framework:', b.is_framework_available())"
   ```

3. **Phase 3.1 Done**: Can extract and decode sample OSV frames
   ```bash
   pytest tests/test_frame_extraction.py -v
   ```

4. **Phase 3.2 Done**: Protobuf metadata parses without error
   ```bash
   pytest tests/test_integration.py::TestMetadataParser -v
   ```

5. **Phase 4 Done**: Single-frame conversion produces valid MP4
   ```bash
   python -m dji_osv_converter.cli sample.OSV output.mp4 --frame-range 0:1
   ```

6. **Phase 5 Done**: All tests pass, output matches reference format
   ```bash
   pytest tests/ -v
   python validate_output.py output.mp4 osv_example/exported_360.mp4
   ```

---

## Known Unknowns & Risks

| Unknown | Mitigation | Impact |
|---------|-----------|--------|
| Exact Objective-C method signatures in DualStitcher | Use class-dump; incremental testing | Medium - will discover via API errors |
| Protobuf field structure for dvtm_oq101 | Run analyze_protobuf.py on sample; reverse-engineer | Medium - can fall back to generic field parsing |
| Metal GPU texture marshaling (bytes ↔ MTLBuffer) | Research PyObjC Metal interop; possible CPU staging | High - GPU path may require iteration |
| Frame synchronization guarantees | Use thread-safe FrameIndex; test with mock streams | Low - well-understood pattern |
| FFmpeg encoder process coordination | Standard subprocess patterns; extensive testing | Low - common practice |

---

## Success Criteria

**MVP (minimum viable):**
- ✅ Framework loads via PyObjC without crashes
- ✅ Frames extract and decode from 8 streams
- ✅ Protobuf metadata parses per-frame calibration
- ✅ Stitcher instantiation succeeds (method call can be attempted)
- ✅ Single frame processes end-to-end (even if output quality poor initially)
- ✅ CLI tool runs without exceptions

**Production (full Phase 1):**
- ✅ All unit & integration tests pass
- ✅ Output MP4 is valid equirectangular (7680×3840, HEVC)
- ✅ Output matches reference quality (SSIM > 0.95 on test frames)
- ✅ Handles full video duration (all 4700+ frames)
- ✅ Performance reasonable (~2-4 hours for 2.7GB file)

---

## Remaining Open Questions

1. **How to marshal frame bytes to Metal textures?** 
   - Research required during Phase 4
   - May need MTLBuffer wrapper or CPU staging path

2. **Is protobuf schema guessable from reverse engineering?**
   - Yes, analyze_protobuf.py designed for this
   - Will know field boundaries/types after running on sample

3. **Do we need to implement optical flow ourselves?**
   - No; DualStitcher.FlowStitchWarpFusion does this
   - We just need to call it correctly

---

## Next Actions After Approval

1. Install dependencies (Phase 1)
2. Create `_objc_bridge.py` with framework loading
3. Rewrite `stitch_dualstitcher.py` to use PyObjC
4. Create `frame_decoder.py` with PyAV integration
5. Reverse-engineer & implement protobuf parser
6. Rewrite `converter.py` orchestration
7. Implement tests & validation
8. Validate against reference output
