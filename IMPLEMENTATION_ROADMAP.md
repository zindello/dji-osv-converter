# Implementation Roadmap

## Phase 1: DualStitcher Wrapper (Option A) - PRIMARY
**Timeline**: 2-3 weeks  
**Status**: Starting now  
**Target**: Production Python library that wraps macOS DualStitcher framework

### Deliverables
- ✅ Python package: `dji_osv_converter`
- ✅ CLI tool: `dji-resolve input.OSV output.mp4`
- ✅ Python API for embedding in other tools
- ✅ Test suite with sample OSV file
- ✅ Documentation

### Milestones

**Milestone 1.1: DualStitcher Bindings (Days 1-2)**
- Reverse-engineer DualStitcher C/Objective-C interface
- Create ctypes bindings to framework functions
- Test basic framework loading and function calls

**Milestone 1.2: OSV Frame Extraction (Days 3-4)**
- Extract 8 HEVC streams from OSV container
- Extract protobuf metadata
- Parse camera calibration for each frame
- Load DualStitcher device parameters (by SN)

**Milestone 1.3: Frame Processing Pipeline (Days 5-7)**
- Feed frames to DualStitcher optical flow stitcher
- Capture stitched output frames
- Validate output resolution (7680×3840)
- Test single frame end-to-end

**Milestone 1.4: Video Encoding (Days 8-9)**
- Encode stitched frames to H.265 with FFmpeg
- Match DJI's codec settings (Main 10, 30fps)
- Add spherical metadata atoms
- Mux with audio track (from input OSV)

**Milestone 1.5: CLI & Testing (Days 10-14)**
- Build command-line interface
- Full integration test with sample OSV
- Performance benchmarking
- Error handling and validation
- Documentation and examples

---

## Phase 2: OpenCV Implementation (Option C) - BACKLOG
**Timeline**: 3-4 weeks (after Phase 1)  
**Status**: Backlog  
**Target**: Cross-platform Python library with GPU acceleration

### Architecture

```
OSV Input
  ↓
Extract 8 streams + metadata
  ↓
GPU-Accelerated Pipeline:
  ├─ Load calibration (OpenCV UMat for GPU)
  ├─ Undistort 8 images (CUDA/Metal/Intel)
  ├─ Compute optical flow (OpenCV/CUDA)
  ├─ Estimate homographies (GPU-accelerated)
  ├─ Warp and blend seams (OpenCV GPU)
  └─ Project to equirectangular (Metal/CUDA)
  ↓
Encode with FFmpeg
  ↓
Output: 7680×3840 Equirectangular MP4
```

### GPU Acceleration Strategy

**Auto-detect & fallback pattern**:
```python
def stitch_with_gpu_acceleration(frames, calibration):
    """Use best available GPU acceleration"""
    
    # Try Metal (macOS M1/2/3)
    try:
        return metal_stitch(frames, calibration)
    except RuntimeError:
        pass
    
    # Try CUDA (NVIDIA)
    try:
        return cuda_stitch(frames, calibration)
    except RuntimeError:
        pass
    
    # Try Intel GPU (oneAPI)
    try:
        return intel_stitch(frames, calibration)
    except RuntimeError:
        pass
    
    # Fallback to CPU
    return cpu_stitch(frames, calibration)
```

### Deliverables

**Phase 2.1: Algorithm Implementation**
- Undistortion pipeline
- Optical flow computation
- Homography estimation
- Seam blending
- Equirectangular projection

**Phase 2.2: GPU Acceleration**
- Metal support (macOS M1/2/3)
- CUDA support (NVIDIA)
- Intel GPU support (oneAPI)
- CPU fallback

**Phase 2.3: Integration**
- Unified API with Phase 1
- Same CLI interface
- Fallback mechanism

---

## Technology Stack

### Phase 1 (DualStitcher)
- **Language**: Python 3.8+
- **Framework binding**: ctypes / PyObjC
- **Video I/O**: FFmpeg (subprocess)
- **Metadata**: protobuf, Pillow (for masks)
- **Packaging**: setuptools, pip
- **Testing**: pytest
- **Dependencies**: ~5 light packages

### Phase 2 (OpenCV)
- **Language**: Python 3.8+
- **Computer Vision**: OpenCV 4.5+ with GPU support
- **GPU/Math**: 
  - Metal (macOS native)
  - CUDA 11+ (NVIDIA)
  - oneAPI (Intel)
  - NumPy/SciPy (CPU math)
- **Video I/O**: FFmpeg
- **Testing**: pytest with GPU support
- **Dependencies**: ~10 packages

---

## Development Process

### Phase 1 Process
1. **Understand DualStitcher API** (Ghidra analysis)
2. **Create ctypes bindings** (Python module)
3. **Build extraction pipeline** (FFmpeg + protobuf)
4. **Integrate & test** (sample OSV)
5. **Optimize & release** (performance tuning)

### Phase 2 Process
1. **Algorithm research** (study DualStitcher + papers)
2. **Prototype CPU version** (validate approach)
3. **Add GPU acceleration** (Metal/CUDA/oneAPI)
4. **Integrate with Phase 1** (unified API)
5. **Benchmark & optimize** (GPU vs CPU comparison)

---

## Testing Strategy

### Phase 1 Tests
- ✅ Framework loading (can load DualStitcher)
- ✅ OSV extraction (8 streams, metadata parsing)
- ✅ Single frame processing (end-to-end)
- ✅ Full video processing (sample OSV)
- ✅ Output validation (compare with exported_360.mp4)
- ✅ CLI interface (command-line parsing)

### Phase 2 Tests
- ✅ CPU stitching (baseline)
- ✅ GPU stitching (Metal/CUDA)
- ✅ Quality comparison (Phase 1 vs Phase 2)
- ✅ Performance benchmarks
- ✅ GPU fallback (GPU unavailable)
- ✅ Cross-platform (Windows/Linux)

---

## Repository Structure

```
djiclireframe/
├── dji_osv_converter/           # Main package
│   ├── __init__.py
│   ├── cli.py                   # CLI entry point
│   ├── osv.py                   # OSV file handling
│   ├── stitch/
│   │   ├── dualstitcher.py      # Phase 1: Framework wrapper
│   │   └── opencv_gpu.py        # Phase 2: OpenCV with GPU
│   ├── encode.py                # FFmpeg encoding
│   └── metadata.py              # Spherical metadata
│
├── tests/
│   ├── test_osv.py
│   ├── test_dualstitcher.py
│   ├── test_opencv.py
│   └── conftest.py
│
├── setup.py
├── requirements.txt
├── requirements-gpu.txt         # GPU acceleration packages
├── README.md
├── CHANGELOG.md
└── docs/
    ├── installation.md
    ├── usage.md
    ├── gpu-acceleration.md
    └── development.md
```

---

## Success Criteria

### Phase 1 MVP
- [ ] Wraps DualStitcher framework successfully
- [ ] Processes sample OSV to equirectangular MP4
- [ ] Output matches DJI Studio quality
- [ ] CLI tool works end-to-end
- [ ] Runs on macOS 10.15+

### Phase 1 Production
- [ ] All tests pass
- [ ] Performance: Process 2.7GB OSV in <2 hours
- [ ] Error handling complete
- [ ] Documentation complete
- [ ] Can handle all Osmo 360 models

### Phase 2 MVP
- [ ] Undistortion works
- [ ] Optical flow computed correctly
- [ ] CPU stitching produces valid output
- [ ] Metal GPU acceleration works on M1/2/3

### Phase 2 Production
- [ ] CUDA support working
- [ ] Intel GPU support working
- [ ] Quality parity with Phase 1
- [ ] Cross-platform (macOS/Windows/Linux)
- [ ] Performance: GPU 3-5x faster than CPU

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| DualStitcher API unclear | Use Ghidra, binary analysis, empirical testing |
| ctypes binding crashes | Extensive error handling, logging |
| GPU memory issues | Stream processing, temporary file management |
| Performance too slow | Optimize hot paths, GPU acceleration |
| DJI updates break it | Maintain backward compatibility, version pinning |

---

## Timeline

```
Week 1 (Phase 1.1-1.3):
├─ Mon-Tue: Reverse-engineer DualStitcher API
├─ Wed-Thu: Create ctypes bindings & test
└─ Fri: Extract frames, validate pipeline

Week 2 (Phase 1.4-1.5):
├─ Mon-Tue: FFmpeg encoding + metadata injection
├─ Wed-Thu: CLI tool + full integration test
└─ Fri: Performance optimization, documentation

Week 3 (Phase 1 Release):
├─ Mon-Tue: Bug fixes from Phase 1 testing
├─ Wed: Package & release v0.1.0
└─ Thu-Fri: Buffer for unexpected issues

Phase 2 (Backlog):
├─ Week 4-5: OpenCV stitching implementation
├─ Week 6-7: GPU acceleration (Metal/CUDA/Intel)
└─ Week 8: Integration & release v0.2.0
```

---

## Current Status

✅ **Research Complete**
- OSV format understood
- Output format confirmed (7680×3840 equirectangular)
- DualStitcher framework found and analyzed
- Implementation strategy finalized

🚀 **Ready to Start Phase 1**
- All prerequisites understood
- Test data available (sample OSV + reference output)
- Dependencies identified
- Architecture documented

**Next**: Begin Milestone 1.1 (DualStitcher bindings)
