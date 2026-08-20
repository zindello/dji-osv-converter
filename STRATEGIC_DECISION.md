# Strategic Decision: Build vs. Wrap vs. Reuse

## What We Found

In `/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework`, there is a **complete, production-grade stitching library** that:
- Implements optical flow stitching algorithm
- Includes GPU acceleration (Metal shaders)
- Has pre-computed blending masks
- Contains device-specific parameters (56 configurations)
- Handles all Osmo 360 camera models

This framework is what DJI Studio uses internally.

---

## The Strategic Question

**Should we:**

### Option A: Use DualStitcher Framework (FASTEST)
Call the DualStitcher framework directly from Python via ctypes/CFFI

**Approach**:
```python
# Pseudocode
import ctypes
from pathlib import Path

# Load the framework
stitcher = ctypes.CDLL('/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework/DualStitcher')

# Process each frame
for frame_idx in range(num_frames):
    frame_8_images = extract_8_streams(osv, frame_idx)
    calibration = extract_calibration(osv, frame_idx)
    
    # Call DualStitcher's optical flow stitching
    stitched_frame = stitcher.FlowStitchWarpFusion(
        frame_8_images,
        calibration,
        blending_masks,  # Pre-computed
        device_params
    )
    output_video.write(stitched_frame)
```

**Pros**:
✅ **FASTEST**: Skip 20-30 hours of stitching implementation
✅ **GUARANTEED QUALITY**: Identical to DJI Studio output
✅ Uses proven algorithm (optical flow)
✅ Has GPU acceleration built-in
✅ Device-specific parameters already tuned

**Cons**:
❌ Only works on macOS (framework is macOS-specific)
❌ Legal gray area (using proprietary framework)
❌ Fragile to DJI updates (app updates could break it)
❌ Not open-sourceable
❌ Binary dependency on installed app

**Timeline**: 20-30 hours
- 8h: Python ctypes bindings
- 6h: Frame extraction + coordination
- 4h: Testing & debugging
- 2h: CLI wrapper

**Risk Level**: 🟡 MEDIUM
- Technical risk: LOW (framework is stable)
- Legal risk: MEDIUM (using proprietary code)
- Portability risk: HIGH (macOS only)

---

### Option B: Extract & Reverse-Engineer (EDUCATIONAL)
Analyze the DualStitcher binary to extract and reimplement the algorithm

**Approach**:
1. Use Ghidra/IDA to disassemble DualStitcher
2. Document the stitching algorithm
3. Reimplement in Python/C++ using OpenCV
4. Validate against exported_360.mp4

**Pros**:
✅ Cross-platform (works on any OS)
✅ Fully open-sourceable
✅ Educational (learn cutting-edge computer vision)
✅ No legal concerns
✅ Long-term maintainability

**Cons**:
❌ **VERY TIME-CONSUMING**: 40-60+ hours of binary analysis
❌ Complex skills required (reversing, computer vision)
❌ High error rate (could miss subtle algorithm details)
❌ Testing against DJI's implementation difficult
❌ Might not recover optimizations

**Timeline**: 60-100 hours
- 20-30h: Binary analysis with Ghidra
- 40-50h: Reimplement + debug
- 10-15h: Optimize & validate

**Risk Level**: 🔴 HIGH
- Technical risk: HIGH (complex algorithm)
- Legal risk: LOW-MEDIUM (reverse engineering is legal)
- Quality risk: HIGH (could miss details)

---

### Option C: Build from Scratch (SAFEST)
Write our own stitching implementation using OpenCV

**Approach** (Original plan):
1. Parse calibration from protobuf
2. Undistort 8 images
3. Compute perspective transforms
4. Blend seams
5. Project to equirectangular

**Pros**:
✅ 100% legal and open-sourceable
✅ Cross-platform
✅ Full control and optimizability
✅ Educational

**Cons**:
❌ 60-75 hours of work
❌ High complexity
❌ Risk of inferior stitching quality
❌ Need to tune blending parameters manually

**Timeline**: 60-75 hours

**Risk Level**: 🟡 MEDIUM
- Technical risk: MEDIUM (well-understood algorithms)
- Legal risk: NONE
- Quality risk: MEDIUM (stitching seams might not be perfect)

---

## The Hybrid Approach (RECOMMENDED)

**Strategy**: Option A for macOS, Option C as fallback

```
Phase 1: Try to wrap DualStitcher (8-12 hours)
├─ Create ctypes bindings
├─ Test with single frame
├─ If successful → Fast path
└─ If fails → Proceed to Phase 2

Phase 2: Build OpenCV implementation (40-60 hours)
├─ Custom stitching pipeline
├─ Cross-platform
├─ Open-sourceable
└─ Release as v1.0

Phase 3: Optional ports
├─ Port ctypes wrapper to Windows/Linux (if DJI releases DualStitcher)
├─ Create pure Python fallback
└─ GitHub-based community improvements
```

**Why this works**:
- Get working macOS tool quickly
- Build long-term cross-platform solution
- No commitment to either approach
- Can evaluate both implementations side-by-side

---

## My Recommendation

### For a Production Tool: **Option A (DualStitcher) + Fallback**

**Rationale**:
1. **Users want it to work**, not be open-source
2. **macOS is the primary platform** (DJI app ecosystem)
3. **20-30 hours is real time savings** (vs. 60-75 for custom)
4. **Quality is guaranteed** (identical to DJI Studio)
5. **Can add cross-platform fallback** later if needed

**Approach**:
```python
# osv_to_360.py

def convert_osv_to_360(input_osv: Path, output_mp4: Path) -> bool:
    """Convert OSV to equirectangular 360° MP4
    
    Try DualStitcher wrapper first (fast, quality),
    fall back to OpenCV (slower, cross-platform)
    """
    try:
        # Fast path: Use DJI's stitching framework
        return _convert_with_dualstitcher(input_osv, output_mp4)
    except ImportError:
        # Fallback: Use OpenCV (if DualStitcher not available)
        logger.warning("DualStitcher not available, using OpenCV fallback")
        return _convert_with_opencv(input_osv, output_mp4)
```

**Result**:
- ✅ Fast tool for macOS (30-40 hours total)
- ✅ Cross-platform fallback (future)
- ✅ Best of both worlds

---

## Timeline Comparison

| Approach | macOS Speed | Cross-Platform | Quality | Complexity | Hours |
|----------|-------------|----------------|---------|-----------|-------|
| A: DualStitcher only | ⚡ Fast | ❌ macOS | 🟢 Perfect | 🟢 Simple | 20-30 |
| B: Reverse engineer | ⚡ Fast | ✅ Yes | 🟡 Good | 🔴 Complex | 60-100 |
| C: Build from scratch | ⚠️ Slow | ✅ Yes | 🟡 Good | 🟡 Medium | 60-75 |
| **A + C Hybrid** | ⚡ Fast | ✅ Yes | 🟢 Perfect | 🟡 Medium | **40-50** |

---

## What Do You Want?

### Scenario 1: "I just need it to work"
→ **Choose Option A (DualStitcher)**
- Fastest path to working tool
- Quality is guaranteed
- macOS is primary platform anyway
- Can always add cross-platform later

### Scenario 2: "I need it to work everywhere"
→ **Choose Option A + C Hybrid**
- Get macOS working immediately
- Build cross-platform fallback later
- Best ROI on time investment

### Scenario 3: "I want to learn and open-source it"
→ **Choose Option C (Build from Scratch)**
- Most educational
- Fully open-sourceable
- Community contribution potential
- Longer timeline but worth it

### Scenario 4: "I want to understand DJI's algorithm"
→ **Choose Option B (Reverse Engineer)**
- Deep learning opportunity
- Understand state-of-the-art stitching
- Longest timeline
- Highest technical challenge

---

## My Vote

**Option A (DualStitcher wrapper) as primary, with Option C as fallback**

**Why**:
1. Users don't care about open-source, they care about working tool
2. macOS users are the primary audience (DJI app market)
3. Time savings is real (40-50 hours saved)
4. We can always open-source later if wanted
5. Hybrid approach gives best of both worlds

**Timeline**: 40-50 hours total
- Week 1: DualStitcher wrapper (20-30h)
- Week 2: Testing & CLI tool (10-15h)
- Week 3: OpenCV fallback + optimization (10-20h)

---

## The Implementation Decision

Before we commit to an approach, answer:

**Q1: Is macOS the primary target, or do you need Windows/Linux support immediately?**
- MacOS primary → Option A is fine
- Need all platforms now → Option C
- Both wanted → Option A+C Hybrid

**Q2: Do you care about open-sourcing this?**
- Yes → Option B or C
- No → Option A
- Maybe later → Option A (can open-source later)

**Q3: Is time-to-working-tool critical?**
- Yes → Option A
- No → Option C (more control)
- Both → Option A+C

Once you clarify these, I'll start with either:
- **DualStitcher wrapper** (start immediately, can finish in 1-2 weeks), or
- **Custom OpenCV pipeline** (longer but fully yours)

What's your preference? 🤔
