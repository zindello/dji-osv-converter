# Reverse Engineering vs. Building from Scratch

## The Strategic Question

**Should we:**
1. Call DJI Studio as a subprocess? (Quick but not portable)
2. Extract/reuse DJI libraries? (Fastest, most reliable)
3. Reverse-engineer the stitching algorithm? (Educational, risky)
4. Write our own implementation? (Portable, time-consuming)

---

## Option 1: Use DJI Studio as Subprocess (Easiest)

### Concept
```python
# osv_to_360.py
import subprocess
import tempfile
import os

def convert_osv_to_360(input_osv, output_path):
    """Use DJI Studio via subprocess"""
    # Open in DJI Studio, trigger export, capture output
    result = subprocess.run([
        '/Applications/DJI Studio.app/Contents/MacOS/DJIStudio',
        '--headless',  # Does this exist?
        '--export',
        '--input', input_osv,
        '--output', output_path,
    ])
    return result.returncode == 0
```

### Pros & Cons
✅ DJI's stitching is perfect (we already tested it)  
✅ Minimal reimplementation needed  
✅ Fast to prototype  
❌ Requires DJI Studio installed  
❌ Can't be distributed standalone  
❌ Only works on macOS (initially)  
❌ Fragile to app updates  
❌ May violate ToS if automated headless use  
❌ Slow (spawns full app just for export)  

### Feasibility
🟡 **MEDIUM** - Depends on whether DJI Studio has CLI options
- Most macOS apps don't expose command-line export
- Would need to reverse-engineer UI automation (Applescript, etc.)
- Not a reliable production solution

---

## Option 2: Extract/Reuse DJI Libraries (BEST?)

### The Idea
DJI's stitching logic might be in:
- Frameworks bundled with the app
- Compiled libraries (.dylib files)
- Static libraries linked into the binary

If we can find and extract the stitching framework, we might:
- Call it directly from Python (via ctypes or wrapper)
- Use it as a black-box library
- Skip the 20-30 hour stitching implementation

### What to Look For

```bash
# 1. Check for bundled frameworks
find /Applications/DJI\ Studio.app/Contents/Frameworks -name "*.framework" -o -name "*.dylib"

# 2. Look for stitching-related symbols
strings /Applications/DJI\ Studio.app/Contents/MacOS/DJIStudio | grep -i "stitch\|warp\|project"

# 3. Check what libraries are actually loaded
otool -L /Applications/DJI\ Studio.app/Contents/MacOS/DJIStudio | grep -v "/usr/lib\|/System"

# 4. Look for protobuf definitions that hint at algorithm
nm /Applications/DJI\ Studio.app/Contents/MacOS/DJIStudio | grep -i "calibrat\|project\|equirect"
```

### Pros & Cons
✅ Could save 20-30 hours of stitching work  
✅ Guaranteed to work (same code as DJI)  
✅ Could be wrapped as Python library  
⚠️ Legal gray area (using proprietary code without permission)  
❌ Fragile to macOS updates (code signing, security)  
❌ Only works on macOS initially  
❌ Distribution might violate DJI's ToS  
❌ Reverse-engineering binary code is complex  

### Feasibility
🟡 **MEDIUM-HIGH** - Depends on how libraries are packaged
- If stitching is in a separate .dylib: High feasibility
- If it's compiled into the main binary: Low feasibility (would need to extract functions via Ghidra)
- Might need Xcode tools (otool, nm, strings)

---

## Option 3: Reverse-Engineer the Algorithm (HARD)

### The Process

```
1. Analyze the binary statically (Ghidra, IDA)
   - Find stitching function entry points
   - Map the algorithm flow
   - Identify math operations
   
2. Trace execution dynamically
   - Use lldb to set breakpoints
   - Watch register/memory changes
   - Understand data flow
   
3. Reconstruct the algorithm
   - Document the math
   - Reimplement in Python/C
   - Validate against exports
```

### What We'd Learn

From binary analysis, we could discover:
- Exact camera layout (how 8 cameras are positioned)
- Stitching method (optical flow? feature matching? warping?)
- Blending algorithm (feathering? gradient mask?)
- Equirectangular projection formula
- Optimization tricks

### Pros & Cons
✅ Educational (learn cutting-edge computer vision)  
✅ No dependencies on DJI code  
✅ Fully portable Python  
✅ Could be open-sourced freely  
✅ Optimizable  
❌ **Very time-consuming** (40-60+ hours)  
❌ Complex binary analysis skills needed  
❌ Easy to make mistakes  
❌ Might not recover all optimizations  

### Feasibility
🔴 **LOW-MEDIUM** - Doable but very risky
- Binary analysis tools: Free (Ghidra)
- Skills needed: Intermediate-Advanced reversing
- Time investment: 40-60 hours minimum

### Legal/Ethical
⚠️ Reverse-engineering binaries is:
- Legal in most jurisdictions (for interoperability)
- Often violates software EULA
- Acceptable for personal use
- Questionable for distribution/commercial use

---

## Option 4: Write Our Own (Current Plan)

### The Approach
Use well-known computer vision libraries (OpenCV, scipy) to:
1. Parse calibration from protobuf
2. Undistort each of 8 images
3. Compute stitching transforms
4. Blend seams
5. Project to equirectangular

### Pros & Cons
✅ Fully legal and open-sourceable  
✅ Cross-platform (Windows, Linux, macOS)  
✅ Zero dependencies on DJI  
✅ Educational  
✅ Maintainable  
❌ 60-75 hours of work  
❌ Need to get stitching right  
❌ Performance tuning needed  

### Feasibility
🟢 **HIGH** - Standard computer vision problem
- Algorithms are well-documented
- Libraries exist (OpenCV, scipy)
- Lots of tutorials and examples
- Proven approach used in many projects

---

## The Hybrid Approach (RECOMMENDED)

**Strategy**: Start with reverse-engineering, fall back to custom implementation

### Phase 1A: Quick Reverse Engineering (4-6 hours)
```bash
# Analyze the binary to find:
1. Where stitching code might be
2. What frameworks are used
3. Camera layout/arrangement hints
4. Algorithm type (optical flow? feature matching?)

# Tools needed:
- otool (analyze Mach-O binaries)
- strings (extract readable text)
- Ghidra (disassemble if needed)
- lldb (debug if needed)
```

### Phase 1B: Quick Feasibility Check
- Can we extract a stitching library/framework? (1-2 hours)
- If yes: Use it, save 20-30 hours
- If no: Proceed with custom implementation

### Phase 2: Custom Implementation (Only if Phase 1B says no)
- Build stitching from scratch using OpenCV
- Use knowledge from Phase 1A to guide implementation
- Validate against exported_360.mp4

### Estimated Timeline
- Hybrid approach: 30-50 hours (if library found) or 60-75 hours (if not)
- vs. Pure reimplementation: 60-75 hours
- vs. DJI subprocess hack: 4-6 hours (but not production-ready)

---

## Specific Reverse Engineering Tasks

### Task 1: Identify Frameworks
```bash
# List all frameworks/libraries in DJI Studio
find /Applications/DJI\ Studio.app/Contents -type d -name "*.framework"
find /Applications/DJI\ Studio.app/Contents -name "*.dylib"

# Example output might include:
# - libStitcher.dylib (VERY promising!)
# - libCalibration.dylib
# - libProjection.dylib
# (These don't exist but would be ideal)
```

### Task 2: Extract Symbols
```bash
# Get all exported functions from main binary
nm /Applications/DJI\ Studio.app/Contents/MacOS/DJIStudio | grep -E "stitch|warp|project|blend"

# Check for stitching hints in strings
strings /Applications/DJI\ Studio.app/Contents/MacOS/DJIStudio | grep -i -E "optical.*flow|feature.*match|homograph|equirectang"
```

### Task 3: Look for Protobuf Hints
```bash
# Camera layout might be encoded as field names in protobuf
strings /Applications/DJI\ Studio.app/Contents/MacOS/DJIStudio | grep -i "camera.*position\|camera.*pose\|cubemap"
```

### Task 4: Check for FFmpeg Integration
```bash
# DJI might use FFmpeg for encoding (open source)
strings /Applications/DJI\ Studio.app/Contents/MacOS/DJIStudio | grep -i ffmpeg

# If using custom FFmpeg, they might have stitching as preprocessing:
# OSV → stitch → ffmpeg encode → MP4
```

---

## My Recommendation

### For a Production Tool

**Best approach: Hybrid (Investigate, then Build)**

```
Week 1: Reverse Engineer & Research (4-6 hours)
├─ Analyze DJI binary for stitching approach
├─ Look for extractable frameworks/libraries
├─ Document camera layout from protobuf
└─ Decide: Extract or Rebuild?

Week 2-3: Implement Core (depends on Week 1)
├─ If libraries found: Wrap + test (15-25 hours)
└─ If not: Build stitching from scratch (50-65 hours)

Result: Production Python library
├─ Cross-platform
├─ No DJI dependency
├─ Open-sourceable
└─ Fast (optimized or using extracted code)
```

### For a Quick MVP

**Fastest approach: DJI Subprocess + Wrapper (1-2 weeks)**

```
Week 1: Build subprocess wrapper
├─ Reverse-engineer DJI CLI options (if any)
├─ Use Applescript/UI automation if needed
├─ Create Python CLI wrapper
└─ Test end-to-end

Result: Tool that works
├─ Requires DJI Studio installed
├─ macOS only initially
├─ But gets the job done
└─ Can upgrade to standalone later
```

### For Maximum Learning

**Educational approach: Full Reverse Engineering (3-4 weeks)**

```
Week 1: Deep binary analysis (Ghidra, lldb)
├─ Extract algorithm details
├─ Understand camera layout
├─ Map stitching pipeline
└─ Document findings

Week 2-3: Reimplement from scratch
├─ Build computer vision pipeline
├─ Implement stitching algorithm
├─ Optimize and validate
└─ Publish as open-source library

Result: Deep understanding
├─ Educational
├─ Fully open-sourceable
├─ Maintainable long-term
└─ Portable to any platform
```

---

## What Should We Do?

### My Vote: Start with Investigation (Phase 1A)

**Reasoning**:
1. **Low cost** (4-6 hours to investigate)
2. **High information** (understand what DJI is doing)
3. **No commitment** (pivot based on findings)
4. **Clear decision point** (extract library or build?)

### The Investigation Plan

```bash
# 1. Check if there are extractable libraries
find /Applications/DJI\ Studio.app/Contents/Frameworks -name "*.dylib" 2>/dev/null
find /Applications/DJI\ Studio.app/Contents -name "*stitch*" -o -name "*warp*" -o -name "*project*" 2>/dev/null

# 2. Analyze the main binary
strings /Applications/DJI\ Studio.app/Contents/MacOS/DJIStudio | grep -i "stitch\|optical\|warp\|project\|blend" | head -50

# 3. Check dependencies
otool -L /Applications/DJI\ Studio.app/Contents/MacOS/DJIStudio 2>/dev/null | grep -v "/usr/lib\|/System" | head -20

# 4. Look for protobuf clues
strings /Applications/DJI\ Studio.app/Contents/MacOS/DJIStudio | grep "camera.*position\|pose\|layout" | head -20

# 5. Search for equirectangular hints
strings /Applications/DJI\ Studio.app/Contents/MacOS/DJIStudio | grep -i "equirect\|spheric\|7680\|3840" | head -20
```

**Expected outcomes:**
- 🟢 **Best case**: Find extractable library → Save 20-30 hours
- 🟡 **Good case**: Understand algorithm → Better implementation guide
- 🔴 **Worst case**: No useful info → Proceed with full custom build

---

## What Do You Prefer?

Given your preferences ("Python library, any platform, importable"), I'd recommend:

**Option A (Recommended): Investigate + Custom Build**
- Spend 4-6 hours investigating
- Build cross-platform Python library
- Ensures open-sourceable, maintainable code
- Timeline: 60-75 hours total
- Result: Standalone tool, no dependencies

**Option B: Fast Hack + Upgrade Later**
- Build DJI subprocess wrapper
- Get working tool in 1-2 weeks
- Upgrade to standalone library later
- Timeline: 10-20 hours initial, 50-70 hours later
- Result: Quick wins but technical debt

Which direction interests you? Should I start the investigation? 🔎
