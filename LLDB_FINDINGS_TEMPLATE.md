# LLDB Debugging Findings - DJI Stitcher Framework

**Date**: [When you ran debugging]  
**DJI Studio Version**: [Check Help > About]  
**Process**: PID 176  
**Framework**: DualStitcher.framework  

## Executive Summary

[Brief summary of what you discovered]

---

## 1. Stitcher Initialization

### Method Calls Observed

```
Sequence of initialization methods called:
1. [Document first init call]
2. [Document second init call]
etc.
```

### Stitcher.initWithStitcherInfo:

**When it's called**: [Describe when this initialization happens]

**Parameters captured**:

```objc
_StitcherSetupInfo {
    // Known fields
    MTLDevice device: 0x...
    MTLCommandQueue queue: 0x...
    // Calibration/masks
    NSString *maskPath: ...
    NSString *calibrationPath: ...
    // Output configuration
    int outputWidth: ...
    int outputHeight: ...
    // Unknown fields (param1-16)
    unknown param1: ...
    unknown param2: ...
    // ... others
}
```

### Stitcher.initWithMetalDevice:

**Metal Device Info**:
- Device name: [e.g., "Apple M1"]
- Device ID: [registry ID if available]
- Memory: [GPU memory info]

---

## 2. Stitching Method Signature

### Observed Method Calls

The main stitching method that fires during operation:

```objc
-[Stitcher simpleStitchSeams:seam1:isCompensate:enableSunGlare:sgGammaTex:]
```

### Method Signature Confirmed

```objc
- (NSImage *)simpleStitchSeams:(NSImage *)seam0
                        seam1:(NSImage *)seam1
                  isCompensate:(BOOL)compensate
                enableSunGlare:(BOOL)sunGlare
                    sgGammaTex:(MTLTexture *)gammaTexture;
```

[Or correct if different]

---

## 3. Input Frame Format

### Left Frame (seam0)

**Type**: NSImage  
**Address**: 0x7fa3b5234568  
**Dimensions**: 3840 × 3840 pixels  
**Pixel Format**: 0x[hex code] (RGBA / RGB / ?)  
**Bytes per Row**: 11520  
**Channels**: 3 (RGB)  
**Sample per Pixel**: 3  
**Has Alpha**: NO  
**Bit Depth**: 8-bit (uint8)  
**Data Pointer**: 0x7faca1234500  

**Sample Pixel Data (first 32 bytes hex)**:
```
0x7faca1234500: 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00
                0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00
                0xff 0xff 0xff 0x00 0xff 0xff 0xff 0x00
                0xAA 0xBB 0xCC 0xDD 0xEE 0xFF 0xAA 0xBB
```

**Interpretation**: [RGB pixel layout analysis]

### Right Frame (seam1)

**Same as left frame** (or note differences):
- Dimensions: 3840 × 3840
- Format: Same as left
- Data layout: Same as left

---

## 4. Parameter Values

### Observed Parameter Values

| Parameter | Type | Value(s) Observed | Notes |
|-----------|------|-------------------|-------|
| seam0 | NSImage* | 3840×3840 RGB | Left/top frame |
| seam1 | NSImage* | 3840×3840 RGB | Right/bottom frame |
| isCompensate | BOOL | YES (true) | Brightness compensation enabled |
| enableSunGlare | BOOL | NO (false) | Sun glare correction disabled |
| sgGammaTex | MTLTexture* | nil | No gamma texture provided |

### Parameter Variations

If parameters changed in different calls:
- Call 1: isCompensate=YES, enableSunGlare=NO
- Call 2: isCompensate=NO, enableSunGlare=YES
- Call 3: [other combinations]

---

## 5. Output Information

### Return Value

**Type**: NSImage  
**Dimensions**: 7680 × 3840 pixels (panorama)  
**Format**: [Same as input / Different?]  
**Channels**: 3 (RGB)  
**Data Size**: 88,473,600 bytes (~84 MB)  

### Output Verification

**Sample output data (first 32 bytes)**:
```
0x...: [hex dump of first few pixels]
```

**Interpretation**: [What the pattern tells us]

---

## 6. Call Sequence

### During Application Startup

```
1. Framework loaded
2. Stitcher.alloc called
3. Stitcher.init called
4. Stitcher.initWithMetalDevice: called
   └─ Metal device created
5. [Waiting for stitching operation...]
```

### During Stitching Operation

```
1. Stitcher.simpleStitchSeams:seam1:... called
   ├─ Input: Left frame 3840×3840
   ├─ Input: Right frame 3840×3840
   ├─ isCompensate: YES
   ├─ enableSunGlare: NO
   ├─ GPU processing...
   └─ Returns: NSImage 7680×3840
2. [Next frame or operation...]
```

### Error Paths (if any)

[Document any error cases you observed]

---

## 7. Pixel Format Analysis

### Format Code Captured

```
pixelFormat value: 0x[hex]
```

### Format Interpretation

Possible formats:
- 0x57524742 = 'WRGB' (?)
- 0x01020304 = Custom?
- Other: [Your finding]

### Data Layout

Based on pixel data inspection:
- Byte order: [RGB / BGR / RGBA / etc]
- Alpha channel: [None / Present]
- Endianness: [Little / Big]
- Packed or planar: [How data is organized]

---

## 8. Metal GPU Setup

### MTLDevice Information

```
Device name: Apple M1 / M2 / etc
Device ID: [Registry ID]
Max threads per group: [if found]
Memory: [Available VRAM]
```

### MTLCommandQueue

```
Queue created: YES
Max commands: [if applicable]
Priority: [if found]
```

### Textures

Other textures involved:
- sgGammaTex: nil (not used)
- Output texture: [if Metal texture used for output]

---

## 9. Calibration and Configuration

### Files Accessed

Paths observed during initialization:
```
Calibration: [path from _StitcherSetupInfo]
Masks: [mask file paths]
Resources: [other resource paths]
```

### Calibration Data

If calibration is loaded:
```
Camera intrinsics (K matrix): [3×3 matrix if found]
Distortion coefficients: [if found]
Extrinsic parameters: [if found]
```

---

## 10. Key Discoveries

### Confirmed Information
- [x] Method signature matches implementation: ✓/✗
- [x] Frame dimensions are 3840×3840: ✓/✗
- [x] RGB format (3 channels, uint8): ✓/✗
- [x] isCompensate parameter is used: ✓/✗
- [x] Output is 7680×3840 panorama: ✓/✗

### Unexpected Findings

Things different from expectations:
1. [Unexpected finding 1]
2. [Unexpected finding 2]
3. [etc]

### Mysteries Solved

Things we didn't understand before:
1. ✓ Mystery: [Question]
   Answer: [Resolution]
2. ✓ Mystery: [Question]
   Answer: [Resolution]

---

## 11. Differences from Implementation

### Current Implementation

File: `dji_osv_converter/stitcher_objc_helper.py`

Current code:
```python
result = stitcher.simpleStitchSeams_seam1_isCompensate_enableSunGlare_sgGammaTex_(
    frame_left,    # seam0
    frame_right,   # seam1
    True,          # isCompensate
    False,         # enableSunGlare
    None           # sgGammaTex
)
```

### Corrections Needed

If different from findings:
1. [Change needed]: [Old value] → [New value]
2. [Change needed]: [Old value] → [New value]
3. [etc]

---

## 12. Verification Checklist

- [ ] Ran debugging session multiple times
- [ ] Captured data from multiple stitching operations
- [ ] Verified parameter consistency across calls
- [ ] Checked frame dimensions match expectations
- [ ] Inspected pixel data format
- [ ] Documented all method calls in sequence
- [ ] Captured _StitcherSetupInfo contents
- [ ] Verified Metal device setup
- [ ] Confirmed output dimensions and format
- [ ] Updated implementation based on findings

---

## 13. Screenshots & Evidence

### LLDB Output Captures

```
(lldb) po $arg2
<NSImage: 0x7fa3b5234568 size=3840,3840>

(lldb) po [$arg2 pixelFormat]
(unsigned long) $3 = 32

(lldb) memory read --size 1 --format x --count 32 0x7faca1234500
0x7faca1234500: 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00
                0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00
                0xff 0xff 0xff 0x00 0xff 0xff 0xff 0x00
                ...
```

[Add actual LLDB output from your session]

---

## 14. Recommendations for Implementation

Based on findings:

1. **Confirm parameter order**: [If order differs]
2. **Update pixel format handling**: [If format different]
3. **Verify frame dimensions**: [If different sizes]
4. **Add initialization step**: [If initialization different]
5. **Configure Metal device**: [If setup differs]

---

## 15. Next Steps

After documenting these findings:

1. [ ] Review findings against current implementation
2. [ ] Create pull request with implementation changes
3. [ ] Run full test suite to verify changes
4. [ ] Test with actual OSV video files
5. [ ] Document in code comments what was learned
6. [ ] Update API documentation

---

## 16. References

### Original Findings Source

**Debugging Session Details**:
- Date: [Date you debugged]
- Duration: [How long]
- Frames tested: [Number of stitching operations captured]
- Variations tested: [Different UI settings if any]

### Related Documentation

- Current implementation: `/Users/zindello/repositories/zindello/djiclireframe/dji_osv_converter/stitcher_objc_helper.py`
- Test suite: `/Users/zindello/repositories/zindello/djiclireframe/tests/test_stitcher_working.py`
- LLDB Guide: `/private/tmp/claude-501/-Users-zindello-repositories-zindello-djiclireframe/7999bdda-14aa-40c9-9856-5e84d9913ee9/scratchpad/LLDB_DEBUGGING_GUIDE.md`

---

## 17. Conclusion

[Summary of what you learned and how it improves the implementation]

This reverse-engineering effort has revealed the actual API used by DJI Studio for stitching, enabling us to:
- Implement the correct parameter order
- Use the correct frame format
- Set proper initialization flags
- Optimize for DJI Studio's actual usage pattern

---

**Status**: [INCOMPLETE / IN PROGRESS / COMPLETE]  
**Next Review**: [Date]  
**Last Updated**: [Date you filled this out]
