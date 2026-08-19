# How to Determine the Actual 360 File Format

## The Quickest Way to Find Out

### Step 1: Export from DJI Studio (5 minutes)
1. Open your OSV file in DJI Studio
2. Look for "Export for DaVinci Resolve" or similar option
3. If found, export to a test directory
4. Note the output filename and location

### Step 2: Inspect the Output (10 minutes)
```bash
# Check what was created
ls -lh ~/path/to/export/

# Get detailed info
ffprobe -v error -show_format -show_streams ~/path/to/export/video.* 2>&1 | head -50

# Check for unusual atoms/metadata
ffprobe -show_format ~/path/to/export/video.* | grep -i "spheric\|spatial\|360\|calibrat"

# Hex dump the header
xxd -l 1000 ~/path/to/export/video.* | head -50
```

### Step 3: Compare Against Known Formats
Use the output from Step 2 to identify:

**Equirectangular?** Look for:
```
Resolution: 4096×2048 or 8192×4096 (2:1 aspect ratio)
Video Streams: 1
Codec: H.264 or H.265
Color Space: BT.709
```

**Cubemap grid?** Look for:
```
Resolution: 8192×8192 or 4096×4096 (square)
Video Streams: 1
Metadata might reference grid layout
```

**Multi-stream cubemap?** Look for:
```
Video Streams: 8
Resolution: 3840×3840 per stream
Similar to original OSV structure
```

**Metadata wrapper?** Look for:
```
Video Streams: 2 (same as OSV)
Audio Streams: 1
Additional metadata atoms/tracks
```

## Automated Detection Script

```bash
#!/bin/bash
# detect_360_format.sh

FILE="$1"
if [ -z "$FILE" ]; then
  echo "Usage: $0 <video_file>"
  exit 1
fi

echo "=== Analyzing 360 File Format ==="
echo "File: $FILE"
echo

# Get resolution
RESOLUTION=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$FILE")
echo "Resolution: $RESOLUTION"

# Get stream count
STREAM_COUNT=$(ffprobe -v error -select_streams v -show_entries stream -of csv=s=x:p=0 "$FILE" | wc -l)
echo "Video Streams: $STREAM_COUNT"

# Get codec
CODEC=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1:p=0 "$FILE")
echo "Video Codec: $CODEC"

# Get duration
DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1:p=0 "$FILE")
echo "Duration: $DURATION seconds"

# Check for metadata atoms
echo
echo "=== Metadata Analysis ==="
ffprobe -v error -show_entries format -of compact=p=0:nk=1 "$FILE" | grep -i "spheric\|spatial\|360\|calibrat\|metadata\|uuid" || echo "(No special 360 metadata found)"

# Check aspect ratio
IFS='x' read -r WIDTH HEIGHT <<< "$RESOLUTION"
RATIO=$(echo "scale=2; $WIDTH / $HEIGHT" | bc)
echo
echo "Aspect Ratio: $RATIO"

# Analysis
echo
echo "=== Format Detection ==="
if [ "$RESOLUTION" = "4096x2048" ] || [ "$RESOLUTION" = "8192x4096" ]; then
  echo "Likely: EQUIRECTANGULAR (360° video)"
  echo "This is the standard 360° video format."
  echo "Action: Resolve should import directly."
elif [ "$RESOLUTION" = "8192x8192" ] || [ "$RESOLUTION" = "4096x4096" ]; then
  echo "Likely: CUBEMAP GRID (8 or 6 faces arranged)"
  echo "This is a cubemap layout for 360° content."
  echo "Action: Check if Resolve can handle cubemap, or convert to equirectangular."
elif [ "$STREAM_COUNT" = "8" ]; then
  echo "Likely: MULTI-STREAM CUBEMAP (8 separate video streams)"
  echo "Similar to original OSV structure."
  echo "Action: May need Resolve plugin or conversion to single-stream format."
else
  echo "Uncertain. Could be:"
  echo "- Custom DJI format"
  echo "- Metadata wrapper around standard format"
  echo "- Proprietary arrangement"
  echo "Action: Inspect with hex editor or compare against sample files."
fi
```

Save as `detect_360_format.sh` and run:
```bash
chmod +x detect_360_format.sh
./detect_360_format.sh exported_video.mp4
```

## If DJI Studio Doesn't Have Export Option

If DJI Studio doesn't expose a "Export for Resolve" option, try:

1. **Check the menu system**
   - File → Export
   - Look for "Video", "Project", "Plugin", "Resolve"

2. **Search DJI Studio help**
   - Help menu → Search for "Resolve" or "360"
   - Look for plugin/export documentation

3. **Check DJI community forums**
   - DJI Support site
   - Ask: "How to export OSV to DaVinci Resolve format?"

4. **Look for sidecar files**
   - Sometimes apps export both video + metadata file
   - Check for `.xml`, `.json`, `.srt`, `.xml.gz` files

## Alternative: Reverse Engineer the Format

If export doesn't reveal it, we can reverse-engineer:

### Method 1: Compare Input and Output
```bash
# Get info about original OSV
ffprobe -v error -show_format -show_streams input.OSV > osv_info.txt

# Use DJI Studio to process it (any operation that saves)
# Then inspect the output
ffprobe -v error -show_format -show_streams output.mov > output_info.txt

# Compare
diff osv_info.txt output_info.txt
```

### Method 2: Hex Comparison
```bash
# Save first 10MB of OSV
dd if=input.OSV bs=1M count=10 of=osv_sample.bin

# Save first 10MB of output
dd if=output.mov bs=1M count=10 of=output_sample.bin

# Look for differences
xxd osv_sample.bin | head -100 > osv.hex
xxd output_sample.bin | head -100 > output.hex
diff osv.hex output.hex
```

### Method 3: Monitor File System
```bash
# In one terminal, monitor file access
fs_usage -w -f filesystem | grep -E "\.mov|\.mp4|\.xml|\.json|export|resolve"

# In another terminal, use DJI Studio to export
# Watch the monitor for file operations
```

## What to Report Back

Once you've determined the format, provide:

1. **File info**: Resolution, codec, stream count
2. **Sample**: Upload small sample file if possible (first 100MB of video)
3. **Metadata**: Any visible metadata tags or special atoms
4. **Source**: "From DJI Studio export" vs "reverse-engineered"
5. **Behavior**: "Resolve can import directly" vs "needs conversion"

With this info, we can immediately start building the converter.

## Expected Outcomes and Next Steps

| Outcome | Complexity | Time to CLI | Next Action |
|---------|------------|------------|-------------|
| Equirectangular MP4 | HIGH | 40-60h | Implement stitching algorithm |
| Cubemap grid MP4 | MEDIUM | 8-12h | Implement grid layout packer |
| Multi-stream with metadata | MEDIUM | 6-10h | Add metadata to 8 streams |
| Just metadata wrapper | LOW | 2-4h | Extract & repackage protobuf |
| Standard Spatial Media MP4 | MEDIUM | 4-8h | Add XML metadata to video |

**Determining the format might save us 40+ hours of wrong-direction development.**
