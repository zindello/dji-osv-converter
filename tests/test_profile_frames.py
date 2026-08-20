"""Profile frame processing to find the bottleneck"""

import sys
from pathlib import Path
import time
import numpy as np

print("=" * 70)
print("Frame Processing Profiler")
print("=" * 70)

from dji_osv_converter.frame_decoder import FrameExtractor
from dji_osv_converter.encoder_opencv import OpenCVEncoder

input_file = Path("osv_example/CAM_20260819204151_0011_D.OSV")
output_file = Path("osv_example/test_profile.mp4")

# Initialize
extractor = FrameExtractor(input_file)
encoder = OpenCVEncoder(str(output_file), width=7680, height=3840, fps=29.97)
encoder.open()

print("\nProcessing frames:")
print("Frame# | Extract | Stitch | Write | Total")
print("-------|---------|--------|-------|-------")

start_loop = time.time()

try:
    for i in range(50):
        frame_start = time.time()

        # Extract
        extract_start = time.time()
        frame_data = extractor.get_frame(i)
        extract_time = time.time() - extract_start

        if not frame_data:
            print(f"{i:6d} | FAILED")
            break

        # Stitch (simple average)
        stitch_start = time.time()
        f0 = np.frombuffer(frame_data[0], dtype=np.uint8).reshape((3840, 3840, 3))
        f1 = np.frombuffer(frame_data[1], dtype=np.uint8).reshape((3840, 3840, 3))
        combined = np.average([f0, f1], axis=0).astype(np.uint8)
        output = np.zeros((3840, 7680, 3), dtype=np.uint8)
        output[:, :3840] = combined
        output[:, 3840:] = combined
        stitch_time = time.time() - stitch_start

        # Write
        write_start = time.time()
        encoder.write_frame(output)
        write_time = time.time() - write_start

        total_time = time.time() - frame_start

        print(f"{i:6d} | {extract_time:7.2f}s | {stitch_time:6.2f}s | {write_time:5.2f}s | {total_time:6.2f}s")

        # Break if taking too long
        if total_time > 5.0:
            print(f"  ⚠ Frame {i} taking {total_time:.2f}s - stopping")
            break

except KeyboardInterrupt:
    print("Interrupted")

elapsed = time.time() - start_loop
encoder.close()
extractor.close()

print(f"\nTotal: {elapsed:.1f}s for {i+1} frames")
if output_file.exists():
    print(f"Output: {output_file.stat().st_size / 1e6:.1f}MB")
