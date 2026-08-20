#!/usr/bin/env python3
"""Detailed profiling: where is time actually spent?"""

import time
import logging
from pathlib import Path

# Verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(message)s'
)

from dji_osv_converter.frame_decoder import FrameExtractor

input_file = Path("osv_example/CAM_20260819204151_0011_D.OSV")

print("\nDetailed Frame Extraction Profile\n")
print("=" * 70)

extractor = FrameExtractor(input_file)
info = extractor.get_stream_info()
print(f"Streams: {info['num_streams']}, Total frames: {info['total_frames']}")

print("\n" + "=" * 70)
print("Extracting frames (watch the logs for timing):")
print("=" * 70 + "\n")

for frame_num in range(10):
    print(f"\n--- Frame {frame_num} ---")
    start = time.time()

    frame_data = extractor.get_frame(frame_num)

    elapsed = time.time() - start

    if frame_data:
        print(f"✓ Got frame {frame_num} in {elapsed:.2f}s")
    else:
        print(f"✗ Frame {frame_num} FAILED after {elapsed:.2f}s")
        break

extractor.close()

print("\n" + "=" * 70)
print("Analysis:")
print("- Frame 0 should be fast (demux thread ramps up)")
print("- Frames 1+ should be steady state")
print("- If frames 1+ are slow, check packet arrival rate or decoder thread starvation")
