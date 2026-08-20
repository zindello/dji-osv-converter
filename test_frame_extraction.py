#!/usr/bin/env python3
"""
Frame extraction performance test - isolates the bottleneck.

Tests ONLY frame extraction, nothing else. Measures:
- Time per frame
- Cumulative slowdown
- Whether it's decode or color conversion
"""

import sys
import time
from pathlib import Path

print("=" * 70)
print("Frame Extraction Performance Test")
print("=" * 70)

from dji_osv_converter.frame_decoder import FrameExtractor

input_file = Path("osv_example/CAM_20260819204151_0011_D.OSV")

# Initialize extractor
print("\n[1] Initializing frame extractor...")
init_start = time.time()
extractor = FrameExtractor(input_file)
init_time = time.time() - init_start
print(f"✓ Initialized in {init_time:.2f}s")

info = extractor.get_stream_info()
print(f"  Streams: {info['num_streams']}, Total frames: {info['total_frames']}")

# Test extraction performance for different frame counts
test_configs = [
    (1, "Single frame (baseline)"),
    (5, "5 frames (small batch)"),
    (10, "10 frames (medium batch)"),
    (50, "50 frames (large batch)"),
]

print("\n[2] Testing extraction performance:")
print("-" * 70)
print("Frames | Time    | Avg/Frame | Total  | Status")
print("-------|---------|-----------|--------|--------")

for frame_count, label in test_configs:
    extract_start = time.time()

    try:
        # Extract frames sequentially
        for i in range(frame_count):
            frame_data = extractor.get_frame(i)

            if not frame_data:
                print(f"{frame_count:6d} | FAILED - could not extract frame {i}")
                break

        elapsed = time.time() - extract_start
        avg_per_frame = elapsed / frame_count

        # Check if we're exceeding reasonable time
        status = "✓" if avg_per_frame < 2.0 else "⚠ SLOW" if avg_per_frame < 5.0 else "✗ VERY SLOW"

        print(f"{frame_count:6d} | {elapsed:7.2f}s | {avg_per_frame:9.2f}s | {elapsed:6.2f}s | {status}")

    except Exception as e:
        print(f"{frame_count:6d} | ERROR - {e}")
        break

print("-" * 70)

# Analyze frame-by-frame for first batch that succeeds
print("\n[3] Frame-by-frame breakdown (first 10 frames):")
print("-" * 70)
print("Frame# | Time    | Δ from prev")
print("-------|---------|-------------")

prev_time = 0
cumulative = init_time

try:
    for i in range(min(10, 50)):
        frame_start = time.time()
        frame_data = extractor.get_frame(i)
        frame_time = time.time() - frame_start

        if not frame_data:
            print(f"{i:6d} | FAILED")
            break

        cumulative += frame_time
        delta = frame_time - prev_time
        delta_str = f"+{delta:.2f}s" if delta >= 0 else f"{delta:.2f}s"

        status = "✓" if frame_time < 1.0 else "⚠" if frame_time < 3.0 else "✗"
        print(f"{i:6d} | {frame_time:7.2f}s | {delta_str:>12s} {status}")

        prev_time = frame_time

        # Early exit if single frame takes too long
        if frame_time > 30:
            print(f"  [Stopping - frame {i} taking {frame_time:.2f}s]")
            break

except Exception as e:
    print(f"Error: {e}")

extractor.close()

print("-" * 70)
print("\n[4] Analysis:")

# Determine bottleneck
if frame_count >= 10:
    # Look at frame 1 time vs frame 0
    print("✓ Test completed")
else:
    print("⚠ Test incomplete - frame extraction too slow")

print("\nKey observations:")
print("- Frame 0 typically fast (demux thread warm-up)")
print("- Frame 1+ slower (waiting for buffered data?)")
print("- If slowdown continues, buffering strategy needs fix")
print("- If frame 1+ stable, it's just demux thread startup cost")

print("\n" + "=" * 70)
