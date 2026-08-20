#!/usr/bin/env python3
"""Test: extract frames WITHOUT waiting for all streams to sync"""

import av
from pathlib import Path
import time
from collections import deque
import threading

input_file = Path("osv_example/CAM_20260819204151_0011_D.OSV")

print("Testing unsynchronized frame extraction:\n")

container = av.open(str(input_file))
video_streams = [s for s in container.streams if s.type == "video" and s.codec_context.name != "mjpeg"]

# Simple single-threaded decode (no sync overhead)
print(f"Sequential decode (no background threads, no sync):")
print("Frame# | Time")
print("-------|-------")

frame_count = 0
for packet in container.demux(video_streams):
    start = time.time()

    for frame in packet.decode():
        frame_time = time.time() - start
        frame_count += 1

        if frame_count <= 20:
            print(f"{frame_count:6d} | {frame_time*1000:6.0f}ms")

        if frame_count >= 50:
            break

    if frame_count >= 50:
        break

container.close()

elapsed = time.time() - start
print(f"\n50 frames took {elapsed:.1f}s")
print(f"Average: {elapsed*1000/50:.0f}ms per frame")
