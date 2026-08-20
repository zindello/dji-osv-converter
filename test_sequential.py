#!/usr/bin/env python3
"""Sequential decode baseline"""

import av
from pathlib import Path
import time

input_file = Path("osv_example/CAM_20260819204151_0011_D.OSV")
container = av.open(str(input_file))
video_streams = [s for s in container.streams if s.type == "video" and s.codec_context.name != "mjpeg"]

print("Baseline: sequential decode (no threads, no sync):\n")

start = time.time()
frame_count = 0

for packet in container.demux(video_streams):
    for frame in packet.decode():
        if frame_count < 5:
            print(f"Frame {frame_count}: decoding...")

        rgb = frame.reformat(format="rgb24").to_ndarray()
        frame_count += 1

        if frame_count >= 50:
            break

    if frame_count >= 50:
        break

elapsed = time.time() - start
container.close()

print(f"\n50 frames sequential: {elapsed:.2f}s ({elapsed*1000/50:.0f}ms per frame)")
print("Expected: ~250ms per frame")
print(f"Actual speedup vs current: {4.5*50 / elapsed:.1f}x slower")
