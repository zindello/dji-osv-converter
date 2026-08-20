#!/usr/bin/env python3
"""Sequential decode with proper frame timing"""

import av
from pathlib import Path
import time

input_file = Path("osv_example/CAM_20260819204151_0011_D.OSV")

print("Sequential decode (baseline - no threads, no sync):\n")

container = av.open(str(input_file))
video_streams = [s for s in container.streams if s.type == "video" and s.codec_context.name != "mjpeg"]

print("Frame# | Packet decode | Color convert | Total")
print("-------|---------------|---------------|-------")

frame_count = 0
total_time = 0

for packet in container.demux(video_streams):
    # Time decode
    decode_start = time.time()
    frames = list(packet.decode())
    decode_time = time.time() - decode_start

    # Time color conversion
    for frame in frames:
        convert_start = time.time()
        rgb = frame.reformat(format="rgb24").to_ndarray()
        convert_time = time.time() - convert_start

        frame_count += 1
        total_time = time.time() - start if frame_count == 1 else total_time + decode_time + convert_time

        if frame_count <= 20:
            print(f"{frame_count:6d} | {decode_time*1000:13.0f}ms | {convert_time*1000:13.0f}ms | {(decode_time+convert_time)*1000:6.0f}ms")

        if frame_count >= 50:
            break

    if frame_count >= 50:
        break

start = time.time()
for packet in container.demux(video_streams):
    for frame in packet.decode():
        frame.reformat(format="rgb24").to_ndarray()

elapsed = time.time() - start
container.close()

print(f"\n50 frames sequential: {elapsed:.1f}s")
print(f"Average: {elapsed*1000/50:.0f}ms per frame")
