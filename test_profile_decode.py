#!/usr/bin/env python3
"""Profile where time is spent in frame extraction"""

import av
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.WARNING)

input_file = Path("osv_example/CAM_20260819204151_0011_D.OSV")

print("Opening container...")
container = av.open(str(input_file))

video_streams = [s for s in container.streams if s.type == "video" and s.codec_context.name != "mjpeg"]
print(f"Found {len(video_streams)} video streams")

print("\nProfiling decode operation on first stream:")
print("=" * 70)

stream = video_streams[0]

# Decode first 5 frames and profile
packet_count = 0
frame_count = 0
decode_times = []
convert_times = []

for packet_num, packet in enumerate(container.demux(video_streams)):
    if packet.stream != stream:
        continue

    # Time the decode
    decode_start = time.time()
    frames = list(packet.decode())  # Force full decode
    decode_time = time.time() - decode_start

    # Time the color conversion for each frame
    for frame in frames:
        convert_start = time.time()
        rgb = frame.reformat(format="rgb24").to_ndarray()
        convert_time = time.time() - convert_start

        decode_times.append(decode_time)
        convert_times.append(convert_time)
        frame_count += 1

        if frame_count <= 5:
            print(f"Frame {frame_count}: decode={decode_time*1000:.1f}ms, convert={convert_time*1000:.1f}ms")

    packet_count += 1

    if frame_count >= 5:
        break

container.close()

print("\n" + "=" * 70)
print(f"Average per frame:")
print(f"  HEVC decode: {sum(decode_times)/len(decode_times)*1000:.1f}ms")
print(f"  Color convert: {sum(convert_times)/len(convert_times)*1000:.1f}ms")
print(f"  Total: {(sum(decode_times)+sum(convert_times))/len(decode_times)*1000:.1f}ms")

# The issue: we're doing BOTH decode + convert for each get_frame call
# And we're doing it on the main thread (now we parallelize with background threads)
