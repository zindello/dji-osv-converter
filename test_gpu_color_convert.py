#!/usr/bin/env python3
"""Test GPU color conversion speed vs CPU"""

import numpy as np
import time
import av
from pathlib import Path

from dji_osv_converter.gpu import factory as gpu_factory

input_file = Path("osv_example/CAM_20260819204151_0011_D.OSV")

print("GPU Color Conversion Performance Test\n")

# Initialize GPU backend
backend = gpu_factory.create_best_backend()
print(f"GPU backend: {backend}")
if backend:
    print(f"Initialized: {backend.initialize()}")

# Get a sample frame
container = av.open(str(input_file))
video_streams = [s for s in container.streams if s.type == "video" and s.codec_context.name != "mjpeg"]

print("\nExtracting sample frame data...")

frame_bytes_list = []
for packet in container.demux(video_streams[:1]):
    for frame in packet.decode():
        frame_bytes_list.append(frame)
        if len(frame_bytes_list) >= 1:
            break
    if len(frame_bytes_list) >= 1:
        break

container.close()

print(f"Got {len(frame_bytes_list)} frames\n")

if frame_bytes_list:
    frame = frame_bytes_list[0]
    print(f"Frame: {frame.width}×{frame.height} {frame.format.name}")

    # Test CPU conversion
    print("\n--- CPU Conversion (PyAV reformat) ---")
    for i in range(3):
        start = time.time()
        rgb_frame = frame.reformat(format="rgb24")
        rgb_array = rgb_frame.to_ndarray()
        elapsed = time.time() - start
        print(f"  Attempt {i+1}: {elapsed*1000:.0f}ms")

    # Test GPU conversion
    if backend:
        print("\n--- GPU Conversion (Metal) ---")
        from dji_osv_converter.gpu.backend import GPUFrame, GPUColorSpace

        yuv_array = frame.to_ndarray()

        for i in range(3):
            start = time.time()
            gpu_frame = GPUFrame(
                gpu_handle=yuv_array,
                width=frame.width,
                height=frame.height,
                color_space=GPUColorSpace.YUV420P,
            )
            rgb_gpu = backend.convert_color_space(gpu_frame, GPUColorSpace.RGB24)
            elapsed = time.time() - start
            print(f"  Attempt {i+1}: {elapsed*1000:.0f}ms")

            if rgb_gpu is None:
                print(f"    ERROR: GPU conversion returned None!")
                break

print("\nTest complete")
