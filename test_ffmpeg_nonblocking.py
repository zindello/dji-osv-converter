#!/usr/bin/env python3
"""FFmpeg subprocess with proper async I/O - no pipe deadlocks"""

import subprocess
import threading
import time
import queue
from pathlib import Path
import numpy as np

input_file = Path("osv_example/CAM_20260819204151_0011_D.OSV")

print("=" * 70)
print("FFmpeg with Non-Blocking I/O Test")
print("=" * 70)

# Decode first 2 streams to raw RGB24 frames
# Using separate threads for reading to avoid deadlocks
cmd = [
    "ffmpeg",
    "-hwaccel", "videotoolbox",  # Hardware acceleration
    "-i", str(input_file),
    "-map", "0:v:0",  # First video stream
    "-map", "0:v:1",  # Second video stream
    "-pix_fmt", "rgb24",
    "-f", "rawvideo",
    "-"  # stdout
]

print(f"\nCommand: {' '.join(cmd)}\n")

# Frame size
WIDTH, HEIGHT = 3840, 3840
FRAME_SIZE = WIDTH * HEIGHT * 3  # RGB24

# Queue for frames
frame_queue = queue.Queue(maxsize=10)
error_queue = queue.Queue()

def read_frames():
    """Read frames from FFmpeg stdout in background thread"""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0  # Unbuffered
        )

        frame_count = 0
        start_time = time.time()

        while True:
            # Read one frame (2 streams × 1 frame each)
            frame_data = process.stdout.read(FRAME_SIZE * 2)

            if not frame_data:
                break

            if len(frame_data) == FRAME_SIZE * 2:
                frame_queue.put((frame_count, frame_data))
                frame_count += 1

                if frame_count % 5 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"  [Reader] Frame {frame_count} (fps: {fps:.1f})")

        process.wait()
        frame_queue.put(None)  # Signal end

    except Exception as e:
        error_queue.put(f"Read error: {e}")

# Start reader thread
print("Starting FFmpeg decoder...")
reader_thread = threading.Thread(target=read_frames, daemon=True)
reader_thread.start()

# Consume frames
print("\nReading frames:")
print("Frame# | Time")
print("-------|-------")

frame_num = 0
start_time = time.time()

try:
    while True:
        # Wait for frame (with timeout)
        try:
            result = frame_queue.get(timeout=5.0)
        except queue.Empty:
            print("✗ Timeout waiting for frame")
            break

        if result is None:
            print(f"\n✓ All frames read")
            break

        frame_num, data = result
        elapsed = time.time() - start_time

        print(f"{frame_num:6d} | {elapsed:7.2f}s")

        if frame_num >= 10:
            print(f"\n... (stopping after 10 frames)")
            break

        # Check for reader errors
        try:
            error = error_queue.get_nowait()
            print(f"✗ {error}")
            break
        except queue.Empty:
            pass

except KeyboardInterrupt:
    print("\n✗ Interrupted")

total_time = time.time() - start_time
print("\n" + "=" * 70)
print(f"Result: {frame_num} frames in {total_time:.2f}s")
if frame_num > 0:
    print(f"Average: {total_time*1000/frame_num:.0f}ms per frame")
print("=" * 70)
