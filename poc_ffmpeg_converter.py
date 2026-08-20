#!/usr/bin/env python3
"""POC: FFmpeg hardware decode + Python stitching + OpenCV encode"""

import subprocess
import threading
import queue
import time
from pathlib import Path
import numpy as np
import cv2

input_file = Path("osv_example/CAM_20260819204151_0011_D.OSV")
output_file = Path("osv_example/poc_output.mp4")

print("=" * 70)
print("POC: FFmpeg Hardware Decode → Stitch → Encode")
print("=" * 70 + "\n")

# Parameters
STREAM_WIDTH = 3840
STREAM_HEIGHT = 3840
OUTPUT_WIDTH = 7680
OUTPUT_HEIGHT = 3840
FPS = 29.97
FRAME_SIZE = STREAM_WIDTH * STREAM_HEIGHT * 3

# Start FFmpeg with hardware acceleration
print("[1] Starting FFmpeg hardware decoder...")
cmd = [
    "ffmpeg",
    "-hwaccel", "videotoolbox",
    "-i", str(input_file),
    "-pix_fmt", "rgb24",
    "-f", "rawvideo",
    "-"
]

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    bufsize=1024*1024*10
)

# Frame queue for async reading
frame_queue = queue.Queue(maxsize=20)
stop_reading = False

def reader_thread_func():
    """Background thread reads frames from FFmpeg"""
    frame_num = 0
    try:
        while not stop_reading:
            # Read both streams (2 × frame_size)
            data = process.stdout.read(FRAME_SIZE * 2)

            if len(data) < FRAME_SIZE * 2:
                frame_queue.put(None)  # Signal EOF
                break

            frame_queue.put((frame_num, data))
            frame_num += 1

            if frame_num % 5 == 0:
                print(f"  [Decode] Frame {frame_num}")
    except Exception as e:
        print(f"  [Decode Error] {e}")
        frame_queue.put(None)

# Start reader thread
print("  ✓ Hardware decoder started")
reader = threading.Thread(target=reader_thread_func, daemon=True)
reader.start()

# Initialize encoder
print("\n[2] Initializing encoder...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(
    str(output_file),
    fourcc,
    FPS,
    (OUTPUT_WIDTH, OUTPUT_HEIGHT)
)
print(f"  ✓ Output: {output_file} ({OUTPUT_WIDTH}×{OUTPUT_HEIGHT} @ {FPS}fps)")

# Main processing loop
print("\n[3] Processing frames...")
print("Frame# | Decode | Stitch | Write | Total")
print("-------|--------|--------|-------|-------")

processed = 0
start_total = time.time()

try:
    for frame_num in range(50):  # Process 50 frames
        frame_start = time.time()

        # Get frame from decoder
        try:
            result = frame_queue.get(timeout=10.0)
        except queue.Empty:
            print(f"{frame_num:6d} | TIMEOUT")
            break

        if result is None:
            print(f"\nDecoding complete at frame {frame_num}")
            break

        _, frame_data = result
        decode_time = time.time() - frame_start

        # Parse frames (2 × 3840×3840×3)
        stitch_start = time.time()

        stream0 = np.frombuffer(frame_data[:FRAME_SIZE], dtype=np.uint8)
        stream0 = stream0.reshape((STREAM_HEIGHT, STREAM_WIDTH, 3))

        stream1 = np.frombuffer(frame_data[FRAME_SIZE:], dtype=np.uint8)
        stream1 = stream1.reshape((STREAM_HEIGHT, STREAM_WIDTH, 3))

        # Simple stitch: average the two streams (mock stitching)
        combined = np.average([stream0, stream1], axis=0).astype(np.uint8)

        # Create output frame (simple stitch: duplicate across width)
        output = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
        output[:, :STREAM_WIDTH] = stream0
        output[:, STREAM_WIDTH:] = stream1

        stitch_time = time.time() - stitch_start

        # Encode frame
        write_start = time.time()
        writer.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        write_time = time.time() - write_start

        total_time = time.time() - frame_start
        processed += 1

        print(f"{frame_num:6d} | {decode_time*1000:6.0f}ms | {stitch_time*1000:6.0f}ms | {write_time*1000:5.0f}ms | {total_time*1000:6.0f}ms")

except KeyboardInterrupt:
    print("\n✗ Interrupted")

# Cleanup
print("\n[4] Finalizing...")
stop_reading = True
reader.join(timeout=5)
process.terminate()
writer.release()

elapsed = time.time() - start_total

print("\n" + "=" * 70)
print(f"Results:")
print(f"  Frames processed: {processed}")
print(f"  Total time: {elapsed:.2f}s")
print(f"  Average per frame: {elapsed*1000/processed:.0f}ms")
print(f"  Output file: {output_file}")
if output_file.exists():
    size_mb = output_file.stat().st_size / 1e6
    print(f"  File size: {size_mb:.1f}MB")
print("=" * 70)

# Verify output
print("\n[5] Verifying output...")
result = subprocess.run(
    ["ffprobe", "-v", "quiet", "-show_format", "-show_streams", str(output_file)],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    print("  ✓ Output is valid MP4")
else:
    print("  ✗ Output verification failed")
