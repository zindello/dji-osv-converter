"""Debug single frame to find exact hang point"""

import sys
from pathlib import Path
import time

print("=" * 70)
print("Single Frame Debug Test")
print("=" * 70)

try:
    print("\n[1] Importing modules...")
    from dji_osv_converter.frame_decoder import FrameExtractor
    from dji_osv_converter.converter import StitchingPipeline
    import numpy as np
    print("✓ Imports done")

    input_file = Path("osv_example/CAM_20260819204151_0011_D.OSV")
    output_file = Path("osv_example/test_single_frame.mp4")

    print("\n[2] Initializing frame extractor...")
    start = time.time()
    extractor = FrameExtractor(input_file)
    print(f"✓ Extractor initialized in {time.time() - start:.2f}s")

    print("\n[3] Getting stream info...")
    info = extractor.get_stream_info()
    print(f"✓ {info['num_streams']} streams, {info['total_frames']} total frames")

    print("\n[4] Extracting frame 0...")
    start = time.time()
    frame_data = extractor.get_frame(0)
    elapsed = time.time() - start
    if frame_data:
        print(f"✓ Frame extracted in {elapsed:.2f}s")
        print(f"  Data: {list(frame_data.keys())}, sizes: {[len(v) for v in frame_data.values()]}")
    else:
        print(f"✗ Failed to extract frame")
        sys.exit(1)

    print("\n[5] Stitching single frame...")
    f0 = np.frombuffer(frame_data[0], dtype=np.uint8).reshape((3840, 3840, 3))
    f1 = np.frombuffer(frame_data[1], dtype=np.uint8).reshape((3840, 3840, 3))

    # Detect dtype
    print(f"  Frame dtype: {f0.dtype}")
    print(f"  Frame min/max: {f0.min()}/{f0.max()}")

    combined = np.average([f0, f1], axis=0).astype(np.uint8)
    output = np.zeros((3840, 7680, 3), dtype=np.uint8)
    output[:, :3840] = combined
    output[:, 3840:] = combined
    stitched_bytes = output.tobytes()
    print(f"✓ Stitched: {len(stitched_bytes)} bytes")

    print("\n[6] Starting FFmpeg encoder...")
    import subprocess

    encoder_cmd = [
        "ffmpeg", "-f", "rawvideo",
        "-video_size", "7680x3840",
        "-pixel_format", "rgb24",
        "-framerate", "29.97",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-b:v", "10M",
        "-pix_fmt", "yuv420p",
        "-y", str(output_file)
    ]

    start = time.time()
    proc = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✓ Encoder started (PID: {proc.pid})")

    print("\n[7] Writing frame to encoder...")
    start = time.time()
    try:
        proc.stdin.write(stitched_bytes)
        proc.stdin.close()
        print(f"✓ Frame written and stdin closed in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"✗ Write failed: {e}")
        proc.kill()
        sys.exit(1)

    print("\n[8] Waiting for encoder...")
    start = time.time()
    ret_code = proc.wait(timeout=30)
    elapsed = time.time() - start

    if ret_code == 0:
        print(f"✓ Encoder finished in {elapsed:.2f}s")
        if output_file.exists():
            size = output_file.stat().st_size
            print(f"✓ Output file: {size/1e6:.1f}MB")
        else:
            print(f"✗ Output file not found")
    else:
        print(f"✗ Encoder failed with code {ret_code}")

    extractor.close()
    print("\n" + "=" * 70)
    print("✓ Single frame test completed")
    print("=" * 70)

except KeyboardInterrupt:
    print("\n✗ Interrupted")
    sys.exit(1)
except Exception as e:
    elapsed = time.time() - start
    print(f"\n✗ Error at {elapsed:.1f}s: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
