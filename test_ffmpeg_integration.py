#!/usr/bin/env python3
"""Integration tests for FFmpeg decoder with full OSV conversion pipeline"""

import logging
import time
from pathlib import Path
import numpy as np
from dji_osv_converter.decoder import factory as decoder_factory
from dji_osv_converter.frame_decoder import FrameExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_ffmpeg_frame_extractor_integration():
    """Test FFmpeg decoder with FrameExtractor"""
    print("\n" + "="*70)
    print("TEST: FFmpeg Integration with FrameExtractor")
    print("="*70)

    osv_path = Path("osv_example/CAM_20260819204151_0011_D.OSV")
    if not osv_path.exists():
        print(f"✗ Test file not found: {osv_path}")
        return False

    try:
        # Create frame extractor with FFmpeg backend
        extractor = FrameExtractor(
            osv_path,
            use_decoder_backend="ffmpeg",
        )

        print(f"✓ FrameExtractor initialized with FFmpeg backend")
        print(f"  Total frames: {extractor.decoder.get_total_frames()}")
        print(f"  Video streams: {extractor.decoder.get_video_stream_count()}")

        # Extract first 10 frames
        print(f"\nExtracting 10 frames...")
        print("Frame# | Stream0 | Stream1 | Time")
        print("-------|---------|---------|-------")

        start_total = time.time()

        for frame_idx in range(10):
            frame_start = time.time()
            frames = extractor.get_frame(frame_idx)
            frame_time = time.time() - frame_start

            if frames is None or len(frames) != 2:
                print(f"{frame_idx:6d} | ERROR")
                return False

            frame0_size = len(frames[0]) if frames[0] else 0
            frame1_size = len(frames[1]) if frames[1] else 0

            print(f"{frame_idx:6d} | {frame0_size:7d} | {frame1_size:7d} | {frame_time*1000:6.0f}ms")

        total_time = time.time() - start_total
        avg_time = total_time / 10

        print(f"\n✓ All 10 frames extracted successfully")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average per frame: {avg_time*1000:.0f}ms")

        extractor.close()
        return True

    except Exception as e:
        logger.exception(f"Integration test failed: {e}")
        return False


def test_ffmpeg_frame_conversion_to_numpy():
    """Test that decoded frames can be converted to numpy arrays"""
    print("\n" + "="*70)
    print("TEST: Frame Conversion to NumPy")
    print("="*70)

    osv_path = Path("osv_example/CAM_20260819204151_0011_D.OSV")
    if not osv_path.exists():
        print(f"✗ Test file not found: {osv_path}")
        return False

    try:
        extractor = FrameExtractor(osv_path, use_decoder_backend="ffmpeg")

        for frame_idx in range(3):
            frames = extractor.get_frame(frame_idx)
            if frames is None or len(frames) != 2:
                print(f"  Frame {frame_idx}: ERROR getting frame")
                return False

            for stream_idx, frame_bytes in frames.items():
                if frame_bytes is None:
                    print(f"  Frame {frame_idx}, Stream {stream_idx}: None")
                    continue

                # Convert to numpy array
                arr = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((3840, 3840, 3))

                print(f"  Frame {frame_idx}, Stream {stream_idx}: "
                      f"shape={arr.shape}, dtype={arr.dtype}, "
                      f"min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")

                # Verify array is valid
                if arr.shape != (3840, 3840, 3):
                    print(f"    ERROR: Unexpected shape")
                    return False

                if arr.dtype != np.uint8:
                    print(f"    ERROR: Unexpected dtype")
                    return False

        extractor.close()
        print("\n✓ Frame conversion to NumPy working correctly")
        return True

    except Exception as e:
        logger.exception(f"NumPy conversion test failed: {e}")
        return False


def test_ffmpeg_vs_best_decoder():
    """Compare FFmpeg with the best available decoder"""
    print("\n" + "="*70)
    print("TEST: FFmpeg vs Best Available Decoder")
    print("="*70)

    osv_path = Path("osv_example/CAM_20260819204151_0011_D.OSV")
    if not osv_path.exists():
        print(f"✗ Test file not found: {osv_path}")
        return False

    try:
        # Test with auto-selected best decoder
        extractor_best = FrameExtractor(osv_path)
        print(f"Best decoder selected: {extractor_best.decoder}")

        # Test with FFmpeg
        extractor_ffmpeg = FrameExtractor(osv_path, use_decoder_backend="ffmpeg")
        print(f"FFmpeg decoder: {extractor_ffmpeg.decoder}")

        # Decode a frame from both
        print("\nDecoding frame 0 from both decoders...")

        start = time.time()
        frames_best = extractor_best.get_frame(0)
        best_time = time.time() - start

        start = time.time()
        frames_ffmpeg = extractor_ffmpeg.get_frame(0)
        ffmpeg_time = time.time() - start

        print(f"  Best decoder: {best_time*1000:.0f}ms")
        print(f"  FFmpeg: {ffmpeg_time*1000:.0f}ms")

        # Verify frames are identical
        if frames_best and frames_ffmpeg:
            for idx in range(len(frames_best)):
                if frames_best[idx] == frames_ffmpeg[idx]:
                    print(f"  Stream {idx}: ✓ Frames match")
                else:
                    print(f"  Stream {idx}: ✗ Frames differ!")
                    print(f"    Best: {len(frames_best[idx])} bytes")
                    print(f"    FFmpeg: {len(frames_ffmpeg[idx])} bytes")

        extractor_best.close()
        extractor_ffmpeg.close()

        print("\n✓ Comparison complete")
        return True

    except Exception as e:
        logger.exception(f"Comparison test failed: {e}")
        return False


def test_ffmpeg_full_pipeline_50frames():
    """Full pipeline test: decode 50 frames with FFmpeg"""
    print("\n" + "="*70)
    print("TEST: Full Pipeline with 50 Frames")
    print("="*70)

    osv_path = Path("osv_example/CAM_20260819204151_0011_D.OSV")
    if not osv_path.exists():
        print(f"✗ Test file not found: {osv_path}")
        return False

    try:
        extractor = FrameExtractor(osv_path, use_decoder_backend="ffmpeg")

        print(f"Processing 50 frames from FFmpeg decoder")
        print("Frame# | Time (ms)")
        print("-------|----------")

        start_total = time.time()
        success_count = 0

        for frame_idx in range(50):
            frame_start = time.time()
            frames = extractor.get_frame(frame_idx)
            frame_time = time.time() - frame_start

            if frames is None or len(frames) != 2:
                print(f"{frame_idx:6d} | TIMEOUT")
                break

            print(f"{frame_idx:6d} | {frame_time*1000:8.0f}")
            success_count += 1

        total_time = time.time() - start_total
        avg_time = total_time / success_count if success_count > 0 else 0

        print("\n" + "="*70)
        print(f"Results:")
        print(f"  Frames processed: {success_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average per frame: {avg_time*1000:.0f}ms")
        print(f"  Throughput: {success_count/total_time:.1f} frames/s")
        print("="*70)

        extractor.close()
        return success_count == 50

    except Exception as e:
        logger.exception(f"Full pipeline test failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FFmpeg Decoder Integration Test Suite")
    print("="*70)

    tests = [
        ("FrameExtractor Integration", test_ffmpeg_frame_extractor_integration),
        ("Frame to NumPy Conversion", test_ffmpeg_frame_conversion_to_numpy),
        ("FFmpeg vs Best Decoder", test_ffmpeg_vs_best_decoder),
        ("Full Pipeline (50 frames)", test_ffmpeg_full_pipeline_50frames),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.exception(f"Test {name} raised exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("Integration Test Summary")
    print("="*70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    print("="*70)
