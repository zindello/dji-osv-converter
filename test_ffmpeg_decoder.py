#!/usr/bin/env python3
"""Test suite for FFmpeg hardware-accelerated decoder"""

import logging
import time
from pathlib import Path
from dji_osv_converter.decoder import factory as decoder_factory
from dji_osv_converter.decoder.detect import detect_available_decoders

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_ffmpeg_availability():
    """Test that FFmpeg is detected"""
    print("\n" + "="*70)
    print("TEST: FFmpeg Availability Detection")
    print("="*70)

    available = detect_available_decoders()
    print(f"Available decoders: {available}")

    if not available.get("ffmpeg"):
        print("WARNING: FFmpeg not available on this system")
        return False

    print("✓ FFmpeg detected")
    return True


def test_ffmpeg_decoder_initialization():
    """Test FFmpeg decoder initialization"""
    print("\n" + "="*70)
    print("TEST: FFmpeg Decoder Initialization")
    print("="*70)

    decoder = decoder_factory.create_decoder("ffmpeg")
    if decoder is None:
        print("✗ Failed to create FFmpeg decoder")
        return False

    print(f"✓ Decoder created: {decoder}")
    decoder.close()
    return True


def test_ffmpeg_file_open():
    """Test opening an OSV file with FFmpeg decoder"""
    print("\n" + "="*70)
    print("TEST: Open OSV File")
    print("="*70)

    # Find test OSV file
    osv_path = Path("osv_example/CAM_20260819204151_0011_D.OSV")
    if not osv_path.exists():
        print(f"✗ Test file not found: {osv_path}")
        return False

    decoder = decoder_factory.create_decoder("ffmpeg")
    if decoder is None:
        print("✗ Failed to create decoder")
        return False

    try:
        success = decoder.open_file(str(osv_path))
        if not success:
            print("✗ Failed to open file")
            return False

        stream_count = decoder.get_stream_count()
        total_frames = decoder.get_total_frames()

        print(f"✓ File opened successfully")
        print(f"  Streams: {stream_count}")
        print(f"  Total frames: {total_frames}")

        for idx in range(stream_count):
            info = decoder.get_stream_info(idx)
            print(f"  Stream {idx}: {info['width']}×{info['height']} @ {info['fps']:.2f}fps")

        return True
    finally:
        decoder.close()


def test_ffmpeg_frame_decoding():
    """Test basic frame decoding"""
    print("\n" + "="*70)
    print("TEST: Frame Decoding (5 frames)")
    print("="*70)

    osv_path = Path("osv_example/CAM_20260819204151_0011_D.OSV")
    if not osv_path.exists():
        print(f"✗ Test file not found: {osv_path}")
        return False

    decoder = decoder_factory.create_decoder("ffmpeg")
    if decoder is None:
        print("✗ Failed to create decoder")
        return False

    try:
        if not decoder.open_file(str(osv_path)):
            print("✗ Failed to open file")
            return False

        stream_count = decoder.get_stream_count()
        print(f"Decoding {5} frames from {stream_count} streams...")

        start_total = time.time()
        success_count = 0

        for frame_num in range(5):
            frame_start = time.time()
            frame_data = decoder.decode_frame_all_streams(frame_num, timeout_s=30)
            frame_time = time.time() - frame_start

            if frame_data is None:
                print(f"  Frame {frame_num}: TIMEOUT")
                continue

            total_bytes = sum(len(data) for data in frame_data.values())
            print(f"  Frame {frame_num}: {frame_time*1000:.0f}ms, {total_bytes} bytes")
            success_count += 1

        total_time = time.time() - start_total
        avg_time = total_time / success_count if success_count > 0 else 0

        print(f"\n✓ Decoded {success_count}/{5} frames")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average per frame: {avg_time*1000:.0f}ms")

        return success_count == 5
    finally:
        decoder.close()


def test_ffmpeg_performance_50frames():
    """Test performance over 50 frames (matches POC)"""
    print("\n" + "="*70)
    print("TEST: Performance (50 frames)")
    print("="*70)

    osv_path = Path("osv_example/CAM_20260819204151_0011_D.OSV")
    if not osv_path.exists():
        print(f"✗ Test file not found: {osv_path}")
        return False

    decoder = decoder_factory.create_decoder("ffmpeg")
    if decoder is None:
        print("✗ Failed to create decoder")
        return False

    try:
        if not decoder.open_file(str(osv_path)):
            print("✗ Failed to open file")
            return False

        print("Frame# | Decode Time | Total Bytes")
        print("-------|-------------|------------")

        start_total = time.time()
        success_count = 0
        total_bytes_decoded = 0

        for frame_num in range(50):
            frame_start = time.time()
            frame_data = decoder.decode_frame_all_streams(frame_num, timeout_s=30)
            frame_time = time.time() - frame_start

            if frame_data is None:
                print(f"{frame_num:6d} | TIMEOUT")
                break

            total_bytes = sum(len(data) for data in frame_data.values())
            total_bytes_decoded += total_bytes
            print(f"{frame_num:6d} | {frame_time*1000:6.0f}ms | {total_bytes:10,d}")
            success_count += 1

        total_time = time.time() - start_total
        mb_decoded = total_bytes_decoded / 1e6
        mb_per_sec = mb_decoded / total_time if total_time > 0 else 0

        print("\n" + "="*70)
        print(f"Results:")
        print(f"  Frames decoded: {success_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average per frame: {total_time*1000/success_count:.0f}ms" if success_count > 0 else "")
        print(f"  Data decoded: {mb_decoded:.1f}MB")
        print(f"  Throughput: {mb_per_sec:.1f}MB/s")
        print("="*70)

        # Success if we decoded all 50 frames
        return success_count == 50

    finally:
        decoder.close()


def test_ffmpeg_sync_across_streams():
    """Test that frames are synchronized across streams"""
    print("\n" + "="*70)
    print("TEST: Stream Synchronization (10 frames)")
    print("="*70)

    osv_path = Path("osv_example/CAM_20260819204151_0011_D.OSV")
    if not osv_path.exists():
        print(f"✗ Test file not found: {osv_path}")
        return False

    decoder = decoder_factory.create_decoder("ffmpeg")
    if decoder is None:
        print("✗ Failed to create decoder")
        return False

    try:
        if not decoder.open_file(str(osv_path)):
            print("✗ Failed to open file")
            return False

        stream_count = decoder.get_stream_count()
        print(f"Testing sync across {stream_count} streams over 10 frames")

        for frame_num in range(10):
            frame_data = decoder.decode_frame_all_streams(frame_num, timeout_s=30)
            if frame_data is None:
                print(f"  Frame {frame_num}: TIMEOUT")
                return False

            # Check all streams present
            if len(frame_data) != stream_count:
                print(f"  Frame {frame_num}: Expected {stream_count} streams, got {len(frame_data)}")
                return False

            # Check all streams have correct size
            expected_size = 3840 * 3840 * 3
            for stream_idx, data in frame_data.items():
                if len(data) != expected_size:
                    print(f"  Frame {frame_num}, Stream {stream_idx}: "
                          f"Expected {expected_size} bytes, got {len(data)}")
                    return False

            print(f"  Frame {frame_num}: ✓ All {stream_count} streams synchronized")

        print("\n✓ Stream synchronization verified")
        return True
    finally:
        decoder.close()


def test_ffmpeg_individual_stream_access():
    """Test accessing individual streams"""
    print("\n" + "="*70)
    print("TEST: Individual Stream Access")
    print("="*70)

    osv_path = Path("osv_example/CAM_20260819204151_0011_D.OSV")
    if not osv_path.exists():
        print(f"✗ Test file not found: {osv_path}")
        return False

    decoder = decoder_factory.create_decoder("ffmpeg")
    if decoder is None:
        print("✗ Failed to create decoder")
        return False

    try:
        if not decoder.open_file(str(osv_path)):
            print("✗ Failed to open file")
            return False

        stream_count = decoder.get_stream_count()

        for frame_num in range(3):
            print(f"Frame {frame_num}:")
            for stream_idx in range(stream_count):
                frame_data = decoder.decode_frame(stream_idx, frame_num, timeout_s=30)
                if frame_data is None:
                    print(f"  Stream {stream_idx}: TIMEOUT")
                    return False

                print(f"  Stream {stream_idx}: {len(frame_data)} bytes")

        print("\n✓ Individual stream access working")
        return True
    finally:
        decoder.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FFmpeg Decoder Test Suite")
    print("="*70)

    tests = [
        ("FFmpeg Availability", test_ffmpeg_availability),
        ("Initialization", test_ffmpeg_decoder_initialization),
        ("File Open", test_ffmpeg_file_open),
        ("Frame Decoding", test_ffmpeg_frame_decoding),
        ("Stream Sync", test_ffmpeg_sync_across_streams),
        ("Individual Stream Access", test_ffmpeg_individual_stream_access),
        ("Performance (50 frames)", test_ffmpeg_performance_50frames),
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
    print("Test Summary")
    print("="*70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    print("="*70)
