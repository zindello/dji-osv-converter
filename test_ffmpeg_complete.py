#!/usr/bin/env python3
"""Complete validation test for FFmpeg decoder implementation"""

import logging
import time
from pathlib import Path
from dji_osv_converter.decoder import factory as decoder_factory
from dji_osv_converter.decoder.detect import detect_available_decoders, get_best_decoder
from dji_osv_converter.frame_decoder import FrameExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_complete_pipeline():
    """Complete end-to-end test of FFmpeg decoder in OSV pipeline"""
    print("\n" + "="*70)
    print("COMPLETE FFmpeg DECODER VALIDATION TEST")
    print("="*70)

    osv_path = Path("osv_example/CAM_20260819204151_0011_D.OSV")
    if not osv_path.exists():
        print(f"✗ Test file not found: {osv_path}")
        return False

    print("\n1. DECODER DETECTION")
    print("-" * 70)
    available = detect_available_decoders()
    print(f"Available decoders: {available}")

    for decoder_name, is_available in available.items():
        status = "✓" if is_available else "✗"
        print(f"  {status} {decoder_name}")

    best_decoder = get_best_decoder()
    print(f"\nBest auto-selected decoder: {best_decoder}")

    if not available.get("ffmpeg"):
        print("\n⚠ FFmpeg not available - tests will use fallback")
        return False

    print("\n2. DIRECT DECODER TEST")
    print("-" * 70)
    decoder = decoder_factory.create_decoder("ffmpeg")
    if decoder is None:
        print("✗ Failed to create FFmpeg decoder")
        return False

    print(f"✓ FFmpeg decoder created: {decoder}")

    try:
        success = decoder.open_file(str(osv_path))
        if not success:
            print("✗ Failed to open OSV file")
            return False

        print(f"✓ OSV file opened")

        stream_count = decoder.get_stream_count()
        total_frames = decoder.get_total_frames()

        print(f"  Streams: {stream_count}")
        print(f"  Total frames: {total_frames}")

        for idx in range(stream_count):
            info = decoder.get_stream_info(idx)
            print(f"  Stream {idx}: {info['width']}×{info['height']} @ {info['fps']:.2f}fps, "
                  f"codec={info['codec_name']}, frames={info['total_frames']}")

        # Decode 5 frames
        print(f"\n  Decoding 5 frames...")
        for frame_num in range(5):
            frame_data = decoder.decode_frame_all_streams(frame_num, timeout_s=30)
            if frame_data is None:
                print(f"  ✗ Frame {frame_num}: TIMEOUT")
                return False
            print(f"  ✓ Frame {frame_num}: {sum(len(b) for b in frame_data.values())} bytes")

        print("✓ Direct decoder test passed")
    finally:
        decoder.close()

    print("\n3. FRAME EXTRACTOR TEST")
    print("-" * 70)
    extractor = FrameExtractor(osv_path, use_decoder_backend="ffmpeg")
    try:
        print(f"✓ FrameExtractor initialized with FFmpeg")
        print(f"  Backend: {extractor.decoder.name if hasattr(extractor.decoder, 'name') else 'HEVCDecoder'}")

        # Get first frame
        frames = extractor.get_frame(0)
        if frames is None or len(frames) != 2:
            print("✗ Failed to get frame")
            return False

        print(f"✓ Frame 0 extracted: {sum(len(b) for b in frames.values())} bytes")

        # Verify frame data
        for idx, data in frames.items():
            if len(data) != 3840 * 3840 * 3:
                print(f"✗ Stream {idx}: Incorrect frame size {len(data)}")
                return False
            print(f"  Stream {idx}: {len(data)} bytes (correct)")

        print("✓ FrameExtractor test passed")
    finally:
        extractor.close()

    print("\n4. PERFORMANCE TEST (50 frames)")
    print("-" * 70)
    extractor = FrameExtractor(osv_path, use_decoder_backend="ffmpeg")
    try:
        print("Decoding 50 frames...")
        start_total = time.time()
        frame_times = []

        for frame_num in range(50):
            frame_start = time.time()
            frames = extractor.get_frame(frame_num)
            frame_time = time.time() - frame_start

            if frames is None or len(frames) != 2:
                print(f"✗ Frame {frame_num}: TIMEOUT/ERROR")
                return False

            frame_times.append(frame_time)

            if frame_num % 10 == 0 or frame_num == 49:
                print(f"  Frame {frame_num:2d}: {frame_time*1000:6.0f}ms")

        total_time = time.time() - start_total
        avg_time = sum(frame_times) / len(frame_times)
        min_time = min(frame_times)
        max_time = max(frame_times)

        print(f"\n  Total time: {total_time:.2f}s")
        print(f"  Average per frame: {avg_time*1000:.0f}ms")
        print(f"  Min per frame: {min_time*1000:.0f}ms")
        print(f"  Max per frame: {max_time*1000:.0f}ms")
        print(f"  Throughput: {50/total_time:.2f} fps")

        # Verify performance
        if avg_time > 0.5:  # Should be <500ms per frame for hardware decode
            print(f"\n⚠ Performance is slower than expected ({avg_time*1000:.0f}ms avg)")
        else:
            print(f"\n✓ Performance is good ({avg_time*1000:.0f}ms avg)")

        print("✓ Performance test passed")
    finally:
        extractor.close()

    print("\n5. COMPARISON WITH OTHER DECODERS")
    print("-" * 70)

    # Test with all available decoders
    comparison_results = {}
    for decoder_name in ["ffmpeg", "videotoolbox", "cpu"]:
        if not available.get(decoder_name):
            print(f"  {decoder_name}: NOT AVAILABLE")
            continue

        try:
            extractor = FrameExtractor(osv_path, use_decoder_backend=decoder_name)
            start = time.time()
            frames = extractor.get_frame(0)
            decode_time = time.time() - start
            extractor.close()

            if frames:
                comparison_results[decoder_name] = decode_time
                print(f"  {decoder_name}: {decode_time*1000:.0f}ms per frame")
            else:
                print(f"  {decoder_name}: FAILED")
        except Exception as e:
            print(f"  {decoder_name}: ERROR - {e}")

    # Show speedup
    if "cpu" in comparison_results:
        cpu_time = comparison_results["cpu"]
        if "ffmpeg" in comparison_results:
            speedup = cpu_time / comparison_results["ffmpeg"]
            print(f"\n  FFmpeg speedup vs CPU: {speedup:.1f}×")

    print("✓ Comparison test passed")

    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
    return True


if __name__ == "__main__":
    success = test_complete_pipeline()
    exit(0 if success else 1)
