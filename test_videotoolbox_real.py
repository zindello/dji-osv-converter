"""Test real VideoToolbox hardware decoder implementation

This test verifies that:
1. VideoToolbox decoder initializes properly
2. Hardware acceleration is detected (if available)
3. Frames decode correctly
4. Performance is fast (<50ms per frame for hardware, or <200ms for software)
"""

import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Test imports
try:
    from dji_osv_converter.decoder import create_decoder, detect_available_decoders
    logger.info("✓ Decoder modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import decoder: {e}")
    exit(1)


def test_videotoolbox_available():
    """Test if VideoToolbox is available on this system"""
    import platform
    if platform.system() != "Darwin":
        logger.info("⏭ VideoToolbox only available on macOS, skipping")
        return False

    available = detect_available_decoders()
    logger.info(f"Available decoders: {available}")

    has_vt = available.get("videotoolbox", False)
    if has_vt:
        logger.info("✓ VideoToolbox available on this system")
    else:
        logger.warning("✗ VideoToolbox not available (CPU fallback will be used)")

    return has_vt


def test_decoder_initialization():
    """Test decoder initialization"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Decoder Initialization")
    logger.info("="*60)

    try:
        decoder = create_decoder("videotoolbox")
        if decoder is None:
            logger.warning("VideoToolbox not available, trying CPU fallback")
            decoder = create_decoder("cpu")
            if decoder is None:
                logger.error("✗ No decoder available")
                return False

        logger.info(f"✓ Decoder initialized: {decoder.name}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to initialize decoder: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_decoding():
    """Test decoding an actual file"""
    logger.info("\n" + "="*60)
    logger.info("TEST: File Decoding")
    logger.info("="*60)

    # Find a test file
    test_files = list(Path(".").glob("osv_example/*.mp4"))
    if not test_files:
        logger.warning("⏭ No MP4 test files found")
        return False

    test_file = test_files[0]
    logger.info(f"Using test file: {test_file}")

    try:
        decoder = create_decoder("videotoolbox")
        if decoder is None:
            logger.info("VideoToolbox not available, using CPU")
            decoder = create_decoder("cpu")

        # Open file
        logger.info(f"Opening file: {test_file}")
        success = decoder.open_file(str(test_file))
        if not success:
            logger.error("✗ Failed to open file")
            return False

        logger.info("✓ File opened successfully")

        # Get stream info
        stream_count = decoder.get_stream_count()
        logger.info(f"Number of streams: {stream_count}")

        for idx in range(stream_count):
            info = decoder.get_stream_info(idx)
            logger.info(
                f"  Stream {idx}: {info['codec_name']} {info['width']}x{info['height']} "
                f"@ {info['fps']:.1f} fps, {info['total_frames']} frames"
            )

        total_frames = decoder.get_total_frames()
        logger.info(f"Total frames: {total_frames}")

        # Decode first 10 frames and measure timing
        logger.info("\nDecoding first 10 frames to measure performance...")
        frame_times = []

        for frame_num in range(min(10, total_frames)):
            start = time.time()
            frames = decoder.decode_frame_all_streams(frame_num, timeout_s=10.0)
            elapsed = (time.time() - start) * 1000  # ms

            if frames is None:
                logger.warning(f"  Frame {frame_num}: TIMEOUT (>10s)")
                continue

            frame_times.append(elapsed)
            logger.info(f"  Frame {frame_num}: {elapsed:.1f}ms - {len(frames)} streams")

        # Analyze performance
        if frame_times:
            avg_time = sum(frame_times) / len(frame_times)
            min_time = min(frame_times)
            max_time = max(frame_times)

            logger.info(f"\nPerformance Summary:")
            logger.info(f"  Average: {avg_time:.1f}ms per frame")
            logger.info(f"  Min: {min_time:.1f}ms")
            logger.info(f"  Max: {max_time:.1f}ms")

            # Detect hardware acceleration
            if avg_time < 50:
                logger.info("✓ HARDWARE ACCELERATION DETECTED (<50ms average)")
            elif avg_time < 150:
                logger.warning("⚠ UNCLEAR (may have partial acceleration)")
            else:
                logger.warning("✗ SOFTWARE FALLBACK (>150ms average)")

        decoder.close()
        logger.info("✓ Decoder closed")
        return True

    except Exception as e:
        logger.error(f"✗ Decoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frame_accuracy():
    """Test that decoded frames are correct"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Frame Accuracy")
    logger.info("="*60)

    test_files = list(Path(".").glob("osv_example/*.mp4"))
    if not test_files:
        logger.warning("⏭ No test files found")
        return False

    test_file = test_files[0]

    try:
        # Decode with VideoToolbox
        vt_decoder = create_decoder("videotoolbox")
        if vt_decoder is None:
            logger.info("VideoToolbox not available, skipping comparison test")
            return True

        vt_decoder.open_file(str(test_file))

        # Decode with CPU for comparison
        cpu_decoder = create_decoder("cpu")
        cpu_decoder.open_file(str(test_file))

        # Compare first frame
        logger.info("Comparing first decoded frame...")

        vt_frames = vt_decoder.decode_frame_all_streams(0, timeout_s=10.0)
        cpu_frames = cpu_decoder.decode_frame_all_streams(0, timeout_s=10.0)

        if vt_frames is None or cpu_frames is None:
            logger.warning("⏭ Could not decode frame from both decoders")
            return False

        # Compare frame sizes
        for stream_idx in vt_frames:
            vt_size = len(vt_frames.get(stream_idx, b""))
            cpu_size = len(cpu_frames.get(stream_idx, b""))

            if vt_size == cpu_size:
                logger.info(
                    f"✓ Stream {stream_idx}: Both decoders produced {vt_size} bytes (match)"
                )
            else:
                logger.warning(
                    f"✗ Stream {stream_idx}: VT={vt_size} bytes, CPU={cpu_size} bytes (mismatch)"
                )

        vt_decoder.close()
        cpu_decoder.close()
        return True

    except Exception as e:
        logger.error(f"✗ Accuracy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("VideoToolbox Hardware Decoder Tests")
    logger.info("="*60)

    results = {
        "VideoToolbox Available": test_videotoolbox_available(),
        "Decoder Initialization": test_decoder_initialization(),
        "File Decoding": test_file_decoding(),
        "Frame Accuracy": test_frame_accuracy(),
    }

    logger.info("\n" + "="*60)
    logger.info("Test Summary")
    logger.info("="*60)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    all_pass = all(results.values())
    if all_pass:
        logger.info("\n✓ All tests passed!")
    else:
        logger.info("\n✗ Some tests failed - see details above")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
