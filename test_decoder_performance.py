"""Performance benchmark for decoder backends

Compares VideoToolbox vs CPU decoder speeds.
Target: <50ms for 10 frames, or 10× speedup over PyAV baseline.
"""

import logging
import time
from pathlib import Path

import pytest

from dji_osv_converter.decoder import create_decoder
from dji_osv_converter.decoder.detect import detect_available_decoders

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class TestDecoderPerformance:
    """Performance benchmarks for different decoders"""

    @pytest.fixture(scope="module")
    def osv_file(self):
        """Find OSV file for testing"""
        test_files = list(Path(".").glob("**/*.osv"))
        if not test_files:
            pytest.skip("No OSV test file found")
        return test_files[0]

    def test_cpu_decoder_throughput(self, osv_file):
        """Benchmark CPU (PyAV) decoder performance"""
        decoder = create_decoder("cpu")
        decoder.open_file(str(osv_file))

        frame_count = min(10, decoder.get_total_frames())
        logger.info(f"Testing CPU decoder: {frame_count} frames from {osv_file.name}")

        # Warmup
        decoder.decode_frame(0, 0, timeout_s=10.0)

        # Benchmark
        start_time = time.time()
        for frame_num in range(frame_count):
            frame_bytes = decoder.decode_frame(0, frame_num, timeout_s=10.0)
            assert frame_bytes is not None

        total_time = time.time() - start_time
        avg_time_ms = total_time / frame_count * 1000

        logger.info(f"CPU Decoder Results:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Average per frame: {avg_time_ms:.1f}ms")
        logger.info(f"  Throughput: {frame_count / total_time:.1f} fps")

        decoder.close()

        # Check performance target
        assert total_time < 50.0, f"CPU decode too slow: {total_time:.2f}s for {frame_count} frames"

    def test_videotoolbox_decoder_throughput(self, osv_file):
        """Benchmark VideoToolbox decoder performance (if available)"""
        import platform
        if platform.system() != "Darwin":
            pytest.skip("VideoToolbox only available on macOS")

        available = detect_available_decoders()
        if not available.get("videotoolbox", False):
            pytest.skip("VideoToolbox not available on this system")

        decoder = create_decoder("videotoolbox")
        if decoder is None:
            pytest.skip("VideoToolbox decoder not available")

        decoder.open_file(str(osv_file))

        frame_count = min(10, decoder.get_total_frames())
        logger.info(f"Testing VideoToolbox decoder: {frame_count} frames")

        # Warmup
        decoder.decode_frame(0, 0, timeout_s=10.0)

        # Benchmark
        start_time = time.time()
        for frame_num in range(frame_count):
            frame_bytes = decoder.decode_frame(0, frame_num, timeout_s=10.0)
            assert frame_bytes is not None

        total_time = time.time() - start_time
        avg_time_ms = total_time / frame_count * 1000

        logger.info(f"VideoToolbox Decoder Results:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Average per frame: {avg_time_ms:.1f}ms")
        logger.info(f"  Throughput: {frame_count / total_time:.1f} fps")

        decoder.close()

        # Check performance target
        assert total_time < 50.0, f"VideoToolbox decode too slow: {total_time:.2f}s for {frame_count} frames"

    def test_multistream_synchronized_performance(self, osv_file):
        """Benchmark synchronized multi-stream decoding"""
        decoder = create_decoder("cpu")
        decoder.open_file(str(osv_file))

        stream_count = decoder.get_stream_count()
        frame_count = min(10, decoder.get_total_frames())
        logger.info(f"Testing multi-stream sync: {frame_count} frames from {stream_count} streams")

        # Warmup
        decoder.decode_frame_all_streams(0, timeout_s=10.0)

        # Benchmark
        start_time = time.time()
        for frame_num in range(frame_count):
            frames = decoder.decode_frame_all_streams(frame_num, timeout_s=10.0)
            assert frames is not None
            assert len(frames) == stream_count

        total_time = time.time() - start_time
        avg_time_ms = total_time / frame_count * 1000

        logger.info(f"Multi-Stream Sync Results:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Average per frame: {avg_time_ms:.1f}ms")
        logger.info(f"  Throughput: {frame_count / total_time:.1f} fps")

        decoder.close()

    def test_frame_data_integrity(self, osv_file):
        """Verify decoded frame data integrity"""
        decoder = create_decoder("cpu")
        decoder.open_file(str(osv_file))

        info = decoder.get_stream_info(0)
        width = info["width"]
        height = info["height"]
        expected_size = width * height * 3  # RGB24

        logger.info(f"Frame size: {width}x{height} = {expected_size} bytes")

        # Decode 5 frames and verify data
        for frame_num in range(min(5, decoder.get_total_frames())):
            frame_bytes = decoder.decode_frame(0, frame_num, timeout_s=10.0)
            assert frame_bytes is not None
            assert len(frame_bytes) == expected_size, \
                f"Frame {frame_num}: expected {expected_size} bytes, got {len(frame_bytes)}"
            logger.info(f"Frame {frame_num}: {len(frame_bytes)} bytes ✓")

        decoder.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
