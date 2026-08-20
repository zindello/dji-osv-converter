"""Unit tests for VideoToolbox decoder backend"""

import logging
import tempfile
import time
from pathlib import Path

import pytest

from dji_osv_converter.decoder import create_decoder
from dji_osv_converter.decoder.detect import detect_available_decoders

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class TestVideoToolboxDetection:
    """Test decoder availability detection"""

    def test_detect_decoders(self):
        """Check what decoders are available"""
        available = detect_available_decoders()
        logger.info(f"Available decoders: {available}")
        assert "cpu" in available
        assert available["cpu"] is True

    def test_videotoolbox_availability(self):
        """Check if VideoToolbox is available on this system"""
        import platform
        if platform.system() != "Darwin":
            pytest.skip("VideoToolbox only available on macOS")

        available = detect_available_decoders()
        logger.info(f"VideoToolbox available: {available.get('videotoolbox', False)}")


class TestVideoToolboxDecoderCreation:
    """Test decoder initialization"""

    def test_create_cpu_decoder(self):
        """Create CPU decoder"""
        decoder = create_decoder("cpu")
        assert decoder is not None
        assert decoder.name == "cpu"
        assert decoder.available is True
        logger.info(f"Created decoder: {decoder}")

    def test_create_videotoolbox_decoder(self):
        """Create VideoToolbox decoder (if available)"""
        import platform
        if platform.system() != "Darwin":
            pytest.skip("VideoToolbox only available on macOS")

        available = detect_available_decoders()
        if not available.get("videotoolbox", False):
            pytest.skip("VideoToolbox not available on this system")

        decoder = create_decoder("videotoolbox")
        assert decoder is not None
        assert decoder.name == "videotoolbox"
        logger.info(f"Created decoder: {decoder}")


class TestDecoderWithOSV:
    """Test decoder with actual OSV file"""

    @pytest.fixture(scope="module")
    def osv_file(self):
        """Find OSV file for testing"""
        # Look for test data
        test_files = list(Path(".").glob("**/*.osv"))
        if not test_files:
            pytest.skip("No OSV test file found")
        return test_files[0]

    def test_open_file(self, osv_file):
        """Test opening an OSV file"""
        decoder = create_decoder("cpu")
        assert decoder is not None

        success = decoder.open_file(str(osv_file))
        assert success is True

        stream_count = decoder.get_stream_count()
        logger.info(f"Found {stream_count} video streams")
        assert stream_count > 0

        total_frames = decoder.get_total_frames()
        logger.info(f"Total frames: {total_frames}")
        assert total_frames > 0

        decoder.close()

    def test_get_stream_info(self, osv_file):
        """Test getting stream metadata"""
        decoder = create_decoder("cpu")
        decoder.open_file(str(osv_file))

        for idx in range(decoder.get_stream_count()):
            info = decoder.get_stream_info(idx)
            logger.info(f"Stream {idx}: {info}")

            assert "width" in info
            assert "height" in info
            assert "fps" in info
            assert "codec_name" in info
            assert "total_frames" in info

        decoder.close()

    def test_decode_single_frame(self, osv_file):
        """Test decoding a single frame"""
        decoder = create_decoder("cpu")
        decoder.open_file(str(osv_file))

        frame_bytes = decoder.decode_frame(0, 0, timeout_s=10.0)
        logger.info(f"Decoded frame: {type(frame_bytes)}, size: {len(frame_bytes) if frame_bytes else 0}")

        assert frame_bytes is not None
        assert len(frame_bytes) > 0

        decoder.close()

    def test_decode_multiple_frames_sequential(self, osv_file):
        """Test decoding multiple frames sequentially"""
        decoder = create_decoder("cpu")
        decoder.open_file(str(osv_file))

        frame_count = min(5, decoder.get_total_frames())
        logger.info(f"Decoding {frame_count} frames...")

        start_time = time.time()
        for frame_num in range(frame_count):
            frame_bytes = decoder.decode_frame(0, frame_num, timeout_s=10.0)
            assert frame_bytes is not None, f"Frame {frame_num} returned None"

        elapsed = time.time() - start_time
        avg_time = elapsed / frame_count * 1000  # ms
        logger.info(f"Average decode time: {avg_time:.1f}ms per frame")

        decoder.close()

    def test_decode_all_streams_synchronized(self, osv_file):
        """Test decoding from all streams synchronously"""
        decoder = create_decoder("cpu")
        decoder.open_file(str(osv_file))

        stream_count = decoder.get_stream_count()
        logger.info(f"Decoding from {stream_count} streams...")

        frames = decoder.decode_frame_all_streams(0, timeout_s=10.0)
        assert frames is not None
        assert len(frames) == stream_count

        for idx, frame_bytes in frames.items():
            assert frame_bytes is not None
            assert len(frame_bytes) > 0
            logger.info(f"Stream {idx}: {len(frame_bytes)} bytes")

        decoder.close()

    def test_decoder_context_manager(self, osv_file):
        """Test using decoder as context manager"""
        decoder = create_decoder("cpu")

        with decoder:
            success = decoder.open_file(str(osv_file))
            assert success is True

            stream_count = decoder.get_stream_count()
            assert stream_count > 0

        # Should be closed after context


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
