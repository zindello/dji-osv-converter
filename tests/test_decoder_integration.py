"""Integration test for decoder backend system

Tests the complete decoder architecture with frame_decoder integration.
"""

import logging
import tempfile
import time
from pathlib import Path

import pytest

from dji_osv_converter.frame_decoder import FrameExtractor, HEVCDecoder
from dji_osv_converter.decoder import (
    detect_available_decoders,
    create_best_decoder,
    create_decoder,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class TestDecoderIntegration:
    """Integration tests for decoder backend system"""

    @pytest.fixture(scope="module")
    def osv_file(self):
        """Find OSV file for testing"""
        test_files = list(Path(".").glob("**/*.osv"))
        if not test_files:
            pytest.skip("No OSV test file found")
        return test_files[0]

    def test_decoder_detection_integration(self):
        """Test decoder detection at high level"""
        available = detect_available_decoders()
        logger.info(f"Available decoders: {available}")

        # CPU should always be available
        assert "cpu" in available
        assert available["cpu"] is True

    def test_create_best_decoder_integration(self):
        """Test creating best decoder"""
        decoder = create_best_decoder()
        assert decoder is not None
        logger.info(f"Best decoder: {decoder.name}")

    def test_hevc_decoder_with_cpu_backend(self, osv_file):
        """Test HEVCDecoder using CPU backend"""
        decoder = HEVCDecoder(str(osv_file), use_decoder_backend="cpu")

        stream_count = decoder.get_video_stream_count()
        logger.info(f"Found {stream_count} video streams")
        assert stream_count > 0

        total_frames = decoder.get_total_frames()
        logger.info(f"Total frames: {total_frames}")
        assert total_frames > 0

        decoder.close()

    def test_hevc_decoder_with_auto_backend(self, osv_file):
        """Test HEVCDecoder with auto-detected backend"""
        decoder = HEVCDecoder(str(osv_file), use_decoder_backend=None)

        stream_count = decoder.get_video_stream_count()
        logger.info(f"Found {stream_count} video streams")
        assert stream_count > 0

        decoder.close()

    def test_hevc_decoder_legacy_mode(self, osv_file):
        """Test HEVCDecoder in legacy mode (no decoder backend)"""
        decoder = HEVCDecoder(str(osv_file), use_decoder_backend="")

        stream_count = decoder.get_video_stream_count()
        logger.info(f"Found {stream_count} video streams")
        assert stream_count > 0

        decoder.close()

    def test_frame_extractor_with_decoder_backend(self, osv_file):
        """Test FrameExtractor with decoder backend"""
        extractor = FrameExtractor(str(osv_file), use_decoder_backend="cpu")

        stream_info = extractor.get_stream_info()
        logger.info(f"Stream info: {stream_info}")

        assert stream_info["num_streams"] > 0
        assert stream_info["total_frames"] > 0

        # Get first frame from all streams
        frames = extractor.get_frame(0)
        assert frames is not None
        assert len(frames) == stream_info["num_streams"]

        for idx, frame_bytes in frames.items():
            assert frame_bytes is not None
            assert len(frame_bytes) > 0
            logger.info(f"Stream {idx}: {len(frame_bytes)} bytes")

        extractor.close()

    def test_frame_extractor_multiple_frames(self, osv_file):
        """Test extracting multiple frames with decoder backend"""
        extractor = FrameExtractor(str(osv_file), use_decoder_backend="cpu")

        frame_count = min(5, extractor.decoder.get_total_frames())
        logger.info(f"Extracting {frame_count} frames...")

        start_time = time.time()
        for frame_num in range(frame_count):
            frames = extractor.get_frame(frame_num)
            assert frames is not None
            assert len(frames) > 0

        elapsed = time.time() - start_time
        avg_time = elapsed / frame_count * 1000  # ms
        logger.info(f"Average extraction time: {avg_time:.1f}ms per frame")

        extractor.close()

    def test_decoder_backend_fallback(self, osv_file):
        """Test that decoder gracefully falls back if backend unavailable"""
        # Try non-existent backend - should fall back to PyAV
        decoder = HEVCDecoder(str(osv_file), use_decoder_backend="invalid")

        stream_count = decoder.get_video_stream_count()
        logger.info(f"Found {stream_count} video streams (fallback)")
        assert stream_count > 0

        decoder.close()

    def test_stream_metadata_consistency(self, osv_file):
        """Verify stream metadata is consistent across backends"""
        # Get info with CPU decoder
        cpu_decoder = HEVCDecoder(str(osv_file), use_decoder_backend="cpu")
        cpu_info = cpu_decoder.get_video_stream_info(0)
        cpu_decoder.close()

        # Get info with auto-detected backend
        auto_decoder = HEVCDecoder(str(osv_file), use_decoder_backend=None)
        auto_info = auto_decoder.get_video_stream_info(0)
        auto_decoder.close()

        # Compare
        assert cpu_info["width"] == auto_info["width"]
        assert cpu_info["height"] == auto_info["height"]
        logger.info(f"Metadata consistent: {cpu_info['width']}x{cpu_info['height']}")

    def test_frame_caching_with_decoder_backend(self, osv_file):
        """Test that FrameExtractor caching works with decoder backend"""
        extractor = FrameExtractor(str(osv_file), use_decoder_backend="cpu")

        # First access (decode)
        start1 = time.time()
        frames1 = extractor.get_frame(0)
        time1 = time.time() - start1

        # Second access (from cache)
        start2 = time.time()
        frames2 = extractor.get_frame(0)
        time2 = time.time() - start2

        logger.info(f"First access: {time1*1000:.1f}ms")
        logger.info(f"Cached access: {time2*1000:.1f}ms")

        # Should be identical
        assert frames1 == frames2

        extractor.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
