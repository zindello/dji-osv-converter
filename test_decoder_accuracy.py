"""Accuracy tests for decoder backends

Verifies that different decoders produce identical (or visually similar) frames.
Compares VideoToolbox vs CPU (PyAV) baseline.
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dji_osv_converter.decoder import create_decoder
from dji_osv_converter.decoder.detect import detect_available_decoders

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class TestDecoderAccuracy:
    """Verify decoder output correctness"""

    @pytest.fixture(scope="module")
    def osv_file(self):
        """Find OSV file for testing"""
        test_files = list(Path(".").glob("**/*.osv"))
        if not test_files:
            pytest.skip("No OSV test file found")
        return test_files[0]

    def test_cpu_decoder_frame_validity(self, osv_file):
        """Verify CPU decoder produces valid RGB24 frames"""
        decoder = create_decoder("cpu")
        decoder.open_file(str(osv_file))

        info = decoder.get_stream_info(0)
        width = info["width"]
        height = info["height"]

        frame_bytes = decoder.decode_frame(0, 0, timeout_s=10.0)
        assert frame_bytes is not None

        # Convert to numpy and verify
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
        logger.info(f"Decoded frame shape: {frame_array.shape}")
        logger.info(f"Dtype: {frame_array.dtype}")
        logger.info(f"Min/Max values: {frame_array.min()}-{frame_array.max()}")

        # Verify it's valid RGB24
        assert frame_array.shape == (height, width, 3)
        assert frame_array.dtype == np.uint8
        assert frame_array.min() >= 0
        assert frame_array.max() <= 255

        decoder.close()

    def test_videotoolbox_vs_cpu_consistency(self, osv_file):
        """Compare VideoToolbox vs CPU decoder outputs"""
        import platform
        if platform.system() != "Darwin":
            pytest.skip("VideoToolbox only available on macOS")

        available = detect_available_decoders()
        if not available.get("videotoolbox", False):
            pytest.skip("VideoToolbox not available on this system")

        # Decode with VideoToolbox
        vt_decoder = create_decoder("videotoolbox")
        if vt_decoder is None:
            pytest.skip("VideoToolbox decoder not available")

        vt_decoder.open_file(str(osv_file))
        vt_frame = vt_decoder.decode_frame(0, 0, timeout_s=10.0)
        vt_decoder.close()

        # Decode with CPU (reference)
        cpu_decoder = create_decoder("cpu")
        cpu_decoder.open_file(str(osv_file))
        cpu_frame = cpu_decoder.decode_frame(0, 0, timeout_s=10.0)
        cpu_decoder.close()

        # Compare
        assert vt_frame is not None, "VideoToolbox frame is None"
        assert cpu_frame is not None, "CPU frame is None"

        info = cpu_decoder.get_stream_info(0)
        width = info["width"]
        height = info["height"]

        vt_array = np.frombuffer(vt_frame, dtype=np.uint8).reshape(height, width, 3)
        cpu_array = np.frombuffer(cpu_frame, dtype=np.uint8).reshape(height, width, 3)

        # Frames might differ slightly due to different decode paths,
        # but should be very similar (within some threshold)
        diff = np.abs(vt_array.astype(np.int32) - cpu_array.astype(np.int32))
        max_diff = diff.max()
        mean_diff = diff.mean()

        logger.info(f"Frame comparison:")
        logger.info(f"  Max difference: {max_diff}")
        logger.info(f"  Mean difference: {mean_diff:.2f}")

        # Allow small differences due to rounding/interpolation differences
        # VideoToolbox might produce slightly different results than software decode
        assert max_diff < 10, f"Frames differ significantly: max_diff={max_diff}"
        assert mean_diff < 1.0, f"Frames differ on average: mean_diff={mean_diff:.2f}"

    def test_all_streams_decode_same_structure(self, osv_file):
        """Verify all streams produce consistent frame structure"""
        decoder = create_decoder("cpu")
        decoder.open_file(str(osv_file))

        stream_count = decoder.get_stream_count()
        logger.info(f"Verifying {stream_count} streams have consistent frame structure...")

        frame_info = {}
        for stream_idx in range(stream_count):
            info = decoder.get_stream_info(stream_idx)
            frame_info[stream_idx] = (info["width"], info["height"])
            logger.info(f"Stream {stream_idx}: {info['width']}x{info['height']}")

        # All streams should have same dimensions
        dims = list(frame_info.values())
        assert all(d == dims[0] for d in dims), "Streams have different dimensions"

        # Decode a frame from each stream
        for stream_idx in range(stream_count):
            frame_bytes = decoder.decode_frame(stream_idx, 0, timeout_s=10.0)
            assert frame_bytes is not None
            width, height = frame_info[stream_idx]
            expected_size = width * height * 3
            assert len(frame_bytes) == expected_size, \
                f"Stream {stream_idx}: expected {expected_size} bytes, got {len(frame_bytes)}"

        decoder.close()

    def test_sequential_frame_data_changes(self, osv_file):
        """Verify that sequential frames contain different data"""
        decoder = create_decoder("cpu")
        decoder.open_file(str(osv_file))

        info = decoder.get_stream_info(0)
        width = info["width"]
        height = info["height"]

        frame_0 = np.frombuffer(decoder.decode_frame(0, 0, timeout_s=10.0), dtype=np.uint8).reshape(height, width, 3)
        frame_1 = np.frombuffer(decoder.decode_frame(0, 1, timeout_s=10.0), dtype=np.uint8).reshape(height, width, 3)

        # Frames should be different
        diff = np.abs(frame_0.astype(np.int32) - frame_1.astype(np.int32))
        mean_diff = diff.mean()

        logger.info(f"Frame 0 vs Frame 1 mean difference: {mean_diff:.2f}")
        assert mean_diff > 0.1, "Consecutive frames appear identical (possible decode error)"

        decoder.close()

    def test_frame_statistics(self, osv_file):
        """Log frame statistics for quality assessment"""
        decoder = create_decoder("cpu")
        decoder.open_file(str(osv_file))

        info = decoder.get_stream_info(0)
        width = info["width"]
        height = info["height"]

        frame_bytes = decoder.decode_frame(0, 0, timeout_s=10.0)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)

        logger.info(f"Frame statistics:")
        logger.info(f"  Mean: {frame_array.mean():.1f}")
        logger.info(f"  Std: {frame_array.std():.1f}")
        logger.info(f"  Min: {frame_array.min()}")
        logger.info(f"  Max: {frame_array.max()}")
        logger.info(f"  Median: {np.median(frame_array):.1f}")

        decoder.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
