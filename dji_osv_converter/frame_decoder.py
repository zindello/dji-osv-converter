"""Multi-stream HEVC frame extraction from OSV files

Handles synchronized decoding of all 8 camera streams in an OSV file.
Uses PyAV for efficient HEVC decoding with frame synchronization.
"""

import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import av

logger = logging.getLogger(__name__)


class FrameIndex:
    """Thread-safe frame index for synchronizing multi-stream decoding"""

    def __init__(self):
        self._frames: Dict[Tuple[int, int], bytes] = {}  # (stream_idx, frame_num) -> frame_data
        self._lock = threading.Lock()
        self._stream_frame_counts = defaultdict(int)  # stream_idx -> max frame number

    def add_frame(self, stream_idx: int, frame_num: int, frame_data: bytes) -> None:
        """Store a decoded frame"""
        with self._lock:
            self._frames[(stream_idx, frame_num)] = frame_data
            self._stream_frame_counts[stream_idx] = max(
                self._stream_frame_counts[stream_idx],
                frame_num + 1
            )

    def get_frame(self, stream_idx: int, frame_num: int) -> Optional[bytes]:
        """Retrieve a decoded frame"""
        with self._lock:
            return self._frames.get((stream_idx, frame_num))

    def has_all_streams_for_frame(self, frame_num: int, num_streams: int) -> bool:
        """Check if all streams have decoded a specific frame"""
        with self._lock:
            for stream_idx in range(num_streams):
                if (stream_idx, frame_num) not in self._frames:
                    return False
            return True

    def get_frame_tuple(
        self, frame_num: int, num_streams: int
    ) -> Optional[Dict[int, bytes]]:
        """Get all streams for a frame (blocks until all available)"""
        max_wait_ms = 10000
        elapsed_ms = 0
        poll_interval_ms = 100

        while elapsed_ms < max_wait_ms:
            if self.has_all_streams_for_frame(frame_num, num_streams):
                with self._lock:
                    return {i: self._frames[(i, frame_num)] for i in range(num_streams)}

            threading.Event().wait(poll_interval_ms / 1000.0)
            elapsed_ms += poll_interval_ms

        return None  # Timeout waiting for all streams

    def clear(self) -> None:
        """Clear all cached frames"""
        with self._lock:
            self._frames.clear()
            self._stream_frame_counts.clear()


class HEVCDecoder:
    """Decodes OSV file with multiple HEVC video streams"""

    def __init__(self, filepath: Path):
        """
        Initialize decoder for OSV file.

        Args:
            filepath: Path to .OSV file

        Raises:
            FileNotFoundError: If file doesn't exist
            av.ProcessError: If FFmpeg fails
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"OSV file not found: {filepath}")

        self._container: Optional[av.container.Container] = None
        self._video_streams = []
        self._frame_index = FrameIndex()
        self._total_frames = 0

        self._open_container()
        self._discover_video_streams()

    def _open_container(self) -> None:
        """Open the OSV file as FFmpeg container"""
        try:
            self._container = av.open(str(self.filepath))
            logger.debug(f"Opened container: {self.filepath}")
        except Exception as e:
            raise av.ProcessError(f"Failed to open {self.filepath}: {e}") from e

    def _discover_video_streams(self) -> None:
        """Discover all video streams in the container"""
        if not self._container:
            raise RuntimeError("Container not open")

        # Include only primary codec video streams (exclude thumbnails/mjpeg)
        self._video_streams = [
            stream for stream in self._container.streams
            if stream.type == "video" and stream.codec_context.name != "mjpeg"
        ]
        logger.info(f"Found {len(self._video_streams)} primary video streams")

        if len(self._video_streams) < 2:
            logger.warning(
                f"Expected at least 2 video streams for OSV, found {len(self._video_streams)}"
            )

    def get_video_stream_count(self) -> int:
        """Get number of video streams"""
        return len(self._video_streams)

    def get_video_stream_info(self, idx: int) -> dict:
        """Get information about a specific video stream"""
        if idx >= len(self._video_streams):
            raise IndexError(f"Stream index {idx} out of range")

        stream = self._video_streams[idx]

        # Calculate FPS from average_rate (can be Fraction)
        fps = 0.0
        if stream.average_rate:
            if isinstance(stream.average_rate, (int, float)):
                fps = float(stream.average_rate)
            else:
                # Handle Fraction or similar
                fps = float(stream.average_rate.numerator) / float(stream.average_rate.denominator)

        return {
            "index": stream.index,
            "codec_name": stream.codec_context.name,
            "width": stream.width,
            "height": stream.height,
            "fps": fps,
            "duration_seconds": float(stream.duration * stream.time_base) if stream.duration else 0,
            "frame_count": stream.frames,
        }

    def decode_frame_synchronously(self, frame_num: int, timeout_s: float = 10.0) -> Optional[Dict[int, bytes]]:
        """
        Decode a specific frame from all streams (synchronously).

        Seeks to frame_num in first stream, then decodes all streams until
        that frame is available from all cameras.

        Args:
            frame_num: Frame number to decode (0-indexed)
            timeout_s: Maximum wait time for all streams

        Returns:
            Dict mapping stream_idx -> RGB24 frame bytes, or None on timeout
        """
        if not self._container or not self._video_streams:
            raise RuntimeError("Decoder not initialized")

        # Seek to approximate position in first stream
        first_stream = self._video_streams[0]
        target_pts = int(frame_num * (first_stream.time_base.denominator /
                                      (first_stream.time_base.numerator * float(first_stream.average_rate))))

        try:
            self._container.seek(target_pts, stream=first_stream)
            logger.debug(f"Seeked to frame {frame_num}")
        except Exception as e:
            logger.warning(f"Seek failed: {e}, decoding linearly instead")

        # Decode until we have the target frame from all streams
        max_frames_to_decode = frame_num + 100  # Reasonable upper bound
        frames_decoded = 0

        for packet in self._container.demux(self._video_streams):
            for frame in packet.decode():
                stream_idx = self._video_streams.index(packet.stream)
                current_frame_num = frame.pts * packet.stream.time_base

                # Convert frame to RGB24 bytes
                rgb_frame = frame.reformat(format="rgb24")
                frame_bytes = rgb_frame.to_ndarray().tobytes()

                self._frame_index.add_frame(stream_idx, int(current_frame_num), frame_bytes)

                frames_decoded += 1

                # Check if we have all streams for target frame
                if self._frame_index.has_all_streams_for_frame(frame_num, len(self._video_streams)):
                    result = self._frame_index.get_frame_tuple(frame_num, len(self._video_streams))
                    logger.debug(f"Decoded frame {frame_num} from {len(self._video_streams)} streams")
                    return result

                # Safety limit
                if frames_decoded > max_frames_to_decode:
                    logger.warning(f"Gave up after {frames_decoded} frames")
                    return None

            # Early exit if we've found the frame
            if self._frame_index.has_all_streams_for_frame(frame_num, len(self._video_streams)):
                result = self._frame_index.get_frame_tuple(frame_num, len(self._video_streams))
                return result

        logger.warning(f"End of file reached without finding frame {frame_num}")
        return None

    def get_total_frames(self) -> int:
        """Get total frame count from first video stream"""
        if not self._video_streams:
            return 0

        first_stream = self._video_streams[0]
        if first_stream.frames:
            return first_stream.frames

        # If frame count not available, estimate from duration and fps
        if first_stream.duration and first_stream.time_base and first_stream.average_rate:
            duration_s = float(first_stream.duration * first_stream.time_base)
            fps = float(first_stream.average_rate)
            return int(duration_s * fps)

        return 0

    def close(self) -> None:
        """Close the container and clean up"""
        if self._container:
            self._container.close()
            self._container = None
        self._frame_index.clear()


class FrameExtractor:
    """High-level interface for extracting synchronized frames from OSV"""

    def __init__(self, osv_filepath: Path):
        """
        Initialize frame extractor for an OSV file.

        Args:
            osv_filepath: Path to .OSV file
        """
        self.decoder = HEVCDecoder(osv_filepath)
        self._frame_cache = {}

    def get_frame(self, frame_num: int) -> Optional[Dict[int, bytes]]:
        """
        Get a decoded frame from all camera streams.

        Args:
            frame_num: Frame number (0-indexed)

        Returns:
            Dict mapping stream_idx -> RGB24 frame bytes (3840×3840),
            or None if frame cannot be decoded
        """
        if frame_num in self._frame_cache:
            return self._frame_cache[frame_num]

        frames = self.decoder.decode_frame_synchronously(frame_num)
        if frames:
            self._frame_cache[frame_num] = frames

        return frames

    def get_stream_info(self) -> dict:
        """Get metadata about all video streams"""
        return {
            "num_streams": self.decoder.get_video_stream_count(),
            "total_frames": self.decoder.get_total_frames(),
            "streams": [
                self.decoder.get_video_stream_info(i)
                for i in range(self.decoder.get_video_stream_count())
            ],
        }

    def close(self) -> None:
        """Close the decoder and clean up"""
        self.decoder.close()


__all__ = [
    "HEVCDecoder",
    "FrameExtractor",
    "FrameIndex",
]
