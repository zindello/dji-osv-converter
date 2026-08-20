"""Multi-stream HEVC frame extraction with PTS-based synchronization

Uses packet-level buffering to synchronize desynchronized streams.
No seeking, no redundant decoding - just stream-aligned buffering.
"""

import logging
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional, Tuple
import av

logger = logging.getLogger(__name__)


class StreamBuffer:
    """Per-stream packet buffer with frame extraction"""

    def __init__(self, stream_idx: int, stream):
        self.stream_idx = stream_idx
        self.stream = stream
        self.packets = deque()
        self.decoded_frames = {}  # frame_num -> bytes
        self.max_frame_seen = -1
        self.lock = threading.Lock()

    def add_packet(self, packet):
        """Add a packet to the buffer"""
        with self.lock:
            self.packets.append(packet)

    def decode_up_to_frame(self, target_frame_num: int) -> bool:
        """
        Decode packets until we have target_frame_num.
        Returns True if target frame is now available.
        """
        with self.lock:
            # If we already have it, return
            if target_frame_num in self.decoded_frames:
                return True

            # Decode packets until we reach target frame
            while self.packets:
                packet = self.packets.popleft()
                try:
                    for frame in packet.decode():
                        if frame.pts is None:
                            continue
                        frame_num = int(frame.pts * self.stream.time_base)

                        # Convert to RGB24
                        rgb_frame = frame.reformat(format="rgb24")
                        frame_bytes = rgb_frame.to_ndarray().tobytes()

                        self.decoded_frames[frame_num] = frame_bytes
                        self.max_frame_seen = max(self.max_frame_seen, frame_num)

                        # If we've reached target, we're done
                        if frame_num >= target_frame_num:
                            return True

                except Exception as e:
                    logger.error(f"Error decoding packet on stream {self.stream_idx}: {e}")
                    continue

            return False

    def get_frame(self, frame_num: int) -> Optional[bytes]:
        """Get a decoded frame if available"""
        with self.lock:
            return self.decoded_frames.get(frame_num)

    def clear_old_frames(self, keep_from_frame: int) -> None:
        """Clear frames older than a certain frame number to save memory"""
        with self.lock:
            frames_to_delete = [f for f in self.decoded_frames if f < keep_from_frame]
            for f in frames_to_delete:
                del self.decoded_frames[f]


class HEVCDecoder:
    """Decodes OSV file with PTS-based multi-stream synchronization"""

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
        self._stream_buffers: Dict[int, StreamBuffer] = {}
        self._total_frames = 0
        self._demux_thread = None
        self._stop_demux = False

        self._open_container()
        self._discover_video_streams()
        self._start_demux_thread()

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

        self._video_streams = [
            stream for stream in self._container.streams
            if stream.type == "video" and stream.codec_context.name != "mjpeg"
        ]
        logger.info(f"Found {len(self._video_streams)} primary video streams")

        # Create per-stream buffers
        for idx, stream in enumerate(self._video_streams):
            self._stream_buffers[idx] = StreamBuffer(idx, stream)

    def _start_demux_thread(self) -> None:
        """Start background thread that demuxes packets into per-stream buffers"""
        def demux_worker():
            """Background thread: demux packets into per-stream buffers"""
            try:
                for packet in self._container.demux(self._video_streams):
                    if self._stop_demux:
                        break

                    # Find which stream this packet belongs to
                    stream_idx = next(
                        (i for i, s in enumerate(self._video_streams) if s == packet.stream),
                        None
                    )
                    if stream_idx is not None:
                        self._stream_buffers[stream_idx].add_packet(packet)

            except Exception as e:
                logger.error(f"Demux thread error: {e}")

        self._demux_thread = threading.Thread(target=demux_worker, daemon=True)
        self._demux_thread.start()
        logger.info("Demux thread started")

    def get_video_stream_count(self) -> int:
        """Get number of video streams"""
        return len(self._video_streams)

    def get_video_stream_info(self, idx: int) -> dict:
        """Get information about a specific video stream"""
        if idx >= len(self._video_streams):
            raise IndexError(f"Stream index {idx} out of range")

        stream = self._video_streams[idx]

        fps = 0.0
        if stream.average_rate:
            if isinstance(stream.average_rate, (int, float)):
                fps = float(stream.average_rate)
            else:
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

        Uses PTS-based buffering to keep streams synchronized.

        Args:
            frame_num: Frame number to decode (0-indexed)
            timeout_s: Maximum wait time for all streams

        Returns:
            Dict mapping stream_idx -> RGB24 frame bytes, or None on failure
        """
        import time

        start_time = time.time()

        # Tell all stream buffers to decode up to target frame
        logger.debug(f"[FRAME {frame_num}] Requesting from {len(self._stream_buffers)} streams")

        while True:
            # Check if all streams have this frame
            all_ready = True
            for idx, buffer in self._stream_buffers.items():
                if buffer.get_frame(frame_num) is None:
                    # Try to decode more packets for this stream
                    buffer.decode_up_to_frame(frame_num)
                    if buffer.get_frame(frame_num) is None:
                        all_ready = False

            if all_ready:
                # Got all streams
                result = {idx: self._stream_buffers[idx].get_frame(frame_num)
                          for idx in range(len(self._stream_buffers))}
                elapsed = time.time() - start_time
                logger.debug(f"[FRAME {frame_num}] Ready in {elapsed*1000:.0f}ms")
                return result

            # Timeout check
            if time.time() - start_time > timeout_s:
                logger.warning(f"[FRAME {frame_num}] Timeout after {timeout_s}s")
                return None

            # Small sleep to avoid busy-waiting
            time.sleep(0.001)

    def get_total_frames(self) -> int:
        """Get total frame count from first video stream"""
        if not self._video_streams:
            return 0

        first_stream = self._video_streams[0]
        if first_stream.frames:
            return first_stream.frames

        if first_stream.duration and first_stream.time_base and first_stream.average_rate:
            duration_s = float(first_stream.duration * first_stream.time_base)
            fps = float(first_stream.average_rate)
            return int(duration_s * fps)

        return 0

    def close(self) -> None:
        """Close the container and clean up"""
        self._stop_demux = True
        if self._demux_thread:
            self._demux_thread.join(timeout=5)
        if self._container:
            self._container.close()
            self._container = None


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
    "StreamBuffer",
]
