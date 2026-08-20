"""CPU-based software HEVC decoder using PyAV

Fallback decoder when hardware acceleration is unavailable.
Baseline performance reference for benchmarking.
"""

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Any
import av

from .backend import DecoderBackend

logger = logging.getLogger(__name__)


class CPUDecoder(DecoderBackend):
    """Software decoder using PyAV (CPU-based)"""

    def __init__(self):
        super().__init__("cpu")
        self.available = False
        self._container: Optional[av.container.Container] = None
        self._video_streams = []
        self._stream_buffers: Dict[int, "CPUStreamBuffer"] = {}
        self._demux_thread = None
        self._stop_demux = False
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialize CPU decoder (PyAV availability check)"""
        try:
            import av
            logger.info("PyAV CPU decoder available")
            self.available = True
            return True
        except ImportError:
            logger.error("PyAV not available for CPU decoder")
            self.available = False
            return False

    def open_file(self, filepath: str) -> bool:
        """Open OSV/MP4 file"""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                return False

            self._container = av.open(str(filepath))
            logger.debug(f"Opened container: {filepath}")

            # Discover video streams
            self._video_streams = [
                stream
                for stream in self._container.streams
                if stream.type == "video" and stream.codec_context.name != "mjpeg"
            ]

            if not self._video_streams:
                logger.error("No video streams found in file")
                return False

            logger.info(f"Found {len(self._video_streams)} video streams")

            # Create per-stream buffers
            for idx in range(len(self._video_streams)):
                self._stream_buffers[idx] = CPUStreamBuffer(
                    idx, self._video_streams[idx]
                )

            # Start demux thread
            self._start_demux_thread()

            return True

        except Exception as e:
            logger.error(f"Failed to open file: {e}")
            return False

    def _start_demux_thread(self) -> None:
        """Start background thread that demuxes packets into per-stream buffers"""
        def demux_worker():
            try:
                for packet in self._container.demux(self._video_streams):
                    if self._stop_demux:
                        break

                    # Find which stream this packet belongs to
                    stream_idx = next(
                        (i for i, s in enumerate(self._video_streams) if s == packet.stream),
                        None,
                    )
                    if stream_idx is not None:
                        self._stream_buffers[stream_idx].add_packet(packet)

            except Exception as e:
                logger.error(f"Demux thread error: {e}")

        self._demux_thread = threading.Thread(target=demux_worker, daemon=True)
        self._demux_thread.start()
        logger.info("Demux thread started")

    def get_stream_count(self) -> int:
        """Get number of video streams"""
        return len(self._video_streams)

    def get_stream_info(self, stream_idx: int) -> Dict[str, Any]:
        """Get stream metadata"""
        if stream_idx >= len(self._video_streams):
            raise IndexError(f"Stream index {stream_idx} out of range")

        stream = self._video_streams[stream_idx]

        fps = 0.0
        if stream.average_rate:
            if isinstance(stream.average_rate, (int, float)):
                fps = float(stream.average_rate)
            else:
                fps = float(stream.average_rate.numerator) / float(
                    stream.average_rate.denominator
                )

        return {
            "index": stream.index,
            "codec_name": stream.codec_context.name,
            "width": stream.width,
            "height": stream.height,
            "fps": fps,
            "duration_seconds": (
                float(stream.duration * stream.time_base) if stream.duration else 0
            ),
            "total_frames": stream.frames,
        }

    def decode_frame(
        self, stream_idx: int, frame_num: int, timeout_s: float = 10.0
    ) -> Optional[bytes]:
        """Decode a frame from a single stream"""
        if stream_idx not in self._stream_buffers:
            logger.error(f"Stream index {stream_idx} out of range")
            return None

        buffer = self._stream_buffers[stream_idx]
        return buffer.get_frame(frame_num, timeout_s)

    def decode_frame_all_streams(
        self, frame_num: int, timeout_s: float = 10.0
    ) -> Optional[Dict[int, bytes]]:
        """Decode a frame from all streams (synchronized)"""
        start_time = time.time()
        logger.debug(f"[FRAME {frame_num}] Waiting from {len(self._stream_buffers)} streams")

        while time.time() - start_time < timeout_s:
            # Check if all streams have decoded this frame
            all_ready = True
            for buffer in self._stream_buffers.values():
                if buffer.get_frame_nowait(frame_num) is None:
                    all_ready = False
                    break

            if all_ready:
                # Got all streams
                result = {
                    idx: self._stream_buffers[idx].get_frame_nowait(frame_num)
                    for idx in range(len(self._stream_buffers))
                }
                elapsed = time.time() - start_time
                logger.debug(f"[FRAME {frame_num}] Ready in {elapsed*1000:.0f}ms")
                return result

            # Small sleep to avoid busy-waiting
            time.sleep(0.001)

        logger.warning(f"[FRAME {frame_num}] Timeout after {timeout_s}s")
        return None

    def get_total_frames(self) -> int:
        """Get total frame count from first video stream"""
        if not self._video_streams:
            return 0

        first_stream = self._video_streams[0]
        if first_stream.frames:
            return first_stream.frames

        if (
            first_stream.duration
            and first_stream.time_base
            and first_stream.average_rate
        ):
            duration_s = float(first_stream.duration * first_stream.time_base)
            fps = float(first_stream.average_rate)
            return int(duration_s * fps)

        return 0

    def close(self) -> None:
        """Close decoder and clean up resources"""
        self._stop_demux = True

        # Stop all decoder threads
        for buffer in self._stream_buffers.values():
            buffer.stop()

        if self._demux_thread:
            self._demux_thread.join(timeout=5)

        # Wait for decoder threads
        for buffer in self._stream_buffers.values():
            if buffer._decoder_thread:
                buffer._decoder_thread.join(timeout=5)

        if self._container:
            self._container.close()
            self._container = None

        logger.info("CPU decoder closed")


class CPUStreamBuffer:
    """Per-stream buffer with background frame decoding using PyAV"""

    def __init__(self, stream_idx: int, stream):
        self.stream_idx = stream_idx
        self.stream = stream
        self.packets = deque()
        self.decoded_frames = {}  # frame_num -> RGB24 bytes
        self.max_frame_seen = -1
        self.lock = threading.Lock()
        self.frame_ready = threading.Condition(self.lock)
        self._decoder_thread = None
        self._stop_decoder = False

        # Start background decoder thread
        self._start_decoder_thread()

    def add_packet(self, packet):
        """Add a packet to the buffer"""
        with self.lock:
            self.packets.append(packet)

    def _start_decoder_thread(self) -> None:
        """Start background thread that decodes packets"""
        def decode_worker():
            while not self._stop_decoder:
                # Get one packet
                packet = None
                with self.lock:
                    if self.packets:
                        packet = self.packets.popleft()

                if packet is None:
                    # No packets ready, sleep briefly
                    time.sleep(0.001)
                    continue

                # Decode packet (outside lock to avoid blocking demux thread)
                try:
                    for frame in packet.decode():
                        if frame.pts is None:
                            continue

                        frame_num = int(frame.pts * self.stream.time_base)

                        # Convert to RGB24 (CPU path)
                        frame_bytes = self._convert_to_rgb(frame)
                        if frame_bytes is None:
                            logger.warning(f"Failed to convert frame {frame_num}, skipping")
                            continue

                        with self.frame_ready:
                            self.decoded_frames[frame_num] = frame_bytes
                            self.max_frame_seen = max(self.max_frame_seen, frame_num)
                            self.frame_ready.notify_all()

                except Exception as e:
                    logger.error(f"Error decoding packet on stream {self.stream_idx}: {e}")
                    continue

        self._decoder_thread = threading.Thread(target=decode_worker, daemon=True)
        self._decoder_thread.start()

    def _convert_to_rgb(self, frame) -> Optional[bytes]:
        """Convert PyAV frame to RGB24 bytes"""
        try:
            rgb_frame = frame.reformat(format="rgb24")
            return rgb_frame.to_ndarray().tobytes()
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return None

    def get_frame(self, frame_num: int, timeout_s: float = 10.0) -> Optional[bytes]:
        """Get a decoded frame with timeout"""
        start_time = time.time()

        while time.time() - start_time < timeout_s:
            with self.lock:
                if frame_num in self.decoded_frames:
                    return self.decoded_frames[frame_num]

            time.sleep(0.001)

        return None

    def get_frame_nowait(self, frame_num: int) -> Optional[bytes]:
        """Get a frame without waiting"""
        with self.lock:
            return self.decoded_frames.get(frame_num)

    def stop(self) -> None:
        """Stop decoder thread"""
        self._stop_decoder = True


__all__ = ["CPUDecoder"]
