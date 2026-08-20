"""Multi-stream HEVC frame extraction with PTS-based synchronization

Uses packet-level buffering to synchronize desynchronized streams.
No seeking, no redundant decoding - just stream-aligned buffering.

Supports multiple decoder backends:
- VideoToolbox: Hardware HEVC decode on M3 Mac (~40-100× speedup)
- CPU (PyAV): Software fallback (always available)

Optional GPU-accelerated color conversion for 50-100× speedup on YUV→RGB.
"""

import logging
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import av

from .decoder import factory as decoder_factory
from .decoder.backend import DecoderBackend

logger = logging.getLogger(__name__)


class StreamBuffer:
    """Per-stream packet buffer with background frame decoding"""

    def __init__(self, stream_idx: int, stream, gpu_backend: Optional[Any] = None):
        self.stream_idx = stream_idx
        self.stream = stream
        self.packets = deque()
        self.decoded_frames = {}  # frame_num -> bytes
        self.max_frame_seen = -1
        self.lock = threading.Lock()
        self.frame_ready = threading.Condition(self.lock)  # Signal when frame decoded
        self.gpu_backend = gpu_backend
        self._decoder_thread = None
        self._stop_decoder = False

        # Start background decoder thread
        self._start_decoder_thread()

    def add_packet(self, packet):
        """Add a packet to the buffer"""
        with self.lock:
            self.packets.append(packet)

    def _start_decoder_thread(self) -> None:
        """Start background thread that decodes packets as they arrive"""
        def decode_worker():
            while not self._stop_decoder:
                # Decode one packet if available
                packet = None
                with self.lock:
                    if self.packets:
                        packet = self.packets.popleft()

                if packet is None:
                    # No packets ready, sleep briefly
                    threading.Event().wait(0.001)
                    continue

                # Decode packet (outside lock to avoid blocking demux thread)
                try:
                    for frame in packet.decode():
                        if frame.pts is None:
                            continue
                        frame_num = int(frame.pts * self.stream.time_base)

                        # Convert to RGB24 (GPU-accelerated if available)
                        frame_bytes = self._convert_to_rgb(frame)
                        if frame_bytes is None:
                            logger.warning(f"Failed to convert frame {frame_num}, skipping")
                            continue

                        with self.frame_ready:
                            self.decoded_frames[frame_num] = frame_bytes
                            self.max_frame_seen = max(self.max_frame_seen, frame_num)
                            self.frame_ready.notify_all()  # Wake up waiters

                except Exception as e:
                    logger.error(f"Error decoding packet on stream {self.stream_idx}: {e}")
                    continue

        self._decoder_thread = threading.Thread(target=decode_worker, daemon=True)
        self._decoder_thread.start()

    def decode_up_to_frame(self, target_frame_num: int, timeout_s: float = 10.0) -> bool:
        """
        Wait for target_frame_num to be decoded by background thread.
        Returns True if frame is now available, False on timeout.
        """
        import time
        start = time.time()

        while time.time() - start < timeout_s:
            with self.lock:
                if target_frame_num in self.decoded_frames:
                    return True

            # Sleep briefly to avoid busy-waiting
            time.sleep(0.001)

        return False

    def get_frame(self, frame_num: int) -> Optional[bytes]:
        """Get a decoded frame if available"""
        with self.lock:
            return self.decoded_frames.get(frame_num)

    def _convert_to_rgb(self, frame) -> Optional[bytes]:
        """
        Convert frame to RGB24 bytes.

        Uses GPU acceleration if backend available, otherwise CPU fallback.
        """
        try:
            if self.gpu_backend:
                # Try GPU-accelerated conversion first
                return self._convert_gpu(frame)
            else:
                # CPU fallback
                return self._convert_cpu(frame)
        except Exception as e:
            logger.error(f"Color conversion failed: {e}")
            return None

    def _convert_gpu(self, frame) -> Optional[bytes]:
        """Convert using GPU backend (50-100× faster)"""
        try:
            from .gpu.backend import GPUFrame, GPUColorSpace
            import numpy as np

            # Get YUV420p array from frame
            yuv_array = frame.to_ndarray()

            # Create GPUFrame
            gpu_frame = GPUFrame(
                gpu_handle=yuv_array,
                width=frame.width,
                height=frame.height,
                color_space=GPUColorSpace.YUV420P,
            )

            # Convert to RGB using GPU
            rgb_frame = self.gpu_backend.convert_color_space(gpu_frame, GPUColorSpace.RGB24)
            if rgb_frame is None:
                logger.debug("GPU conversion returned None, falling back to CPU")
                return self._convert_cpu(frame)

            # Extract bytes from GPU frame
            if isinstance(rgb_frame.gpu_handle, np.ndarray):
                return rgb_frame.gpu_handle.tobytes()
            else:
                # Backend may return other types (Metal texture, etc.)
                # Convert to numpy first
                rgb_array = np.asarray(rgb_frame.gpu_handle)
                return rgb_array.tobytes()

        except Exception as e:
            logger.debug(f"GPU conversion failed: {e}, falling back to CPU")
            return self._convert_cpu(frame)

    def _convert_cpu(self, frame) -> Optional[bytes]:
        """CPU-based conversion (PyAV fallback)"""
        try:
            rgb_frame = frame.reformat(format="rgb24")
            return rgb_frame.to_ndarray().tobytes()
        except Exception as e:
            logger.error(f"CPU conversion failed: {e}")
            return None

    def clear_old_frames(self, keep_from_frame: int) -> None:
        """Clear frames older than a certain frame number to save memory"""
        with self.lock:
            frames_to_delete = [f for f in self.decoded_frames if f < keep_from_frame]
            for f in frames_to_delete:
                del self.decoded_frames[f]


class HEVCDecoder:
    """Decodes OSV file with PTS-based multi-stream synchronization

    Supports both old PyAV-based streaming decoder and new decoder backends
    (VideoToolbox for hardware decode, CPU for fallback).
    """

    def __init__(
        self,
        filepath: Path,
        gpu_backend: Optional[Any] = None,
        use_decoder_backend: Optional[str] = None,
    ):
        """
        Initialize decoder for OSV file.

        Args:
            filepath: Path to .OSV file
            gpu_backend: Optional GPU backend for accelerated color conversion
            use_decoder_backend: Optional decoder backend name ("videotoolbox", "cpu").
                                If None, will auto-detect best available.
                                Set to "" to disable and use legacy PyAV decoder.

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
        self._gpu_backend = gpu_backend

        # New decoder backend support
        self._decoder_backend: Optional[DecoderBackend] = None
        self._use_decoder_backend = use_decoder_backend

        # Try to initialize decoder backend if requested
        self._init_decoder_backend()

        # If decoder backend available, use it; otherwise fall back to PyAV
        if self._decoder_backend is not None:
            logger.info(f"Using {self._decoder_backend.name} decoder backend")
            self._decoder_backend.open_file(str(self.filepath))
        else:
            # Fallback to legacy PyAV-based decoder
            logger.info("Using legacy PyAV decoder (no hardware acceleration)")
            self._open_container()
            self._discover_video_streams()
            self._start_demux_thread()

    def _init_decoder_backend(self) -> None:
        """Try to initialize the specified decoder backend"""
        if self._use_decoder_backend == "":
            # Explicitly disabled
            logger.debug("Decoder backend explicitly disabled")
            return

        try:
            if self._use_decoder_backend is None:
                # Auto-detect best available
                logger.debug("Auto-detecting best decoder backend...")
                self._decoder_backend = decoder_factory.create_best_decoder()
            else:
                # Use specified backend
                logger.debug(f"Trying to create {self._use_decoder_backend} decoder backend...")
                self._decoder_backend = decoder_factory.create_decoder(self._use_decoder_backend)

            if self._decoder_backend is not None:
                logger.info(f"Decoder backend {self._decoder_backend.name} initialized")
            else:
                logger.warning("No suitable decoder backend available, falling back to PyAV")
        except Exception as e:
            logger.warning(f"Failed to initialize decoder backend: {e}, falling back to PyAV")
            self._decoder_backend = None

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
            self._stream_buffers[idx] = StreamBuffer(idx, stream, gpu_backend=self._gpu_backend)

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
        if self._decoder_backend is not None:
            return self._decoder_backend.get_stream_count()
        return len(self._video_streams)

    def get_video_stream_info(self, idx: int) -> dict:
        """Get information about a specific video stream"""
        if self._decoder_backend is not None:
            info = self._decoder_backend.get_stream_info(idx)
            # Normalize keys for backward compatibility
            return {
                "index": info.get("index", idx),
                "codec_name": info.get("codec_name", "unknown"),
                "width": info["width"],
                "height": info["height"],
                "fps": info.get("fps", 0.0),
                "duration_seconds": info.get("duration_seconds", 0),
                "frame_count": info.get("total_frames", 0),
            }

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
        Wait for a frame from all streams.

        Background decoder threads decode packets asynchronously.
        This method waits for frames to be ready.

        Args:
            frame_num: Frame number (0-indexed)
            timeout_s: Maximum wait time for all streams

        Returns:
            Dict mapping stream_idx -> RGB24 frame bytes, or None on failure/timeout
        """
        import time

        # Use decoder backend if available
        if self._decoder_backend is not None:
            return self._decoder_backend.decode_frame_all_streams(frame_num, timeout_s=timeout_s)

        # Fallback to legacy PyAV decoder
        start_time = time.time()
        logger.debug(f"[FRAME {frame_num}] Waiting from {len(self._stream_buffers)} streams")

        while time.time() - start_time < timeout_s:
            # Check if all streams have decoded this frame
            all_ready = True
            for idx, buffer in self._stream_buffers.items():
                if buffer.get_frame(frame_num) is None:
                    all_ready = False
                    break

            if all_ready:
                # Got all streams
                result = {idx: self._stream_buffers[idx].get_frame(frame_num)
                          for idx in range(len(self._stream_buffers))}
                elapsed = time.time() - start_time
                logger.debug(f"[FRAME {frame_num}] Ready in {elapsed*1000:.0f}ms")
                return result

            # Small sleep to avoid busy-waiting
            time.sleep(0.001)

        logger.warning(f"[FRAME {frame_num}] Timeout after {timeout_s}s")
        return None

    def get_total_frames(self) -> int:
        """Get total frame count from first video stream"""
        if self._decoder_backend is not None:
            return self._decoder_backend.get_total_frames()

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
        # Close decoder backend if active
        if self._decoder_backend is not None:
            self._decoder_backend.close()
            self._decoder_backend = None
            return

        # Legacy PyAV cleanup
        self._stop_demux = True

        # Stop all decoder threads
        for buffer in self._stream_buffers.values():
            buffer._stop_decoder = True

        if self._demux_thread:
            self._demux_thread.join(timeout=5)

        # Wait for decoder threads
        for buffer in self._stream_buffers.values():
            if buffer._decoder_thread:
                buffer._decoder_thread.join(timeout=5)

        if self._container:
            self._container.close()
            self._container = None


class FrameExtractor:
    """High-level interface for extracting synchronized frames from OSV"""

    def __init__(
        self,
        osv_filepath: Path,
        gpu_backend: Optional[Any] = None,
        use_decoder_backend: Optional[str] = None,
    ):
        """
        Initialize frame extractor for an OSV file.

        Args:
            osv_filepath: Path to .OSV file
            gpu_backend: Optional GPU backend for accelerated color conversion
            use_decoder_backend: Optional decoder backend name ("videotoolbox", "cpu").
                                If None, will auto-detect best available.
        """
        self.decoder = HEVCDecoder(
            osv_filepath,
            gpu_backend=gpu_backend,
            use_decoder_backend=use_decoder_backend,
        )
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
