"""FFmpeg hardware-accelerated video decoder for HEVC streams

Production-grade implementation with:
- Subprocess-based FFmpeg with VideoToolbox hardware acceleration (-hwaccel videotoolbox)
- Background thread reading raw RGB24 frames
- Per-stream synchronized frame delivery
- Comprehensive error handling and timeout management
- Proper cleanup (subprocess termination, thread joining)
- Detailed performance logging

Architecture:
- FFmpeg subprocess decodes HEVC streams to raw RGB24 video data
- Background reader thread reads frames from subprocess stdout
- Frame queue with configurable buffer size
- Per-stream frame delivery synchronized to match decode timing
- Graceful fallback on FFmpeg unavailability
"""

import logging
import subprocess
import threading
import queue
import time
from pathlib import Path
from typing import Dict, Optional, Any
import av
import numpy as np

from .backend import DecoderBackend

logger = logging.getLogger(__name__)

# Frame size constants (3840×3840×3 bytes per stream)
STREAM_WIDTH = 3840
STREAM_HEIGHT = 3840
BYTES_PER_PIXEL = 3
FRAME_SIZE_BYTES = STREAM_WIDTH * STREAM_HEIGHT * BYTES_PER_PIXEL


class FFmpegDecoder(DecoderBackend):
    """Hardware-accelerated FFmpeg decoder with VideoToolbox support"""

    def __init__(self):
        super().__init__("ffmpeg")
        self.available = False
        self._filepath: Optional[Path] = None
        self._container: Optional[av.container.Container] = None
        self._video_streams = []
        self._stream_count = 0

        # FFmpeg subprocess
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_reading = False
        self._frame_queue: Optional[queue.Queue] = None
        self._decode_error: Optional[str] = None

        # Per-stream frame buffers
        self._stream_frames: Dict[int, Dict[int, bytes]] = {}
        self._stream_locks = {}
        self._stream_conditions = {}
        self._max_frame_seen = {}

        # Timing and statistics
        self._decode_start_time: Optional[float] = None
        self._frames_decoded = 0
        self._total_bytes_read = 0
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialize and check FFmpeg availability"""
        try:
            # Check FFmpeg is available
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                logger.error("FFmpeg not found or not executable")
                self.available = False
                return False

            # Check for VideoToolbox support
            result = subprocess.run(
                ["ffmpeg", "-hwaccels"],
                capture_output=True,
                timeout=5,
                text=True,
            )

            has_videotoolbox = "videotoolbox" in result.stdout.lower()

            logger.info(f"FFmpeg available, VideoToolbox support: {has_videotoolbox}")
            self.available = True
            return True

        except FileNotFoundError:
            logger.error("FFmpeg not found - install with 'brew install ffmpeg'")
            self.available = False
            return False
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg check timed out")
            self.available = False
            return False
        except Exception as e:
            logger.error(f"FFmpeg initialization failed: {e}")
            self.available = False
            return False

    def open_file(self, filepath: str) -> bool:
        """Open OSV/MP4 file and start FFmpeg decode"""
        try:
            self._filepath = Path(filepath)
            if not self._filepath.exists():
                logger.error(f"File not found: {filepath}")
                return False

            logger.info(f"Opening file: {filepath}")

            # Use PyAV to discover streams and get metadata
            self._container = av.open(str(self._filepath))

            # Find video streams
            self._video_streams = [
                stream for stream in self._container.streams
                if stream.type == "video" and stream.codec_context.name != "mjpeg"
            ]

            if not self._video_streams:
                logger.error("No video streams found")
                return False

            self._stream_count = len(self._video_streams)
            logger.info(f"Found {self._stream_count} video streams")

            # Initialize per-stream buffers
            for idx in range(self._stream_count):
                self._stream_frames[idx] = {}
                self._stream_locks[idx] = threading.Lock()
                self._stream_conditions[idx] = threading.Condition(self._stream_locks[idx])
                self._max_frame_seen[idx] = -1

            # Verify stream dimensions (OSV should be 3840×3840)
            for idx, stream in enumerate(self._video_streams):
                logger.info(f"  Stream {idx}: {stream.width}×{stream.height} @ "
                          f"{float(stream.average_rate):.2f}fps, codec={stream.codec_context.name}")
                if stream.width != STREAM_WIDTH or stream.height != STREAM_HEIGHT:
                    logger.warning(f"  Stream {idx} has unexpected dimensions: "
                                 f"{stream.width}×{stream.height} (expected {STREAM_WIDTH}×{STREAM_HEIGHT})")

            # Start FFmpeg subprocess
            if not self._start_ffmpeg_subprocess():
                return False

            # Start reader thread
            self._start_reader_thread()

            logger.info("FFmpeg decoder ready")
            return True

        except Exception as e:
            logger.error(f"Failed to open file: {e}")
            self.close()
            return False

    def _start_ffmpeg_subprocess(self) -> bool:
        """Start FFmpeg subprocess with hardware acceleration"""
        try:
            self._decode_start_time = time.time()
            self._frames_decoded = 0
            self._total_bytes_read = 0
            self._frame_queue = queue.Queue(maxsize=50)  # Large buffer for smooth decode
            self._stop_reading = False
            self._decode_error = None

            # Build FFmpeg command with hardware acceleration
            cmd = [
                "ffmpeg",
                "-hwaccel", "videotoolbox",      # Request hardware acceleration
                "-i", str(self._filepath),        # Input file
                "-pix_fmt", "rgb24",              # Output format: RGB24
                "-f", "rawvideo",                 # Output as raw video
                "-",                              # Write to stdout
            ]

            logger.debug(f"Starting FFmpeg: {' '.join(cmd)}")

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1024 * 1024 * 10,  # 10MB buffer for smooth reading
            )

            logger.info("FFmpeg subprocess started with VideoToolbox hardware acceleration")
            return True

        except FileNotFoundError:
            logger.error("FFmpeg executable not found")
            return False
        except Exception as e:
            logger.error(f"Failed to start FFmpeg subprocess: {e}")
            return False

    def _start_reader_thread(self) -> None:
        """Start background thread that reads frames from FFmpeg"""
        def reader_worker():
            """Background thread: reads raw RGB24 frames from FFmpeg stdout"""
            frame_num = 0
            try:
                while not self._stop_reading and self._process:
                    # Read one frame worth of data for ALL streams
                    # OSV files have 2 streams, so read 2 × FRAME_SIZE_BYTES
                    frame_bytes = self._process.stdout.read(FRAME_SIZE_BYTES * self._stream_count)

                    if not frame_bytes or len(frame_bytes) < FRAME_SIZE_BYTES * self._stream_count:
                        # EOF reached
                        logger.info(f"FFmpeg EOF at frame {frame_num}")
                        self._frame_queue.put(None)  # Signal EOF
                        break

                    # Frame data is complete
                    try:
                        self._frame_queue.put((frame_num, frame_bytes), timeout=5.0)
                        with self._lock:
                            self._total_bytes_read += len(frame_bytes)
                        frame_num += 1
                    except queue.Full:
                        logger.warning("Frame queue full, dropping frame")
                        continue

            except Exception as e:
                logger.error(f"Reader thread error: {e}")
                self._decode_error = str(e)
                self._frame_queue.put(None)  # Signal EOF on error
            finally:
                logger.debug(f"Reader thread finished after {frame_num} frames")

        self._reader_thread = threading.Thread(target=reader_worker, daemon=True)
        self._reader_thread.start()
        logger.debug("FFmpeg reader thread started")

    def get_stream_count(self) -> int:
        """Get number of video streams"""
        return self._stream_count

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
                fps = float(stream.average_rate.numerator) / float(stream.average_rate.denominator)

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

    def decode_frame(self, stream_idx: int, frame_num: int, timeout_s: float = 10.0) -> Optional[bytes]:
        """Decode a frame from a specific stream"""
        if stream_idx not in self._stream_frames:
            logger.error(f"Invalid stream index: {stream_idx}")
            return None

        # First check if frame is already cached
        with self._stream_locks[stream_idx]:
            if frame_num in self._stream_frames[stream_idx]:
                return self._stream_frames[stream_idx][frame_num]

        # Frame not cached, need to decode
        frame_data = self.decode_frame_all_streams(frame_num, timeout_s)
        if frame_data is None:
            return None

        return frame_data.get(stream_idx)

    def decode_frame_all_streams(
        self, frame_num: int, timeout_s: float = 10.0
    ) -> Optional[Dict[int, bytes]]:
        """Decode a frame from all streams (synchronized)"""
        start_time = time.time()

        logger.debug(f"[FRAME {frame_num}] Waiting for frame from {self._stream_count} streams")

        while time.time() - start_time < timeout_s:
            # Check if we have decoded this frame
            frame_data = self._try_get_frame(frame_num)
            if frame_data is not None:
                elapsed = time.time() - start_time
                logger.debug(f"[FRAME {frame_num}] Ready in {elapsed*1000:.0f}ms, "
                           f"{sum(len(b) for b in frame_data.values())} bytes")
                return frame_data

            # Frame not ready yet - wait a bit
            time.sleep(0.001)

        # Timeout
        logger.warning(f"[FRAME {frame_num}] Timeout after {timeout_s}s waiting for decode")
        if self._decode_error:
            logger.error(f"  Decoder error: {self._decode_error}")
        return None

    def _try_get_frame(self, frame_num: int) -> Optional[Dict[int, bytes]]:
        """Try to get a frame without blocking"""
        try:
            frame_num_to_check, frame_bytes = self._frame_queue.get_nowait()

            if frame_bytes is None:
                # EOF marker
                return None

            if frame_num_to_check != frame_num:
                # We got a frame, but it's not the one we want
                # This shouldn't happen with sequential decoding
                logger.warning(f"Got frame {frame_num_to_check}, expected {frame_num}")
                return None

            # Split combined frame data into per-stream buffers
            result = {}
            for idx in range(self._stream_count):
                start = idx * FRAME_SIZE_BYTES
                end = start + FRAME_SIZE_BYTES
                result[idx] = frame_bytes[start:end]

            # Cache in per-stream buffers
            for idx, data in result.items():
                with self._stream_conditions[idx]:
                    self._stream_frames[idx][frame_num] = data
                    self._max_frame_seen[idx] = max(self._max_frame_seen[idx], frame_num)
                    self._stream_conditions[idx].notify_all()

            with self._lock:
                self._frames_decoded += 1

            return result

        except queue.Empty:
            return None

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
        """Close decoder and clean up resources"""
        logger.info("Closing FFmpeg decoder")

        # Signal reader thread to stop
        self._stop_reading = True

        # Terminate FFmpeg subprocess
        if self._process:
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("FFmpeg subprocess didn't terminate gracefully, killing")
                    self._process.kill()
                    self._process.wait()
            except Exception as e:
                logger.error(f"Error terminating FFmpeg: {e}")
            finally:
                self._process = None

        # Wait for reader thread
        if self._reader_thread:
            self._reader_thread.join(timeout=5)
            self._reader_thread = None

        # Close PyAV container
        if self._container:
            self._container.close()
            self._container = None

        # Log statistics
        if self._decode_start_time:
            elapsed = time.time() - self._decode_start_time
            mb_decoded = self._total_bytes_read / 1e6
            logger.info(f"Decode stats: {self._frames_decoded} frames, "
                       f"{mb_decoded:.1f}MB, {elapsed:.1f}s, "
                       f"{mb_decoded/elapsed if elapsed > 0 else 0:.1f}MB/s")

        logger.info("FFmpeg decoder closed")

    def __repr__(self) -> str:
        return f"FFmpegDecoder(available={self.available}, streams={self._stream_count})"


__all__ = ["FFmpegDecoder"]
