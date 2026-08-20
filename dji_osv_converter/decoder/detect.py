"""Decoder auto-detection and availability probing

Separate from backend abstraction - just detects what's available.
Caller explicitly selects which decoder to use by passing target name to factory.
"""

import logging
import platform
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def detect_available_decoders() -> Dict[str, bool]:
    """
    Probe system for available decoder backends.

    Returns:
        Dict mapping decoder name -> availability (True/False)
        Example: {"ffmpeg": True, "videotoolbox": True, "cpu": True}
    """
    available = {"cpu": True}  # CPU (PyAV) always available as fallback

    system = platform.system()

    # FFmpeg (platform-agnostic, prioritized first)
    available["ffmpeg"] = _check_ffmpeg()

    # VideoToolbox on macOS
    if system == "Darwin":
        available["videotoolbox"] = _check_videotoolbox()

    logger.info(f"Decoder detection: {available}")
    return available


def _check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and has hardware acceleration support"""
    try:
        import subprocess

        # Check if ffmpeg is available
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.debug("FFmpeg not available or not executable")
            return False

        # Check for hardware acceleration support (VideoToolbox on macOS)
        result = subprocess.run(
            ["ffmpeg", "-hwaccels"],
            capture_output=True,
            timeout=5,
            text=True,
        )

        has_hwaccel = result.returncode == 0 and len(result.stdout) > 0
        logger.debug(f"FFmpeg available, hardware acceleration: {has_hwaccel}")
        return True

    except FileNotFoundError:
        logger.debug("FFmpeg executable not found")
        return False
    except subprocess.TimeoutExpired:
        logger.debug("FFmpeg check timed out")
        return False
    except Exception as e:
        logger.debug(f"FFmpeg detection failed: {e}")
        return False


def _check_videotoolbox() -> bool:
    """Check if VideoToolbox framework is available (macOS only)"""
    try:
        import platform
        if platform.system() != "Darwin":
            return False

        # Try importing PyObjC VideoToolbox binding
        try:
            import objc
            objc.loadBundle(
                "VideoToolbox",
                globals(),
                "/System/Library/Frameworks/VideoToolbox.framework",
            )
            logger.debug("VideoToolbox framework available")
            return True
        except ImportError:
            logger.debug("PyObjC not available for VideoToolbox")
            return False
        except Exception as e:
            logger.debug(f"VideoToolbox check failed: {e}")
            return False

    except Exception as e:
        logger.debug(f"VideoToolbox detection failed: {e}")
        return False


def get_best_decoder() -> str:
    """
    Recommend the best available decoder backend.

    Priority: ffmpeg > videotoolbox > cpu

    Returns:
        Decoder name to use (e.g., "ffmpeg", "videotoolbox", "cpu")
    """
    available = detect_available_decoders()

    # Priority order (fastest to slowest)
    for decoder in ["ffmpeg", "videotoolbox"]:
        if available.get(decoder, False):
            logger.info(f"Auto-selected decoder backend: {decoder}")
            return decoder

    logger.info("No hardware decoders available, using CPU (PyAV)")
    return "cpu"


__all__ = ["detect_available_decoders", "get_best_decoder"]
