"""GPU auto-detection and availability probing

Separate from backend abstraction - just detects what's available.
Caller explicitly selects which GPU to use by passing target name to backend factory.
"""

import logging
import platform
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def detect_available_gpus() -> Dict[str, bool]:
    """
    Probe system for available GPU backends.

    Returns:
        Dict mapping backend name -> availability (True/False)
        Example: {"metal": True, "cuda": False, "oneapi": False, "cpu": True}
    """
    available = {"cpu": True}  # CPU always available

    system = platform.system()

    # Apple Silicon / Intel Mac
    if system == "Darwin":
        available["metal"] = _check_metal()

    # NVIDIA GPU (any OS)
    available["cuda"] = _check_cuda()

    # Intel iGPU (any OS)
    available["oneapi"] = _check_oneapi()

    logger.info(f"GPU detection: {available}")
    return available


def _check_metal() -> bool:
    """Check if Metal framework is available (macOS only)"""
    try:
        import platform
        if platform.system() != "Darwin":
            return False

        # Try importing PyObjC Metal binding
        try:
            import objc
            from Metal import MTLCreateSystemDefaultDevice

            device = MTLCreateSystemDefaultDevice()
            if device is not None:
                logger.debug(f"Metal device found: {device.name()}")
                return True
        except ImportError:
            logger.debug("PyObjC Metal bindings not available")

        return False
    except Exception as e:
        logger.debug(f"Metal check failed: {e}")
        return False


def _check_cuda() -> bool:
    """Check if NVIDIA CUDA is available"""
    try:
        import cv2

        if not hasattr(cv2, "cuda"):
            logger.debug("OpenCV CUDA module not available")
            return False

        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        if device_count > 0:
            logger.debug(f"CUDA found: {device_count} device(s)")
            return True

        return False
    except Exception as e:
        logger.debug(f"CUDA check failed: {e}")
        return False


def _check_oneapi() -> bool:
    """Check if Intel oneAPI (Intel Media SDK / Quick Sync) is available"""
    try:
        # Try importing Intel Media SDK or oneAPI
        # This is platform-specific and not widely available in Python
        # For now, return False - can be extended later

        import platform
        system = platform.system()

        # Quick Sync is available on Windows with Intel iGPU
        if system == "Windows":
            try:
                # Try to detect Quick Sync via ffmpeg
                import subprocess

                result = subprocess.run(
                    ["ffmpeg", "-decoders"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if "hevc_qsv" in result.stdout or "h264_qsv" in result.stdout:
                    logger.debug("Intel Quick Sync (oneAPI) available via FFmpeg")
                    return True
            except Exception as e:
                logger.debug(f"Quick Sync check failed: {e}")

        return False
    except Exception as e:
        logger.debug(f"oneAPI check failed: {e}")
        return False


def get_best_gpu() -> str:
    """
    Recommend the best available GPU backend.

    Priority: CUDA > Metal > oneAPI > CPU

    Returns:
        Backend name to use (e.g., "cuda", "metal", "cpu")
    """
    available = detect_available_gpus()

    # Priority order (fastest to slowest)
    for backend in ["cuda", "metal", "oneapi"]:
        if available.get(backend, False):
            logger.info(f"Auto-selected GPU backend: {backend}")
            return backend

    logger.info("No GPU backends available, using CPU")
    return "cpu"
