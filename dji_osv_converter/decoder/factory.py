"""Decoder backend factory - instantiates the right decoder by name"""

import logging
from typing import Optional

from .backend import DecoderBackend
from .detect import detect_available_decoders, get_best_decoder

logger = logging.getLogger(__name__)


def create_decoder(target: str) -> Optional[DecoderBackend]:
    """
    Create a decoder backend instance by name.

    Args:
        target: Decoder name ("videotoolbox", "cpu")

    Returns:
        Initialized DecoderBackend instance, or None if unavailable
    """
    available = detect_available_decoders()

    if target not in available:
        logger.error(f"Unknown decoder backend: {target}")
        return None

    if not available[target]:
        logger.error(f"Decoder backend not available: {target}")
        return None

    # Lazy import to avoid dependency on specific frameworks
    if target == "videotoolbox":
        from .videotoolbox import VideoToolboxDecoder

        decoder = VideoToolboxDecoder()
    elif target == "cpu":
        from .cpu import CPUDecoder

        decoder = CPUDecoder()
    else:
        logger.error(f"No implementation for decoder: {target}")
        return None

    # Initialize decoder
    if not decoder.initialize():
        logger.error(f"Failed to initialize {target} decoder")
        return None

    logger.info(f"Initialized {target} decoder backend")
    return decoder


def create_best_decoder() -> Optional[DecoderBackend]:
    """
    Create the best available decoder backend (auto-detected).

    Priority: videotoolbox > cpu

    Returns:
        Initialized DecoderBackend instance
    """
    best = get_best_decoder()
    return create_decoder(best)


__all__ = ["create_decoder", "create_best_decoder"]
