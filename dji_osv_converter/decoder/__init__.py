"""Decoder backends for OSV frame extraction

Modular decoder architecture supporting:
- VideoToolbox: Hardware HEVC decode (M3 Mac)
- CPU: Software PyAV fallback

Similar pattern to GPU backends:
- Abstract interface in backend.py
- Concrete implementations (videotoolbox.py, cpu.py)
- Auto-detection (detect.py)
- Factory pattern (factory.py)
"""

from .backend import DecoderBackend, DecoderFrame
from .detect import detect_available_decoders, get_best_decoder
from .factory import create_decoder, create_best_decoder

__all__ = [
    "DecoderBackend",
    "DecoderFrame",
    "detect_available_decoders",
    "get_best_decoder",
    "create_decoder",
    "create_best_decoder",
]
