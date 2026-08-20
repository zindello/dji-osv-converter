"""GPU acceleration abstraction layer

Modular GPU backends for different hardware targets:
- Metal (Apple Silicon)
- CUDA (NVIDIA)
- oneAPI (Intel iGPU)
- CPU (fallback)

Backends are selected explicitly by the caller, not auto-detected here.
Auto-detection is handled separately by gpu.detect module.
"""

from .backend import GPUBackend, GPUFrame, GPUCodec
from .detect import detect_available_gpus, get_best_gpu

__all__ = [
    "GPUBackend",
    "GPUFrame",
    "GPUCodec",
    "detect_available_gpus",
    "get_best_gpu",
]
