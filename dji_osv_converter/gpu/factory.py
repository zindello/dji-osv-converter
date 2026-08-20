"""GPU backend factory - instantiates the right backend by name"""

import logging
from typing import Optional

from .backend import GPUBackend
from .detect import detect_available_gpus

logger = logging.getLogger(__name__)


def create_backend(target: str) -> Optional[GPUBackend]:
    """
    Create a GPU backend instance by name.

    Args:
        target: Backend name ("metal", "cuda", "oneapi", "cpu")

    Returns:
        Initialized GPUBackend instance, or None if unavailable
    """
    available = detect_available_gpus()

    if target not in available:
        logger.error(f"Unknown GPU backend: {target}")
        return None

    if not available[target]:
        logger.error(f"GPU backend not available: {target}")
        return None

    # Lazy import to avoid dependency on specific frameworks
    if target == "metal":
        from .metal_backend import MetalBackend

        backend = MetalBackend()
    elif target == "cuda":
        from .cuda_backend import CUDABackend

        backend = CUDABackend()
    elif target == "oneapi":
        from .oneapi_backend import OneAPIBackend

        backend = OneAPIBackend()
    elif target == "cpu":
        from .cpu_backend import CPUBackend

        backend = CPUBackend()
    else:
        logger.error(f"No implementation for backend: {target}")
        return None

    # Initialize backend
    if not backend.initialize():
        logger.error(f"Failed to initialize {target} backend")
        return None

    logger.info(f"Initialized {target} GPU backend")
    return backend


def create_best_backend() -> Optional[GPUBackend]:
    """
    Create the best available GPU backend (auto-detected).

    Priority: CUDA > Metal > oneAPI > CPU

    Returns:
        Initialized GPUBackend instance
    """
    from .detect import get_best_gpu

    best = get_best_gpu()
    return create_backend(best)
