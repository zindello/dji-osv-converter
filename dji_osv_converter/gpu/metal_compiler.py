"""Metal shader compiler for macOS

Compiles .metal source files to Metal library (.metallib).
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def compile_metal_shaders(shader_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
    """
    Compile Metal shader source (.metal) to Metal library (.metallib).

    Uses xcrun metal and metallib commands (part of Xcode).

    Args:
        shader_path: Path to .metal source file
        output_path: Path to save .metallib (default: same dir, .metallib extension)

    Returns:
        Path to compiled .metallib, or None on failure
    """
    if not shader_path.exists():
        logger.error(f"Shader file not found: {shader_path}")
        return None

    if output_path is None:
        output_path = shader_path.with_suffix(".metallib")

    try:
        # Compile to .air (intermediate representation)
        air_path = shader_path.with_suffix(".air")

        logger.debug(f"Compiling Metal shader: {shader_path} → {air_path}")

        # Step 1: Compile .metal to .air
        compile_cmd = [
            "xcrun",
            "-sdk", "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o", str(air_path),
        ]

        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.error(f"Metal compilation failed: {result.stderr}")
            return None

        logger.debug(f"Compiled to AIR: {air_path}")

        # Step 2: Link .air to .metallib
        link_cmd = [
            "xcrun",
            "-sdk", "macosx",
            "metallib",
            str(air_path),
            "-o", str(output_path),
        ]

        result = subprocess.run(link_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.error(f"Metal linking failed: {result.stderr}")
            return None

        logger.info(f"Compiled Metal library: {output_path}")

        # Clean up .air file
        air_path.unlink(missing_ok=True)

        return output_path

    except subprocess.TimeoutExpired:
        logger.error("Metal compilation timed out")
        return None
    except FileNotFoundError:
        logger.error("xcrun not found - install Xcode Command Line Tools")
        logger.error("Run: xcode-select --install")
        return None
    except Exception as e:
        logger.error(f"Metal compilation failed: {e}")
        return None


def load_metal_library(library_path: Path):
    """
    Load a compiled Metal library.

    Args:
        library_path: Path to .metallib file

    Returns:
        Metal library object, or None on failure
    """
    if not library_path.exists():
        logger.error(f"Metal library not found: {library_path}")
        return None

    try:
        from Metal import MTLCreateSystemDefaultDevice

        device = MTLCreateSystemDefaultDevice()
        if device is None:
            logger.error("Failed to get Metal device")
            return None

        # Load from file
        error = None
        library = device.newLibraryWithFile_error_(str(library_path), error)

        if library is None:
            logger.error(f"Failed to load Metal library: {error}")
            return None

        logger.info(f"Loaded Metal library: {library_path}")
        return library

    except ImportError:
        logger.error("PyObjC Metal bindings not available")
        return None
    except Exception as e:
        logger.error(f"Failed to load Metal library: {e}")
        return None
