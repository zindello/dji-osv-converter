#!/usr/bin/env python3
"""Subprocess Helper for DJI Stitcher Dynamic Seam Stitching

This script runs in a subprocess with proper DYLD environment variables.
It loads the DualStitcher framework and calls the dynamicStitchSeams method
with Metal textures for seam masks and optical flow data.

Stdin: pickle-encoded dict with:
  - 'left': numpy array (3840×3840×3 uint8)
  - 'right': numpy array (3840×3840×3 uint8)
  - 'seam_mask_1': numpy array (7680×292×4 uint8 BGRA8Unorm)
  - 'seam_mask_2': numpy array (7680×292×4 uint8 BGRA8Unorm)
  - 'flow_texture': numpy array (789×394×4 float32 RGBA16Float)
  - 'param_path': string path to framework resources
  - 'camera_sn': camera serial number for calibration

Stdout: pickle-encoded numpy array (7680×3840×3 uint8)

This implementation:
1. Imports PyObjC and loads DualStitcher.framework
2. Creates MTLDevice and MTLCommandQueue for GPU operations
3. Creates Metal textures for seam masks and flow data
4. Instantiates Stitcher class
5. Calls dynamicStitchSeams with Metal textures
6. Converts output to numpy array and returns via pickle
"""

import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Setup logging to stderr
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def load_stitcher_framework() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Load DualStitcher framework and initialize Stitcher class.

    Returns:
        Tuple of (Stitcher_class, metal_device, autorelease_pool) or (None, None, None) on error
    """
    try:
        import objc
        from Foundation import NSBundle, NSAutoreleasePool, NSData
        from Metal import MTLCreateSystemDefaultDevice

        logger.debug("PyObjC and Metal imports successful")
    except ImportError as e:
        logger.error(f"PyObjC not available: {e}")
        return None, None, None

    framework_path = Path("/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework")

    try:
        # Create autorelease pool
        pool = NSAutoreleasePool.alloc().init()
        logger.debug("Created autorelease pool")

        # Load framework
        logger.debug(f"Loading framework from {framework_path}")
        bundle = NSBundle.bundleWithPath_(str(framework_path))
        if not bundle:
            logger.error("Failed to load framework via NSBundle")
            pool.release()
            return None, None, None

        logger.debug("Framework loaded via NSBundle")
        bundle.load()

        # Get Stitcher class
        try:
            stitcher_class = objc.lookUpClass("Stitcher")
            logger.info(f"Found Stitcher class: {stitcher_class}")
        except objc.error as e:
            logger.error(f"Could not find Stitcher class: {e}")
            pool.release()
            return None, None, None

        # Create Metal device
        try:
            metal_device = MTLCreateSystemDefaultDevice()
            if metal_device:
                logger.info("Metal device created successfully")
            else:
                logger.warning("MTLCreateSystemDefaultDevice returned None")
        except Exception as e:
            logger.warning(f"Failed to create Metal device: {e}")
            metal_device = None

        return stitcher_class, metal_device, pool

    except Exception as e:
        logger.error(f"Framework loading failed: {e}", exc_info=True)
        return None, None, None


def create_stitcher_instance(stitcher_class, metal_device) -> Optional[Any]:
    """Create an instance of the Stitcher class.

    Args:
        stitcher_class: The Stitcher Objective-C class
        metal_device: Metal device (MTLDevice)

    Returns:
        Stitcher instance or None on error
    """
    try:
        logger.debug("Creating Stitcher instance...")

        # Try simple alloc().init()
        try:
            logger.debug("Trying Stitcher.alloc().init()...")
            stitcher = stitcher_class.alloc().init()
            if stitcher:
                logger.info("Stitcher instance created with alloc().init()")
                return stitcher
        except Exception as e:
            logger.debug(f"alloc().init() failed: {e}")

        # Try with Metal device if available
        if metal_device:
            try:
                logger.debug("Trying initWithMetalDevice:...")
                stitcher = stitcher_class.alloc().initWithMetalDevice_(metal_device)
                if stitcher:
                    logger.info("Stitcher instance created with Metal device")
                    return stitcher
            except Exception as e:
                logger.debug(f"Metal device initialization failed: {e}")

        logger.error("All Stitcher initialization methods failed")
        return None

    except Exception as e:
        logger.error(f"Failed to create Stitcher instance: {e}", exc_info=True)
        return None


def create_metal_texture_from_array(
    metal_device: Any,
    array: np.ndarray,
    width: int,
    height: int,
    pixel_format_str: str = "BGRA8Unorm"
) -> Optional[Any]:
    """Create a Metal texture from a numpy array.

    Args:
        metal_device: MTLDevice
        array: numpy array (height×width×channels)
        width: texture width
        height: texture height
        pixel_format_str: Metal pixel format ("BGRA8Unorm", "RGBA16Float", etc.)

    Returns:
        MTLTexture or None on error
    """
    try:
        from Metal import (
            MTLTextureDescriptor,
            MTLPixelFormatBGRA8Unorm,
            MTLPixelFormatRGBA16Float,
            MTLResourceStorageModeManaged,
        )
        from Foundation import NSData

        logger.debug(f"Creating Metal texture: {width}×{height} {pixel_format_str}")

        # Map format strings to Metal formats
        format_map = {
            "BGRA8Unorm": MTLPixelFormatBGRA8Unorm,
            "RGBA8Unorm": MTLPixelFormatBGRA8Unorm,  # Fallback
            "RGBA16Float": MTLPixelFormatRGBA16Float,
        }

        pixel_format = format_map.get(pixel_format_str, MTLPixelFormatBGRA8Unorm)
        logger.debug(f"Using pixel format: {pixel_format}")

        # Create texture descriptor
        desc = MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_(
            pixel_format, width, height
        )
        desc.setResourceOptions_(MTLResourceStorageModeManaged)

        # Create texture
        texture = metal_device.newTextureWithDescriptor_(desc)
        if not texture:
            logger.error("Failed to create Metal texture")
            return None

        logger.debug(f"Created texture: {texture}")

        # Copy data to texture
        array_contig = np.ascontiguousarray(array)
        bytes_per_row = width * array.shape[2] if len(array.shape) > 2 else width

        try:
            from Foundation import NSData

            # Create NSData from array
            array_bytes = array_contig.tobytes()
            ns_data = NSData.dataWithBytesNoCopy_length_(
                array_bytes,
                len(array_bytes)
            )

            # Get region for the entire texture
            from Metal import MTLRegionMake2D
            region = MTLRegionMake2D(0, 0, width, height)

            # Copy data to texture
            texture.replaceRegion_mipmapLevel_withBytes_(
                region, 0, array_bytes
            )

            logger.debug(f"Copied data to texture: {len(array_bytes)} bytes")
            return texture
        except Exception as e:
            logger.warning(f"Failed to copy data via NSData: {e}")
            # Try direct bytes method
            try:
                array_bytes = array_contig.tobytes()
                region = MTLRegionMake2D(0, 0, width, height) if 'MTLRegionMake2D' in dir() else (0, 0, width, height)
                texture.replaceRegion_mipmapLevel_withBytes_(
                    region, 0, array_bytes
                )
                logger.debug(f"Copied data using direct bytes: {len(array_bytes)} bytes")
                return texture
            except Exception as e2:
                logger.error(f"Failed to copy data to texture: {e2}")
                return None

    except Exception as e:
        logger.error(f"Failed to create Metal texture: {e}", exc_info=True)
        return None


def stitch_frames_dynamic(
    stitcher: Any,
    frame_left: np.ndarray,
    frame_right: np.ndarray,
    seam_mask_1: np.ndarray,
    seam_mask_2: np.ndarray,
    flow_texture_array: np.ndarray,
    metal_device: Any,
) -> Optional[np.ndarray]:
    """Perform stitching using dynamicStitchSeams method.

    This is the key method that uses optical flow and seam masks for proper stitching.

    Args:
        stitcher: Stitcher instance
        frame_left: Left frame (3840×3840×3 uint8)
        frame_right: Right frame (3840×3840×3 uint8)
        seam_mask_1: Seam mask 1 (7680×292×4 uint8)
        seam_mask_2: Seam mask 2 (7680×292×4 uint8)
        flow_texture_array: Flow texture (789×394×4 float32)
        metal_device: MTLDevice for texture creation

    Returns:
        Stitched frame (7680×3840×3 uint8) or None on error
    """
    try:
        import objc

        # Ensure contiguous arrays
        frame_left = np.ascontiguousarray(frame_left, dtype=np.uint8)
        frame_right = np.ascontiguousarray(frame_right, dtype=np.uint8)
        seam_mask_1 = np.ascontiguousarray(seam_mask_1, dtype=np.uint8)
        seam_mask_2 = np.ascontiguousarray(seam_mask_2, dtype=np.uint8)
        flow_texture_array = np.ascontiguousarray(flow_texture_array, dtype=np.float32)

        logger.debug(f"Frames: left={frame_left.shape} {frame_left.dtype}, right={frame_right.shape} {frame_right.dtype}")
        logger.debug(f"Seam masks: {seam_mask_1.shape} {seam_mask_1.dtype}")
        logger.debug(f"Flow texture: {flow_texture_array.shape} {flow_texture_array.dtype}")

        # Try to create Metal textures for seam masks and flow data
        logger.debug("Creating Metal textures...")

        seam_tex_1 = create_metal_texture_from_array(
            metal_device, seam_mask_1, 7680, 292, "BGRA8Unorm"
        )
        seam_tex_2 = create_metal_texture_from_array(
            metal_device, seam_mask_2, 7680, 292, "BGRA8Unorm"
        )
        flow_tex = create_metal_texture_from_array(
            metal_device, flow_texture_array, 789, 394, "RGBA16Float"
        )

        if not seam_tex_1 or not seam_tex_2:
            logger.warning("Failed to create seam textures, trying fallback...")

        # Try dynamicStitchSeams with Metal textures
        logger.debug("Calling dynamicStitchSeams:seam1:isCompensate:enableSunGlare:sgGammaTex:flowTex:...")
        try:
            result = stitcher.dynamicStitchSeams_seam1_isCompensate_enableSunGlare_sgGammaTex_flowTex_(
                seam_tex_1 if seam_tex_1 else seam_mask_1,  # seam0/left texture
                seam_tex_2 if seam_tex_2 else seam_mask_2,  # seam1/right texture
                True,   # isCompensate
                False,  # enableSunGlare
                None,   # sgGammaTex
                flow_tex if flow_tex else flow_texture_array  # flowTex
            )
            if result is not None:
                logger.info("dynamicStitchSeams returned non-None result")
                return convert_objc_result_to_numpy(result)
            else:
                logger.debug("dynamicStitchSeams returned None")
        except Exception as e:
            logger.debug(f"dynamicStitchSeams call failed: {e}")

        # Try alternative method: simpleStitchSeams (fallback)
        logger.debug("Falling back to simpleStitchSeams...")
        try:
            result = stitcher.simpleStitchSeams_seam1_isCompensate_enableSunGlare_sgGammaTex_(
                frame_left,
                frame_right,
                True,   # isCompensate
                False,  # enableSunGlare
                None    # sgGammaTex
            )
            if result is not None:
                logger.info("simpleStitchSeams succeeded as fallback")
                return convert_objc_result_to_numpy(result)
        except Exception as e:
            logger.debug(f"simpleStitchSeams fallback failed: {e}")

        # Try direct method lookup
        logger.debug("Trying direct method lookup...")
        method_names = [
            "stitchSeams_seam1_isCompensate_enableSunGlare_sgGammaTex_flowTex_",
            "dynamicStitchSeams:seam1:isCompensate:enableSunGlare:sgGammaTex:flowTex:",
        ]

        for method_name in method_names:
            try:
                logger.debug(f"Trying {method_name}...")
                if hasattr(stitcher, method_name):
                    result = getattr(stitcher, method_name)(
                        seam_tex_1, seam_tex_2, True, False, None, flow_tex
                    )
                    if result is not None:
                        logger.info(f"{method_name} succeeded")
                        return convert_objc_result_to_numpy(result)
            except Exception as e:
                logger.debug(f"{method_name} failed: {e}")

        logger.error("No stitching method succeeded")
        return None

    except Exception as e:
        logger.error(f"Stitching failed: {e}", exc_info=True)
        return None


def convert_objc_result_to_numpy(result: Any) -> Optional[np.ndarray]:
    """Convert Objective-C result to numpy array.

    Args:
        result: Result from Objective-C stitcher (could be NSData, MTLTexture, etc.)

    Returns:
        numpy array (7680×3840×3 uint8) or None on error
    """
    try:
        logger.debug(f"Result type: {type(result)}, {result.__class__.__name__}")

        # If already numpy array
        if isinstance(result, np.ndarray):
            logger.debug(f"Result is already numpy array: {result.shape}")
            return result

        # If NSData
        if hasattr(result, 'bytes'):
            logger.debug("Converting NSData to bytes...")
            data_bytes = result.bytes()
            array = np.frombuffer(data_bytes, dtype=np.uint8).copy()
            expected_size = 7680 * 3840 * 3

            if array.size == expected_size:
                array = array.reshape((7680, 3840, 3))
                logger.info(f"Converted NSData to numpy array: {array.shape}")
                return array
            else:
                logger.warning(f"NSData size mismatch: {array.size} vs expected {expected_size}")
                try:
                    array = array.reshape((3840, 7680, 3))
                    logger.info(f"Reshaped to (3840, 7680, 3)")
                    return array
                except:
                    pass

        # If buffer-like
        if hasattr(result, '__buffer__'):
            logger.debug("Result is buffer-like, converting...")
            array = np.frombuffer(result, dtype=np.uint8).copy()
            expected_size = 7680 * 3840 * 3

            if array.size == expected_size:
                array = array.reshape((7680, 3840, 3))
                logger.info(f"Converted buffer to numpy array: {array.shape}")
                return array

        logger.error(f"Cannot convert result type: {type(result)}")
        return None

    except Exception as e:
        logger.error(f"Failed to convert Objective-C result: {e}", exc_info=True)
        return None


def main():
    """Main entry point for subprocess stitcher"""
    try:
        # Read input
        logger.debug("Reading input from stdin...")
        input_data = pickle.load(sys.stdin.buffer)

        frame_left = input_data['left']
        frame_right = input_data['right']
        seam_mask_1 = input_data['seam_mask_1']
        seam_mask_2 = input_data['seam_mask_2']
        flow_texture = input_data['flow_texture']
        param_path = Path(input_data['param_path'])
        camera_sn = input_data.get('camera_sn', 'default')

        logger.debug(f"Loaded frames: left={frame_left.shape}, right={frame_right.shape}")
        logger.debug(f"Loaded seam masks: {seam_mask_1.shape}, {seam_mask_2.shape}")
        logger.debug(f"Loaded flow texture: {flow_texture.shape}")

        # Verify inputs
        assert frame_left.shape == (3840, 3840, 3), f"Invalid left shape: {frame_left.shape}"
        assert frame_right.shape == (3840, 3840, 3), f"Invalid right shape: {frame_right.shape}"
        assert seam_mask_1.shape == (292, 7680, 4), f"Invalid seam mask 1: {seam_mask_1.shape}"
        assert seam_mask_2.shape == (292, 7680, 4), f"Invalid seam mask 2: {seam_mask_2.shape}"
        assert flow_texture.shape == (394, 789, 4), f"Invalid flow texture: {flow_texture.shape}"

        # Ensure contiguous
        frame_left = np.ascontiguousarray(frame_left)
        frame_right = np.ascontiguousarray(frame_right)
        seam_mask_1 = np.ascontiguousarray(seam_mask_1)
        seam_mask_2 = np.ascontiguousarray(seam_mask_2)
        flow_texture = np.ascontiguousarray(flow_texture)

        # Load framework
        logger.info("Loading DualStitcher framework...")
        stitcher_class, metal_device, pool = load_stitcher_framework()

        if not stitcher_class:
            logger.error("Failed to load Stitcher class")
            sys.exit(1)

        # Create instance
        logger.info("Creating Stitcher instance...")
        stitcher = create_stitcher_instance(stitcher_class, metal_device)

        if not stitcher:
            logger.error("Failed to create Stitcher instance")
            sys.exit(1)

        # Check initialization
        if hasattr(stitcher, 'isAllAllocated'):
            try:
                is_allocated = stitcher.isAllAllocated()
                logger.info(f"Stitcher.isAllAllocated(): {is_allocated}")
            except Exception as e:
                logger.warning(f"Could not call isAllAllocated: {e}")

        # Perform stitching
        logger.info("Performing dynamic seam stitching...")
        output_buffer = stitch_frames_dynamic(
            stitcher, frame_left, frame_right,
            seam_mask_1, seam_mask_2, flow_texture,
            metal_device
        )

        if output_buffer is None:
            logger.error("Dynamic stitching failed")
            sys.exit(1)

        # Verify output
        logger.debug(f"Stitching output shape: {output_buffer.shape}")
        logger.debug(f"Output dtype: {output_buffer.dtype}")
        logger.debug(f"Output value range: [{output_buffer.min()}, {output_buffer.max()}]")

        # Ensure correct shape
        if output_buffer.shape != (7680, 3840, 3):
            if output_buffer.shape == (3840, 7680, 3):
                logger.warning("Output is rotated (3840×7680 instead of 7680×3840)")
                output_buffer = np.transpose(output_buffer, (1, 0, 2))
            else:
                logger.warning(f"Unexpected output shape: {output_buffer.shape}")
                expected_size = 7680 * 3840 * 3
                if output_buffer.size == expected_size:
                    output_buffer = output_buffer.reshape((7680, 3840, 3))
                    logger.info("Reshaped output to (7680, 3840, 3)")
                else:
                    logger.error(f"Output size mismatch: {output_buffer.size} != {expected_size}")
                    sys.exit(1)

        # Ensure uint8
        if output_buffer.dtype != np.uint8:
            logger.debug(f"Converting output from {output_buffer.dtype} to uint8")
            output_buffer = output_buffer.astype(np.uint8)

        # Write output
        logger.debug(f"Writing output to stdout: {output_buffer.shape}")
        pickle.dump(output_buffer, sys.stdout.buffer)
        logger.info("Stitching completed successfully")

    except Exception as e:
        logger.error(f"Error in stitcher: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
