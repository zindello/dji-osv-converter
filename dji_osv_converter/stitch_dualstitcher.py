"""DualStitcher Framework Wrapper - Complete API Implementation

This module provides comprehensive Python bindings to DJI's DualStitcher framework
for optical flow-based stitching of multi-camera 360° video.

Based on reverse-engineering of DualStitcher.framework (macOS binary analysis).
Supports both Objective-C API (Metal GPU) and C API (ML models).

API Signatures Discovered:
- Objective-C: Stitcher class with Metal texture-based methods
- C API: ML_ImageStitcher* functions for CPU-based processing
- Configuration: device-specific calibration files in Stitcher-Mac.bundle
"""

import ctypes
import logging
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from enum import IntEnum
import numpy as np

try:
    import objc
    from Foundation import NSBundle, NSString, NSData
    PYOBJC_AVAILABLE = True
except ImportError:
    PYOBJC_AVAILABLE = False

logger = logging.getLogger(__name__)

# Framework paths
FRAMEWORK_PATH = Path("/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework")
FRAMEWORK_BINARY = FRAMEWORK_PATH / "Versions/A/DualStitcher"
FRAMEWORK_RESOURCES = FRAMEWORK_PATH / "Versions/A/Resources/Stitcher-Mac.bundle/case_00"
FRAMEWORKS_DIR = Path("/Applications/DJI Studio.app/Contents/Frameworks")

# Stitching method types
class StitchingMethod(IntEnum):
    """Available stitching algorithms"""
    EQUAL_WEIGHTS = 0
    SIMPLE_BLEND = 1
    FLOW_STITCH = 2
    AI_STITCH = 3
    HIGH_PRECISION = 4
    FLOW_STITCH_CALC = 5


class DeviceType(IntEnum):
    """GPU device types"""
    APPLE_GPU = 0  # Metal (macOS)
    NVIDIA_GPU = 1  # CUDA (Windows/Linux)


class DualStitcherError(Exception):
    """Exception raised for DualStitcher-related errors"""
    pass


class CalibrationData:
    """Camera calibration data from intrinsic_extrinsic JSON"""

    def __init__(self, json_data: Dict[str, Any]):
        """Initialize from JSON structure

        Args:
            json_data: Parsed JSON dictionary with intrinsic00, extrinsic00, etc.
        """
        self.intrinsic00 = json_data.get("intrinsic00", {})
        self.intrinsic01 = json_data.get("intrinsic01", {})
        self.extrinsic00 = json_data.get("extrinsic00", {})
        self.extrinsic01 = json_data.get("extrinsic01", {})

    def get_camera_matrix_left(self) -> np.ndarray:
        """Get intrinsic camera matrix for left camera (K matrix)"""
        intr = self.intrinsic00
        return np.array([
            [intr.get('fx', 1777), 0, intr.get('cx', 1912)],
            [0, intr.get('fy', 1777), intr.get('cy', 1927)],
            [0, 0, 1]
        ], dtype=np.float32)

    def get_camera_matrix_right(self) -> np.ndarray:
        """Get intrinsic camera matrix for right camera"""
        intr = self.intrinsic01
        return np.array([
            [intr.get('fx', 1777), 0, intr.get('cx', 1900)],
            [0, intr.get('fy', 1777), intr.get('cy', 1933)],
            [0, 0, 1]
        ], dtype=np.float32)

    def get_distortion_coeffs_left(self) -> np.ndarray:
        """Get distortion coefficients for left camera"""
        intr = self.intrinsic00
        return np.array([
            intr.get('k1', -0.183),
            intr.get('k2', 0.0184),
            intr.get('p1', 0.0001),
            intr.get('p2', 0.00016),
            intr.get('k3', 0.0),
            intr.get('k4', 0.0),
        ], dtype=np.float32)

    def get_rotation_extrinsic_left(self) -> Dict[str, float]:
        """Get rotation (yaw, pitch, roll) for left camera"""
        return self.extrinsic00

    def get_rotation_extrinsic_right(self) -> Dict[str, float]:
        """Get rotation (yaw, pitch, roll) for right camera"""
        return self.extrinsic01


class StitcherConfig:
    """Configuration for DualStitcher"""

    def __init__(
        self,
        camera_sn: str = "default",
        stitching_method: StitchingMethod = StitchingMethod.AI_STITCH,
        device_type: DeviceType = DeviceType.APPLE_GPU,
        enable_sun_glare_correction: bool = True,
        enable_lsc: bool = False,  # Lens shading correction
        enable_ai: bool = True,
        dynamic_stitching: bool = True,
    ):
        """Initialize stitcher configuration.

        Args:
            camera_sn: Camera serial number (matches device-specific parameters)
            stitching_method: Which stitching algorithm to use
            device_type: GPU device type (Metal on macOS)
            enable_sun_glare_correction: Enable sun glare detection/removal
            enable_lsc: Enable lens shading correction
            enable_ai: Use AI-based improvements
            dynamic_stitching: Handle moving scenes
        """
        self.camera_sn = camera_sn
        self.stitching_method = stitching_method
        self.device_type = device_type
        self.enable_sun_glare_correction = enable_sun_glare_correction
        self.enable_lsc = enable_lsc
        self.enable_ai = enable_ai
        self.dynamic_stitching = dynamic_stitching

        # Load device-specific parameters
        self.device_param_index = self._load_device_params()
        self.calibration_data = self._load_calibration()

    def _load_device_params(self) -> int:
        """Load device-specific parameter index from framework resources"""
        params_file = FRAMEWORK_RESOURCES / "param_list.json"
        if not params_file.exists():
            logger.warning(f"Parameter list not found: {params_file}")
            return 0

        try:
            with open(params_file, 'r') as f:
                params = json.load(f)

            # Find this device's parameter index
            for entry in params.get("param_list", []):
                if entry["sn_name"] == self.camera_sn:
                    logger.debug(f"Loaded parameters for {self.camera_sn}: index {entry['input']}")
                    return entry["input"]

            # Default to index 0
            logger.warning(f"Camera {self.camera_sn} not found in param_list, using default")
            return 0

        except Exception as e:
            logger.error(f"Failed to load device parameters: {e}")
            return 0

    def _load_calibration(self) -> Optional[CalibrationData]:
        """Load camera calibration from framework resources"""
        calib_file = FRAMEWORK_RESOURCES / "intrinsic_extrinsic.json"
        if not calib_file.exists():
            logger.warning(f"Calibration file not found: {calib_file}")
            return None

        try:
            with open(calib_file, 'r') as f:
                calib_data = json.load(f)
            return CalibrationData(calib_data)
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return None


class DualStitcherObjC:
    """High-level Objective-C wrapper for Metal GPU stitching

    Requires PyObjC and Metal framework (macOS only).
    Provides the best quality and performance through GPU acceleration.
    """

    def __init__(self, config: Optional[StitcherConfig] = None):
        """Initialize ObjC-based DualStitcher.

        Args:
            config: StitcherConfig object (uses defaults if None)

        Raises:
            DualStitcherError: If framework cannot be loaded
        """
        if not PYOBJC_AVAILABLE:
            raise DualStitcherError(
                "PyObjC not available. Install with: pip install pyobjc"
            )

        if config is None:
            config = StitcherConfig()

        self.config = config
        self._stitcher_class = None
        self._stitcher_instance = None
        self._bundle = None

        self._load_framework()

    def _load_framework(self):
        """Load DualStitcher framework via NSBundle"""
        if not FRAMEWORK_PATH.exists():
            raise DualStitcherError(
                f"DualStitcher framework not found. "
                f"Please install DJI Studio at {FRAMEWORK_PATH.parent.parent}"
            )

        try:
            # Load dependencies first (openmcv2, ml_fis)
            for dep_name in ["openmcv2", "ml_fis"]:
                dep_path = FRAMEWORKS_DIR / f"{dep_name}.framework"
                if dep_path.exists():
                    try:
                        objc.loadBundle(
                            dep_name,
                            globals(),
                            bundle_path=str(dep_path)
                        )
                        logger.debug(f"Loaded dependency: {dep_name}")
                    except Exception as e:
                        logger.warning(f"Could not load dependency {dep_name}: {e}")

            # Load main framework
            self._bundle = NSBundle.bundleWithPath_(str(FRAMEWORK_PATH))
            if not self._bundle:
                raise DualStitcherError("NSBundle returned None for DualStitcher")

            # Get Stitcher class - try different methods for compatibility
            try:
                # Method 1: Direct objc_getClass (newer PyObjC)
                if hasattr(objc, 'objc_getClass'):
                    self._stitcher_class = objc.objc_getClass("Stitcher")
                # Method 2: Use objc.lookUpClass (older PyObjC)
                elif hasattr(objc, 'lookUpClass'):
                    self._stitcher_class = objc.lookUpClass("Stitcher")
                # Method 3: Load from bundle dynamically
                else:
                    objc.loadBundle(
                        "DualStitcher",
                        globals(),
                        bundle_path=str(FRAMEWORK_PATH)
                    )
                    # Try to access via framework name
                    self._stitcher_class = globals().get("Stitcher")
            except Exception as e:
                logger.warning(f"Could not load Stitcher class: {e}")
                self._stitcher_class = None

            if not self._stitcher_class:
                logger.warning("Stitcher class not found, ObjC API will not be available")

            logger.debug(f"Loaded DualStitcher framework from {FRAMEWORK_PATH}")

        except DualStitcherError:
            raise
        except Exception as e:
            raise DualStitcherError(
                f"Failed to load DualStitcher framework: {e}. "
                "Make sure DJI Studio is installed."
            ) from e

    def is_available(self) -> bool:
        """Check if DualStitcher framework is available"""
        return (
            self._stitcher_class is not None
            and self._bundle is not None
            and FRAMEWORK_BINARY.exists()
        )

    def stitch_frame_ai(
        self,
        img_left: Any,  # MTLTexture
        img_right: Any,  # MTLTexture
        compensate: bool = True,
        enable_sun_glare: bool = True,
        sg_gamma_tex: Optional[Any] = None,
    ) -> Optional[Any]:  # Returns MTLTexture
        """Stitch two images using AI-based seam stitching.

        This is the recommended method for best quality.
        Uses neural networks for optical flow refinement.

        Args:
            img_left: MTLTexture for left fisheye image
            img_right: MTLTexture for right fisheye image
            compensate: Enable exposure/color compensation
            enable_sun_glare: Enable sun glare correction
            sg_gamma_tex: MTLTexture for sun glare gamma (optional)

        Returns:
            MTLTexture with stitched equirectangular image
        """
        if not self.is_available():
            raise DualStitcherError("Stitcher not available")

        if not self._stitcher_instance:
            raise DualStitcherError("Stitcher not initialized")

        try:
            # Call Objective-C method:
            # -[Stitcher aiStitchSeams:seam1:isCompensate:useLSC:enableSunGlare:sgGammaTex:]
            result = self._stitcher_instance.aiStitchSeams_seam1_isCompensate_useLSC_enableSunGlare_sgGammaTex_(
                img_left,
                img_right,
                compensate,
                self.config.enable_lsc,
                enable_sun_glare,
                sg_gamma_tex
            )
            return result
        except Exception as e:
            logger.error(f"aiStitchSeams failed: {e}")
            return None

    def stitch_frame_flow(
        self,
        img_left: Any,  # MTLTexture
        img_right: Any,  # MTLTexture
        is_dynamic: bool = True,
        compensate: bool = True,
        enable_sun_glare: bool = True,
    ) -> Optional[Any]:  # Returns MTLTexture
        """Stitch using optical flow-based seam stitching.

        Good balance between quality and performance.
        Uses bidirectional optical flow for seam detection.

        Args:
            img_left: MTLTexture for left image
            img_right: MTLTexture for right image
            is_dynamic: Handle moving scenes
            compensate: Enable compensation
            enable_sun_glare: Enable sun glare correction

        Returns:
            MTLTexture with stitched result
        """
        if not self.is_available():
            raise DualStitcherError("Stitcher not available")

        if not self._stitcher_instance:
            raise DualStitcherError("Stitcher not initialized")

        try:
            # -[Stitcher FlowStitchSeams:seam1:isDynamic:isCompensate:useLSC:deviceType:enableSunGlare:sgGammaTex:]
            result = self._stitcher_instance.FlowStitchSeams_seam1_isDynamic_isCompensate_useLSC_deviceType_enableSunGlare_sgGammaTex_(
                img_left,
                img_right,
                is_dynamic,
                compensate,
                self.config.enable_lsc,
                int(self.config.device_type),
                enable_sun_glare,
                None  # sgGammaTex
            )
            return result
        except Exception as e:
            logger.error(f"FlowStitchSeams failed: {e}")
            return None

    def stitch_frame_simple(
        self,
        img_left: Any,  # MTLTexture
        img_right: Any,  # MTLTexture
    ) -> Optional[Any]:  # Returns MTLTexture
        """Stitch using simple equal-weights blending.

        Fastest method, suitable for preview/testing.

        Args:
            img_left: MTLTexture for left image
            img_right: MTLTexture for right image

        Returns:
            MTLTexture with stitched result
        """
        if not self.is_available():
            raise DualStitcherError("Stitcher not available")

        if not self._stitcher_instance:
            raise DualStitcherError("Stitcher not initialized")

        try:
            result = self._stitcher_instance.equalWeightsStitchSeams_seam1_(
                img_left,
                img_right
            )
            return result
        except Exception as e:
            logger.error(f"equalWeightsStitchSeams failed: {e}")
            return None

    def get_info(self) -> dict:
        """Get information about the loaded DualStitcher framework"""
        return {
            "framework_path": str(FRAMEWORK_BINARY),
            "is_available": self.is_available(),
            "camera_sn": self.config.camera_sn,
            "device_param_index": self.config.device_param_index,
            "device_type": "Metal" if self.config.device_type == DeviceType.APPLE_GPU else "CUDA",
            "stitching_method": self.config.stitching_method.name,
            "config": {
                "enable_sun_glare_correction": self.config.enable_sun_glare_correction,
                "enable_lsc": self.config.enable_lsc,
                "enable_ai": self.config.enable_ai,
                "dynamic_stitching": self.config.dynamic_stitching,
            },
            "pyobjc_available": PYOBJC_AVAILABLE,
            "calibration_loaded": self.config.calibration_data is not None,
        }

    def __repr__(self) -> str:
        return (
            f"DualStitcherObjC(camera={self.config.camera_sn}, "
            f"method={self.config.stitching_method.name}, "
            f"available={self.is_available()})"
        )


class DualStitcherC:
    """Low-level C API wrapper for ML-based stitching

    Uses ctypes to call C functions directly.
    More portable but requires understanding of C data types.
    """

    def __init__(self, config: Optional[StitcherConfig] = None):
        """Initialize C-based DualStitcher.

        Args:
            config: StitcherConfig object (uses defaults if None)

        Raises:
            DualStitcherError: If framework cannot be loaded
        """
        if config is None:
            config = StitcherConfig()

        self.config = config
        self._lib = None
        self._load_library()

    def _load_library(self):
        """Load DualStitcher binary via ctypes"""
        if not FRAMEWORK_BINARY.exists():
            raise DualStitcherError(
                f"DualStitcher framework not found. "
                f"Please install DJI Studio."
            )

        try:
            # Set DYLD_FRAMEWORK_PATH to help resolve @rpath dependencies
            env = os.environ.copy()
            env['DYLD_FRAMEWORK_PATH'] = str(FRAMEWORKS_DIR)

            # Load the main library
            self._lib = ctypes.CDLL(str(FRAMEWORK_BINARY), use_errno=True)
            logger.debug(f"Loaded DualStitcher via ctypes from {FRAMEWORK_BINARY}")

            # Define C function signatures
            self._define_c_functions()

        except OSError as e:
            raise DualStitcherError(
                f"Failed to load DualStitcher C library: {e}. "
                "Make sure DJI Studio is installed."
            ) from e

    def _define_c_functions(self):
        """Define C function signatures for ctypes"""
        if not self._lib:
            return

        # ML_ImageStitcherInit
        self._ML_ImageStitcherInit = self._lib.ML_ImageStitcherInit
        self._ML_ImageStitcherInit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self._ML_ImageStitcherInit.restype = ctypes.c_int

        # ML_ImageStitcherDeInit
        self._ML_ImageStitcherDeInit = self._lib.ML_ImageStitcherDeInit
        self._ML_ImageStitcherDeInit.argtypes = []
        self._ML_ImageStitcherDeInit.restype = ctypes.c_int

        # ML_ImageStitcherProcessStepA
        self._ML_ImageStitcherProcessStepA = self._lib.ML_ImageStitcherProcessStepA
        self._ML_ImageStitcherProcessStepA.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._ML_ImageStitcherProcessStepA.restype = ctypes.c_int

        # ML_ImageStitcherProcessStepB
        self._ML_ImageStitcherProcessStepB = self._lib.ML_ImageStitcherProcessStepB
        self._ML_ImageStitcherProcessStepB.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._ML_ImageStitcherProcessStepB.restype = ctypes.c_int

    def is_available(self) -> bool:
        """Check if C library is loaded"""
        return self._lib is not None and FRAMEWORK_BINARY.exists()

    def process_step_a(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        width: int,
        height: int,
        stride: int,
    ) -> int:
        """Process step A (feature extraction/optical flow)

        Args:
            img_left: Left image as uint8 numpy array
            img_right: Right image as uint8 numpy array
            width: Image width
            height: Image height
            stride: Bytes per row

        Returns:
            Error code (0 = success)
        """
        if not self.is_available():
            raise DualStitcherError("C library not available")

        try:
            left_ptr = img_left.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            right_ptr = img_right.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

            result = self._ML_ImageStitcherProcessStepA(
                left_ptr,
                right_ptr,
                width,
                height,
                stride,
            )

            if result != 0:
                logger.warning(f"ML_ImageStitcherProcessStepA returned error code: {result}")

            return result
        except Exception as e:
            logger.error(f"process_step_a failed: {e}")
            return -1

    def process_step_b(
        self,
        output_buffer: np.ndarray,
        output_width: int,
        output_height: int,
    ) -> int:
        """Process step B (blending/seam placement)

        Args:
            output_buffer: Output buffer for stitched image (uint8)
            output_width: Output width (typically 7680)
            output_height: Output height (typically 3840)

        Returns:
            Error code (0 = success)
        """
        if not self.is_available():
            raise DualStitcherError("C library not available")

        try:
            output_ptr = output_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

            result = self._ML_ImageStitcherProcessStepB(
                output_ptr,
                output_width,
                output_height,
            )

            if result != 0:
                logger.warning(f"ML_ImageStitcherProcessStepB returned error code: {result}")

            return result
        except Exception as e:
            logger.error(f"process_step_b failed: {e}")
            return -1

    def get_info(self) -> dict:
        """Get information about the C library"""
        return {
            "framework_path": str(FRAMEWORK_BINARY),
            "is_available": self.is_available(),
            "camera_sn": self.config.camera_sn,
            "device_param_index": self.config.device_param_index,
            "calibration_loaded": self.config.calibration_data is not None,
        }

    def __repr__(self) -> str:
        return (
            f"DualStitcherC(camera={self.config.camera_sn}, "
            f"available={self.is_available()})"
        )


# Convenience factory function
def create_stitcher(
    use_objc: bool = True,
    camera_sn: str = "default",
    stitching_method: StitchingMethod = StitchingMethod.AI_STITCH,
) -> Any:  # Union[DualStitcherObjC, DualStitcherC]
    """Create a DualStitcher instance.

    Args:
        use_objc: Use Objective-C wrapper (True) or C wrapper (False)
        camera_sn: Camera serial number
        stitching_method: Stitching algorithm to use

    Returns:
        DualStitcher instance (ObjC or C variant)

    Raises:
        DualStitcherError: If framework cannot be loaded
    """
    config = StitcherConfig(
        camera_sn=camera_sn,
        stitching_method=stitching_method,
    )

    if use_objc:
        return DualStitcherObjC(config)
    else:
        return DualStitcherC(config)


# Export main classes and functions
__all__ = [
    "DualStitcher",
    "DualStitcherObjC",
    "DualStitcherC",
    "StitcherConfig",
    "CalibrationData",
    "DualStitcherError",
    "StitchingMethod",
    "DeviceType",
    "create_stitcher",
]

# Backward compatibility alias
DualStitcher = DualStitcherObjC
