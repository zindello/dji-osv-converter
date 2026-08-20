"""PyObjC bridge for DualStitcher framework

Loads macOS frameworks using PyObjC's Objective-C bindings.
Handles dependency resolution (openmcv2, ml_fis) and provides
singleton access to the DualStitcher Stitcher class.
"""

import logging
from pathlib import Path
from typing import Optional
import objc
from Foundation import NSBundle

logger = logging.getLogger(__name__)

# Framework paths
FRAMEWORKS_DIR = Path("/Applications/DJI Studio.app/Contents/Frameworks")
DUALSTITCHER_FRAMEWORK = FRAMEWORKS_DIR / "DualStitcher.framework"
OPENMCV2_FRAMEWORK = FRAMEWORKS_DIR / "openmcv2.framework"
ML_FIS_FRAMEWORK = FRAMEWORKS_DIR / "ml_fis.framework"


class FrameworkLoadError(Exception):
    """Raised when framework loading fails"""
    pass


class ObjCBridge:
    """Singleton manager for Objective-C framework access"""

    _instance: Optional["ObjCBridge"] = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._bundle = None
        self._stitcher_class = None
        self._dependencies_loaded = False
        self._stitcher_info_class = None

        self._load_dependencies()
        self._load_framework()
        self._initialize_classes()
        ObjCBridge._initialized = True

    def _load_dependencies(self) -> None:
        """Load framework dependencies before main framework"""
        dependencies = [
            (OPENMCV2_FRAMEWORK, "openmcv2"),
            (ML_FIS_FRAMEWORK, "ml_fis"),
        ]

        for framework_path, name in dependencies:
            if not framework_path.exists():
                logger.warning(f"Optional dependency not found: {name} at {framework_path}")
                continue

            try:
                bundle = NSBundle.bundleWithPath_(str(framework_path))
                if bundle:
                    logger.debug(f"Loaded dependency framework: {name}")
                else:
                    logger.warning(f"Failed to load dependency framework: {name}")
            except Exception as e:
                logger.warning(f"Could not load dependency {name}: {e}")

        self._dependencies_loaded = True

    def _load_framework(self) -> None:
        """Load the main DualStitcher framework"""
        if not DUALSTITCHER_FRAMEWORK.exists():
            raise FrameworkLoadError(
                f"DualStitcher framework not found at {DUALSTITCHER_FRAMEWORK}. "
                "Please ensure DJI Studio is installed."
            )

        try:
            self._bundle = NSBundle.bundleWithPath_(str(DUALSTITCHER_FRAMEWORK))
            if not self._bundle:
                raise FrameworkLoadError("NSBundle returned None for DualStitcher framework")

            # NSBundle.bundleWithPath_() automatically loads the bundle,
            # no explicit load() call needed
            logger.debug(f"Loaded DualStitcher framework from {DUALSTITCHER_FRAMEWORK}")

        except Exception as e:
            raise FrameworkLoadError(f"Failed to load DualStitcher framework: {e}") from e

    def _initialize_classes(self) -> None:
        """Discover and cache Objective-C classes from framework"""
        if not self._bundle:
            raise FrameworkLoadError("Framework not loaded")

        # Get the main Stitcher class from the framework
        try:
            self._stitcher_class = objc.objc_getClass("Stitcher")
            if not self._stitcher_class:
                raise FrameworkLoadError("Could not find Stitcher class in framework")
            logger.debug("Found Stitcher class")
        except Exception as e:
            raise FrameworkLoadError(f"Failed to get Stitcher class: {e}") from e

        # Try to get StitcherInfo if it exists
        try:
            self._stitcher_info_class = objc.objc_getClass("StitcherInfo")
            if self._stitcher_info_class:
                logger.debug("Found StitcherInfo class")
        except Exception:
            logger.debug("StitcherInfo class not found (may be internal structure)")

    def is_framework_available(self) -> bool:
        """Check if framework was successfully loaded"""
        return (
            self._bundle is not None
            and self._stitcher_class is not None
            and DUALSTITCHER_FRAMEWORK.exists()
        )

    def get_stitcher_class(self):
        """Get the Objective-C Stitcher class"""
        if not self._stitcher_class:
            raise FrameworkLoadError("Stitcher class not loaded")
        return self._stitcher_class

    def get_stitcher_info_class(self):
        """Get the Objective-C StitcherInfo class if available"""
        return self._stitcher_info_class

    def get_framework_path(self) -> Path:
        """Get path to DualStitcher framework"""
        return DUALSTITCHER_FRAMEWORK

    def get_framework_bundle(self):
        """Get NSBundle for DualStitcher framework"""
        return self._bundle

    def get_info(self) -> dict:
        """Get information about loaded frameworks"""
        return {
            "framework_available": self.is_framework_available(),
            "framework_path": str(DUALSTITCHER_FRAMEWORK),
            "framework_exists": DUALSTITCHER_FRAMEWORK.exists(),
            "dependencies_loaded": self._dependencies_loaded,
            "stitcher_class_available": self._stitcher_class is not None,
            "stitcher_info_class_available": self._stitcher_info_class is not None,
        }


def get_objc_bridge() -> ObjCBridge:
    """Get or create singleton ObjC bridge instance"""
    return ObjCBridge()


__all__ = [
    "ObjCBridge",
    "FrameworkLoadError",
    "get_objc_bridge",
    "FRAMEWORKS_DIR",
    "DUALSTITCHER_FRAMEWORK",
]
