#!/usr/bin/env python3
"""Test script for DualStitcher API bindings

Tests both Objective-C and C API implementations.
Verifies framework loading, configuration parsing, and basic functionality.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from dji_osv_converter.stitch_dualstitcher import (
    DualStitcherObjC,
    DualStitcherC,
    StitcherConfig,
    CalibrationData,
    StitchingMethod,
    DeviceType,
    create_stitcher,
    FRAMEWORK_PATH,
    FRAMEWORK_BINARY,
    FRAMEWORK_RESOURCES,
)


def test_framework_paths():
    """Test that framework paths exist and are accessible"""
    print("\n" + "="*60)
    print("TEST: Framework Paths")
    print("="*60)

    print(f"Framework path: {FRAMEWORK_PATH}")
    print(f"  Exists: {FRAMEWORK_PATH.exists()}")

    print(f"Binary path: {FRAMEWORK_BINARY}")
    print(f"  Exists: {FRAMEWORK_BINARY.exists()}")

    print(f"Resources path: {FRAMEWORK_RESOURCES}")
    print(f"  Exists: {FRAMEWORK_RESOURCES.exists()}")

    if FRAMEWORK_PATH.exists():
        print("\nFramework structure:")
        for item in sorted(FRAMEWORK_PATH.rglob("*"))[:10]:
            if item.is_file():
                size = item.stat().st_size
                print(f"  {item.relative_to(FRAMEWORK_PATH)}: {size} bytes")

    return FRAMEWORK_PATH.exists() and FRAMEWORK_BINARY.exists()


def test_config_loading():
    """Test configuration and calibration loading"""
    print("\n" + "="*60)
    print("TEST: Configuration Loading")
    print("="*60)

    try:
        config = StitcherConfig(camera_sn="default")
        print(f"Config created successfully")
        print(f"  Camera SN: {config.camera_sn}")
        print(f"  Device param index: {config.device_param_index}")
        print(f"  Stitching method: {config.stitching_method.name}")
        print(f"  Device type: {config.device_type.name}")
        print(f"  Enable AI: {config.enable_ai}")
        print(f"  Enable sun glare: {config.enable_sun_glare_correction}")

        if config.calibration_data:
            print("\nCalibration data loaded:")
            calib = config.calibration_data
            intr_left = calib.intrinsic00
            print(f"  Left camera intrinsics:")
            print(f"    fx={intr_left.get('fx', 'N/A')}, fy={intr_left.get('fy', 'N/A')}")
            print(f"    cx={intr_left.get('cx', 'N/A')}, cy={intr_left.get('cy', 'N/A')}")
            print(f"    k1={intr_left.get('k1', 'N/A')}, k2={intr_left.get('k2', 'N/A')}")
            print(f"  Right camera intrinsics:")
            intr_right = calib.intrinsic01
            print(f"    fx={intr_right.get('fx', 'N/A')}, fy={intr_right.get('fy', 'N/A')}")

        return True
    except Exception as e:
        logger.error(f"Config loading failed: {e}")
        return False


def test_c_api_loading():
    """Test C API library loading"""
    print("\n" + "="*60)
    print("TEST: C API Library Loading")
    print("="*60)

    try:
        stitcher_c = DualStitcherC()
        print(f"DualStitcherC instance created")
        print(f"  Available: {stitcher_c.is_available()}")

        info = stitcher_c.get_info()
        print(f"\nLibrary info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        return stitcher_c.is_available()
    except Exception as e:
        logger.error(f"C API loading failed: {e}")
        logger.error("This is expected if framework dependencies are not properly set up.")
        return False


def test_objc_api_loading():
    """Test Objective-C API loading"""
    print("\n" + "="*60)
    print("TEST: Objective-C API Loading")
    print("="*60)

    try:
        stitcher_objc = DualStitcherObjC()
        print(f"DualStitcherObjC instance created")
        print(f"  Available: {stitcher_objc.is_available()}")

        info = stitcher_objc.get_info()
        print(f"\nObjC framework info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        return stitcher_objc.is_available()
    except Exception as e:
        logger.error(f"ObjC API loading failed: {e}")
        logger.error("This is expected if PyObjC is not installed or framework not available.")
        return False


def test_factory_function():
    """Test convenience factory function"""
    print("\n" + "="*60)
    print("TEST: Factory Function")
    print("="*60)

    try:
        # Test C API
        stitcher = create_stitcher(use_objc=False)
        print(f"Created via factory (C): {stitcher}")
        print(f"  Type: {type(stitcher).__name__}")
        print(f"  Available: {stitcher.is_available()}")

        # Test ObjC API
        try:
            stitcher = create_stitcher(use_objc=True)
            print(f"\nCreated via factory (ObjC): {stitcher}")
            print(f"  Type: {type(stitcher).__name__}")
            print(f"  Available: {stitcher.is_available()}")
        except Exception as e:
            print(f"\nObjC factory failed (expected if PyObjC not available): {e}")

        return True
    except Exception as e:
        logger.error(f"Factory function test failed: {e}")
        return False


def test_calibration_data():
    """Test calibration data extraction"""
    print("\n" + "="*60)
    print("TEST: Calibration Data Processing")
    print("="*60)

    try:
        config = StitcherConfig()
        if not config.calibration_data:
            print("Calibration data not available")
            return False

        calib = config.calibration_data

        # Test camera matrix extraction
        K_left = calib.get_camera_matrix_left()
        print(f"Left camera matrix (K):")
        print(K_left)

        K_right = calib.get_camera_matrix_right()
        print(f"\nRight camera matrix (K):")
        print(K_right)

        # Test distortion coefficients
        dist_left = calib.get_distortion_coeffs_left()
        print(f"\nLeft camera distortion coefficients:")
        print(dist_left)

        # Test rotation extraction
        rot_left = calib.get_rotation_extrinsic_left()
        print(f"\nLeft camera rotation (yaw, pitch, roll):")
        print(f"  {rot_left}")

        return True
    except Exception as e:
        logger.error(f"Calibration data test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("DualStitcher API Test Suite")
    print("="*60)

    results = []

    # Test framework paths
    results.append(("Framework Paths", test_framework_paths()))

    # Test configuration
    results.append(("Configuration Loading", test_config_loading()))

    # Test calibration
    results.append(("Calibration Data", test_calibration_data()))

    # Test C API
    results.append(("C API Loading", test_c_api_loading()))

    # Test ObjC API
    results.append(("Objective-C API Loading", test_objc_api_loading()))

    # Test factory
    results.append(("Factory Function", test_factory_function()))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print("="*60)
    print(f"Total: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
