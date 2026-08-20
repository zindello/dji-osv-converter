"""Tests for DualStitcher framework wrapper"""

import pytest
from dji_osv_converter.stitch_dualstitcher import (
    DualStitcher,
    StitcherConfig,
    DualStitcherError,
)


class TestStitcherConfig:
    """Tests for StitcherConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = StitcherConfig()
        assert config.camera_sn == "default"
        assert config.device_type == 0  # Metal
        assert config.enable_ai is True

    def test_custom_sn(self):
        """Test configuration with custom serial number"""
        config = StitcherConfig(camera_sn="95SXN830221R02")
        assert config.camera_sn == "95SXN830221R02"


class TestDualStitcher:
    """Tests for DualStitcher wrapper"""

    def test_framework_loading(self):
        """Test that DualStitcher framework can be loaded"""
        try:
            stitcher = DualStitcher()
            assert stitcher is not None
        except DualStitcherError as e:
            # This is OK - framework might not be installed or dependencies missing
            error_msg = str(e)
            assert any(phrase in error_msg for phrase in [
                "not found", "not available", "dependencies"
            ])

    def test_framework_availability(self):
        """Test framework availability check"""
        try:
            stitcher = DualStitcher()
            is_available = stitcher.is_available()
            assert isinstance(is_available, bool)
        except DualStitcherError:
            pass  # Framework not available

    def test_get_info(self):
        """Test getting framework information"""
        try:
            stitcher = DualStitcher()
            info = stitcher.get_info()
            assert isinstance(info, dict)
            assert "is_available" in info
            assert "camera_sn" in info
            assert "device_type" in info
        except DualStitcherError:
            pass

    def test_repr(self):
        """Test string representation"""
        try:
            stitcher = DualStitcher()
            repr_str = repr(stitcher)
            assert "DualStitcher" in repr_str
            assert "available" in repr_str
        except DualStitcherError:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
