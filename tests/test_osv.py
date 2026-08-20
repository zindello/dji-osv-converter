"""Tests for OSV file handling"""

import pytest
from pathlib import Path
from dji_osv_converter.osv import OSVFile, OSVMetadata


@pytest.fixture
def sample_osv_path():
    """Path to sample OSV file for testing"""
    return Path(__file__).parent.parent / "osv_example" / "CAM_20260819204151_0011_D.OSV"


class TestOSVMetadata:
    """Tests for OSVMetadata"""

    def test_default_metadata(self):
        """Test default metadata initialization"""
        metadata = OSVMetadata()
        assert metadata.camera_sn == "unknown"
        assert metadata.num_cameras == 8
        assert metadata.frame_count == 0
        assert metadata.fps == 29.97
        assert metadata.resolution == (3840, 3840)


class TestOSVFile:
    """Tests for OSVFile reader"""

    def test_file_not_found(self):
        """Test error when file doesn't exist"""
        with pytest.raises(FileNotFoundError):
            OSVFile(Path("/nonexistent/file.OSV"))

    def test_load_sample_osv(self, sample_osv_path):
        """Test loading sample OSV file"""
        if not sample_osv_path.exists():
            pytest.skip("Sample OSV file not found")

        osv = OSVFile(sample_osv_path)
        assert osv.filepath == sample_osv_path
        assert osv.filepath.exists()

    def test_parse_metadata(self, sample_osv_path):
        """Test parsing metadata from OSV"""
        if not sample_osv_path.exists():
            pytest.skip("Sample OSV file not found")

        osv = OSVFile(sample_osv_path)
        metadata = osv.metadata

        # Check that metadata was parsed
        assert metadata.frame_count > 0
        assert metadata.duration_seconds > 0
        assert metadata.fps > 0
        assert metadata.num_cameras >= 8

    def test_repr(self, sample_osv_path):
        """Test string representation"""
        if not sample_osv_path.exists():
            pytest.skip("Sample OSV file not found")

        osv = OSVFile(sample_osv_path)
        repr_str = repr(osv)
        assert "OSVFile" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
