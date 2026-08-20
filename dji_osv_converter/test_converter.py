"""Test framework for converter with timeouts and validation"""

import subprocess
import logging
import time
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class ConverterTest:
    """Time-gated converter test with validation"""

    def __init__(self, timeout_seconds: int = 300):
        """
        Initialize test.

        Args:
            timeout_seconds: Hard cutoff timeout (kill process if exceeded)
        """
        self.timeout = timeout_seconds
        self.process = None

    def run_conversion(
        self,
        input_file: Path,
        output_file: Path,
        frame_range: Tuple[int, int],
        gpu_backend: Optional[str] = None,
    ) -> bool:
        """
        Run converter with timeout.

        Args:
            input_file: Input OSV file
            output_file: Output MP4 file
            frame_range: (start, end) frames to convert
            gpu_backend: GPU backend to use

        Returns:
            True if conversion succeeded
        """
        try:
            # Build command
            cmd = [
                "python3",
                "-c",
                f"""
import sys
from pathlib import Path
from dji_osv_converter.converter import StitchingPipeline

input_file = Path('{input_file}')
output_file = Path('{output_file}')
gpu_backend = {repr(gpu_backend)}

pipeline = StitchingPipeline(input_file, output_file, gpu_backend=gpu_backend)
result = pipeline.run(frame_range={frame_range}, cleanup=True)
sys.exit(0 if result else 1)
""",
            ]

            # Run with timeout
            print(f"Starting converter... (timeout: {self.timeout}s)")
            start_time = time.time()

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                stdout, stderr = self.process.communicate(timeout=self.timeout)
                elapsed = time.time() - start_time

                if self.process.returncode == 0:
                    print(f"✓ Converter finished in {elapsed:.1f}s")
                    return True
                else:
                    print(f"✗ Converter failed (exit code {self.process.returncode})")
                    if stderr:
                        print(f"  Error: {stderr.decode()[:200]}")
                    return False

            except subprocess.TimeoutExpired:
                elapsed = time.time() - start_time
                print(f"✗ Converter timeout after {elapsed:.1f}s (limit: {self.timeout}s)")
                self.process.kill()
                self.process.wait()
                return False

        except Exception as e:
            print(f"✗ Failed to run converter: {e}")
            return False

    def validate_output(
        self,
        output_file: Path,
        frame_count: int,
        fps: float = 29.97,
        width: int = 7680,
        height: int = 3840,
    ) -> Dict[str, bool]:
        """
        Validate output file.

        Args:
            output_file: Path to output MP4
            frame_count: Expected number of frames
            fps: Expected frames per second
            width: Expected frame width
            height: Expected frame height

        Returns:
            Dict of validation results
        """
        results = {
            "file_exists": False,
            "file_size_valid": False,
            "duration_valid": False,
            "format_valid": False,
            "resolution_valid": False,
            "all_valid": False,
        }

        try:
            # Check file exists
            if not output_file.exists():
                print(f"✗ Output file not found: {output_file}")
                return results

            results["file_exists"] = True
            print(f"✓ File exists: {output_file}")

            # Check file size (min 100KB per frame)
            file_size = output_file.stat().st_size
            min_expected = frame_count * 100_000  # Conservative: 100KB/frame
            if file_size < min_expected * 0.1:  # At least 10% of expected
                print(f"✗ File too small: {file_size/1e6:.1f}MB (expected ≥ {min_expected/1e6:.1f}MB)")
            else:
                results["file_size_valid"] = True
                print(f"✓ File size reasonable: {file_size/1e6:.1f}MB")

            # Validate with ffprobe
            probe_results = self._ffprobe(output_file)
            if not probe_results:
                print(f"✗ ffprobe validation failed")
                return results

            # Check format
            if probe_results.get("codec") == "h264":
                results["format_valid"] = True
                print(f"✓ Codec: {probe_results['codec']}")
            else:
                print(f"✗ Unexpected codec: {probe_results.get('codec')}")

            # Check resolution
            if (probe_results.get("width") == width and
                probe_results.get("height") == height):
                results["resolution_valid"] = True
                print(f"✓ Resolution: {width}×{height}")
            else:
                print(f"✗ Wrong resolution: {probe_results.get('width')}×{probe_results.get('height')}")

            # Check duration
            expected_duration = frame_count / fps
            actual_duration = probe_results.get("duration", 0)
            duration_error = abs(actual_duration - expected_duration) / expected_duration if expected_duration > 0 else 1.0

            if duration_error < 0.05:  # Within 5%
                results["duration_valid"] = True
                print(f"✓ Duration: {actual_duration:.2f}s (expected {expected_duration:.2f}s)")
            else:
                print(f"✗ Duration mismatch: {actual_duration:.2f}s vs {expected_duration:.2f}s")

            # Overall result
            results["all_valid"] = all([
                results["file_exists"],
                results["file_size_valid"],
                results["format_valid"],
                results["resolution_valid"],
                results["duration_valid"],
            ])

        except Exception as e:
            print(f"✗ Validation error: {e}")

        return results

    def _ffprobe(self, file_path: Path) -> Optional[Dict]:
        """
        Get video information using ffprobe.

        Returns:
            Dict with codec, width, height, duration, fps
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height,r_frame_rate",
                "-show_entries", "format=duration",
                "-of", "json",
                str(file_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)

            # Extract values
            stream = data.get("streams", [{}])[0]
            fmt = data.get("format", {})

            # Parse frame rate
            fps_str = stream.get("r_frame_rate", "30/1")
            try:
                num, den = map(float, fps_str.split("/"))
                fps = num / den if den > 0 else 30.0
            except:
                fps = 30.0

            return {
                "codec": stream.get("codec_name", ""),
                "width": stream.get("width", 0),
                "height": stream.get("height", 0),
                "duration": float(fmt.get("duration", 0)),
                "fps": fps,
            }

        except Exception as e:
            logger.error(f"ffprobe error: {e}")
            return None


def main():
    """Run converter test"""
    import sys

    # Setup
    input_file = Path("osv_example/CAM_20260819204151_0011_D.OSV")
    output_file = Path("osv_example/test_converter_output.mp4")
    frame_count = 50
    timeout = 120  # 2 minute timeout

    print("=" * 70)
    print("Converter Test Framework")
    print("=" * 70)

    # Run test
    test = ConverterTest(timeout_seconds=timeout)

    print(f"\n[1/3] Running conversion (timeout: {timeout}s)...")
    success = test.run_conversion(
        input_file,
        output_file,
        frame_range=(0, frame_count),
        gpu_backend="metal",
    )

    if not success:
        print("\n" + "=" * 70)
        print("✗ CONVERSION FAILED")
        print("=" * 70)
        return False

    print(f"\n[2/3] Validating output file...")
    results = test.validate_output(output_file, frame_count)

    print(f"\n[3/3] Test Results:")
    print("-" * 70)
    for key, value in results.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")

    print("=" * 70)
    if results["all_valid"]:
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
