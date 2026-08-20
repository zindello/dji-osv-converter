"""Comprehensive tests for optical flow stitching with real OSV data

Tests the complete optical flow stitching pipeline:
1. Extract frames from real OSV file
2. Compute optical flow
3. Warp and stitch frames
4. Generate output images
5. Measure performance and quality metrics
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import cv2
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from dji_osv_converter.stitch_optical_flow import OpticalFlowStitcher, create_optical_flow_stitcher
from dji_osv_converter.frame_decoder import FrameExtractor

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_stitch_optical_flow.log'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Output directory for test results
OUTPUT_DIR = Path(__file__).parent / "test_output_optical_flow"
OUTPUT_DIR.mkdir(exist_ok=True)


class OpticalFlowStitchingTest:
    """Comprehensive test suite for optical flow stitching"""

    def __init__(self, osv_path: Path):
        """
        Initialize test suite.

        Args:
            osv_path: Path to the OSV file
        """
        self.osv_path = osv_path
        self.frame_extractor: Optional[FrameExtractor] = None
        self.stitcher: Optional[OpticalFlowStitcher] = None
        self.test_frames: List[Tuple[np.ndarray, np.ndarray]] = []

        # Performance metrics
        self.metrics = {
            "extraction_time": 0.0,
            "flow_times": [],
            "warp_times": [],
            "blend_times": [],
            "total_stitch_times": [],
            "output_psnr": [],
            "output_ssim": [],
        }

    def setup(self) -> bool:
        """Initialize frame extractor and stitcher."""
        try:
            logger.info(f"Setting up test with OSV file: {self.osv_path}")

            if not self.osv_path.exists():
                logger.error(f"OSV file not found: {self.osv_path}")
                return False

            # Initialize frame extractor
            logger.info("Initializing frame extractor...")
            self.frame_extractor = FrameExtractor(self.osv_path)

            # Initialize stitcher with optical flow
            logger.info("Initializing optical flow stitcher...")
            self.stitcher = create_optical_flow_stitcher(
                enable_warping=True,
                enable_poisson_blend=True,
                seam_method="flow_guided",
            )

            logger.info("Setup completed successfully")
            return True

        except Exception as e:
            logger.exception(f"Setup failed: {e}")
            return False

    def extract_test_frames(self, num_frames: int = 5) -> bool:
        """
        Extract frames from the OSV file.

        Args:
            num_frames: Number of frames to extract

        Returns:
            True if successful
        """
        try:
            if self.frame_extractor is None:
                logger.error("Frame extractor not initialized")
                return False

            logger.info(f"Extracting {num_frames} frames from OSV file...")
            extract_start = time.time()

            # Get stream info
            stream_info = self.frame_extractor.get_stream_info()
            total_frames = stream_info["total_frames"]
            logger.info(f"Total frames available: {total_frames}")

            if total_frames < 2:
                logger.error("Not enough frames in the OSV file")
                return False

            # Extract frame pairs
            frame_count = 0
            for frame_num in range(min(num_frames, total_frames)):
                # Get frames from all streams (should be 2 for OSV)
                frame_bytes_dict = self.frame_extractor.get_frame(frame_num)

                if frame_bytes_dict is None:
                    logger.warning(f"Failed to extract frame {frame_num}")
                    continue

                if len(frame_bytes_dict) < 2:
                    logger.warning(f"Frame {frame_num} has only {len(frame_bytes_dict)} streams (expected 2)")
                    continue

                # Convert bytes to numpy arrays
                frame0 = np.frombuffer(frame_bytes_dict[0], dtype=np.uint8).reshape((3840, 3840, 3))
                frame1 = np.frombuffer(frame_bytes_dict[1], dtype=np.uint8).reshape((3840, 3840, 3))

                self.test_frames.append((frame0, frame1))
                frame_count += 1
                logger.info(f"Extracted frame pair {frame_count}: {frame0.shape}, {frame1.shape}")

            self.metrics["extraction_time"] = time.time() - extract_start

            if len(self.test_frames) == 0:
                logger.error("No frames extracted")
                return False

            logger.info(f"Successfully extracted {len(self.test_frames)} frame pairs in {self.metrics['extraction_time']:.2f}s")
            return True

        except Exception as e:
            logger.exception(f"Frame extraction failed: {e}")
            return False

    def test_optical_flow(self) -> bool:
        """Test optical flow computation on extracted frames."""
        try:
            if not self.test_frames:
                logger.error("No test frames available")
                return False

            logger.info("Testing optical flow computation...")

            for i, (frame0, frame1) in enumerate(self.test_frames):
                logger.info(f"\nTesting optical flow on frame pair {i + 1}...")

                # Convert to grayscale
                gray0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)

                # Compute optical flow
                flow_start = time.time()
                flow = cv2.calcOpticalFlowFarneback(
                    gray0, gray1, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2,
                    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
                )
                flow_time = time.time() - flow_start

                logger.info(f"  Flow computed: shape={flow.shape}, time={flow_time:.3f}s")

                # Analyze flow statistics
                flow_magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
                logger.info(f"  Flow magnitude - min: {flow_magnitude.min():.2f}, max: {flow_magnitude.max():.2f}, mean: {flow_magnitude.mean():.2f}")

                # Visualize flow
                self._visualize_flow(flow, i, "optical_flow")

                return True

            return True

        except Exception as e:
            logger.exception(f"Optical flow test failed: {e}")
            return False

    def test_warping(self) -> bool:
        """Test image warping based on optical flow."""
        try:
            if not self.test_frames:
                logger.error("No test frames available")
                return False

            logger.info("Testing image warping...")

            for i, (frame0, frame1) in enumerate(self.test_frames):
                logger.info(f"\nTesting warping on frame pair {i + 1}...")

                # Compute optical flow
                gray0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    gray0, gray1, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2,
                    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
                )

                # Warp using flow
                warp_start = time.time()
                h, w = frame0.shape[:2]
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                x = x.astype(np.float32)
                y = y.astype(np.float32)

                x_warped = x + flow[:, :, 0]
                y_warped = y + flow[:, :, 1]

                frame0_warped = cv2.remap(frame0, x_warped, y_warped, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                warp_time = time.time() - warp_start

                logger.info(f"  Warping completed: time={warp_time:.3f}s")

                # Save warped frames
                self._save_frame(frame0, i, "frame0_original")
                self._save_frame(frame0_warped, i, "frame0_warped")

                return True

            return True

        except Exception as e:
            logger.exception(f"Warping test failed: {e}")
            return False

    def test_stitching_full_pipeline(self) -> bool:
        """Test complete stitching pipeline on extracted frames."""
        try:
            if self.stitcher is None:
                logger.error("Stitcher not initialized")
                return False

            if not self.test_frames:
                logger.error("No test frames available")
                return False

            logger.info("Testing complete stitching pipeline...")

            for i, (frame0, frame1) in enumerate(self.test_frames):
                logger.info(f"\nStitching frame pair {i + 1}...")

                stitch_start = time.time()
                output = self.stitcher.stitch_frame(frame0, frame1)
                stitch_time = time.time() - stitch_start

                if output is None:
                    logger.error(f"Stitching failed for frame pair {i + 1}")
                    return False

                logger.info(f"  Output shape: {output.shape}, time: {stitch_time:.3f}s")

                # Get performance metrics
                metrics = self.stitcher.get_performance_metrics()
                self.metrics["flow_times"].append(metrics["flow_time"])
                self.metrics["warp_times"].append(metrics["warp_time"])
                self.metrics["blend_times"].append(metrics["blend_time"])
                self.metrics["total_stitch_times"].append(stitch_time)

                logger.info(f"  Metrics: flow={metrics['flow_time']:.3f}s, warp={metrics['warp_time']:.3f}s, blend={metrics['blend_time']:.3f}s")

                # Save output image
                self._save_frame(output, i, "stitched_output")

            logger.info("Stitching pipeline test completed successfully")
            return True

        except Exception as e:
            logger.exception(f"Stitching pipeline test failed: {e}")
            return False

    def test_quality_metrics(self) -> bool:
        """Compute and compare quality metrics for stitched output."""
        try:
            if not self.test_frames:
                logger.error("No test frames available")
                return False

            logger.info("Computing quality metrics...")

            for i, (frame0, frame1) in enumerate(self.test_frames):
                logger.info(f"\nComputing metrics for frame pair {i + 1}...")

                # Stitch the frames
                output = self.stitcher.stitch_frame(frame0, frame1)
                if output is None:
                    logger.error(f"Stitching failed for frame pair {i + 1}")
                    continue

                # Compute PSNR for output
                # Use the left frame as reference (simplified metric)
                ref = np.concatenate([frame0, frame0[:, :self.stitcher.OUTPUT_WIDTH - 3840]], axis=1)
                psnr = self._compute_psnr(output.astype(np.float32), ref.astype(np.float32))
                logger.info(f"  Output PSNR: {psnr:.2f} dB")
                self.metrics["output_psnr"].append(psnr)

                # Compute SSIM
                ssim = self._compute_ssim(output, ref)
                logger.info(f"  Output SSIM: {ssim:.4f}")
                self.metrics["output_ssim"].append(ssim)

            return True

        except Exception as e:
            logger.exception(f"Quality metrics test failed: {e}")
            return False

    def test_edge_cases(self) -> bool:
        """Test edge cases and error handling."""
        try:
            logger.info("Testing edge cases...")

            if self.stitcher is None:
                logger.error("Stitcher not initialized")
                return False

            # Test 1: Wrong frame size
            logger.info("Test 1: Wrong frame size...")
            wrong_size = np.zeros((1920, 1920, 3), dtype=np.uint8)
            output = self.stitcher.stitch_frame(wrong_size, wrong_size)
            if output is not None:
                logger.error("Expected None for wrong size, got output")
                return False
            logger.info("  PASS: Wrong size handled correctly")

            # Test 2: Wrong dtype
            logger.info("Test 2: Wrong dtype...")
            wrong_dtype = np.zeros((3840, 3840, 3), dtype=np.float32)
            output = self.stitcher.stitch_frame(wrong_dtype, wrong_dtype)
            if output is not None:
                logger.error("Expected None for wrong dtype, got output")
                return False
            logger.info("  PASS: Wrong dtype handled correctly")

            # Test 3: Black frames (low-quality input)
            logger.info("Test 3: Black frames...")
            black_frame = np.zeros((3840, 3840, 3), dtype=np.uint8)
            output = self.stitcher.stitch_frame(black_frame, black_frame)
            if output is not None:
                logger.info("  PASS: Black frames processed (output shape: {})".format(output.shape))
            else:
                logger.warning("  WARNING: Black frames returned None")

            logger.info("Edge case testing completed")
            return True

        except Exception as e:
            logger.exception(f"Edge case testing failed: {e}")
            return False

    def generate_report(self) -> str:
        """Generate a test report."""
        report = []
        report.append("\n" + "=" * 80)
        report.append("OPTICAL FLOW STITCHING TEST REPORT")
        report.append("=" * 80)

        report.append(f"\nTest Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"OSV File: {self.osv_path}")
        report.append(f"Output Directory: {OUTPUT_DIR}")

        # Frame extraction
        report.append(f"\n[FRAME EXTRACTION]")
        report.append(f"  Frames extracted: {len(self.test_frames)}")
        report.append(f"  Total extraction time: {self.metrics['extraction_time']:.2f}s")
        if len(self.test_frames) > 0:
            report.append(f"  Avg frame size: {self.test_frames[0][0].shape}")

        # Performance metrics
        if self.metrics["total_stitch_times"]:
            report.append(f"\n[PERFORMANCE]")
            report.append(f"  Frames stitched: {len(self.metrics['total_stitch_times'])}")
            report.append(f"  Avg stitch time: {np.mean(self.metrics['total_stitch_times']):.3f}s")
            report.append(f"  Avg flow time: {np.mean(self.metrics['flow_times']):.3f}s")
            report.append(f"  Avg warp time: {np.mean(self.metrics['warp_times']):.3f}s")
            report.append(f"  Avg blend time: {np.mean(self.metrics['blend_times']):.3f}s")

            # Calculate FPS
            if len(self.metrics['total_stitch_times']) > 0:
                fps = 1.0 / np.mean(self.metrics['total_stitch_times'])
                report.append(f"  Throughput: {fps:.2f} fps")

        # Quality metrics
        if self.metrics["output_psnr"]:
            report.append(f"\n[QUALITY METRICS]")
            report.append(f"  Avg PSNR: {np.mean(self.metrics['output_psnr']):.2f} dB")
            report.append(f"  Min PSNR: {np.min(self.metrics['output_psnr']):.2f} dB")
            report.append(f"  Max PSNR: {np.max(self.metrics['output_psnr']):.2f} dB")

        if self.metrics["output_ssim"]:
            report.append(f"  Avg SSIM: {np.mean(self.metrics['output_ssim']):.4f}")
            report.append(f"  Min SSIM: {np.min(self.metrics['output_ssim']):.4f}")
            report.append(f"  Max SSIM: {np.max(self.metrics['output_ssim']):.4f}")

        # Output files
        report.append(f"\n[OUTPUT FILES]")
        output_files = list(OUTPUT_DIR.glob("*.png"))
        report.append(f"  Generated images: {len(output_files)}")
        for f in sorted(output_files)[:10]:
            report.append(f"    - {f.name}")

        report.append("\n" + "=" * 80)
        return "\n".join(report)

    def run_all_tests(self) -> bool:
        """Run all tests in sequence."""
        try:
            logger.info("Starting comprehensive test suite...")

            # Setup
            if not self.setup():
                logger.error("Setup failed")
                return False

            # Extract frames
            if not self.extract_test_frames(5):
                logger.error("Frame extraction failed")
                return False

            # Run tests
            tests = [
                ("Optical Flow", self.test_optical_flow),
                ("Warping", self.test_warping),
                ("Full Pipeline", self.test_stitching_full_pipeline),
                ("Quality Metrics", self.test_quality_metrics),
                ("Edge Cases", self.test_edge_cases),
            ]

            results = {}
            for test_name, test_func in tests:
                logger.info(f"\n--- Running: {test_name} ---")
                try:
                    result = test_func()
                    results[test_name] = "PASS" if result else "FAIL"
                    logger.info(f"Result: {results[test_name]}")
                except Exception as e:
                    logger.exception(f"{test_name} failed with exception")
                    results[test_name] = "ERROR"

            # Generate report
            report = self.generate_report()
            logger.info(report)

            # Save report
            report_path = OUTPUT_DIR / "test_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {report_path}")

            # Print results summary
            logger.info("\n--- TEST RESULTS SUMMARY ---")
            for test_name, result in results.items():
                logger.info(f"  {test_name}: {result}")

            all_passed = all(r == "PASS" for r in results.values())
            if all_passed:
                logger.info("\n✓ All tests passed!")
            else:
                logger.warning("\n✗ Some tests failed")

            return all_passed

        except Exception as e:
            logger.exception(f"Test suite failed: {e}")
            return False

    # Helper methods

    def _save_frame(self, frame: np.ndarray, frame_idx: int, prefix: str) -> None:
        """Save a frame as PNG image."""
        try:
            output_path = OUTPUT_DIR / f"{prefix}_{frame_idx:03d}.png"

            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame

            cv2.imwrite(str(output_path), frame_bgr)
            logger.debug(f"Saved: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save frame: {e}")

    def _visualize_flow(self, flow: np.ndarray, frame_idx: int, prefix: str) -> None:
        """Visualize optical flow as an image."""
        try:
            # Compute magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Create HSV visualization
            hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)  # Hue from angle
            hsv[..., 1] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Saturation from magnitude
            hsv[..., 2] = 255  # Value

            # Convert HSV to BGR
            flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Save
            output_path = OUTPUT_DIR / f"{prefix}_{frame_idx:03d}.png"
            cv2.imwrite(str(output_path), flow_vis)
            logger.debug(f"Saved flow visualization: {output_path}")

        except Exception as e:
            logger.error(f"Failed to visualize flow: {e}")

    @staticmethod
    def _compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute PSNR between two images."""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_val = 255.0
        return 20 * np.log10(max_val) - 10 * np.log10(mse)

    @staticmethod
    def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute structural similarity between two images."""
        # Simplified SSIM (without sliding window)
        c1, c2 = 6.5025, 58.5225

        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        return np.mean(ssim_map)


def main():
    """Main test entry point."""
    # Path to sample OSV file
    osv_path = Path(__file__).parent / "osv_example" / "CAM_20260819204151_0011_D.OSV"

    if not osv_path.exists():
        logger.error(f"Sample OSV file not found: {osv_path}")
        logger.info("Please ensure the sample OSV file is available in osv_example/")
        return False

    # Create and run test suite
    test_suite = OpticalFlowStitchingTest(osv_path)
    success = test_suite.run_all_tests()

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
