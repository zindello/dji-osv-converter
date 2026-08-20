#!/usr/bin/env python3
"""Performance benchmark: FFmpeg vs other decoders"""

import logging
import time
from pathlib import Path
from dji_osv_converter.decoder import factory as decoder_factory
from dji_osv_converter.decoder.detect import detect_available_decoders
from dji_osv_converter.frame_decoder import FrameExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark_decoder(decoder_name: str, num_frames: int = 50) -> dict:
    """Benchmark a specific decoder"""
    osv_path = Path("osv_example/CAM_20260819204151_0011_D.OSV")
    if not osv_path.exists():
        logger.error(f"Test file not found: {osv_path}")
        return None

    print(f"\n{'='*70}")
    print(f"Benchmarking: {decoder_name}")
    print(f"{'='*70}")

    try:
        # Create extractor with specific decoder
        extractor = FrameExtractor(osv_path, use_decoder_backend=decoder_name)
        print(f"Backend: {extractor.decoder}")

        # Warmup: decode first frame
        print("Warmup: decoding first frame...", end="", flush=True)
        start = time.time()
        _ = extractor.get_frame(0)
        warmup_time = time.time() - start
        print(f" {warmup_time*1000:.0f}ms")

        # Benchmark: decode multiple frames
        print(f"Decoding {num_frames} frames...")
        print("Frame# |   Time   | Cumulative")
        print("-------|----------|------------")

        frame_times = []
        start_total = time.time()

        for frame_idx in range(1, num_frames + 1):
            frame_start = time.time()
            frames = extractor.get_frame(frame_idx)
            frame_time = time.time() - frame_start

            if frames is None or len(frames) != 2:
                print(f"{frame_idx:6d} | TIMEOUT")
                extractor.close()
                return None

            frame_times.append(frame_time)
            cumulative = time.time() - start_total

            if frame_idx % 5 == 0 or frame_idx == num_frames:
                print(f"{frame_idx:6d} | {frame_time*1000:6.0f}ms | {cumulative:8.2f}s")

        total_time = time.time() - start_total

        # Calculate statistics
        frame_times_ms = [t * 1000 for t in frame_times]
        avg_time = sum(frame_times) / len(frame_times)
        min_time = min(frame_times)
        max_time = max(frame_times)

        # Calculate throughput
        total_bytes = num_frames * 2 * 3840 * 3840 * 3  # 2 streams, RGB24
        mb_total = total_bytes / 1e6
        mb_per_sec = mb_total / total_time if total_time > 0 else 0
        fps = num_frames / total_time if total_time > 0 else 0

        result = {
            "decoder": decoder_name,
            "num_frames": num_frames,
            "total_time": total_time,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "fps": fps,
            "mb_per_sec": mb_per_sec,
        }

        # Print summary
        print(f"\n{'='*70}")
        print(f"Results ({decoder_name}):")
        print(f"{'='*70}")
        print(f"Total time:        {total_time:.2f}s")
        print(f"Average per frame: {avg_time*1000:.0f}ms")
        print(f"Min per frame:     {min_time*1000:.0f}ms")
        print(f"Max per frame:     {max_time*1000:.0f}ms")
        print(f"Frames/second:     {fps:.2f} fps")
        print(f"Data throughput:   {mb_per_sec:.1f} MB/s")
        print(f"Total data:        {mb_total:.0f} MB")
        print(f"{'='*70}")

        extractor.close()
        return result

    except Exception as e:
        logger.exception(f"Benchmark failed: {e}")
        return None


def benchmark_all_decoders(num_frames: int = 50):
    """Benchmark all available decoders"""
    print("\n" + "="*70)
    print("Decoder Performance Benchmark Suite")
    print("="*70)

    available = detect_available_decoders()
    print(f"\nAvailable decoders: {available}")

    # Priority order for benchmarking
    decoder_priority = ["ffmpeg", "videotoolbox", "cpu"]
    decoders_to_test = [d for d in decoder_priority if available.get(d, False)]

    if not decoders_to_test:
        print("ERROR: No decoders available!")
        return

    results = {}
    for decoder_name in decoders_to_test:
        result = benchmark_decoder(decoder_name, num_frames)
        if result:
            results[decoder_name] = result

    # Comparison table
    if len(results) > 1:
        print("\n" + "="*70)
        print("COMPARISON TABLE")
        print("="*70)
        print(f"{'Decoder':<20} | {'Frames':<7} | {'Avg (ms)':<10} | {'FPS':<8} | {'MB/s':<10}")
        print("-" * 70)

        for decoder_name in decoders_to_test:
            if decoder_name not in results:
                continue

            r = results[decoder_name]
            print(f"{decoder_name:<20} | {r['num_frames']:<7} | {r['avg_time']*1000:>8.0f} | "
                  f"{r['fps']:>6.2f} | {r['mb_per_sec']:>8.1f}")

        # Calculate speedup vs CPU
        if "cpu" in results and "ffmpeg" in results:
            speedup = results["cpu"]["avg_time"] / results["ffmpeg"]["avg_time"]
            print(f"\nFFmpeg speedup vs CPU: {speedup:.1f}×")

        if "cpu" in results and "videotoolbox" in results:
            speedup = results["cpu"]["avg_time"] / results["videotoolbox"]["avg_time"]
            print(f"VideoToolbox speedup vs CPU: {speedup:.1f}×")

        print("="*70)

    return results


def analyze_frame_timing_distribution(decoder_name: str, num_frames: int = 50):
    """Analyze frame timing distribution"""
    print(f"\n{'='*70}")
    print(f"Frame Timing Analysis: {decoder_name}")
    print(f"{'='*70}")

    osv_path = Path("osv_example/CAM_20260819204151_0011_D.OSV")
    if not osv_path.exists():
        logger.error(f"Test file not found: {osv_path}")
        return

    try:
        extractor = FrameExtractor(osv_path, decoder_backend=decoder_name)

        # Decode frames
        frame_times = []
        for frame_idx in range(num_frames):
            start = time.time()
            frames = extractor.get_frame(frame_idx)
            frame_time = time.time() - start

            if frames is None:
                break

            frame_times.append(frame_time)

        extractor.close()

        if not frame_times:
            print("No frames decoded")
            return

        # Statistics
        frame_times_ms = [t * 1000 for t in frame_times]
        sorted_times = sorted(frame_times_ms)

        print(f"Frames analyzed:   {len(frame_times)}")
        print(f"Mean:              {sum(frame_times_ms)/len(frame_times_ms):.1f}ms")
        print(f"Median:            {sorted_times[len(sorted_times)//2]:.1f}ms")
        print(f"Min:               {min(frame_times_ms):.1f}ms")
        print(f"Max:               {max(frame_times_ms):.1f}ms")
        print(f"Std Dev:           {(sum((x - sum(frame_times_ms)/len(frame_times_ms))**2 for x in frame_times_ms) / len(frame_times_ms))**0.5:.1f}ms")

        # Percentiles
        p50 = sorted_times[int(len(sorted_times) * 0.50)]
        p75 = sorted_times[int(len(sorted_times) * 0.75)]
        p90 = sorted_times[int(len(sorted_times) * 0.90)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]

        print(f"P50 (median):      {p50:.1f}ms")
        print(f"P75:               {p75:.1f}ms")
        print(f"P90:               {p90:.1f}ms")
        print(f"P99:               {p99:.1f}ms")

        print("="*70)

    except Exception as e:
        logger.exception(f"Analysis failed: {e}")


if __name__ == "__main__":
    # Run benchmarks
    results = benchmark_all_decoders(num_frames=50)

    # Analyze timing distribution for FFmpeg if available
    available = detect_available_decoders()
    if available.get("ffmpeg"):
        analyze_frame_timing_distribution("ffmpeg", num_frames=50)

    print("\n" + "="*70)
    print("Benchmark Complete")
    print("="*70)
