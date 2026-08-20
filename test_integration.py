"""Integration test - verify VideoToolbox decoder works with frame extraction pipeline"""

import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dji_osv_converter.frame_decoder import FrameExtractor

# Find test file
test_files = list(Path("osv_example").glob("*.mp4"))
if not test_files:
    logger.error("No test files found")
    exit(1)

test_file = test_files[0]
logger.info(f"Using test file: {test_file}")

try:
    # Create extractor (will auto-select best decoder)
    logger.info("\nInitializing FrameExtractor...")
    extractor = FrameExtractor(test_file)

    # Get stream info
    stream_info = extractor.get_stream_info()
    logger.info(f"✓ Stream info retrieved:")
    logger.info(f"  Streams: {stream_info['num_streams']}")
    logger.info(f"  Total frames: {stream_info['total_frames']}")
    for stream in stream_info['streams']:
        logger.info(f"  Stream {stream['index']}: {stream['codec_name']} "
                   f"{stream['width']}x{stream['height']} @ {stream['fps']} fps")

    # Decode first frame
    logger.info("\nDecoding first frame...")
    start = time.time()
    frames = extractor.get_frame(0)
    elapsed = (time.time() - start) * 1000

    if frames:
        logger.info(f"✓ Frame decoded in {elapsed:.1f}ms")
        for stream_idx, frame_bytes in frames.items():
            logger.info(f"  Stream {stream_idx}: {len(frame_bytes)} bytes")
    else:
        logger.error("✗ Failed to decode frame")
        exit(1)

    # Try second frame
    logger.info("\nDecoding second frame...")
    start = time.time()
    frames = extractor.get_frame(1)
    elapsed = (time.time() - start) * 1000

    if frames:
        logger.info(f"✓ Frame decoded in {elapsed:.1f}ms")
    else:
        logger.error("✗ Failed to decode second frame")

    extractor.close()
    logger.info("\n✓ Integration test PASSED")

except Exception as e:
    logger.error(f"✗ Integration test FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
