"""Quick test of VideoToolbox decoder"""

import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dji_osv_converter.decoder import create_decoder, detect_available_decoders

# Check available decoders
available = detect_available_decoders()
logger.info(f"Available decoders: {available}")

# Test VideoToolbox
if available.get("videotoolbox"):
    logger.info("\nTesting VideoToolbox decoder...")
    decoder = create_decoder("videotoolbox")
    if decoder:
        logger.info(f"✓ Created {decoder.name} decoder")

        # Try to open a test file
        test_files = list(Path("osv_example").glob("*.mp4"))
        if test_files:
            test_file = test_files[0]
            logger.info(f"Opening: {test_file}")

            if decoder.open_file(str(test_file)):
                logger.info(f"✓ Opened successfully")
                logger.info(f"  Streams: {decoder.get_stream_count()}")
                logger.info(f"  Total frames: {decoder.get_total_frames()}")

                # Try to decode frame 0
                logger.info("Decoding frame 0...")
                start = time.time()
                frames = decoder.decode_frame_all_streams(0, timeout_s=5.0)
                elapsed = (time.time() - start) * 1000

                if frames:
                    logger.info(f"✓ Frame decoded in {elapsed:.1f}ms")
                    logger.info(f"  Streams: {len(frames)}")
                else:
                    logger.warning(f"✗ Frame decode timeout after {elapsed:.1f}ms")

                decoder.close()
            else:
                logger.error("Failed to open file")
        else:
            logger.warning("No test files found")
    else:
        logger.error("Failed to create VideoToolbox decoder")
else:
    logger.info("VideoToolbox not available")

# Test CPU fallback
logger.info("\n\nTesting CPU (PyAV) decoder...")
decoder = create_decoder("cpu")
if decoder:
    logger.info(f"✓ Created {decoder.name} decoder")

    test_files = list(Path("osv_example").glob("*.mp4"))
    if test_files:
        test_file = test_files[0]
        logger.info(f"Opening: {test_file}")

        if decoder.open_file(str(test_file)):
            logger.info(f"✓ Opened successfully")
            logger.info(f"  Streams: {decoder.get_stream_count()}")
            logger.info(f"  Total frames: {decoder.get_total_frames()}")

            # Try to decode frame 0
            logger.info("Decoding frame 0...")
            start = time.time()
            frames = decoder.decode_frame_all_streams(0, timeout_s=10.0)
            elapsed = (time.time() - start) * 1000

            if frames:
                logger.info(f"✓ Frame decoded in {elapsed:.1f}ms")
                logger.info(f"  Streams: {len(frames)}")
            else:
                logger.warning(f"✗ Frame decode timeout after {elapsed:.1f}ms")

            decoder.close()
        else:
            logger.error("Failed to open file")
else:
    logger.error("Failed to create CPU decoder")
