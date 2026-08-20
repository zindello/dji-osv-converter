#!/usr/bin/env python3
"""Debug frame extraction - check if GPU is being used"""

import logging
from pathlib import Path
import time
import signal

# Enable debug logging to see GPU usage
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Test exceeded time limit")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

try:
    from dji_osv_converter.frame_decoder import FrameExtractor
    from dji_osv_converter.gpu import factory as gpu_factory

    # Check GPU backend
    print("\n=== GPU Backend Check ===")
    backend = gpu_factory.create_best_backend()
    print(f"Best backend: {backend}")
    if backend:
        print(f"Backend name: {backend.name}")
        print(f"Initialize: {backend.initialize()}")

    input_file = Path("osv_example/CAM_20260819204151_0011_D.OSV")

    print("\n=== Frame Extraction with Debug ===")
    extractor = FrameExtractor(input_file, gpu_backend=backend)

    print("\nExtracting first 5 frames:")
    for i in range(5):
        start = time.time()
        frame_data = extractor.get_frame(i)
        elapsed = time.time() - start

        if not frame_data:
            print(f"Frame {i}: FAILED")
            break

        status = "✓" if elapsed < 1.0 else "⚠" if elapsed < 3.0 else "✗"
        print(f"Frame {i}: {elapsed:.2f}s {status}")

    extractor.close()

except TimeoutError:
    print("\n[TIMEOUT] - test exceeded 30 seconds")
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
