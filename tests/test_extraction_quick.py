#!/usr/bin/env python3
"""Quick frame extraction diagnostic - 60 second timeout"""

import sys
import time
import signal
from pathlib import Path

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Test exceeded time limit")

# Set 60 second timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)

try:
    print("Frame Extraction Diagnostic (60s timeout)\n")

    from dji_osv_converter.frame_decoder import FrameExtractor

    input_file = Path("osv_example/CAM_20260819204151_0011_D.OSV")

    print("Initializing...")
    extractor = FrameExtractor(input_file)

    print("Extracting frames:")
    print("Frame | Time   | Status")
    print("------|--------|-------")

    for i in range(50):
        start = time.time()

        frame_data = extractor.get_frame(i)

        elapsed = time.time() - start

        if not frame_data:
            print(f"{i:5d} | FAIL")
            break

        status = "✓" if elapsed < 1.0 else "⚠" if elapsed < 3.0 else "✗ SLOW"
        print(f"{i:5d} | {elapsed:6.2f}s | {status}")

        if elapsed > 10:
            print("\n[Stopping - frame extraction taking >10s]")
            break

    extractor.close()

    print("\nTest completed")

except TimeoutError:
    print("\n[TIMEOUT] - test exceeded 60 seconds")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] {e}")
    sys.exit(1)
