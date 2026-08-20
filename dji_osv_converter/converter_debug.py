"""Debug converter to find the hang point"""

import logging
import time
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)

from dji_osv_converter.converter import StitchingPipeline

input_file = Path("osv_example/CAM_20260819204151_0011_D.OSV")
output_file = Path("osv_example/test_debug.mp4")

print("Starting debug conversion with detailed logging...")
print("Watch for where output stops (that's where it hangs)\n")

start = time.time()
try:
    pipeline = StitchingPipeline(input_file, output_file, gpu_backend='metal')
    result = pipeline.run(frame_range=(0, 50), cleanup=True)
    elapsed = time.time() - start

    if result:
        print(f"\n✓ Conversion succeeded in {elapsed:.1f}s")
    else:
        print(f"\n✗ Conversion failed after {elapsed:.1f}s")

except KeyboardInterrupt:
    elapsed = time.time() - start
    print(f"\n✗ Interrupted after {elapsed:.1f}s")
    sys.exit(1)
except Exception as e:
    elapsed = time.time() - start
    print(f"\n✗ Exception after {elapsed:.1f}s: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
