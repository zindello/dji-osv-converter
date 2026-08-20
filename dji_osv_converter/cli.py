"""Command-line interface for DJI OSV Converter

Usage:
    dji-resolve input.OSV output.mp4 [options]
    dji-resolve --help
"""

import argparse
import logging
import sys
from pathlib import Path

from . import __version__, OSVConverter


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )


def main() -> int:
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        prog="dji-resolve",
        description="Convert DJI OSV 360° video to equirectangular MP4 for DaVinci Resolve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dji-resolve input.OSV output.mp4
  dji-resolve input.OSV output.mp4 --quality high
  dji-resolve input.OSV output.mp4 --frame-range 0:100
  dji-resolve input.OSV output.mp4 --verbose
        """,
    )

    parser.add_argument(
        "input",
        help="Input OSV file path",
    )

    parser.add_argument(
        "output",
        help="Output MP4 file path",
    )

    parser.add_argument(
        "-q", "--quality",
        choices=["low", "medium", "high"],
        default="high",
        help="Encoding quality (default: high)",
    )

    parser.add_argument(
        "-r", "--frame-range",
        help="Process only a range of frames (e.g., '0:100' or '100:200')",
    )

    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="Temporary directory for intermediate files",
    )

    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep temporary files after conversion",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info(f"DJI OSV Converter {__version__}")

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    if input_path.suffix.upper() != ".OSV":
        logger.warning(f"Input file may not be an OSV file (suffix: {input_path.suffix})")

    # Validate output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert
    try:
        converter = OSVConverter()

        logger.info(f"Converting: {input_path.name}")
        logger.info(f"Output: {output_path.name}")

        options = {
            "quality": args.quality,
            "frame_range": args.frame_range,
            "temp_dir": args.temp_dir,
            "cleanup": not args.no_cleanup,
        }

        success = converter.convert(str(input_path), str(output_path), **options)

        if success:
            logger.info(f"Conversion complete!")
            logger.info(f"Output file: {output_path}")
            logger.info(f"File size: {output_path.stat().st_size / (1024**3):.2f} GB")
            return 0
        else:
            logger.error("Conversion failed")
            return 1

    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Conversion error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
