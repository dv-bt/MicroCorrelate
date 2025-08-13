#!/usr/bin/env python3
"""
Script to stitch together images produced by the ThermoFisher Maps software
"""

import argparse
from pathlib import Path
from microcorrelate.stitching import stitch_images


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Stitch images together")

    parser.add_argument(
        "-s",
        "--source",
        required=True,
        help="Tilset directory irectory containing images to stitch",
    )

    parser.add_argument(
        "-d",
        "--dest",
        required=True,
        help="Destination file to save the stitched image. Must be a .tif image",
    )

    parser.add_argument(
        "-c",
        "--compress",
        action="store_true",
        help="Enable image compression (zlib)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    """Main function to run the script."""
    args = parse_arguments()
    tileset_path = Path(args.source)
    dest_path = Path(args.dest)
    dest_path.parent.mkdir(exist_ok=True)
    stitch_images(tileset_path, dest_path, args.compress, args.verbose)


if __name__ == "__main__":
    main()
