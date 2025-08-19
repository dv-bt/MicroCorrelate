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
        type=Path,
        help="Tilset directory irectory containing images to stitch",
    )

    parser.add_argument(
        "-d",
        "--dest",
        required=True,
        type=Path,
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

    parser.add_argument(
        "-w",
        "--working_dir",
        required=False,
        type=Path,
        default=Path.cwd(),
        help=(
            "Optional working directory used for parsing source and destinatione path "
            "if they are relative."
        ),
    )

    args = parser.parse_args()

    # Update paths
    args.source = (args.working_dir / args.source).resolve()
    args.dest = (args.working_dir / args.dest).resolve()

    return args


def main():
    """Main function to run the script."""
    args = parse_arguments()
    args.dest.parent.mkdir(exist_ok=True)
    stitch_images(args.source, args.dest, args.compress, args.verbose)


if __name__ == "__main__":
    main()
