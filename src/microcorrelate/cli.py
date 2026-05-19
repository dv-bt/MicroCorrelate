"""Command-line interface for MicroCorrelate."""

import argparse
from pathlib import Path

from microcorrelate.stitching import stitch_images


def _parse_stitch_args() -> argparse.Namespace:
    """Parse arguments for the stitch-maps command."""
    parser = argparse.ArgumentParser(
        description="Stitch tiled images from a Thermo Fisher Maps acquisition."
    )
    parser.add_argument(
        "-s",
        "--source",
        required=True,
        type=Path,
        help="Tileset directory containing images to stitch.",
    )
    parser.add_argument(
        "-d",
        "--dest",
        required=True,
        type=Path,
        help="Destination file (.tif or .zarr).",
    )
    parser.add_argument(
        "-g",
        "--group_path",
        type=str,
        default=None,
        help="Zarr group path to write into (optional).",
    )
    parser.add_argument(
        "-p",
        "--pyramid_levels",
        type=int,
        default=1,
        help="Number of pyramid levels when saving to .zarr (default: 1).",
    )
    parser.add_argument(
        "-c",
        "--compress",
        action="store_true",
        help="Enable zlib compression (TIFF only).",
    )
    parser.add_argument(
        "-n",
        "--no-crop",
        dest="crop_borders",
        action="store_false",
        help="Disable cropping of empty tile borders (cropping is on by default).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )
    parser.add_argument(
        "-w",
        "--working_dir",
        type=Path,
        default=Path.cwd(),
        help="Working directory used to resolve relative paths (default: cwd).",
    )
    parser.set_defaults(crop_borders=True)

    args = parser.parse_args()
    args.source = (args.working_dir / args.source).resolve()
    args.dest = (args.working_dir / args.dest).resolve()
    return args


def stitch_maps() -> None:
    """Entry point for the ``stitch-maps`` command."""
    args = _parse_stitch_args()
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    stitch_images(
        tileset_path=args.source,
        dest_path=args.dest,
        compression=args.compress,
        group_path=args.group_path,
        pyramid_levels=args.pyramid_levels,
        crop_borders=args.crop_borders,
        verbose=args.verbose,
    )