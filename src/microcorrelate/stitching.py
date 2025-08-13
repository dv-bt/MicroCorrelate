"""
This module contains functions for stitching together microscopy images
"""

import re
from pathlib import Path

import numpy as np
import xmltodict
from tifffile import TiffWriter, imread
from tqdm import tqdm

from microcorrelate.utils import extract_integers, vprint


def stitch_images(
    tileset_path: Path | str,
    dest_path: Path | str,
    compression: bool = True,
    verbose: bool = False,
) -> None:
    """Stitch together tiled images from a tileset acquired via the Thermo Fisher Maps
    software and save the stitched image as a single TIFF file. The tileset is expected
    to be in the directory structure produced by Maps, and have been already stitched
    by the acquisition software (that is, there's no overlap or gaps between tiles).

    Parameters
    ----------
    tileset_path : Path | str
        Path to the tileset directory.
    dest_path : Path | str
        Path to save the stitched image.
    compression : bool
        Save image with compression (zlib). (default = True).
    verbose : bool
        Enable verbose output (default = False).

    Returns
    -------
    None

    """
    pyramid_path = _find_pyramid(tileset_path)
    vprint(f"Stitchig image at {pyramid_path}", verbose)

    tile_size, pixel_size = _get_metadata(pyramid_path)
    image_stitch = _generate_empty_image(pyramid_path, tile_size)
    tile_list = _get_tile_list(pyramid_path)

    for image_path in tqdm(tile_list, "Stitching tiles", disable=not verbose):
        image_stitch = _stitch_tile(image_stitch, image_path, tile_size)

    _save_stitch(dest_path, image_stitch, pixel_size, compression)
    vprint(f"Stitched image saved at {dest_path}", verbose)


def _find_pyramid(tileset_path: Path) -> Path:
    """Find the path to the image pyramid. The image pyramid is detected by looking for
    a pyramid.xml file. Raises an error if there are multiple image pyramids under
    tileset_path"""

    tileset_path = Path(tileset_path)
    xml_matches = list(tileset_path.rglob("*pyramid.xml"))

    # Check if pyramid.xml is unique
    if len(xml_matches) > 1:
        raise ValueError(
            f"Multiple pyramid.xml files found in '{tileset_path}': {len(xml_matches)}."
            " Only one XML file is expected per directory."
        )

    return xml_matches[0].parent


def _get_pyramid_level(pyramid_path: Path) -> int:
    """Get maximum (highest resolution) pyramid level"""
    levels = extract_integers(pyramid_path, r"l_(\d+)", "l_*")
    return max(levels)


def _get_tile_list(pyramid_path: Path) -> list[Path]:
    """Get list of file paths to the image tiles"""
    max_level = _get_pyramid_level(pyramid_path)
    return list(pyramid_path.glob(f"l_{max_level}/*/tile*"))


def _generate_empty_image(pyramid_path: Path, tile_size: int) -> np.ndarray:
    """Generate an empty image with the correct size to house the stitched image"""

    max_level = _get_pyramid_level(pyramid_path)
    cols = extract_integers(pyramid_path, r"c_(\d+)", f"l_{max_level}/c_*")
    rows = extract_integers(pyramid_path, r"tile_(\d+)", f"l_{max_level}/*/tile*")

    width = (max(cols) + 1) * tile_size
    height = (max(rows) + 1) * tile_size
    return np.zeros((height, width), dtype="uint8")


def _stitch_tile(
    image_stitch: np.ndarray, image_path: Path, tile_size: int
) -> np.ndarray:
    """
    Stitches a single tile into the main image.
    """
    image_path = str(image_path)
    image = imread(image_path)
    col = int(re.search(r"c_(\d+)", image_path).group(1)) * tile_size
    row = int(re.search(r"tile_(\d+)", image_path).group(1)) * tile_size
    image_stitch[row : row + tile_size, col : col + tile_size] = image
    return image_stitch


def _get_metadata(pyramid_path: Path) -> tuple[int, float]:
    """
    Get tileset metadata from the pyramid.xml file. Takes as argument the root of the
    image pyramid structure (not the path to the xml file!) and returns a tuple of
    (tile size, pixel size). Pixel size is reported in micrometers.
    """

    with open(pyramid_path / "pyramid.xml", "r") as file:
        text = file.read()

    pyramid_metadata = xmltodict.parse(text)["root"]
    tile_size = int(pyramid_metadata["imageset"]["@tileWidth"])
    pixel_size = float(pyramid_metadata["metadata"]["pixelsize"]["x"]) * 1e6

    return tile_size, pixel_size


def _save_stitch(
    dest_path: Path, image_stitch: np.ndarray, pixel_size: float, compression: bool
) -> None:
    """Saves the stitched image"""

    metadata = {
        "axes": "YX",
        "PhysicalSizeX": pixel_size,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": pixel_size,
        "PhysicalSizeYUnit": "µm",
    }
    options = {
        "tile": (256, 256),
        "resolutionunit": "CENTIMETER",
        "maxworkers": 2,
    }
    if compression:
        options.update(
            compression="zlib",
            compressionargs={"level": 6},
            predictor=2,
        )
    with TiffWriter(dest_path, bigtiff=True) as tif:
        tif.write(
            image_stitch,
            resolution=(1e4 / pixel_size, 1e4 / pixel_size),
            metadata=metadata,
            **options,
        )
