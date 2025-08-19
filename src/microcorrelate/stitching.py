"""
This module contains functions for stitching together microscopy images
"""

import re
from pathlib import Path
import json
from typing import Any
from dataclasses import dataclass

import numpy as np
import xmltodict
from tifffile import TiffWriter, imread, TiffFile
from tqdm import tqdm
from imageio.v3 import immeta

from microcorrelate.utils import (
    extract_integers,
    vprint,
    find_common_vals,
    flatten_dict,
)


@dataclass(frozen=True)
class StitchConfig:
    tag_metadata_common: int = 65000
    tag_metadata_all: int = 65001


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

    acquisition_metadata = _get_acquisition_metadata(pyramid_path, verbose)
    _save_stitch(dest_path, image_stitch, pixel_size, compression, acquisition_metadata)
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
    dest_path: Path,
    image_stitch: np.ndarray,
    pixel_size: float,
    compression: bool,
    acquisition_metadata: dict,
) -> None:
    """Saves the stitched image. Acquisition metadata are saved in custom tiff tags as
    JSON-formatted string. The actual tiff tags used are set in StitchConfig"""

    metadata_common, metadata_all = _format_metadata(acquisition_metadata)

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
            extratags=[
                (StitchConfig.tag_metadata_common, 7, 1, metadata_common, True),
                (StitchConfig.tag_metadata_all, 7, 1, metadata_all, True),
            ],
            **options,
        )


def _find_orig_images(pyramid_path: Path) -> list[Path] | None:
    """Find the path to the orignal images before stitching. This is useful because the
    tiles in the Maps image pyramid don't have acquisition metadata. Returns none
    if the directory is not found.
    NOTE: for this to work, the stitched tile set should be named
    'Tileset_name (stitched)' and the corresponding original tileset should be named
    'Tileset_name'. This is the standard pattern produced by the software."""

    pyramid_path = str(pyramid_path)
    orig_path = pyramid_path[: pyramid_path.rfind(" (stitched)")]
    orig_path = Path(orig_path)
    if orig_path.exists():
        return list(orig_path.glob("*.tif"))


def _get_acquisition_metadata(pyramid_path: Path, verbose: bool = True) -> dict | None:
    """Reads the acquisition metadata from the original images and returns it as a
    dictionary where each entry is the metadata for one image.
    If the directory was not found reading metadata is skipped and the
    function returns None"""

    orig_list = _find_orig_images(pyramid_path)

    if not orig_list:
        vprint(
            "Original tileset not found. It could have been moved or renamed. "
            "Skipped parsing acquisition metadata",
            verbose,
        )
        return
    metadata_all = {}
    for image_path in tqdm(
        orig_list, "Reading acquisition metadata", disable=not verbose
    ):
        metadata = immeta(image_path)
        # Remove unnecessary keys introduced by imageio
        metadata_clean = {
            key: val
            for key, val in metadata.items()
            if ("is_" not in key) and (key != "byteorder")
        }
        metadata_all[image_path.name] = metadata_clean
    return metadata_all


def _find_common_metadata(metadata: dict[str, dict[str, Any]]) -> dict:
    """Reduce metadata to a single common dictionary"""

    metadata_list = [flatten_dict(val) for val in metadata.values()]
    metadata_common = metadata_list[0]
    for d in metadata_list[1:]:
        metadata_common = find_common_vals(metadata_common, d)
    return metadata_common


def _format_metadata(metadata: dict[str, dict[str, Any]] | None) -> tuple[str, str]:
    """Formats the acquisition metadata as two JSON-formatted strings:
    - the common acquisition metadata
    - the detailed acquisition metadata for each image
    Returns empty strings if None is passed as input metadata"""

    if not metadata:
        return "", ""
    metadata_common = json.dumps(_find_common_metadata(metadata)).encode("utf-8")
    metadata_all = json.dumps(metadata).encode("utf-8")
    return metadata_common, metadata_all


def read_metadata(stitch_path: Path) -> tuple[dict, dict]:
    """Read acquisition metadata from stitched images"""

    def read_tag_json(tif: TiffFile, tag_id: int) -> dict:
        """Read tag value and parse it as a json string"""
        return json.loads(tif.pages[0].tags[tag_id].value)

    with TiffFile(stitch_path) as tif:
        metadata_common = read_tag_json(tif, StitchConfig.tag_metadata_common)
        metadata_all = read_tag_json(tif, StitchConfig.tag_metadata_all)
    return metadata_common, metadata_all
