"""
This module contains the core functionality of the microcorrelate package.
"""

import re
from pathlib import Path

import numpy as np
import xmltodict
from imageio.v3 import imread
from tifffile import TiffWriter

from microcorrelate.utils import extract_integers


def stitch_images(tileset_path: Path | str, dest_path: Path | str) -> None:
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

    Returns
    -------
    None

    """

    if isinstance(tileset_path, str):
        tileset_path = Path(tileset_path)

    xml_path = list(tileset_path.rglob("*/**/*pyramid.xml"))[0]
    pyramid_path = xml_path.parent
    print(f"Stitchig image at {xml_path}")

    with open(xml_path, "r") as file:
        text = file.read()

    pyramid_metadata = xmltodict.parse(text)["root"]

    tile_size = int(pyramid_metadata["imageset"]["@tileWidth"])

    levels = extract_integers(pyramid_path, r"l_(\d+)", "l_*")
    max_level = max(levels)

    cols = extract_integers(pyramid_path, r"c_(\d+)", f"l_{max_level}/c_*")
    rows = extract_integers(pyramid_path, r"tile_(\d+)", f"l_{max_level}/*/tile*", True)

    width = (max(cols) + 1) * tile_size
    height = (max(rows) + 1) * tile_size

    image_stitch = np.zeros((height, width), dtype="uint8")

    for image_path in pyramid_path.glob(f"l_{max_level}/*/tile*"):
        image_path = str(image_path)
        image = imread(image_path)
        col = int(re.search(r"c_(\d+)", image_path).group(1)) * tile_size
        row = int(re.search(r"tile_(\d+)", image_path).group(1)) * tile_size
        image_stitch[row : row + tile_size, col : col + tile_size] = image

    pixelsize = float(pyramid_metadata["metadata"]["pixelsize"]["x"]) * 1e6
    with TiffWriter(dest_path, bigtiff=True) as tif:
        metadata = {
            "axes": "YX",
            "PhysicalSizeX": pixelsize,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixelsize,
            "PhysicalSizeYUnit": "µm",
        }
        options = {
            "tile": (128, 128),
            "resolutionunit": "CENTIMETER",
            "maxworkers": 2,
        }
        tif.write(
            image_stitch,
            resolution=(1e4 / pixelsize, 1e4 / pixelsize),
            metadata=metadata,
            **options,
        )
