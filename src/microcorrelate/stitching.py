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
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from ome_zarr.writer import write_image
from ome_zarr.scale import Scaler

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
    group_path: str | None = None,
    pyramid_levels: int = 1,
    verbose: bool = False,
) -> None:
    """Stitch together tiled images from a tileset acquired via the Thermo Fisher Maps
    software and save the stitched image as a single TIFF file or Zarr file.
    The tileset is expected to be in the directory structure produced by Maps, and have
    been already stitched by the acquisition software
    (that is, there's no overlap or gaps between tiles).

    Parameters
    ----------
    tileset_path : Path | str
        Path to the tileset directory.
    dest_path : Path | str
        Path to save the stitched image. It can be either a .tif file or a .zarr store.
    compression : bool
        Save Tiff image with compression (zlib). This parameter is only used when
        output is TIff. Zarr files are always compressed. (default = True).
    group_path : str | None
        Path to the group in the Zarr array where images should be saved. If None, save
        in root. This parameter is only used if output is Zarr. (default = None).
    pyramid_levels : int
        Number or pyramid levels used when saving to Zarr multiscales. If 1, no
        downscaled levels are calculated. (default = 1).
    verbose : bool
        Enable verbose output (default = False).

    Returns
    -------
    None

    """
    pyramid_path = _find_pyramid(tileset_path)
    vprint(f"Stitchig image at {pyramid_path}", verbose)

    tile_shape, spacing = _get_metadata(pyramid_path)
    vprint(f"Dataset spacing [m]: {spacing}", verbose)
    image_stitch = _generate_empty_image(pyramid_path, tile_shape)
    tile_list = _get_tile_list(pyramid_path)

    vprint("Saving image...", verbose)
    for image_path in tqdm(tile_list, "Stitching tiles", disable=not verbose):
        image_stitch = _stitch_tile(image_stitch, image_path, tile_shape)

    if dest_path.suffix in [".tif", ".tiff"]:
        acquisition_metadata = _get_acquisition_metadata(pyramid_path, verbose)
        _save_stitch_tiff(
            dest_path, image_stitch, spacing, compression, acquisition_metadata
        )
    elif dest_path.suffix == ".zarr":
        _save_stitch_zarr(
            dest_path,
            image_stitch,
            spacing,
            group_path=group_path,
            pyramid_levels=pyramid_levels,
        )
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

    try:
        return xml_matches[0].parent
    except IndexError:
        raise ValueError(f"No pyramid.xml found in '{tileset_path}'.")


def _get_pyramid_level(pyramid_path: Path) -> int:
    """Get maximum (highest resolution) pyramid level"""
    levels = extract_integers(pyramid_path, r"l_(\d+)", "l_*")
    return max(levels)


def _get_tile_list(pyramid_path: Path) -> list[Path]:
    """Get list of file paths to the image tiles"""
    max_level = _get_pyramid_level(pyramid_path)
    return list(pyramid_path.glob(f"l_{max_level}/*/tile*"))


def _generate_empty_image(
    pyramid_path: Path, tile_shape: tuple[int, int]
) -> np.ndarray:
    """Generate an empty image with the correct size to house the stitched image"""

    max_level = _get_pyramid_level(pyramid_path)
    cols = extract_integers(pyramid_path, r"c_(\d+)", f"l_{max_level}/c_*")
    rows = extract_integers(pyramid_path, r"tile_(\d+)", f"l_{max_level}/*/tile*")

    height = (max(rows) + 1) * tile_shape[0]
    width = (max(cols) + 1) * tile_shape[1]
    return np.zeros((height, width), dtype="uint8")


def _stitch_tile(
    image_stitch: np.ndarray, image_path: Path, tile_shape: tuple[int, int]
) -> np.ndarray:
    """
    Stitches a single tile into the main image.
    """
    image_path = str(image_path)
    image = imread(image_path)
    col = int(re.search(r"c_(\d+)", image_path).group(1)) * tile_shape[1]
    row = int(re.search(r"tile_(\d+)", image_path).group(1)) * tile_shape[0]
    image_stitch[row : row + tile_shape[0], col : col + tile_shape[1]] = image
    return image_stitch


def _get_metadata(pyramid_path: Path) -> tuple[tuple[int, int], tuple[float, float]]:
    """
    Get tileset metadata from the pyramid.xml file. Takes as argument the root of the
    image pyramid structure (not the path to the xml file!) and returns a tuple of
    (tile_shape, spacing). Spacing is reported in the native units (assumed meters).
    """

    with open(pyramid_path / "pyramid.xml", "rb") as f:
        pyramid_metadata = xmltodict.parse(f)["root"]

    tile_shape = (
        int(pyramid_metadata["imageset"]["@tileHeight"]),
        int(pyramid_metadata["imageset"]["@tileWidth"]),
    )
    spacing = (
        float(pyramid_metadata["metadata"]["pixelsize"]["y"]),
        float(pyramid_metadata["metadata"]["pixelsize"]["x"]),
    )

    return tile_shape, spacing


def _save_stitch_tiff(
    dest_path: Path,
    image_stitch: np.ndarray,
    spacing: float,
    compression: bool,
    acquisition_metadata: dict,
) -> None:
    """Saves the stitched image. Acquisition metadata are saved in custom tiff tags as
    JSON-formatted string. The actual tiff tags used are set in StitchConfig"""

    metadata_common, metadata_all = _format_metadata(acquisition_metadata)

    # Convert spacing from m to µm
    spacing_um = tuple(i * 1e6 for i in spacing)

    metadata = {
        "axes": "YX",
        "PhysicalSizeX": spacing_um[1],
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": spacing_um[0],
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
            resolution=(1e4 / spacing_um[1], 1e4 / spacing_um[0]),
            metadata=metadata,
            extratags=[
                (StitchConfig.tag_metadata_common, 7, 1, metadata_common, True),
                (StitchConfig.tag_metadata_all, 7, 1, metadata_all, True),
            ],
            **options,
        )


def _save_stitch_zarr(
    dest_path: Path,
    image_stitch: np.ndarray,
    spacing: tuple[float, float],
    group_path: str | None = None,
    zarr_chunks: tuple[int] = (512, 512),
    pyramid_levels: int = 1,
    pyramid_factor: int = 2,
    pyramid_method: str = "gaussian",
) -> None:
    """Saves the stitched image as a zarr file compatible with OME-NGFF v0.5
    specifications"""
    root = zarr.open_group(dest_path)
    if group_path:
        image_group = root.require_group(group_path)
    else:
        image_group = root

    # Convert spacing from m to nm
    spacing_nm = tuple(i * 1e9 for i in spacing)

    compressor = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)

    scaler = Scaler(
        downscale=pyramid_factor, max_layer=pyramid_levels - 1, method=pyramid_method
    )

    storage_options = [
        {"chunks": zarr_chunks, "compressors": compressor}
    ] * pyramid_levels
    axes = [
        {"name": "y", "type": "space", "unit": "nanometer"},
        {"name": "x", "type": "space", "unit": "nanometer"},
    ]
    coordinate_transforms = _get_multiscale_props(pyramid_levels, spacing_nm)

    write_image(
        image=image_stitch,
        group=image_group,
        scaler=scaler,
        axes=axes,
        coordinate_transformations=coordinate_transforms,
        storage_options=storage_options,
    )

    # Add spacing to array attributes
    scale_mapping = _get_scale_mapping(image_group)
    for arr_key, arr_spacing in scale_mapping.items():
        array = image_group.get(arr_key)
        array.attrs["spacing"] = arr_spacing


def _get_multiscale_props(pyramid_levels: int, spacing_nm: tuple[int | float]) -> list:
    """Get multiscale scales and coordinate transformations"""
    coordinate_transformations = []
    for level_index in range(pyramid_levels):
        factor = 2**level_index
        scale_nm = [spacing_nm[0] * factor, spacing_nm[1] * factor]
        coordinate_transformations.append([{"type": "scale", "scale": scale_nm}])
    return coordinate_transformations


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


def _get_scale_mapping(group: zarr.Group) -> dict[str, list[int | float]]:
    """Get scale mappings from zarr group containing OME multiscales"""

    def get_scale(dataset: dict) -> list[int | float]:
        """Get scale from dataset"""
        transforms = dataset["coordinateTransformations"]
        for transform in transforms:
            if transform["type"] == "scale":
                return transform["scale"]
        return []

    multiscales = group.attrs["ome"]["multiscales"]
    datasets = multiscales[0]["datasets"]
    return {i["path"]: get_scale(i) for i in datasets}
