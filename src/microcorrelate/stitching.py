"""
This module contains functions for stitching together microscopy images
"""

import re
from pathlib import Path

import numpy as np
import xmltodict
from tifffile import TiffWriter, imread
from tqdm import tqdm
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from ome_zarr.writer import write_image
from ome_zarr.scale import Scaler

from microcorrelate.utils import (
    extract_integers,
    vprint,
    get_crop_idx,
)


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

    # Crop black borders, if present
    image_stitch = image_stitch[get_crop_idx(image_stitch > 0)]

    if dest_path.suffix in [".tif", ".tiff"]:
        _save_stitch_tiff(dest_path, image_stitch, spacing, compression)
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


def _get_pyramid_level(pyramid_path: Path) -> str:
    """Get maximum (highest resolution) pyramid level"""
    levels = [i.name for i in sorted(pyramid_path.glob("l_*/"))]
    return levels[-1]


def _get_tile_list(pyramid_path: Path) -> list[Path]:
    """Get list of file paths to the image tiles"""
    max_level = _get_pyramid_level(pyramid_path)
    return list(pyramid_path.glob(f"{max_level}/*/tile*"))


def _generate_empty_image(
    pyramid_path: Path, tile_shape: tuple[int, int]
) -> np.ndarray:
    """Generate an empty image with the correct size to house the stitched image"""

    max_level = _get_pyramid_level(pyramid_path)
    cols = extract_integers(pyramid_path, r"c_(\d+)", f"{max_level}/c_*")
    rows = extract_integers(pyramid_path, r"tile_(\d+)", f"{max_level}/*/tile*")

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
) -> None:
    """Saves the stitched image as tiff."""

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
        image=image_stitch.astype(np.float32),
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
