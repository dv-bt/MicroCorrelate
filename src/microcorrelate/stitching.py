"""
This module contains functions for stitching together microscopy images
"""

import re
from pathlib import Path
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import warnings

import numpy as np
import xmltodict
from tifffile import TiffWriter, imread
from tqdm import tqdm
import ngff_zarr as nz

from microcorrelate.utils import (
    extract_integers,
    vprint,
    get_crop_idx,
)


class Length:
    """A physical length with lossless unit conversion.

    Stores a length in metres as the canonical representation and exposes
    read-only properties for common unit conversions. Intended to eliminate
    scattered unit conversion factors across the stitching pipeline.

    Parameters
    ----------
    meters : float
        Length in metres.

    Examples
    --------
    .. code-block:: python

        l = Length(1.5e-7)
        l.m   # 1.5e-07
        l.um  # 0.15
        l.nm  # 150.0
    """

    def __init__(self, meters: float) -> None:
        self._m = meters

    @property
    def m(self) -> float:
        return self._m

    @property
    def cm(self) -> float:
        return self._m * 1e2

    @property
    def mm(self) -> float:
        return self._m * 1e3

    @property
    def um(self) -> float:
        return self._m * 1e6

    @property
    def nm(self) -> float:
        return self._m * 1e9

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Length):
            return NotImplemented
        return self._m == other._m

    def __repr__(self) -> str:
        return f"Length({self._m} m)"


@dataclass(frozen=True)
class PixelSize:
    """Physical size of a single pixel.

    Parameters
    ----------
    y : Length
        Pixel size along the Y axis.
    x : Length
        Pixel size along the X axis.

    Examples
    --------
    .. code-block:: python

        pixel_size = PixelSize(y=Length(1.5e-7), x=Length(1.5e-7))
        pixel_size.y.nm  # 150.0
        pixel_size.x.um  # 0.15
    """

    y: Length
    x: Length


@dataclass(frozen=True)
class StagePosition:
    """Global stage coordinates of the top-left corner of a stitched tileset.

    Coordinates are extracted from ``MapsProject.xml`` and converted from
    the center-based convention used by Maps to top-left, so that pixel-to-physical
    coordinate mapping is a direct offset from index ``[0, 0]``.
    Units in the ``MapsProject.xml`` are assumed to be meters.

    Parameters
    ----------
    y : Length
        Stage Y coordinate of the top-left corner.
    x : Length
        Stage X coordinate of the top-left corner.

    Examples
    --------
    .. code-block:: python

        stage = StagePosition(y=Length(-0.020), x=Length(-0.011))
        stage.x.um  # -11000.0
        stage.y.nm  # -20000000.0
    """

    y: Length
    x: Length


@dataclass(frozen=True)
class TilesetMetadata:
    """Physical metadata of a Maps stitched tileset.

    Aggregates pixel size and optional stage position into a single object returned by
    ``_get_metadata``. Stage position is ``None`` when ``MapsProject.xml`` is
    unavailable or the tileset cannot be matched.

    Parameters
    ----------
    pixel_size : PixelSize
        Physical size of a single pixel.
    stage_position : StagePosition or None
        Global stage coordinates of the top-left corner, or ``None`` if
        unavailable.
    """

    pixel_size: PixelSize
    stage_position: StagePosition | None


def stitch_images(
    tileset_path: Path | str,
    dest_path: Path | str,
    compression: bool = True,
    group_path: str | None = None,
    pyramid_levels: int = 1,
    crop_borders: bool = True,
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
    crop_borders : bool
        Crop black borders in the stitched dataset, which normally arise from a mismatch
        in the tile size and image size used by Maps. Note that this will briefly cause
        two copies of the image array to be held in memory, it should be used with
        caution when operating close to memory limits. (default = True).
    verbose : bool
        Enable verbose output (default = False).

    Returns
    -------
    None

    """
    # Enforce Path type
    tileset_path = Path(tileset_path)
    dest_path = Path(dest_path)

    pyramid_path = _find_pyramid(tileset_path)
    vprint(f"Stitchig image at {pyramid_path}", verbose)

    tile_shape, metadata = _get_metadata(pyramid_path)
    vprint(
        f"Dataset spacing [m]: ({metadata.pixel_size.x.m}, {metadata.pixel_size.y.m})",
        verbose,
    )
    tile_list = _get_tile_list(pyramid_path)
    image_stitch = _generate_empty_image(pyramid_path, tile_shape)

    for image_path in tqdm(tile_list, "Stitching tiles", disable=not verbose):
        image_stitch = _stitch_tile(image_stitch, image_path, tile_shape)

    # Crop black borders, if present
    if crop_borders:
        vprint("Cropping black border from empty tiles", verbose)
        image_stitch, metadata = _crop_image(image=image_stitch, metadata=metadata)

    vprint("Saving image...", verbose)
    if dest_path.suffix in [".tif", ".tiff"]:
        _save_stitch_tiff(dest_path, image_stitch, metadata, compression)
    elif dest_path.suffix == ".zarr":
        _save_stitch_zarr(
            dest_path,
            image_stitch,
            metadata,
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
    tile_list = list(pyramid_path.glob(f"{max_level}/*/tile*"))
    if tile_list == []:
        raise FileNotFoundError(
            f"No tiff tiles found under {str(pyramid_path)}\n",
            "Tiles are expected to be in a data/level/column/tile_row.tif structure",
        )
    return tile_list


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


def _get_metadata(pyramid_path: Path) -> tuple[tuple[int, int], TilesetMetadata]:
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
    pixel_size = PixelSize(
        y=Length(float(pyramid_metadata["metadata"]["pixelsize"]["y"])),
        x=Length(float(pyramid_metadata["metadata"]["pixelsize"]["x"])),
    )
    stage_position = _get_stage_position(pyramid_path=pyramid_path)
    metadata = TilesetMetadata(pixel_size=pixel_size, stage_position=stage_position)

    return tile_shape, metadata


def _save_stitch_tiff(
    dest_path: Path,
    image_stitch: np.ndarray,
    metadata: TilesetMetadata,
    compression: bool,
) -> None:
    """Save the stitched image as an OME-TIFF file.

    Physical pixel size is written as standard OME ``Pixels`` attributes.
    If a stage position is available, it is written as an OME ``Plane``
    position in micrometres.

    Parameters
    ----------
    dest_path : Path
        Output file path.
    image_stitch : np.ndarray
        Stitched image array.
    metadata : TilesetMetadata
        Physical metadata including pixel size and optional stage position.
    compression : bool
        Whether to apply zlib compression.
    """
    ome_metadata = {
        "axes": "YX",
        "PhysicalSizeX": metadata.pixel_size.x.um,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": metadata.pixel_size.y.um,
        "PhysicalSizeYUnit": "µm",
    }
    if metadata.stage_position is not None:
        ome_metadata["Plane"] = {
            "PositionX": metadata.stage_position.x.um,
            "PositionXUnit": "µm",
            "PositionY": metadata.stage_position.y.um,
            "PositionYUnit": "µm",
        }

    options = {
        "tile": (256, 256),
        "resolutionunit": "CENTIMETER",
        "maxworkers": 2,
    }
    if compression:
        options.update(compression="zlib", compressionargs={"level": 6}, predictor=2)

    with TiffWriter(dest_path, bigtiff=True) as tif:
        tif.write(
            image_stitch,
            resolution=(1 / metadata.pixel_size.x.cm, 1 / metadata.pixel_size.y.cm),
            metadata=ome_metadata,
            **options,
        )


def _save_stitch_zarr(
    dest_path: Path,
    image_stitch: np.ndarray,
    metadata: TilesetMetadata,
    group_path: str | None = None,
    zarr_chunks: tuple[int, int] = (512, 512),
    pyramid_levels: int = 1,
    pyramid_factor: int = 2,
) -> None:
    """Save the stitched image as an OME-NGFF v0.5 Zarr file.

    Physical pixel size and stage position are written as standard OME-NGFF
    scale and translation coordinate transformations in nanometres.
    If ``group_path`` is provided, the image is written to a subgroup of the
    store, allowing multiple OME-NGFF images to coexist in a single store.

    Parameters
    ----------
    dest_path : Path
        Output Zarr store path.
    image_stitch : np.ndarray
        Stitched image array.
    metadata : TilesetMetadata
        Physical metadata including pixel size and optional stage position.
    group_path : str or None
        Path to a subgroup within the store. Each subgroup is a self-contained
        OME-NGFF image. If None, writes to the root of the store.
    zarr_chunks : tuple of int
        Chunk shape for the Zarr arrays. Default is ``(512, 512)``.
    pyramid_levels : int
        Number of multiscale pyramid levels. Default is 1.
    pyramid_factor : int
        Downscale factor between pyramid levels. Default is 2.
    """
    translation = (
        {"y": metadata.stage_position.y.nm, "x": metadata.stage_position.x.nm}
        if metadata.stage_position is not None
        else None
    )

    image = nz.to_ngff_image(
        data=image_stitch,
        dims=["y", "x"],
        scale={"y": metadata.pixel_size.y.nm, "x": metadata.pixel_size.x.nm},
        translation=translation,
        axes_units={"y": "nanometer", "x": "nanometer"},
    )

    multiscales = nz.to_multiscales(
        image,
        scale_factors=[pyramid_factor**i for i in range(1, pyramid_levels)],
    )

    store = str(dest_path / group_path) if group_path else str(dest_path)
    nz.to_ngff_zarr(store, multiscales, chunks=zarr_chunks)


def _get_tileset_guid(pyramid_path: Path) -> str | None:
    """Get tileset GUID from MultiChannelParams.xml, return None if GUID or file
    not found"""
    xml_path = pyramid_path.parents[1] / "MultiChannelParams.xml"
    if not xml_path.exists():
        return None
    with open(xml_path, "rb") as f:
        data = xmltodict.parse(f)["MultiChannelParameters"]
    guid = data.get("Channel", {}).get("Guid")
    return guid.strip() if guid is not None else None


def _find_maps_project(start_path: Path, max_levels: int = 6) -> Path | None:
    """Walk up the directory tree to find MapsProject.xml.

    Parameters
    ----------
    start_path : Path
        Directory to start searching from.
    max_levels : int
        Maximum number of parent directories to search. Default is 6. In the canonical
        Maps directory structure, the MapsProject.xml is 4 levels up from the pyramid
        path.

    Returns
    -------
    Path | None
        Path to MapsProject.xml, or None if not found within ``max_levels``.
    """
    current = start_path
    for _ in range(max_levels):
        candidate = current / "MapsProject.xml"
        if candidate.exists():
            return candidate
        current = current.parent
    return None


def _get_stage_position(pyramid_path: Path) -> StagePosition | None:
    """Extract the global stage position of a stitched tileset from MapsProject.xml.

    Walks up from ``pyramid_path`` to find ``MapsProject.xml``, then matches
    the tileset to its layer entry using the channel GUID from
    ``MultiChannelParams.xml``. Returns None silently if any file is missing
    or no matching entry is found, so stitching can proceed without it.

    Parameters
    ----------
    pyramid_path : Path
        Directory containing ``pyramid.xml``.

    Returns
    -------
    StagePosition | None
        Global stage position, or None if the lookup fails at any step.
    """
    _PROJ = "http://schemas.datacontract.org/2004/07/Fei.Applications.Perseus.Project"
    _SAL = "http://schemas.datacontract.org/2004/07/Fei.Applications.SAL"
    _ZID = "{http://schemas.microsoft.com/2003/10/Serialization/}Id"
    _ZREF = "{http://schemas.microsoft.com/2003/10/Serialization/}Ref"
    _INIL = "{http://www.w3.org/2001/XMLSchema-instance}nil"
    _ITYPE = "{http://www.w3.org/2001/XMLSchema-instance}type"

    def _p(ns: str, tag: str) -> str:
        return f"{{{ns}}}{tag}"

    def _resolve(id_map: dict[str, str], el: ET.Element | None) -> str | None:
        if el is None:
            return None
        if el.get(_INIL) == "true":
            ref = el.get(_ZREF)
            return id_map.get(ref) if ref else None
        ref = el.get(_ZREF)
        if ref:
            return id_map.get(ref)
        val = el.get("Value")
        if val:
            return val
        return el.text.strip() if el.text and el.text.strip() else None

    channel_guid = _get_tileset_guid(pyramid_path)
    if channel_guid is None:
        warnings.warn(
            "Stage position unavailable: "
            "MultiChannelParams.xml not found or contains no GUID.",
            UserWarning,
            stacklevel=2,
        )
        return None

    project_path = _find_maps_project(pyramid_path)
    if project_path is None:
        warnings.warn(
            "Stage position unavailable: MapsProject.xml not found.",
            UserWarning,
            stacklevel=2,
        )
        return None

    tree = ET.parse(project_path)
    root = tree.getroot()

    # Pass 1: build z:Id to value map to resolve WCF cross-references
    id_map: dict[str, str] = {}
    for el in root.iter():
        zid = el.get(_ZID)
        if zid:
            if el.text and el.text.strip():
                id_map[zid] = el.text.strip()
            val = el.get("Value")
            if val:
                id_map[zid] = val

    # Pass 2: find the stitched ImagePyramidLayer matching the channel GUID
    for el in root.iter():
        if el.get(_ITYPE) != "ImagePyramidLayer":
            continue
        if _resolve(id_map, el.find(_p(_PROJ, "pyramidAsSource"))) != "true":
            continue
        ch_defs = el.find(_p(_PROJ, "NewChannelDefinitions"))
        if ch_defs is None:
            continue
        guids = [
            _resolve(id_map, ch.find(_p(_PROJ, "guid")))
            for ch in ch_defs.findall(_p(_PROJ, "ChannelDefinition"))
        ]
        if channel_guid not in guids:
            continue

        sp = el.find(_p(_PROJ, "StagePosition"))
        if sp is None:
            return None
        center_x = _resolve(id_map, sp.find(_p(_SAL, "x")))
        center_y = _resolve(id_map, sp.find(_p(_SAL, "y")))
        hfw = _resolve(id_map, el.find(_p(_PROJ, "HorizontalFieldWidth")))
        vfw = _resolve(id_map, el.find(_p(_PROJ, "VerticalFieldWidth")))
        if center_x is None or center_y is None or hfw is None or vfw is None:
            return None

        topleft_x = float(center_x) - float(hfw) / 2
        topleft_y = float(center_y) - float(vfw) / 2
        return StagePosition(x=Length(topleft_x), y=Length(topleft_y))

    warnings.warn(
        "Stage position unavailable: tileset GUID not found in MapsProject.xml.",
        UserWarning,
        stacklevel=2,
    )
    return None


def _crop_image(
    image: np.ndarray, metadata: TilesetMetadata
) -> tuple[np.ndarray, TilesetMetadata]:
    """Crop black borders from a stitched image and update metadata accordingly.

    Computes the bounding box of non-zero pixels and crops the image to it.
    If the top-left corner shifts, the stage position in the returned metadata is
    corrected by the corresponding physical offset. If the stage position is navailable
    or the origin does not move (i.e. only bottom or right borders are removed), the
    original metadata is returned unchanged.

    .. note::
        This will produce a copy of the array, so this function briefly
        holds two full copies of the image in memory. This should be kept in mind when
        working close to memory limits.

    Parameters
    ----------
    image : np.ndarray
        Stitched image array, possibly with black borders from incomplete tiles.
    metadata : TilesetMetadata
        Physical metadata associated with the image prior to cropping.

    Returns
    -------
    image_crop : np.ndarray
        Image cropped to the bounding box of non-zero pixels.
    metadata : TilesetMetadata
        Updated metadata with corrected stage position if the origin moved,
        otherwise the original metadata unchanged.
    """
    indexer = get_crop_idx(image)
    origin_moves = any(i.start != 0 for i in indexer)
    image_crop = image[indexer]

    new_metadata = None
    if origin_moves and metadata.stage_position is not None:
        new_origin_y = Length(
            metadata.stage_position.y.m + indexer[0].start * metadata.pixel_size.y.m
        )
        new_origin_x = Length(
            metadata.stage_position.x.m + indexer[1].start * metadata.pixel_size.x.m
        )
        new_metadata = TilesetMetadata(
            pixel_size=metadata.pixel_size,
            stage_position=StagePosition(x=new_origin_x, y=new_origin_y),
        )
    return image_crop, new_metadata or metadata
