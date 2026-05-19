"""Tests for microcorrelate.stitching module."""

from pathlib import Path

import numpy as np
import pytest
import zarr
from tifffile import imwrite, imread
from imageio.v3 import immeta

from microcorrelate.stitching import (
    Length,
    PixelSize,
    StagePosition,
    TilesetMetadata,
    _crop_image,
    _find_maps_project,
    _find_pyramid,
    _get_stage_position,
    _get_tile_list,
    _get_tileset_guid,
    stitch_images,
)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
TILE_SIZE = 128
PIXEL_SIZE_M = 1.5e-7

TEST_GUID = "127ed61e-bcfd-4011-b4ae-8a6fe3f4aeeb"
_DIFFERENT_GUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

STAGE_CENTER_X_M = 0.0062771374126911689
STAGE_CENTER_Y_M = -0.025597698487458617
HFW_M = 0.0008589
VFW_M = 0.0008622

STAGE_X_M = STAGE_CENTER_X_M - HFW_M / 2
STAGE_Y_M = STAGE_CENTER_Y_M - VFW_M / 2

PYRAMID_XML = f"""\
<?xml version="1.0" encoding="utf-8"?>
<root>
  <imageset tileWidth="{TILE_SIZE}" tileHeight="{TILE_SIZE}"/>
  <metadata>
    <pixelsize>
      <x>{PIXEL_SIZE_M}</x>
      <y>{PIXEL_SIZE_M}</y>
    </pixelsize>
  </metadata>
</root>"""

MULTICHANNEL_XML = f"""\
<MultiChannelParameters Count="1" AdditiveAlpha="-1">
  <Channel Index="0">
    <Guid>{TEST_GUID}</Guid>
    <Name>CBS; All</Name>
  </Channel>
</MultiChannelParameters>"""

MAPS_PROJECT_XML = f"""\
<?xml version="1.0" encoding="utf-8"?>
<PerseusProject
  xmlns="http://schemas.datacontract.org/2004/07/Fei.Applications.Perseus.Project"
  xmlns:i="http://www.w3.org/2001/XMLSchema-instance"
  xmlns:z="http://schemas.microsoft.com/2003/10/Serialization/">
  <LayerGroups z:Size="1">
    <LayerGroup>
      <Layers z:Size="1" xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays">
        <a:anyType i:type="ImagePyramidLayer">
          <StagePosition xmlns:b="http://schemas.datacontract.org/2004/07/Fei.Applications.SAL">
            <b:x>{STAGE_CENTER_X_M}</b:x>
            <b:y>{STAGE_CENTER_Y_M}</b:y>
          </StagePosition>
          <NewChannelDefinitions z:Size="1">
            <ChannelDefinition>
              <guid>{TEST_GUID}</guid>
            </ChannelDefinition>
          </NewChannelDefinitions>
          <HorizontalFieldWidth Value="{HFW_M}"/>
          <VerticalFieldWidth Value="{VFW_M}"/>
          <pyramidAsSource>true</pyramidAsSource>
        </a:anyType>
      </Layers>
    </LayerGroup>
  </LayerGroups>
</PerseusProject>"""


# --------------------------------------------------------------------------------------
# Tileset helpers and fixtures
# --------------------------------------------------------------------------------------


def make_tiles(n_cols: int, n_rows: int) -> dict[tuple[int, int], np.ndarray]:
    """Generate a dict of tiles keyed by (col, row)."""
    return {
        (col, row): np.full((TILE_SIZE, TILE_SIZE), row * 64 + col, dtype=np.uint8)
        for col in range(n_cols)
        for row in range(n_rows)
    }


def make_ground_truth(n_cols: int, n_rows: int) -> np.ndarray:
    """Assemble the expected stitched image from generated tiles."""
    tiles = make_tiles(n_cols, n_rows)
    return np.block(
        [[tiles[(col, row)] for col in range(n_cols)] for row in range(n_rows)]
    )


def make_tileset(
    root: Path,
    n_cols: int = 3,
    n_rows: int = 3,
    missing_tiles: list[tuple[int, int]] | None = None,
    include_multichannel_xml: bool = True,
    include_pyramid_xml: bool = True,
    include_maps_project_xml: bool = True,
    n_channel_folders: int = 1,
    malformed_pyramid_xml: bool = False,
) -> Path:
    missing = set(missing_tiles or [])
    tileset_path = root / "LayersData" / "tileset"
    tiles = make_tiles(n_cols, n_rows)

    for ch_idx in range(n_channel_folders):
        channel_path = tileset_path / f"T000-Z000-CH{ch_idx}"
        pyramid_path = channel_path / "data"
        level_path = pyramid_path / "l_0"
        level_path.mkdir(parents=True)

        for (col, row), tile in tiles.items():
            if (col, row) in missing:
                continue
            col_path = level_path / f"c_{col}"
            col_path.mkdir(exist_ok=True)
            imwrite(col_path / f"tile_{row}.tif", tile)

        if include_pyramid_xml:
            xml = "<<invalid>>" if malformed_pyramid_xml else PYRAMID_XML
            (pyramid_path / "pyramid.xml").write_text(xml)

    if include_multichannel_xml:
        (tileset_path / "MultiChannelParams.xml").write_text(MULTICHANNEL_XML)

    if include_maps_project_xml:
        (root / "MapsProject.xml").write_text(MAPS_PROJECT_XML)

    return tileset_path


@pytest.fixture
def mock_tileset(tmp_path: Path) -> Path:
    return make_tileset(tmp_path, missing_tiles=[(2, r) for r in range(3)])


@pytest.fixture
def mock_tileset_without_multichannel_xml(tmp_path: Path) -> Path:
    return make_tileset(
        tmp_path,
        missing_tiles=[(2, r) for r in range(3)],
        include_multichannel_xml=False,
    )


@pytest.fixture
def mock_tileset_without_project(tmp_path: Path) -> Path:
    return make_tileset(
        tmp_path,
        missing_tiles=[(2, r) for r in range(3)],
        include_maps_project_xml=False,
    )


# --------------------------------------------------------------------------------------
# Length
# --------------------------------------------------------------------------------------


class TestLenght:
    val_m = 1.5e-7
    length = Length(val_m)

    def test_length_metre_property(self):
        assert self.length.m == self.val_m

    def test_length_um_conversion(self):
        assert self.length.um == pytest.approx(self.val_m * 1e6)

    def test_length_nm_conversion(self):
        assert self.length.nm == pytest.approx(self.val_m * 1e9)

    def test_length_cm_conversion(self):
        assert self.length.cm == pytest.approx(self.val_m * 1e2)

    def test_length_eq_matching_values(self):
        assert self.length == self.length

    def test_length_eq_non_length_type(self):
        result = self.length.__eq__(self.val_m)
        assert result is NotImplemented

    def test_length_repr_contains_value(self):
        assert str(self.val_m) in repr(self.length)


# ---------------------------------------------------------------------------
# _get_tileset_guid
# ---------------------------------------------------------------------------


def test_get_tileset_guid_returns_guid_from_multichannel_xml(mock_tileset):
    tileset_path = mock_tileset
    pyramid_path = tileset_path / "T000-Z000-CH0" / "data"
    assert _get_tileset_guid(pyramid_path) == TEST_GUID


def test_get_tileset_guid_returns_none_when_multichannel_xml_absent(
    mock_tileset_without_multichannel_xml,
):
    tileset_path = mock_tileset_without_multichannel_xml
    pyramid_path = tileset_path / "T000-Z000-CH0" / "data"
    assert _get_tileset_guid(pyramid_path) is None


# --------------------------------------------------------------------------------------
# _find_maps_project
# --------------------------------------------------------------------------------------


def test_find_maps_project_finds_file_four_levels_up(mock_tileset, tmp_path):
    # pyramid_path is data/, four hops to tmp_path where MapsProject.xml lives
    pyramid_path = mock_tileset / "T000-Z000-CH0" / "data"
    result = _find_maps_project(pyramid_path)
    assert result == tmp_path / "MapsProject.xml"


def test_find_maps_project_returns_none_when_file_absent(mock_tileset_without_project):
    pyramid_path = mock_tileset_without_project / "T000-Z000-CH0" / "data"
    assert _find_maps_project(pyramid_path) is None


def test_find_maps_project_returns_none_when_file_is_beyond_max_levels(mock_tileset):
    # MapsProject.xml is 4 hops up, max_level=3 never reaches it
    pyramid_path = mock_tileset / "T000-Z000-CH0" / "data"
    assert _find_maps_project(pyramid_path, max_levels=3) is None


# --------------------------------------------------------------------------------------
# _get_stage_position
# --------------------------------------------------------------------------------------


def test_get_stage_position_returns_correct_topleft_coordinates(
    mock_tileset,
):
    pyramid_path = mock_tileset / "T000-Z000-CH0" / "data"
    result = _get_stage_position(pyramid_path)
    assert result is not None
    assert result.x.m == pytest.approx(STAGE_X_M)
    assert result.y.m == pytest.approx(STAGE_Y_M)


def test_get_stage_position_returns_none_when_multichannel_xml_absent(
    mock_tileset_without_multichannel_xml,
):
    pyramid_path = mock_tileset_without_multichannel_xml / "T000-Z000-CH0" / "data"
    with pytest.warns(UserWarning, match="MultiChannelParams.xml"):
        assert _get_stage_position(pyramid_path) is None


def test_get_stage_position_returns_none_when_maps_project_absent(
    mock_tileset_without_project,
):
    pyramid_path = mock_tileset_without_project / "T000-Z000-CH0" / "data"
    with pytest.warns(UserWarning, match="MapsProject.xml not found"):
        assert _get_stage_position(pyramid_path) is None


def test_get_stage_position_returns_none_when_guid_not_in_project(tmp_path):
    tileset_path = make_tileset(tmp_path)
    wrong_project = MAPS_PROJECT_XML.replace(TEST_GUID, _DIFFERENT_GUID)
    (tmp_path / "MapsProject.xml").write_text(wrong_project)
    pyramid_path = tileset_path / "T000-Z000-CH0" / "data"
    with pytest.warns(UserWarning, match="GUID not found"):
        assert _get_stage_position(pyramid_path) is None


# --------------------------------------------------------------------------------------
# _find_pyramid
# --------------------------------------------------------------------------------------


def test_find_pyramid_returns_data_directory_path(mock_tileset):
    expected = mock_tileset / "T000-Z000-CH0" / "data"
    assert _find_pyramid(mock_tileset) == expected


def test_find_pyramid_raises_when_no_pyramid_xml_present(tmp_path):
    tileset_path = make_tileset(tmp_path, include_pyramid_xml=False)
    with pytest.raises(ValueError, match="No pyramid.xml"):
        _find_pyramid(tileset_path)


def test_find_pyramid_raises_when_multiple_pyramid_xml_present(tmp_path):
    tileset_path = make_tileset(tmp_path, n_channel_folders=2)
    with pytest.raises(ValueError, match="Multiple pyramid.xml"):
        _find_pyramid(tileset_path)


# --------------------------------------------------------------------------------------
# _get_tile_list
# --------------------------------------------------------------------------------------


def test_get_tile_list_returns_all_existing_tile_paths(mock_tileset):
    # 3x3 grid, c_2 directory exists but is empty: 6 tiles across c_0 and c_1
    pyramid_path = mock_tileset / "T000-Z000-CH0" / "data"
    tiles = _get_tile_list(pyramid_path)
    assert len(tiles) == 6


def test_get_tile_list_raises_when_tile_directory_is_empty(tmp_path):
    all_missing = [(c, r) for c in range(3) for r in range(3)]
    tileset_path = make_tileset(tmp_path, missing_tiles=all_missing)
    pyramid_path = tileset_path / "T000-Z000-CH0" / "data"
    with pytest.raises(FileNotFoundError):
        _get_tile_list(pyramid_path)


# --------------------------------------------------------------------------------------
# _crop_image
# --------------------------------------------------------------------------------------


class TestCropImage:
    def __init__(self):
        self.right_border_image = np.zeros(
            (3 * TILE_SIZE, 3 * TILE_SIZE), dtype=np.uint8
        )
        self.right_border_image[:, : 2 * TILE_SIZE] = 128

        self.left_border_image = np.zeros(
            (3 * TILE_SIZE, 3 * TILE_SIZE), dtype=np.uint8
        )
        self.left_border_image[:, TILE_SIZE:] = 128

        self.stage = StagePosition(x=Length(STAGE_X_M), y=Length(STAGE_Y_M))

    def _make_metadata(self, stage_pos=None):
        return TilesetMetadata(
            pixel_size=PixelSize(x=Length(PIXEL_SIZE_M), y=Length(PIXEL_SIZE_M)),
            stage_position=stage_pos,
        )

    def test_pixel_size_unchanged(self):
        _, out = _crop_image(self.right_border_image, self._make_metadata(self.stage))
        assert out.pixel_size.x == Length(PIXEL_SIZE_M)

    def test_right_border_removed(self):
        cropped, _ = _crop_image(self.right_border_image, self._make_metadata())
        assert cropped.shape == (3 * TILE_SIZE, 2 * TILE_SIZE)

    def test_stage_position_updated_on_left_border(self):
        _, out = _crop_image(self.left_border_image, self._make_metadata(self.stage))
        assert out.stage_position.x.m == pytest.approx(
            STAGE_X_M + TILE_SIZE * PIXEL_SIZE_M
        )

    def test_stage_position_unchanged_on_right_border_only(self):
        metadata = self._make_metadata(self.stage)
        _, out = _crop_image(self.right_border_image, metadata)
        assert out is metadata

    def test_no_raise_when_stage_position_is_none(self):
        _, out = _crop_image(self.right_border_image, self._make_metadata())
        assert out.stage_position is None


# --------------------------------------------------------------------------------------
# stitch_images — TIFF output
# --------------------------------------------------------------------------------------


def test_stitch_tiff_full_image_matches_ground_truth(tmp_path):
    tileset_path = make_tileset(tmp_path)
    dest = tmp_path / "out.tif"
    stitch_images(tileset_path, dest, crop_borders=False, compression=False)
    np.testing.assert_array_equal(imread(dest), make_ground_truth(3, 3))


def test_stitch_tiff_cropped_image_matches_ground_truth(mock_tileset, tmp_path):
    dest = tmp_path / "out.tif"
    stitch_images(mock_tileset, dest, crop_borders=True, compression=False)
    np.testing.assert_array_equal(imread(dest), make_ground_truth(2, 3))


def test_stitch_tiff_physical_size_in_metadata(mock_tileset, tmp_path):
    dest = tmp_path / "out.tif"
    stitch_images(mock_tileset, dest, crop_borders=False, compression=False)
    meta = immeta(dest)
    assert meta["PhysicalSizeX"] == pytest.approx(PIXEL_SIZE_M * 1e6)
    assert meta["PhysicalSizeXUnit"] == "µm"


def test_stitch_tiff_plane_position_written_when_stage_available(
    mock_tileset, tmp_path
):
    dest = tmp_path / "out.tif"
    stitch_images(mock_tileset, dest, crop_borders=False, compression=False)
    meta = immeta(dest)
    assert meta.get("Plane", {}).get("PositionX") is not None


def test_stitch_tiff_plane_position_absent_when_stage_unavailable(
    mock_tileset_without_project, tmp_path
):
    dest = tmp_path / "out.tif"
    with pytest.warns(UserWarning, match="MapsProject.xml not found"):
        stitch_images(
            mock_tileset_without_project, dest, crop_borders=False, compression=False
        )
    meta = immeta(dest)
    assert "Plane" not in meta


# --------------------------------------------------------------------------------------
# stitch_images — Zarr output
# --------------------------------------------------------------------------------------


def test_stitch_zarr_format_version_is_ngff_v05(mock_tileset, tmp_path):
    dest = tmp_path / "out.zarr"
    stitch_images(mock_tileset, dest, crop_borders=False)
    store = zarr.open(str(dest), mode="r")
    assert store.attrs["ome"]["version"] == "0.5"


def test_stitch_zarr_scale_transform_matches_pixel_size_in_nanometres(
    mock_tileset, tmp_path
):
    dest = tmp_path / "out.zarr"
    stitch_images(mock_tileset, dest, crop_borders=False)
    store = zarr.open(str(dest), mode="r")
    datasets = store.attrs["ome"]["multiscales"][0]["datasets"]
    scale = next(
        t for t in datasets[0]["coordinateTransformations"] if t["type"] == "scale"
    )["scale"]
    expected = PIXEL_SIZE_M * 1e9
    assert scale == pytest.approx([expected, expected])


def test_stitch_zarr_translation_transform_matches_stage_position_in_nanometres(
    mock_tileset, tmp_path
):
    dest = tmp_path / "out.zarr"
    stitch_images(mock_tileset, dest, crop_borders=False)
    store = zarr.open(str(dest), mode="r")
    datasets = store.attrs["ome"]["multiscales"][0]["datasets"]
    translation = next(
        t
        for t in datasets[0]["coordinateTransformations"]
        if t["type"] == "translation"
    )["translation"]
    assert translation == pytest.approx([STAGE_Y_M * 1e9, STAGE_X_M * 1e9])


def test_stitch_zarr_full_image_matches_ground_truth(tmp_path):
    tileset_path = make_tileset(tmp_path)
    dest = tmp_path / "out.zarr"
    stitch_images(tileset_path, dest, crop_borders=False)
    store = zarr.open(str(dest), mode="r")
    path = store.attrs["ome"]["multiscales"][0]["datasets"][0]["path"]
    np.testing.assert_array_equal(store[path][:], make_ground_truth(3, 3))


def test_stitch_zarr_translation_is_zero_when_stage_position_unavailable(
    mock_tileset_without_project, tmp_path
):
    dest = tmp_path / "out.zarr"
    with pytest.warns(UserWarning, match="MapsProject.xml not found"):
        stitch_images(mock_tileset_without_project, dest, crop_borders=False)
    store = zarr.open(str(dest), mode="r")
    datasets = store.attrs["ome"]["multiscales"][0]["datasets"]
    translation = next(
        t
        for t in datasets[0]["coordinateTransformations"]
        if t["type"] == "translation"
    )["translation"]
    assert translation == pytest.approx([0.0, 0.0])


# --------------------------------------------------------------------------------------
# stitch_images — failure modes
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".tif", ".zarr"])
def test_stitch_raises_value_error_when_no_pyramid_xml(tmp_path, ext):
    tileset_path = make_tileset(tmp_path, include_pyramid_xml=False)
    with pytest.raises(ValueError):
        stitch_images(tileset_path, tmp_path / f"out{ext}")


@pytest.mark.parametrize("ext", [".tif", ".zarr"])
def test_stitch_raises_value_error_when_multiple_pyramid_xml(tmp_path, ext):
    tileset_path = make_tileset(tmp_path, n_channel_folders=2)
    with pytest.raises(ValueError):
        stitch_images(tileset_path, tmp_path / f"out{ext}")


@pytest.mark.parametrize("ext", [".tif", ".zarr"])
def test_stitch_raises_file_not_found_when_tile_directory_empty(tmp_path, ext):
    all_missing = [(c, r) for c in range(3) for r in range(3)]
    tileset_path = make_tileset(tmp_path, missing_tiles=all_missing)
    with pytest.raises(FileNotFoundError):
        stitch_images(tileset_path, tmp_path / f"out{ext}")
