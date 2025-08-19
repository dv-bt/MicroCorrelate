import numpy as np
import pytest
from imageio.v3 import immeta, imwrite
from tifffile import TiffFile

from microcorrelate.stitching import stitch_images

# Constants for the mock tileset
TILE_SIZE = 128
PIXEL_SIZE = 1.0e-7


@pytest.fixture
def mock_tileset(tmp_path):
    # Create a mock tileset directory structure with dummy images and metadata
    tileset_path = tmp_path / "tileset"
    tileset_path.mkdir()

    channel_path = tileset_path / "channel"
    channel_path.mkdir()

    pyramid_path = channel_path / "data"
    pyramid_path.mkdir()

    level_path = pyramid_path / "l_0"
    level_path.mkdir()

    for col in range(3):
        col_path = level_path / f"c_{col}"
        col_path.mkdir()
        for row in range(3):
            tile_path = col_path / f"tile_{row}.png"
            if (col, row) != (2, 2):
                image = np.full(
                    (TILE_SIZE, TILE_SIZE), fill_value=row * 64 + col, dtype=np.uint8
                )
                imwrite(tile_path, image)

    xml_content = f"""
    <root>
        <imageset tileWidth="{TILE_SIZE}" tileHeight="{TILE_SIZE}"/>
        <metadata>
            <physicalsize>
                <x>{PIXEL_SIZE * TILE_SIZE * 3}</x>
                <y>{PIXEL_SIZE * TILE_SIZE * 3}</y>
            </physicalsize>
            <pixelsize>
                <x>{PIXEL_SIZE}</x>
                <y>{PIXEL_SIZE}</y>
            </pixelsize>
        </metadata>
    </root>
    """
    xml_path = pyramid_path / "pyramid.xml"
    xml_path.write_text(xml_content)

    return tileset_path


def test_stitch_images(mock_tileset, tmp_path):
    dest_path = tmp_path / "stitched_image.tiff"
    stitch_images(mock_tileset, dest_path)

    with TiffFile(dest_path) as tif:
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.shape == (TILE_SIZE * 3, TILE_SIZE * 3)
        assert page.dtype == np.uint8

        # Check some pixel values to ensure stitching is correct
        image = page.asarray()
        assert image[0, 0] == 0
        assert image[0, 128] == 1
        assert image[0, 256] == 2
        assert image[128, 0] == 64
        assert image[128, 128] == 65
        assert image[128, 256] == 66
        assert image[256, 0] == 128
        assert image[256, 128] == 129
        assert image[256, 256] == 0

    metadata = immeta(dest_path)
    assert metadata["PhysicalSizeX"] == PIXEL_SIZE * 1e6
    assert metadata["PhysicalSizeY"] == PIXEL_SIZE * 1e6
    assert metadata["PhysicalSizeXUnit"] == "µm"
    assert metadata["PhysicalSizeYUnit"] == "µm"
