"""Tests for the utility functions in microcorrelate.utils."""

import pytest  # noqa: F401
import skimage as ski

from microcorrelate.utils import extract_integers, read_downscaled, round_up_multiple


def test_round_up_multiple():
    assert round_up_multiple(5, 3) == 6
    assert round_up_multiple(10, 5) == 10
    assert round_up_multiple(14, 4) == 16
    assert round_up_multiple(0, 1) == 0
    assert round_up_multiple(7, 7) == 7


def test_extract_integers(tmp_path):
    # Setup test files
    (tmp_path / "file_1.txt").write_text("dummy content")
    (tmp_path / "file_2.txt").write_text("dummy content")
    (tmp_path / "file_10.txt").write_text("dummy content")
    (tmp_path / "file_20.txt").write_text("dummy content")

    regex_pattern = r"file_(\d+)\.txt"
    glob_pattern = "file_*.txt"

    result = extract_integers(tmp_path, regex_pattern, glob_pattern)
    assert sorted(result) == [1, 2, 10, 20]

    # Test with recursive flag
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file_30.txt").write_text("dummy content")
    result = extract_integers(tmp_path, regex_pattern, glob_pattern, recursive=True)
    assert sorted(result) == [1, 2, 10, 20, 30]

    # Test with no matches
    regex_pattern = r"nonexistent_(\d+)\.txt"
    result = extract_integers(tmp_path, regex_pattern, glob_pattern)
    assert result == []


def test_read_downscaled(tmp_path):
    # Create a test image
    image = ski.data.camera()
    image_path = tmp_path / "test_image.png"
    ski.io.imsave(image_path, image)

    # Test downscaling by a factor of 2
    downscale_factor = 2
    downscaled_image = read_downscaled(image_path, downscale_factor)
    assert downscaled_image.shape == (image.shape[0] // 2, image.shape[1] // 2)

    # Test downscaling by a factor of 4
    downscale_factor = 4
    downscaled_image = read_downscaled(image_path, downscale_factor)
    assert downscaled_image.shape == (image.shape[0] // 4, image.shape[1] // 4)

    # Test with an RGB image
    rgb_image = ski.data.astronaut()
    rgb_image_path = tmp_path / "test_rgb_image.png"
    ski.io.imsave(rgb_image_path, rgb_image)

    downscale_factor = 2
    downscaled_rgb_image = read_downscaled(rgb_image_path, downscale_factor)
    assert downscaled_rgb_image.shape == (
        rgb_image.shape[0] // 2,
        rgb_image.shape[1] // 2,
    )
    assert downscaled_rgb_image.ndim == 2
