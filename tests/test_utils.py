"""Tests for the utility functions in microcorrelate.utils."""

import numpy as np

from microcorrelate.utils import (
    extract_integers,
    find_common_vals,
    flatten_dict,
    get_crop_idx,
    round_up_multiple,
)


def test_round_up_multiple():
    assert round_up_multiple(5, 3) == 6
    assert round_up_multiple(10, 5) == 10
    assert round_up_multiple(14, 4) == 16
    assert round_up_multiple(0, 1) == 0
    assert round_up_multiple(7, 7) == 7


def test_extract_integers_returns_matches_from_root_dir(tmp_path):
    (tmp_path / "file_1.txt").write_text("dummy")
    (tmp_path / "file_2.txt").write_text("dummy")
    (tmp_path / "file_10.txt").write_text("dummy")
    (tmp_path / "file_20.txt").write_text("dummy")
    result = extract_integers(tmp_path, r"file_(\d+)\.txt", "file_*.txt")
    assert sorted(result) == [1, 2, 10, 20]


def test_extract_integers_finds_files_in_subdirectories(tmp_path):
    # extract_integers uses rglob unconditionally, so subdirs are always searched
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file_30.txt").write_text("dummy")
    result = extract_integers(tmp_path, r"file_(\d+)\.txt", "file_*.txt")
    assert 30 in result


def test_extract_integers_returns_empty_list_when_no_matches(tmp_path):
    (tmp_path / "file_1.txt").write_text("dummy")
    result = extract_integers(tmp_path, r"nonexistent_(\d+)\.txt", "file_*.txt")
    assert result == []


def test_get_crop_idx_returns_bounding_box_of_non_zero_region():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    row_slice, col_slice = get_crop_idx(mask)
    assert row_slice == slice(2, 5)
    assert col_slice == slice(3, 7)


def test_get_crop_idx_full_image_is_unchanged():
    mask = np.ones((8, 8), dtype=np.uint8)
    row_slice, col_slice = get_crop_idx(mask)
    assert row_slice == slice(0, 8)
    assert col_slice == slice(0, 8)


def test_flatten_dict_collapses_nested_keys_with_dot_separator():
    nested = {"a": {"b": {"c": 1}, "d": 2}, "e": 3}
    result = flatten_dict(nested)
    assert result == {"a.b.c": 1, "a.d": 2, "e": 3}


def test_flatten_dict_already_flat_dict_is_unchanged():
    flat = {"x": 1, "y": 2}
    assert flatten_dict(flat) == flat


def test_find_common_vals_returns_keys_with_equal_values():
    d1 = {"a": 1, "b": 2, "c": 3}
    d2 = {"a": 1, "b": 99, "c": 3}
    result = find_common_vals(d1, d2)
    assert result == {"a": 1, "c": 3}


def test_find_common_vals_returns_empty_when_no_shared_values():
    d1 = {"a": 1, "b": 2}
    d2 = {"a": 9, "b": 9}
    assert find_common_vals(d1, d2) == {}


def test_find_common_vals_ignores_keys_absent_from_second_dict():
    d1 = {"a": 1, "b": 2}
    d2 = {"a": 1}
    result = find_common_vals(d1, d2)
    assert result == {"a": 1}
