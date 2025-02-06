import pytest
from pathlib import Path
from microcorrelate.utils import round_up_multiple, extract_integers


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