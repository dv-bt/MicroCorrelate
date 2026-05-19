"""Tests for the CLI argument parser in microcorrelate.cli."""

import pytest

from microcorrelate.cli import _parse_stitch_args

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

BASE_ARGS = ["stitch-maps", "-s", "tileset", "-d", "out.tif"]


def parse(*extra: str, monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setattr("sys.argv", [*BASE_ARGS, *extra])
    return _parse_stitch_args()


# --------------------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------------------


def test_defaults(monkeypatch):
    args = parse(monkeypatch=monkeypatch)
    assert args.group_path is None
    assert args.pyramid_levels == 1
    assert args.compress is False
    assert args.crop_borders is True
    assert args.verbose is False


# --------------------------------------------------------------------------------------
# Required arguments
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_flag", ["-s", "--source"]
)
@pytest.mark.parametrize(
    "dest_flag", ["-d", "--dest"]
)
def test_required_args(source_flag, dest_flag, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["stitch-maps", source_flag, "tileset", dest_flag, "out.tif"],
    )
    args = _parse_stitch_args()
    assert args.source.name == "tileset"
    assert args.dest.name == "out.tif"


# --------------------------------------------------------------------------------------
# Optional flags — long and short forms
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["-g", "--group_path"])
def test_group_path(flag, monkeypatch):
    args = parse(flag, "my/group", monkeypatch=monkeypatch)
    assert args.group_path == "my/group"


@pytest.mark.parametrize("flag", ["-p", "--pyramid_levels"])
def test_pyramid_levels(flag, monkeypatch):
    args = parse(flag, "4", monkeypatch=monkeypatch)
    assert args.pyramid_levels == 4


@pytest.mark.parametrize("flag", ["-c", "--compress"])
def test_compress(flag, monkeypatch):
    args = parse(flag, monkeypatch=monkeypatch)
    assert args.compress is True


@pytest.mark.parametrize("flag", ["-n", "--no-crop"])
def test_no_crop(flag, monkeypatch):
    args = parse(flag, monkeypatch=monkeypatch)
    assert args.crop_borders is False


def test_crop_borders_true_by_default(monkeypatch):
    args = parse(monkeypatch=monkeypatch)
    assert args.crop_borders is True


@pytest.mark.parametrize("flag", ["-v", "--verbose"])
def test_verbose(flag, monkeypatch):
    args = parse(flag, monkeypatch=monkeypatch)
    assert args.verbose is True


# --------------------------------------------------------------------------------------
# Path resolution via working_dir
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["-w", "--working_dir"])
def test_working_dir_resolves_relative_paths(flag, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "stitch-maps",
            "-s", "tileset",
            "-d", "out.tif",
            flag, str(tmp_path),
        ],
    )
    args = _parse_stitch_args()
    assert args.source == (tmp_path / "tileset").resolve()
    assert args.dest == (tmp_path / "out.tif").resolve()
