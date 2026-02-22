"""
This module is a placeholder for the utils module of the microcorrelate package.
"""

import re
from pathlib import Path
from typing import Iterator, Any
import numpy as np
from skimage.transform import resize
import cv2


def round_up_multiple(num: int, base: int) -> int:
    """Round up a number to a multiple of base"""
    return num + (base - num) % base


def extract_integers(
    base_path: Path, regex_pattern: str, glob_pattern: str
) -> list[int]:
    """Extract integers matching the group in a regex pattern from filenames matching
    the glob pattern
    """
    try:
        matches = [
            int(re.search(regex_pattern, str(i)).group(1))
            for i in base_path.rglob(glob_pattern)
        ]
    except AttributeError:
        matches = []
    return matches


def vprint(text: str, verbose: bool) -> None:
    """Helper function for cleanly handling print statements with a verbose option"""
    if verbose:
        print(text)


def deep_items(
    d: dict[str, Any], prefix: tuple[str, ...] = ()
) -> Iterator[tuple[str, Any]]:
    """Recursively yield (key.path, value) for all non-dict values
    in a nested dictionary."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from deep_items(v, prefix + (k,))
        else:
            path = prefix + (k,)
            yield ".".join(path), v


def flatten_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested dictionary. Key hierarchy is preserved by assigning new
    keys as 'a.b.c'."""
    return dict(deep_items(d))


def find_common_vals(d1: dict[str, Any], d2: dict[str, Any]) -> dict[str, Any]:
    """Returns a dictionary with the keys share the same value between the two input
    dictionaries"""
    return {key: d1[key] for key in d1.keys() if d1[key] == d2.get(key)}


def resample_to_spacing(
    array: np.ndarray,
    input_spacing: float | tuple[float, float],
    output_spacing: float | tuple[float, float],
    channel_axis: int | None = None,
) -> np.ndarray:
    """Resample an array to a target pixel spacing.

    Parameters
    ----------
    array : np.ndarray
        Input image array.
    input_spacing : float or tuple of float
        Current pixel spacing. A scalar is treated as isotropic.
    output_spacing : float or tuple of float
        Target pixel spacing. A scalar is treated as isotropic.
    channel_axis : int or None, optional
        Axis containing channels, excluded from resampling. Default is None.

    Returns
    -------
    np.ndarray
        Resampled array with the same dtype as the input.
    """
    if np.isscalar(input_spacing):
        input_spacing = (input_spacing, input_spacing)
    if np.isscalar(output_spacing):
        output_spacing = (output_spacing, output_spacing)

    spatial_axes = [i for i in range(array.ndim) if i != channel_axis]
    output_shape = list(array.shape)
    for ax, (in_sp, out_sp) in zip(spatial_axes, zip(input_spacing, output_spacing)):
        output_shape[ax] = int(np.round(array.shape[ax] * in_sp / out_sp))

    return resize(array, output_shape, preserve_range=True)


def get_crop_idx(mask: np.ndarray, margin: int = 0) -> tuple[slice, ...]:
    """
    Compute array slices to crop a numpy array to the bounding box of non-zero mask regions.

    Useful for removing empty borders (e.g. black frames from image stitching).

    Parameters
    ----------
    mask : np.ndarray
        2D binary mask where non-zero pixels define the region of interest.
    margin : int, optional
        Number of pixels to expand the bounding box on each side, by default 0.
        Clamped to array boundaries.

    Returns
    -------
    tuple[slice, ...]
        A tuple of slices ``(row_slice, col_slice)`` that can be used to index
        a numpy array directly, e.g. ``image[get_crop_idx(mask)]``.

    Examples
    --------
    .. code-block:: python

        cropped = image[get_crop_idx(mask, margin=10)]
    """
    coords = cv2.findNonZero(mask.astype(int))
    x, y, w, h = cv2.boundingRect(coords)

    x = max(x - margin, 0)
    y = max(y - margin, 0)
    w = min(w + 2 * margin, mask.shape[1])
    h = min(h + 2 * margin, mask.shape[0])

    return np.s_[y : y + h, x : x + w]
