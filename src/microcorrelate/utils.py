"""
This module is a placeholder for the utils module of the microcorrelate package.
"""

import re
from pathlib import Path

import numpy as np
import skimage as ski


def round_up_multiple(num: int, base: int) -> int:
    """Round up a number to a multiple of base"""
    return num + (base - num) % base


def extract_integers(
    base_path: Path, regex_pattern: str, glob_pattern: str, recursive: bool = False
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


def read_downscaled(image_path: Path, downscale_factor: int) -> np.ndarray[float]:
    """Read an image from the given path, downscale it by the specified factor,
    and convert it to grayscale if it is an RGB image. The downscaled image is
    returned as a floating point image.

    Parameters:
    -----------
    image_path : Path
        The path to the image file.
    downscale_factor : int
        The factor by which to downscale the image.

    Returns:
    --------
    image_ds : np.ndarray
        The downscaled grayscale image.
    """

    image_full = ski.io.imread(image_path)
    if image_full.ndim == 2:
        downscale_tuple = (downscale_factor, downscale_factor)
    else:
        downscale_tuple = (downscale_factor, downscale_factor, 1)

    image_ds = ski.transform.downscale_local_mean(image_full, downscale_tuple)

    # If the image is 3-channel, assume it's RGB and covert it to grayscale
    if image_full.ndim == 3:
        image_ds = ski.color.rgb2gray(image_ds)

    return image_ds
