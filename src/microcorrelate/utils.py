"""
This module is a placeholder for the utils module of the microcorrelate package.
"""

import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813
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


def create_itk_image(
    image: np.ndarray,
    scaling: float | int = 1,
    spacing_old: float = 1.0,
    origin_old: tuple[float] = (0.0, 0.0),
) -> sitk.Image:
    """Converts a NumPy array to a SimpleITK Image with optional scaling.
    Only single-channel images with isotropic pixels are supported.
    For a reference of the SimpleITK Image orgin and spacing, see:
    https://itk.org/ITKSoftwareGuide/html/Book1/ITKSoftwareGuide-Book1ch4.html

    Parameters:
    -----------
    image : np.ndarray
        The input image as a NumPy array.
    scaling : float or int, optional
        The scaling factor for the image spacing. Default is 1 (no scaling).
    spacing_old : float, optional
        The original spacing of the image. Default is 1.0.
    origin_old : tuple of float, optional
        The original origin of the image. Default is (0.0, 0.0).

    Returns:
    --------
    image_itk : SimpleITK.Image
        The resulting SimpleITK Image with updated spacing and origin if scaling
        is applied.
    """

    image_itk = sitk.GetImageFromArray(image)

    if scaling != 1:
        spacing_new = spacing_old * scaling
        origin_scaling = (spacing_new - spacing_old) / 2
        origin_new = tuple(i + origin_scaling for i in origin_old)
        image_itk.SetOrigin(origin_new)
        image_itk.SetSpacing((spacing_new, spacing_new))

    return image_itk
