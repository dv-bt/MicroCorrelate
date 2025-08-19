"""
This module is a placeholder for the utils module of the microcorrelate package.
"""

import re
from pathlib import Path
from typing import Iterator, Any

import numpy as np
import SimpleITK as sitk  # noqa: N813
import skimage as ski
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


def read_downscaled(image_path: Path, downscale_factor: int = 1) -> np.ndarray[float]:
    """Read an image from the given path, optinally downscaling it by the specified factor,
    and convert it to grayscale if it is an RGB image. The downscaled image is
    returned as a floating point image.

    Parameters:
    -----------
    image_path : Path
        The path to the image file.
    downscale_factor : int
        The factor by which to downscale the image. Dowscale if larger than 1.
        Default = 1

    Returns:
    --------
    image_ds : np.ndarray
        The downscaled grayscale image.
    """

    image_full = ski.io.imread(image_path)

    # If downscale factor is 1, return the original image
    if downscale_factor <= 1:
        return image_full

    # Calculate new dimensions for the resized image
    height, width = image_full.shape[:2]
    new_height, new_width = height // downscale_factor, width // downscale_factor

    # Resize the image using OpenCV
    image_ds = cv2.resize(
        image_full, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    # If the image is 3-channel, assume it's RGB and covert it to grayscale
    if image_full.ndim == 3:
        image_ds = cv2.cvtColor(image_ds, cv2.COLOR_RGB2GRAY)

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


def read_itk_image(
    image_path: Path,
    downscale_factor: int,
    spacing: float = 1.0,
    origin: tuple[float] = (0.0, 0.0),
):
    """
    Read an ITK image from the given path and return it as an ITK image object.
    This function reads a microscopy image using the `read_downscaled` function
    and converts it to an ITK image using `create_itk_image`.

    Parameters
    ----------
    image_path : Path
        Path to the image file to read.
    downscale_factor : int
        Factor by which the image should be downscaled during reading.
    spacing : float, optional
        The spacing between pixels in the image, by default 1.0.
    origin : tuple[float], optional
        The physical coordinate of the origin (0,0) pixel, by default (0.0, 0.0).

    Returns
    -------
    itk.Image
        The read and processed ITK image object.

    See Also
    --------
    read_downscaled : Function to read and downscale an image.
    create_itk_image : Function to create an ITK image from a numpy array.
    """

    image_np = read_downscaled(image_path, downscale_factor)
    image_itk = create_itk_image(image_np, downscale_factor, spacing, origin)
    return image_itk


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
    return {key: d1[key] for key in d1.keys() if d1[key] == d2[key]}
