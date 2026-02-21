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
    spacing: float | tuple[float, float] = (1.0, 1.0),
    origin: tuple[float, float] = (0.0, 0.0),
    channel_axis: int | None = None,
) -> sitk.Image:
    """Convert a NumPy array to a SimpleITK Image.

    This is a simple conversion function for creating ITK images from
    in-memory arrays (e.g., from HDF5 files, zarr arrays, or other sources).
    For reading and downscaling image files, use read_itk_image instead.

    Both single- and multi-channel images are supported. Pixel spacing
    can be isotropic (scalar) or anisotropic (tuple).
    For a reference of the SimpleITK Image origin and spacing, see:
    https://itk.org/ITKSoftwareGuide/html/Book1/ITKSoftwareGuide-Book1ch4.html

    Parameters
    ----------
    image : np.ndarray
        The input image as a NumPy array.
    spacing : float | tuple of float, optional
        The pixel spacing in physical units. For isotropic images, this
        is the distance between pixel centers. Default is (1.0, 1.0).
    origin : tuple of float, optional
        The physical coordinates of the origin (0,0) pixel. Default is (0.0, 0.0).
    channel_axis : int | None, optional
        The axis in image that contains channel information, if the image is
        multichannel. If None, the image is treated as single channel.
        Default is None.

    Returns
    -------
    image_itk : SimpleITK.Image
        SimpleITK Image object with specified spacing and origin.

    Examples
    --------
    >>> import numpy as np
    >>> from microcorrelate.utils import create_itk_image
    >>>
    >>> # Create from HDF5 array
    >>> import h5py
    >>> with h5py.File('data.h5', 'r') as f:
    >>>     array = f['images/layer1'][:]
    >>>     image = create_itk_image(array, spacing=0.5)
    >>>
    >>> # Create from zarr array
    >>> import zarr
    >>> z = zarr.open('data.zarr', mode='r')
    >>> image = create_itk_image(z['images/0'][:], spacing=1.2)
    """
    if np.isscalar(spacing):
        spacing = (spacing, spacing)

    multichannel = channel_axis is not None

    if multichannel:
        # sitk expects channel as last axis; move from wherever it is
        image = np.moveaxis(image, channel_axis, -1)

    image_itk = sitk.GetImageFromArray(image, isVector=multichannel)
    image_itk.SetSpacing(spacing)
    image_itk.SetOrigin(origin)
    return image_itk


def read_itk_image(
    image_path: Path,
    downscale_factor: int = 1,
    spacing: float = 1.0,
    origin: tuple[float] = (0.0, 0.0),
    use_binning: bool = True,
) -> sitk.Image:
    """Read an image file and convert it to a SimpleITK Image with optional downscaling.

    This function uses SimpleITK's native resampling to handle downscaling,
    which automatically adjusts spacing and origin. This is more robust than
    manual numpy-based downscaling as it preserves image metadata correctly.

    Parameters
    ----------
    image_path : Path
        Path to the image file to read.
    downscale_factor : int, optional
        Factor by which to downscale the image. For example, downscale_factor=2
        reduces the image size by half in each dimension. Default is 1 (no downscaling).
    spacing : float, optional
        The original pixel spacing in physical units before any downscaling.
        Default is 1.0.
    origin : tuple of float, optional
        The physical coordinates of the origin (0,0) pixel. Default is (0.0, 0.0).
    use_binning : bool, optional
        If True, use BinShrink (faster, averages pixel values during downsampling).
        If False, use Resample with linear interpolation (smoother but slower).
        Default is True.

    Returns
    -------
    sitk.Image
        SimpleITK Image object, downscaled if requested. Spacing and origin are
        automatically adjusted by SimpleITK to maintain physical coordinate system.

    Notes
    -----
    When downscaling by factor N:
    - Image dimensions are reduced by factor N
    - Spacing is automatically increased by factor N
    - Physical size of the image remains the same
    - SimpleITK handles all metadata adjustments

    Examples
    --------
    >>> from pathlib import Path
    >>> from microcorrelate.utils import read_itk_image
    >>>
    >>> # Read image at full resolution
    >>> image = read_itk_image(Path('image.tif'))
    >>>
    >>> # Read with 4x downscaling for faster processing
    >>> image_small = read_itk_image(Path('image.tif'), downscale_factor=4)
    >>>
    >>> # Read with custom spacing (e.g., microscope pixel size)
    >>> image = read_itk_image(Path('image.tif'), spacing=0.325, downscale_factor=2)

    See Also
    --------
    read_downscaled : Function to read and downscale an image using OpenCV.
    create_itk_image : Function to create an ITK image from a numpy array.
    """
    # Try to read image using SimpleITK (handles various formats)
    try:
        image_itk = sitk.ReadImage(str(image_path))
    except RuntimeError:
        # If SimpleITK can't read it, fall back to our custom reader
        # (handles RGB conversion, etc.)
        image_np = read_downscaled(image_path, downscale_factor=1)
        image_itk = sitk.GetImageFromArray(image_np)

    # Set the original spacing and origin
    image_itk.SetSpacing((spacing, spacing))
    image_itk.SetOrigin(origin)

    # Apply downscaling if requested
    if downscale_factor > 1:
        if use_binning:
            # BinShrink: Faster, averages pixels (good for reducing noise)
            # Automatically updates spacing by the shrink factor
            image_itk = sitk.BinShrink(image_itk, [downscale_factor, downscale_factor])
        else:
            # Resample: Smoother but slower, uses interpolation
            original_size = image_itk.GetSize()
            new_size = [s // downscale_factor for s in original_size]

            image_itk = sitk.Resample(
                image_itk,
                new_size,
                sitk.Transform(),  # Identity transform
                sitk.sitkLinear,  # Linear interpolation
                image_itk.GetOrigin(),
                [spacing * downscale_factor] * 2,  # New spacing
                image_itk.GetDirection(),
                0.0,  # Default pixel value
                image_itk.GetPixelID(),
            )

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
    return {key: d1[key] for key in d1.keys() if d1[key] == d2.get(key)}
