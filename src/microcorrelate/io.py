"""
This module contains various functions to read image and create image objects.
"""

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
from skimage.io import imread
import SimpleITK as sitk
import h5py
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


def create_itk_image(
    image: np.ndarray,
    spacing: float | tuple[float, float] = (1.0, 1.0),
    origin: tuple[float, float] = (0.0, 0.0),
    channel_axis: int | None = None,
) -> sitk.Image:
    """Convert a NumPy array to a SimpleITK Image.

    This is a simple conversion function for creating ITK images from
    in-memory arrays (e.g., from HDF5 files, zarr arrays, or other sources).

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


def read_tpef(
    image_path: Path, max_level: int = 3
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Read TPEF image data and discover pixel spacing from metadata file.

    Searches for `exp_prop.txt` in the image directory and up to `max_level`
    parent directories to extract pixel spacing information.

    Parameters
    ----------
    image_path : Path
        Path to the TPEF image file.
    max_level : int, optional
        Maximum number of parent directory levels to search for `exp_prop.txt`.
        Default is 3.

    Returns
    -------
    image : np.ndarray
        Image array with shape (Z, channels, Y, X) or (channels, Y, X).
    spacing : tuple[float, float]
        Pixel spacing in micrometers (Y, X).

    Raises
    ------
    FileNotFoundError
        If `exp_prop.txt` is not found within the search scope.

    See Also
    --------
    extract_tpef_spacing : Extract spacing from `exp_prop.txt` file.
    """

    image = imread(image_path)

    level = 0
    found = False
    while level <= max_level:
        prop_path = image_path.parents[level] / "exp_prop.txt"
        if prop_path.exists():
            spacing = extract_tpef_spacing(prop_path)
            found = True
            break
        level += 1

    if not found:
        raise FileNotFoundError(
            "exp_prop.txt file not found within discovery scope with "
            f"max_level={max_level}. "
            "Check that the file exists, or increase max_level"
        )

    return image, spacing


def read_laicp_hdf5(
    filepath: Path | str,
) -> tuple[np.ndarray, tuple[float, float], list[str]]:
    """
    Read LA-ICP-MS data from a TOFWERK HDF5 file.

    Reads the reconstructed image data, infers pixel spacing from the
    position grid, and only keep user selected channels and total ion count.

    Parameters
    ----------
    path : Path | str
        Path to the HDF5 file.

    Returns
    -------
    data : np.ndarray
        Image array of shape (n_channels, rows, cols). Only user selected channels
        (ending with `'\n'` in the channel labels) and total ion count are kept.
    spacing : tuple[float, float]
        Pixel spacing in micrometers (row, col).
    labels : list[str]
        Channel labels corresponding to axis 0 of the preserved data.
    """
    with h5py.File(filepath, "r") as f:
        data = f["tof data 1/image 1/out_default_Area_recon"][:]
        pos = f["point cloud 1/TOFPilot sample data 1/PositionGrid"][:]
        raw_labels = f["tof data 1/NLabel"][:]

    data = data.squeeze()
    pos = pos.squeeze()

    labels = [lab[0].decode("utf-8") for lab in raw_labels]

    # Keep TIC (first channel) and user-selected channels (ending with \n)
    keep_idx = [0]
    keep_labels = [labels[0]]

    for i, lab in enumerate(labels[1:], start=1):
        if lab.endswith("\n") and not lab.startswith("padding"):
            keep_idx.append(i)
            keep_labels.append(lab.strip())

    data = data[keep_idx]

    x, y = pos[:, 0], pos[:, 1]
    spacing = (np.diff(np.unique(y)).mean(), np.diff(np.unique(x)).mean())

    return data, spacing, keep_labels


def read_ome_zarr(zarr_path: Path | str) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Read highest resolution level from OME-Zarr with spacing.

    Parameters
    ----------
    zarr_path : Path or str
        Path to the zarr store.

    Returns
    -------
    data : np.ndarray
        Image array at highest resolution.
    spacing : tuple[float, float]
        Pixel spacing in micrometers (Y, X).

    Raises
    ------
    ValueError
        If unit is not nanometer or micrometer.
    """

    reader = Reader(parse_url(zarr_path))
    nodes = list(reader())

    # First node contains the image
    node = nodes[0]

    # Highest resolution is first level (index 0)
    data = np.array(node.data[0])

    axes = node.metadata["axes"]
    coord_transforms = node.metadata["coordinateTransformations"]
    scale = coord_transforms[0][0]["scale"]

    # Get unit from axes (assuming both y and x have same unit)
    unit = axes[0]["unit"]

    if unit == "nanometer":
        spacing = (scale[0] / 1000, scale[1] / 1000)
    elif unit == "micrometer":
        spacing = (scale[0], scale[1])
    else:
        raise ValueError(f"Unsupported unit '{unit}'")

    return data, spacing


def extract_tpef_spacing(filepath: Path | str) -> tuple[float, float]:
    """
    Extract YX spacing values for a TPEF experiment.

    Input should be an `exp_prop.txt` file, which contains acquisition properties for
    the experiment. This function currently ignores Z spacing.

    Parameters
    ----------
    filepath : Path | str
        Path to `exp_prop.txt` file. The file should contain spacing in the format
        `" X 512     1.381068 µm"`

    Returns
    -------
    spacing : tuple[float, float]
        Spacing in micrometers (Y, X).

    Raises
    ------
    ValueError
        If X or Y spacing was not detected in the file.
    """
    pattern = re.compile(r"^\s*([XY])\s+\d+\s+([\d.]+)\s+µm")

    x_spacing = None
    y_spacing = None

    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            match = pattern.match(line)
            if match:
                axis, spacing = match.groups()
                if axis == "X":
                    x_spacing = float(spacing)
                elif axis == "Y":
                    y_spacing = float(spacing)

    if x_spacing is None or y_spacing is None:
        raise ValueError(
            f"Spacing not found: X={'found' if x_spacing else 'missing'}, "
            f"Y={'found' if y_spacing else 'missing'}"
        )

    return (y_spacing, x_spacing)
