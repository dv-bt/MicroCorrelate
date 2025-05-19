"""
This module contains the core functionality of the microcorrelate package.
"""

import re
from pathlib import Path

import numpy as np
import xmltodict
from imageio.v3 import imread
from tifffile import TiffWriter
import SimpleITK as sitk
from microcorrelate.callbacks import (
    metric_end_plot,
    metric_start_plot,
    metric_plot_values,
)
import matplotlib.pyplot as plt
from skimage.util import compare_images

from microcorrelate.utils import extract_integers


def stitch_images(tileset_path: Path | str, dest_path: Path | str) -> None:
    """Stitch together tiled images from a tileset acquired via the Thermo Fisher Maps
    software and save the stitched image as a single TIFF file. The tileset is expected
    to be in the directory structure produced by Maps, and have been already stitched
    by the acquisition software (that is, there's no overlap or gaps between tiles).

    Parameters
    ----------
    tileset_path : Path | str
        Path to the tileset directory.
    dest_path : Path | str
        Path to save the stitched image.

    Returns
    -------
    None

    """

    if isinstance(tileset_path, str):
        tileset_path = Path(tileset_path)

    xml_path = list(tileset_path.rglob("*/**/*pyramid.xml"))[0]
    pyramid_path = xml_path.parent
    print(f"Stitchig image at {xml_path}")

    with open(xml_path, "r") as file:
        text = file.read()

    pyramid_metadata = xmltodict.parse(text)["root"]

    tile_size = int(pyramid_metadata["imageset"]["@tileWidth"])

    levels = extract_integers(pyramid_path, r"l_(\d+)", "l_*")
    max_level = max(levels)

    cols = extract_integers(pyramid_path, r"c_(\d+)", f"l_{max_level}/c_*")
    rows = extract_integers(pyramid_path, r"tile_(\d+)", f"l_{max_level}/*/tile*", True)

    width = (max(cols) + 1) * tile_size
    height = (max(rows) + 1) * tile_size

    image_stitch = np.zeros((height, width), dtype="uint8")

    for image_path in pyramid_path.glob(f"l_{max_level}/*/tile*"):
        image_path = str(image_path)
        image = imread(image_path)
        col = int(re.search(r"c_(\d+)", image_path).group(1)) * tile_size
        row = int(re.search(r"tile_(\d+)", image_path).group(1)) * tile_size
        image_stitch[row : row + tile_size, col : col + tile_size] = image

    pixelsize = float(pyramid_metadata["metadata"]["pixelsize"]["x"]) * 1e6
    with TiffWriter(dest_path, bigtiff=True) as tif:
        metadata = {
            "axes": "YX",
            "PhysicalSizeX": pixelsize,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixelsize,
            "PhysicalSizeYUnit": "µm",
        }
        options = {
            "tile": (128, 128),
            "resolutionunit": "CENTIMETER",
            "maxworkers": 2,
        }
        tif.write(
            image_stitch,
            resolution=(1e4 / pixelsize, 1e4 / pixelsize),
            metadata=metadata,
            **options,
        )


def _get_tranform_function(transform_type: str) -> sitk.Transform:
    """Get the SimpleITK transform function from the transform type string."""
    if transform_type == "affine":
        return sitk.AffineTransform(2)
    elif transform_type == "rigid":
        return sitk.Euler2DTransform()
    elif transform_type == "similarity":
        return sitk.Similarity2DTransform()
    else:
        raise ValueError("Invalid transform type")


def register_images(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    transform_function: str = "affine",
    num_histogram_bins: int = 50,
    learning_rate: float = 1.0,
    max_iterations: int = 200,
) -> sitk.Transform:
    """Register two images using SimpleITK. The fixed image is the reference image to
    which the moving image is registered. Currently supported transformations are
    affine, rigid, and similarity. The registration is performed using the Mattes
    mutual information metric and gradient descent with line search optimization.

    Parameters
    fixed_image : np.ndarray
        Fixed image in the registration.
    moving_image : np.ndarray
        Moving image to register to the fixed image.
    transform_function : str
        Type of transformation to use. Options are 'affine', 'rigid', and 'similarity'.
    num_histogram_bins : int
        Number of histogram bins for the Mattes mutual information metric.
    learning_rate : float
        Learning rate for the gradient descent optimizer.
    max_iterations : int
        Maximum number of iterations for the gradient descent optimizer.

    Returns
    -------
    final_transform : sitk.Transform
        The final transformation.

    """

    initial_transform = sitk.CenteredTransformInitializer(
        sitk.Cast(fixed_image, moving_image.GetPixelID()),
        moving_image,
        _get_tranform_function(transform_function),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration_method = sitk.ImageRegistrationMethod()

    registration_method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=num_histogram_bins
    )

    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsGradientDescentLineSearch(
        learningRate=learning_rate, numberOfIterations=max_iterations
    )

    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Set the initial moving and optimized transforms.
    optimized_transform = _get_tranform_function(transform_function)
    registration_method.SetMovingInitialTransform(initial_transform)
    registration_method.SetInitialTransform(optimized_transform)

    # Set callbacks for online plotting
    # TO DO

    registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32),
    )

    # Need to compose the transformations after registration.
    final_transform = sitk.CompositeTransform(optimized_transform)
    final_transform.AddTransform(initial_transform)

    # Generate report
    # TO DO

    return final_transform


class ImageRegistration(sitk.ImageRegistrationMethod):
    """Class to register images"""

    def __init__(
        self,
        transform_function: str = "affine",
        num_histogram_bins: int = 30,
        learning_rate: float = 1.0,
        max_iterations: int = 200,
        plot_metrics: bool = True,
    ):
        super().__init__()

        # Initialize optimization parameters
        self.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=num_histogram_bins
        )

        self.SetOptimizerAsGradientDescentLineSearch(
            learningRate=learning_rate, numberOfIterations=max_iterations
        )

        self.SetOptimizerScalesFromPhysicalShift()

        # Initialize transform
        self.SetInterpolator(sitk.sitkLinear)
        self.transform_function = transform_function

        # Initialize final transform as none
        self.final_transform = None

        # Set callbacks
        self.plot_metrics = plot_metrics
        if self.plot_metrics:
            self.AddCommand(sitk.sitkStartEvent, metric_start_plot)
            self.AddCommand(sitk.sitkEndEvent, metric_end_plot)
            self.AddCommand(sitk.sitkIterationEvent, lambda: metric_plot_values(self))

    def register_images(
        self, fixed_image: sitk.Image, moving_image: sitk.Image
    ) -> None:
        initial_transform = sitk.CenteredTransformInitializer(
            sitk.Cast(fixed_image, moving_image.GetPixelID()),
            moving_image,
            _get_tranform_function(self.transform_function),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        optimized_transform = _get_tranform_function(self.transform_function)
        self.SetMovingInitialTransform(initial_transform)
        self.SetInitialTransform(optimized_transform)

        self.Execute(
            sitk.Cast(fixed_image, sitk.sitkFloat32),
            sitk.Cast(moving_image, sitk.sitkFloat32),
        )

        # Need to compose the transformations after registration.
        self.final_transform = sitk.CompositeTransform(optimized_transform)
        self.final_transform.AddTransform(initial_transform)

        if self.plot_metrics:
            self._plot_comparison(fixed_image, moving_image)

    def _plot_comparison(
        self, fixed_image: sitk.Image, moving_image: sitk.Image
    ) -> None:
        """Plot image comparison at the end of registration"""
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(fixed_image)
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetTransform(self.final_transform)

        final_image = resample.Execute(moving_image)
        fig, ax = plt.subplots(dpi=200)
        ax.imshow(
            compare_images(
                sitk.GetArrayFromImage(fixed_image),
                sitk.GetArrayFromImage(final_image),
                method="checkerboard",
            ),
            cmap="gray",
        )

    def apply_registration(
        self, fixed_image: sitk.Image, moving_image: sitk.Image
    ) -> np.ndarray | None:
        if not self.final_transform:
            print(
                "Registration transformation is not yet defined.\n"
                + "Run ImageRegistration.register_images"
            )
            return
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(fixed_image)
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetTransform(self.final_transform)
        moving_image_final = resample.Execute(moving_image)

        final_image = sitk.GetArrayViewFromImage(moving_image_final)
        return final_image
