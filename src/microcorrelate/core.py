"""
This module contains the core functionality of the microcorrelate package.
"""

import numpy as np
import SimpleITK as sitk
from microcorrelate.callbacks import (
    metric_end_plot,
    metric_start_plot,
    metric_plot_values,
)
import matplotlib.pyplot as plt
from skimage.util import compare_images


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
        resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetTransform(self.final_transform)
        moving_image_final = resample.Execute(moving_image)

        final_image = sitk.GetArrayViewFromImage(moving_image_final)
        return final_image
