"""
This module contains the core functionality of the microcorrelate package.
"""

from typing import Literal
import numpy as np
import SimpleITK as sitk
from microcorrelate.callbacks import (
    metric_end_plot,
    metric_start_plot,
    metric_plot_values,
)
import matplotlib.pyplot as plt
from skimage.util import compare_images


def _get_transform_function(
    transform_type: Literal["affine", "rigid", "similarity"],
    ndim: int = 2,
) -> sitk.Transform:
    """Get the SimpleITK transform function from the transform type string."""
    if ndim == 2:
        if transform_type == "affine":
            return sitk.AffineTransform(2)
        elif transform_type == "rigid":
            return sitk.Euler2DTransform()
        elif transform_type == "similarity":
            return sitk.Similarity2DTransform()
        else:
            raise ValueError("Invalid transform type")
    elif ndim == 3:
        if transform_type == "affine":
            return sitk.AffineTransform(3)
        elif transform_type == "rigid":
            return sitk.Euler3DTransform()
        elif transform_type == "similarity":
            return sitk.Similarity3DTransform()
        else:
            raise ValueError("Invalid transform type")
    else:
        raise ValueError(f"Unsupported dimensionality: {ndim}. Must be 2 or 3.")


def _create_rotation_transform(
    angle_degrees: float | tuple[float, float, float],
    image: sitk.Image,
) -> sitk.Euler2DTransform | sitk.Euler3DTransform:
    """
    Create a rotation transform centered on the image center.

    Parameters
    ----------
    angle_degrees : float or tuple[float, float, float]
        For 2D: a single rotation angle in degrees (counterclockwise).
        For 3D: a tuple ``(rx, ry, rz)`` of rotation angles in degrees around
        the X, Y, and Z axes respectively.
    image : sitk.Image
        Reference image used to determine the rotation center.

    Returns
    -------
    sitk.Euler2DTransform or sitk.Euler3DTransform
        Rotation transform configured to rotate around the image center.
    """
    ndim = image.GetDimension()
    size = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()

    center = [origin[i] + (size[i] - 1) * spacing[i] / 2.0 for i in range(ndim)]

    if ndim == 2:
        transform = sitk.Euler2DTransform()
        transform.SetCenter(center)
        transform.SetAngle(np.deg2rad(angle_degrees))
    else:
        if not isinstance(angle_degrees, (tuple, list)) or len(angle_degrees) != 3:
            raise ValueError(
                "For 3D rotation, angle_degrees must be a tuple of 3 angles "
                "(rx, ry, rz) in degrees."
            )
        transform = sitk.Euler3DTransform()
        transform.SetCenter(center)
        transform.SetRotation(
            np.deg2rad(angle_degrees[0]),
            np.deg2rad(angle_degrees[1]),
            np.deg2rad(angle_degrees[2]),
        )

    return transform


def register_images(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    transform_function: str = "affine",
    num_histogram_bins: int = 50,
    learning_rate: float = 1.0,
    max_iterations: int = 200,
    initial_rotation: float | tuple[float, float, float] | None = None,
) -> sitk.Transform:
    """Register two images using SimpleITK. The fixed image is the reference image to
    which the moving image is registered. Currently supported transformations are
    affine, rigid, and similarity. The registration is performed using the Mattes
    mutual information metric and gradient descent with line search optimization.
    Both 2D and 3D images are supported; dimensionality is inferred automatically
    from the input images.

    Parameters
    ----------
    fixed_image : sitk.Image
        Fixed image in the registration.
    moving_image : sitk.Image
        Moving image to register to the fixed image.
    transform_function : str
        Type of transformation to use. Options are 'affine', 'rigid', and 'similarity'.
    num_histogram_bins : int
        Number of histogram bins for the Mattes mutual information metric.
    learning_rate : float
        Learning rate for the gradient descent optimizer.
    max_iterations : int
        Maximum number of iterations for the gradient descent optimizer.
    initial_rotation : float or None
        Initial rotation angle in degrees (counterclockwise) to apply before
        registration. Useful for coarse manual alignment. If None or 0, no
        rotation is applied. Default is None.

    Returns
    -------
    final_transform : sitk.Transform
        The final transformation composed of manual rotation (if specified),
        centered initialization, and optimized registration.

    """

    ndim = fixed_image.GetDimension()

    initial_transform = sitk.CenteredTransformInitializer(
        sitk.Cast(fixed_image, moving_image.GetPixelID()),
        moving_image,
        _get_transform_function(transform_function, ndim),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Create manual rotation transform if specified
    manual_transform = None
    if initial_rotation is not None and initial_rotation != 0:
        manual_transform = _create_rotation_transform(
            initial_rotation,
            moving_image,
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
    optimized_transform = _get_transform_function(transform_function, ndim)

    if manual_transform is not None:
        # Compose manual rotation with centered initializer
        pre_optimization_transform = sitk.CompositeTransform(ndim)
        pre_optimization_transform.AddTransform(manual_transform)
        pre_optimization_transform.AddTransform(initial_transform)
        registration_method.SetMovingInitialTransform(pre_optimization_transform)
    else:
        registration_method.SetMovingInitialTransform(initial_transform)

    registration_method.SetInitialTransform(optimized_transform)

    # Set callbacks for online plotting
    # TO DO

    registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32),
    )

    # Need to compose the transformations after registration.
    if manual_transform is not None:
        final_transform = sitk.CompositeTransform(optimized_transform)
        final_transform.AddTransform(initial_transform)
        final_transform.AddTransform(manual_transform)
    else:
        final_transform = sitk.CompositeTransform(optimized_transform)
        final_transform.AddTransform(initial_transform)

    # Generate report
    # TO DO

    return final_transform


class ImageRegistration(sitk.ImageRegistrationMethod):
    """Intensity-based image registration using SimpleITK.

    Wraps :class:`SimpleITK.ImageRegistrationMethod` with Mattes mutual
    information metric and gradient descent with line search optimisation.
    Optionally plots the metric value at each iteration.

    Parameters
    ----------
    transform_function : {"affine", "rigid", "similarity"}
        Type of spatial transform to estimate. Default is ``"affine"``.
    num_histogram_bins : int
        Number of histogram bins for the Mattes mutual information metric.
        Default is 30.
    learning_rate : float
        Learning rate for the gradient descent optimiser. Default is 1.0.
    max_iterations : int
        Maximum number of optimiser iterations. Default is 200.
    plot_metrics : bool
        If True, plot the metric value live during registration. Default is True.
    initial_rotation : float or None
        Initial rotation angle in degrees (counterclockwise) applied before
        optimisation. If None, no pre-rotation is applied. Default is None.

    Attributes
    ----------
    final_transform : sitk.CompositeTransform or None
        Estimated transform after calling :meth:`register_images`.
        ``None`` until registration has been run.
    """

    def __init__(
        self,
        transform_function: Literal["affine", "rigid", "similarity"] = "affine",
        num_histogram_bins: int = 30,
        learning_rate: float = 1.0,
        max_iterations: int = 200,
        plot_metrics: bool = True,
        initial_rotation: float | tuple[float, float, float] | None = None,
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

        # Store initial rotation
        self.initial_rotation = initial_rotation

        # Set callbacks
        self.plot_metrics = plot_metrics
        if self.plot_metrics:
            self.AddCommand(sitk.sitkStartEvent, metric_start_plot)
            self.AddCommand(sitk.sitkEndEvent, metric_end_plot)
            self.AddCommand(sitk.sitkIterationEvent, lambda: metric_plot_values(self))

    def register_images(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        initial_rotation: float | tuple[float, float, float] | None = None,
    ) -> None:
        """
        Register moving image to fixed image.

        Parameters
        ----------
        fixed_image : sitk.Image
            Reference image.
        moving_image : sitk.Image
            Image to be registered.
        initial_rotation : float or tuple[float, float, float] or None
            Initial rotation angle(s) in degrees. For 2D, a single float. For 3D,
            a tuple ``(rx, ry, rz)``. If provided, overrides the initial_rotation
            set in __init__. If None, uses the value from __init__. Default is None.
        """
        # Use method parameter if provided, otherwise use instance variable
        rotation = (
            initial_rotation if initial_rotation is not None else self.initial_rotation
        )

        ndim = fixed_image.GetDimension()

        initial_transform = sitk.CenteredTransformInitializer(
            sitk.Cast(fixed_image, moving_image.GetPixelID()),
            moving_image,
            _get_transform_function(self.transform_function, ndim),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        # Create manual rotation transform if specified
        manual_transform = None
        if rotation is not None and rotation != 0:
            manual_transform = _create_rotation_transform(
                rotation,
                moving_image,
            )

        optimized_transform = _get_transform_function(self.transform_function, ndim)

        if manual_transform is not None:
            # Compose manual rotation with centered initializer
            pre_optimization_transform = sitk.CompositeTransform(ndim)
            pre_optimization_transform.AddTransform(manual_transform)
            pre_optimization_transform.AddTransform(initial_transform)
            self.SetMovingInitialTransform(pre_optimization_transform)
        else:
            self.SetMovingInitialTransform(initial_transform)

        self.SetInitialTransform(optimized_transform)

        self.Execute(
            sitk.Cast(fixed_image, sitk.sitkFloat32),
            sitk.Cast(moving_image, sitk.sitkFloat32),
        )

        # Need to compose the transformations after registration.
        if manual_transform is not None:
            self.final_transform = sitk.CompositeTransform(optimized_transform)
            self.final_transform.AddTransform(initial_transform)
            self.final_transform.AddTransform(manual_transform)
        else:
            self.final_transform = sitk.CompositeTransform(optimized_transform)
            self.final_transform.AddTransform(initial_transform)

        if self.plot_metrics:
            self._plot_comparison(fixed_image, moving_image)

    def _plot_comparison(
        self, fixed_image: sitk.Image, moving_image: sitk.Image
    ) -> None:
        """Plot image comparison at the end of registration"""
        if fixed_image.GetDimension() != 2:
            return  # checkerboard comparison only supported for 2D images
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
        """Apply the estimated transform to resample a moving image onto the fixed grid.

        Parameters
        ----------
        fixed_image : sitk.Image
            Reference image defining the output grid.
        moving_image : sitk.Image
            Image to be resampled.

        Returns
        -------
        np.ndarray or None
            Registered image array, or None if no transform has been estimated yet.
        """
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
