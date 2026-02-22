"""
This module contains functions intendend for interactive landmark-based registration
with Napari.
"""

from __future__ import annotations
import logging
import sys
from typing import Literal
from dataclasses import dataclass
import numpy as np
import napari
import SimpleITK as sitk
from microcorrelate.core import _get_transform_function
from microcorrelate.io import create_itk_image
from microcorrelate.utils import vprint


@dataclass
class ViewerParams:
    """Visual parameters for napari viewer."""

    fixed_color: str = "green"
    moving_color: str = "magenta"
    pts_size: float = 10.0
    pts_border_width: float = 0.2
    pts_border_color: str = "white"
    text_color: str = "white"
    text_size: int = 12
    text_anchor: str = "upper_right"


class NapariRegistrator:
    def __init__(
        self,
        fixed_data: np.ndarray,
        moving_data: np.ndarray,
        fixed_spacing: tuple[float, float],
        moving_spacing: tuple[float, float],
        transform_type: Literal["affine", "rigid", "similarity"] = "affine",
        fixed_channel_axis: int | None = None,
        moving_channel_axis: int | None = None,
        viewer_params: ViewerParams | None = None,
    ) -> None:
        """Set up the Napari-based interactive registrator.

        Parameters
        ----------
        fixed_data : np.ndarray
            Reference image. The moving image will be registered onto this grid.
        moving_data : np.ndarray
            Moving image to be registered.
        fixed_spacing : tuple[float, float]
            Pixel pitch of the fixed image (row, col) in physical units.
        moving_spacing : tuple[float, float]
            Pixel pitch of the moving image (row, col) in physical units.
        transform_type : {"affine", "rigid", "similarity"}
            Type of spatial transform to estimate. Default is "affine".
        fixed_channel_axis : int or None
            Channel axis index in `fixed_data`, or None for single-channel.
        moving_channel_axis : int or None
            Channel axis index in `moving_data`, or None for single-channel.
        viewer_params : ViewerParams or None
            Parameters for the napari viewer. If None, uses the default values for
            ViewerParams.


        Notes
        -----
        After calling :meth:`run`, the estimated transform is stored in
        ``self.transform`` and the registered moving image in ``self.resampled_data``.
        """
        self.fixed_data = fixed_data
        self.fixed_spacing = fixed_spacing
        self.fixed_image = create_itk_image(
            fixed_data, spacing=fixed_spacing, channel_axis=fixed_channel_axis
        )

        self.moving_data = moving_data
        self.moving_spacing = moving_spacing
        self.moving_channel_axis = moving_channel_axis

        self.transform_type = transform_type
        self.transform = _get_transform_function(transform_type)
        self.resampled_data = np.zeros(shape=fixed_data.shape, dtype=moving_data.dtype)

        self._viewer_params = viewer_params if viewer_params else ViewerParams()

    def run(self, verbose=False) -> None:
        """Open a Napari viewer for interactive landmark placement and run registration.

        Place matching point pairs on the ``landmarks_fixed`` and
        ``landmarks_moving`` layers, then close the viewer window to trigger
        transform estimation and resampling of the moving image.

        Parameters
        ----------
        verbose : bool
            If True, print progress messages. Default is False.

        Raises
        ------
        ValueError
            If landmark counts don't match or fewer than 3 points provided.
        """

        pts_fixed, pts_moving = self._show_landmarks()

        vprint("Estimating transform", verbose)
        self.transform = self._estimate_transform(
            pts_fixed=pts_fixed, pts_moving=pts_moving
        )

        vprint("Resampling moving image", verbose)
        self.resampled_data = self.register_image(
            moving_array=self.moving_data,
            moving_spacing=self.moving_spacing,
            moving_channel_axis=self.moving_channel_axis,
        )

    def _estimate_transform(
        self,
        pts_fixed: np.ndarray,
        pts_moving: np.ndarray,
    ) -> sitk.Transform:
        """
        Estimate affine transform from napari landmarks in physical coordinates.

        Parameters
        ----------
        pts_fixed : np.ndarray
            Landmarks in fixed image, napari order (row, col).
        pts_moving : np.ndarray
            Landmarks in moving image, napari order (row, col).

        Returns
        -------
        sitk.Transform
            Transform in physical coordinates.

        Raises
        ------
        ValueError
            If landmark counts don't match or fewer than 3 points provided.
        """
        # Validate landmarks
        if len(pts_fixed) != len(pts_moving):
            raise ValueError(
                f"Landmark count mismatch: {len(pts_fixed)} fixed points "
                f"vs {len(pts_moving)} moving points"
            )

        if len(pts_fixed) < 3:
            raise ValueError(
                f"At least 3 landmark pairs required, got {len(pts_fixed)}"
            )

        moving_phys = (
            (pts_moving[:, ::-1] * self.moving_spacing[::-1]).flatten().tolist()
        )
        fixed_phys = (pts_fixed[:, ::-1] * self.fixed_spacing[::-1]).flatten().tolist()

        transform = _get_transform_function(self.transform_type)
        transform = sitk.LandmarkBasedTransformInitializer(
            transform, fixed_phys, moving_phys
        )

        return transform

    def register_image(
        self,
        moving_array: np.ndarray,
        moving_spacing: tuple[float, float],
        moving_channel_axis: int | None = None,
        default_value: float = 0.0,
    ) -> np.ndarray:
        """
        Apply transform and resample moving image onto fixed image grid.

        Parameters
        ----------
        moving_array : np.ndarray
            Moving image array.
        moving_spacing : Tuple[float, float]
            Moving image pixel pitch (row, col).
        moving_channel_axis : int | None
            Index of the channel axis of the moving image, if present. If None, the
            image is treated as single channel.
        default_value : float
            Fill value for regions outside moving image bounds.

        Returns
        -------
        np.ndarray
            Registered image on fixed image grid.
        """
        moving_image = create_itk_image(
            moving_array, moving_spacing, channel_axis=moving_channel_axis
        )

        registered_sitk = sitk.Resample(
            moving_image,
            self.fixed_image,
            self.transform,
            sitk.sitkLinear,
            default_value,
            moving_image.GetPixelID(),
        )

        result = sitk.GetArrayFromImage(registered_sitk)

        if moving_channel_axis is not None:
            # sitk returns channel as last axis; move it back to original position
            result = np.moveaxis(result, -1, moving_channel_axis)

        return result

    def _create_viewer(self) -> napari.Viewer:
        """Create napari viewer with plugin warnings suppressed."""
        # Suppress output logging for napari plugins with incorrect schemas.
        # This is safe for our use case since they are not used.
        suppress = ["npe2.manifest.schema", "npe2.manifest", "npe2", "napari"]
        saved = {name: logging.getLogger(name).level for name in suppress}
        for name in suppress:
            logging.getLogger(name).setLevel(logging.CRITICAL)
        try:
            viewer = napari.Viewer()
        finally:
            for name, level in saved.items():
                logging.getLogger(name).setLevel(level)
        return viewer

    def _show_landmarks(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Show napari viewer for interactive landmark placement.

        Displays moving and fixed images with point layers for landmark selection.
        Returns landmark coordinates in physical units when the viewer is closed.

        Returns
        -------
        fixed_landmarks : np.ndarray
            Fixed landmark coordinates in physical units.
        moving_landmarks : np.ndarray
            Moving landmark coordinates in physical units.
        """

        viewer = self._create_viewer()

        viewer.add_image(
            self.fixed_data,
            name="image_fixed",
            colormap=self._viewer_params.fixed_color,
            scale=self.fixed_spacing,
        )
        viewer.add_image(
            self.moving_data,
            name="image_moving",
            colormap=self._viewer_params.moving_color,
            scale=self.moving_spacing,
        )

        pts_fixed = viewer.add_points(
            name="landmarks_fixed",
            scale=self.fixed_spacing,
            face_color=self._viewer_params.fixed_color,
            border_color=self._viewer_params.pts_border_color,
            border_width=self._viewer_params.pts_border_width,
            border_width_is_relative=True,
        )
        pts_moving = viewer.add_points(
            name="landmarks_moving",
            scale=self.moving_spacing,
            face_color=self._viewer_params.moving_color,
            border_color=self._viewer_params.pts_border_color,
            border_width=self._viewer_params.pts_border_width,
            border_width_is_relative=True,
        )

        def update_pts_text(layer):
            """Text update callback for points layers"""
            layer.text = {
                "string": [str(i + 1) for i in range(len(layer.data))],
                "color": self._viewer_params.text_color,
                "size": self._viewer_params.text_size,
                "anchor": self._viewer_params.text_anchor,
            }

        # Connect to data change events
        pts_fixed.events.data.connect(lambda e: update_pts_text(pts_fixed))
        pts_moving.events.data.connect(lambda e: update_pts_text(pts_moving))

        # Initialize
        update_pts_text(pts_fixed)
        update_pts_text(pts_moving)

        _napari_blocking_run(viewer)

        return pts_fixed.data, pts_moving.data

    def show_registered(self):
        """Show the resampled image overlaid to the fixed image"""

        viewer = self._create_viewer()

        viewer.add_image(
            self.fixed_data,
            name="image_fixed",
            colormap=self._viewer_params.fixed_color,
            scale=self.fixed_spacing,
        )
        viewer.add_image(
            self.resampled_data,
            name="image_registered",
            colormap=self._viewer_params.moving_color,
            scale=self.fixed_spacing,
        )

        _napari_blocking_run(viewer)


def _napari_blocking_run(viewer: napari.Viewer) -> None:
    """
    Create a napari blocking run that works for both scripts and interactive envs.

    napari.run() is non-blocking in Jupyter/IPython environments where the Qt event loop
    is already managed by IPython (%gui qt). Mirror napari's own internal check and
    block explicitly via a local QEventLoop until the viewer window is destroyed.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari Viewer object to control.
    """
    napari.run()

    ipy = sys.modules.get("IPython")
    if (
        ipy
        and (shell := ipy.get_ipython())
        and getattr(shell, "active_eventloop", None) == "qt"
    ):
        from qtpy.QtCore import QEventLoop

        loop = QEventLoop()
        viewer.window._qt_window.destroyed.connect(loop.quit)

        try:
            loop.exec_()
        except RuntimeError:
            # Ignore Qt cleanup race condition
            pass
