"""Voxelisation of surface meshes onto a regular grid.

These helpers turn a closed surface mesh into a boolean voxel mask, which can
then be combined into a marker field for :class:`~eshelbyx.grid.Grid`. They
depend on PyVista, which is an optional dependency; see :mod:`eshelbyx.io`.

The three ``voxelize_*`` functions differ only in the inclusion rule and
therefore in how they treat the interface:

===========================  ==========================================
``voxelize_centers``         voxel centre inside the mesh
``voxelize_all_corners``     all eight corners inside (erodes the phase)
``voxelize_any_corner``      at least one corner inside (dilates it)
===========================  ==========================================

Centre-based inclusion is the usual choice. The corner-based variants bracket
it and are useful for estimating the sensitivity of a homogenised property to
interface voxelisation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "make_image_grid",
    "voxelize_all_corners",
    "voxelize_any_corner",
    "voxelize_centers",
]


def _corner_mask(mesh, image) -> np.ndarray:
    """Boolean inside/outside test at every grid *point* (voxel corner)."""
    from eshelbyx.io import _pyvista

    pv = _pyvista()
    nx, ny, nz = image.dimensions
    points = pv.PolyData(image.points)
    inside = points.select_enclosed_points(mesh)["SelectedPoints"].astype(bool)
    return inside.reshape(nx, ny, nz)


def _corner_octets(inside: np.ndarray):
    """The eight corner arrays of every voxel, each of shape ``dims - 1``."""
    return (
        inside[:-1, :-1, :-1],
        inside[1:, :-1, :-1],
        inside[:-1, 1:, :-1],
        inside[1:, 1:, :-1],
        inside[:-1, :-1, 1:],
        inside[1:, :-1, 1:],
        inside[:-1, 1:, 1:],
        inside[1:, 1:, 1:],
    )


def make_image_grid(bounds, dims):
    """Build a PyVista ``ImageData`` spanning ``bounds`` with ``dims`` points.

    Parameters
    ----------
    bounds : sequence of float
        ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
    dims : tuple of int
        Number of grid *points* per direction; the voxel count is ``dims - 1``.
    """
    from eshelbyx.io import _pyvista

    pv = _pyvista()
    nx, ny, nz = dims
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    image = pv.ImageData()
    image.dimensions = (nx, ny, nz)
    image.origin = (xmin, ymin, zmin)
    image.spacing = (
        (xmax - xmin) / (nx - 1),
        (ymax - ymin) / (ny - 1),
        (zmax - zmin) / (nz - 1),
    )
    return image


def voxelize_centers(mesh, image) -> np.ndarray:
    """Voxels whose centre lies inside ``mesh``.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``image.dimensions - 1``.
    """
    centers = image.cell_centers()
    enclosed = centers.select_enclosed_points(mesh, tolerance=1e-16)
    inside = enclosed["SelectedPoints"].view(np.bool_)

    nx, ny, nz = image.dimensions
    return inside.reshape((nx - 1, ny - 1, nz - 1), order="F")


def voxelize_all_corners(mesh, image) -> np.ndarray:
    """Voxels with all eight corners inside ``mesh`` (eroded phase)."""
    octets = _corner_octets(_corner_mask(mesh, image))
    return np.logical_and.reduce(octets)


def voxelize_any_corner(mesh, image) -> np.ndarray:
    """Voxels with at least one corner inside ``mesh`` (dilated phase)."""
    octets = _corner_octets(_corner_mask(mesh, image))
    return np.logical_or.reduce(octets)
