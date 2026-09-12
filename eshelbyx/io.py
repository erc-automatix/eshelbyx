"""Reading and writing of voxel grids and result fields.

All PyVista use is confined to this module and imported lazily, so that
``import eshelbyx`` works without a visualisation stack installed. Install the
optional dependency with ``pip install eshelbyx[viz]``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = ["plot", "read_vti", "to_pyvista", "write_vti"]


def _pyvista():
    """Import PyVista on demand with an actionable error message."""
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "This feature requires PyVista, which is an optional dependency of "
            "eshelbyx. Install it with `pip install eshelbyx[viz]`."
        ) from exc
    return pv


def _flatten_cells(field: np.ndarray, ndim: int) -> np.ndarray:
    """Flatten a cell field to the ``(n_cells, n_components)`` layout VTK wants.

    Two orderings have to be got right, and they differ:

    - **Spatial.** VTK numbers cells with the first index varying fastest, so
      the spatial axes are reversed and then flattened in C order.
    - **Components.** Any trailing component axes are flattened in C order
      *here*, explicitly. Handing VTK an ``(n, 3, 3)`` array instead makes it
      ravel the component block in Fortran order, which silently stores the
      transpose of each tensor. That is invisible for symmetric stress and
      strain and wrong for anything else, so it is not left to VTK.
    """
    field = np.asarray(field)
    spatial = tuple(range(ndim))
    components = tuple(range(ndim, field.ndim))
    reordered = np.transpose(field, spatial[::-1] + components)
    n_cells = int(np.prod(field.shape[:ndim]))
    n_components = int(np.prod(field.shape[ndim:])) if components else 1
    flat = np.ascontiguousarray(reordered).reshape(n_cells, n_components)
    return flat[:, 0] if not components else flat


def to_pyvista(grid, fields: dict[str, np.ndarray] | None = None):
    """Convert a :class:`~eshelbyx.grid.Grid` to a PyVista ``ImageData``.

    Parameters
    ----------
    grid : Grid
        The grid to convert. All of its fields are attached as cell data.
    fields : dict, optional
        Extra fields to attach, e.g. solver results. Keys override grid fields
        of the same name.
    """
    pv = _pyvista()

    image = pv.ImageData()
    image.dimensions = tuple(n + 1 for n in grid.shape)
    image.origin = tuple(grid.origin)
    image.spacing = tuple(grid.spacing)

    all_fields = dict(grid.fields)
    all_fields.update(fields or {})
    for name, field in all_fields.items():
        image.cell_data[name] = _flatten_cells(field, grid.ndim)

    return image


def write_vti(grid, filename: str, fields: dict[str, np.ndarray] | None = None):
    """Write a grid and optional result fields to a VTK image file."""
    to_pyvista(grid, fields).save(filename)


def plot(grid, field: str = "markers", **kwargs):
    """Quick plot of one cell field."""
    to_pyvista(grid).plot(scalars=field, **kwargs)


def read_vti(
    filename: str,
    marker_field: str | None = None,
    material_map: dict | None = None,
):
    """Read a voxel grid from a VTK image file.

    Parameters
    ----------
    filename : str
        Path to the ``.vti`` file.
    marker_field : str, optional
        Name of the cell array holding the phase markers. If omitted, a cell
        array literally named ``"markers"`` is used when present; otherwise the
        first cell array in the file is used and a warning is emitted, since
        that choice depends on file ordering.
    material_map : dict, optional
        ``{marker_id: {property: value}}`` applied via
        :meth:`~eshelbyx.grid.Grid.apply_material_map`.
    """
    from eshelbyx.grid import Grid

    pv = _pyvista()
    image = pv.read(filename)

    shape = tuple(int(d) - 1 for d in image.dimensions)
    ndim = len(shape)
    size = tuple(float(image.spacing[i]) * n for i, n in enumerate(shape))
    origin = tuple(float(o) for o in image.origin)

    fields = {}
    for name, data in image.cell_data.items():
        array = np.asarray(data)
        # VTK records a component count, not a tensor rank, so a 3x3 field comes
        # back with a trailing axis of 9. Components are in C order (see
        # _flatten_cells), so the caller can simply reshape to (..., 3, 3).
        components = array.shape[1:]
        # Inverse of _flatten_cells: unflatten to reversed spatial axes, then
        # restore the natural (i, j, k) ordering.
        reordered = array.reshape(shape[::-1] + components)
        spatial = tuple(range(ndim))
        fields[name] = jnp.asarray(
            np.transpose(reordered, spatial[::-1] + tuple(range(ndim, reordered.ndim)))
        )

    if marker_field is not None:
        if marker_field not in fields:
            raise KeyError(
                f"No cell array named {marker_field!r} in {filename}. Available: {sorted(fields)}"
            )
        fields["markers"] = fields[marker_field]
    elif "markers" not in fields:
        if not fields:
            raise ValueError(f"{filename} contains no cell data.")
        first = next(iter(fields))
        import warnings

        warnings.warn(
            f"No 'markers' cell array in {filename}; using {first!r}. "
            "Pass marker_field= to select it explicitly.",
            stacklevel=2,
        )
        fields["markers"] = fields[first]

    grid = Grid(shape=shape, size=size, fields=fields, origin=origin)
    if material_map:
        grid = grid.apply_material_map(material_map)
    return grid
