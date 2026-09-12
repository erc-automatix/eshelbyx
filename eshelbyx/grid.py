"""Regular voxel grid carrying geometry and per-cell fields."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["Grid"]


class Grid(eqx.Module):
    """Regular Cartesian voxel grid with named cell fields.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> grid = Grid.from_function(shape=(32, 32, 32), marker_fn=sphere)
    >>> grid = grid.apply_material_map({0: {"E": 1.0, "nu": 0.3}})
    >>> grid["E"]
    """

    # Geometry
    shape: tuple[int, ...]
    size: tuple[float, ...]
    origin: tuple[float, ...] = (0.0, 0.0, 0.0)
    periodic: tuple[bool, ...] = (True, True, True)

    # Data
    fields: dict[str, jax.Array]
    metadata: dict[str, any] = eqx.field(default_factory=dict, static=True)

    def __init__(
        self,
        shape: tuple[int, ...],
        size: tuple[float, ...] = (1.0, 1.0, 1.0),
        fields: dict[str, jax.Array] | None = None,
        origin: tuple[float, ...] = (0.0, 0.0, 0.0),
        periodic: tuple[bool, ...] | None = None,
        **field_kwargs,
    ):
        """
        Parameters
        ----------
        shape : tuple of int
            Number of voxels per direction.
        size : tuple of float
            Physical extent of the cell per direction.
        fields : dict, optional
            Named cell fields of shape ``shape + components``.
        origin : tuple of float
            Coordinates of the lower corner.
        periodic : tuple of bool, optional
            Periodicity per direction. Defaults to periodic in all directions.
        **field_kwargs
            Additional fields given as keyword arguments.
        """
        self.shape = shape
        self.size = size
        self.origin = origin
        self.periodic = periodic if periodic else tuple([True] * len(shape))
        self.fields = {"markers": 0}
        self.fields.update(fields or {})
        self.metadata = {}

        for key, value in field_kwargs.items():
            self.fields[key] = self._validate_field(value)

    def _validate_field(self, data: jax.Array) -> jax.Array:
        data = jnp.asarray(data)
        if data.shape[: self.ndim] != self.shape:
            raise ValueError(f"Field shape {data.shape} incompatible with grid {self.shape}")
        return data

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.shape)

    @property
    def dim(self) -> int:
        """Alias for :attr:`ndim`."""
        return self.ndim

    @property
    def spacing(self) -> tuple[float, ...]:
        """Voxel size per direction."""
        return tuple(L / N for L, N in zip(self.size, self.shape, strict=True))

    @property
    def num_cells(self) -> int:
        """Total number of voxels."""
        return int(np.prod(self.shape))

    @property
    def volume_fractions(self) -> dict:
        """Volume fraction of each marker value."""
        return {int(v): jnp.mean(self.markers == v) for v in jnp.unique(self.markers)}

    def cell_centers(self) -> jnp.ndarray:
        """Coordinates of the voxel centres, shape ``shape + (ndim,)``."""
        h = self.spacing
        ranges = [
            jnp.arange(N) * h[i] + h[i] / 2 + self.origin[i] for i, N in enumerate(self.shape)
        ]
        return jnp.stack(jnp.meshgrid(*ranges, indexing="ij"), axis=-1)

    # ------------------------------------------------------------------ #
    # Field access
    # ------------------------------------------------------------------ #
    def __getitem__(self, key: str) -> jax.Array:
        return self.fields[key]

    def add_field(self, name: str, data: jax.Array) -> Grid:
        """Return a new grid with ``name`` added or replaced."""
        new_fields = {**self.fields, name: self._validate_field(data)}
        return Grid(self.shape, self.size, new_fields, self.origin, self.periodic)

    def get_field(self, name: str, default=None) -> jax.Array | None:
        """Return a field, or ``default`` if absent."""
        return self.fields.get(name, default)

    def has_field(self, name: str) -> bool:
        """Whether a field of this name exists."""
        return name in self.fields

    @property
    def markers(self) -> jax.Array | None:
        """Phase marker field."""
        return self.get_field("markers")

    def apply_material_map(self, material_map: dict[int, dict[str, float]]) -> Grid:
        """Return a new grid with per-phase properties expanded into fields.

        Parameters
        ----------
        material_map : dict
            ``{marker_id: {property_name: value}}``.

        Examples
        --------
        >>> grid = grid.apply_material_map(          # doctest: +SKIP
        ...     {0: {"E": 1.0, "nu": 0.3}, 1: {"E": 1e-6, "nu": 0.4}}
        ... )
        """
        if "markers" not in self.fields:
            raise ValueError("Grid must have a 'markers' field")

        markers = self.fields["markers"]
        new_fields = {**self.fields}

        prop_names = set()
        for props in material_map.values():
            prop_names.update(props.keys())
        prop_names.discard("name")

        for prop_name in prop_names:
            field = jnp.zeros(self.shape)
            for marker_id, props in material_map.items():
                if prop_name in props:
                    field = jnp.where(markers == marker_id, props[prop_name], field)
            new_fields[prop_name] = field

        return Grid(self.shape, self.size, new_fields, self.origin, self.periodic)

    def broadcast_to_grid(self, tensor):
        """Broadcast a scalar or jaxmat tensor to a field over the grid."""
        from jaxmat.tensors import Tensor

        if isinstance(tensor, Tensor):
            return tensor.__class__(
                array=jnp.broadcast_to(tensor.array, self.shape + tensor.array.shape)
            )
        return jnp.broadcast_to(tensor, self.shape)

    # ------------------------------------------------------------------ #
    # Input / output (delegates to eshelbyx.io, which owns PyVista)
    # ------------------------------------------------------------------ #
    @classmethod
    def from_vti(
        cls,
        filename: str,
        marker_field: str | None = None,
        material_map: dict | None = None,
    ) -> Grid:
        """Read a grid from a VTK image file. See :func:`eshelbyx.io.read_vti`."""
        from eshelbyx.io import read_vti

        return read_vti(filename, marker_field, material_map)

    def to_pyvista(self):
        """Convert to a PyVista ``ImageData``. See :func:`eshelbyx.io.to_pyvista`."""
        from eshelbyx.io import to_pyvista

        return to_pyvista(self)

    def save(self, filename: str, fields: dict[str, jax.Array] | None = None):
        """Write the grid, and optionally extra result fields, to a VTK file."""
        from eshelbyx.io import write_vti

        write_vti(self, filename, fields)

    def plot(self, field: str = "markers", **kwargs):
        """Quick plot of one field."""
        from eshelbyx.io import plot

        plot(self, field, **kwargs)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_function(
        cls,
        shape: tuple[int, ...],
        marker_fn: Callable,
        size: tuple[float, ...] = (1.0, 1.0, 1.0),
        **field_fns,
    ) -> Grid:
        """Build a grid by evaluating functions at the voxel centres.

        Parameters
        ----------
        shape : tuple of int
            Number of voxels per direction.
        marker_fn : callable
            ``(x, y, z) -> markers``, evaluated on the voxel-centre coordinates.
        size : tuple of float
            Physical extent of the cell.
        **field_fns
            Further ``(x, y, z) -> field`` callables, stored under their keyword.
        """
        origin = tuple(0.0 for _ in shape)

        h = tuple(L / N for L, N in zip(size, shape, strict=True))
        ranges = [jnp.arange(N) * h[i] + h[i] / 2 for i, N in enumerate(shape)]
        coords = jnp.stack(jnp.meshgrid(*ranges, indexing="ij"), axis=-1)

        if len(shape) == 3:
            x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        elif len(shape) == 2:
            x, y = coords[..., 0], coords[..., 1]
            z = jnp.zeros_like(x)
        else:
            raise NotImplementedError("Only 2D and 3D grids are supported")

        fields = {"markers": marker_fn(x, y, z)}
        for name, fn in field_fns.items():
            fields[name] = fn(x, y, z)

        return cls(shape, size, fields, origin)

    def __repr__(self) -> str:
        return f"Grid(shape={self.shape}, fields={list(self.fields.keys())})"
