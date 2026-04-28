import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pyvista as pv
from typing import Tuple, Optional, Dict, Callable


class Grid(eqx.Module):
    """
    Unified grid class with geometry, fields, materials, and PyVista integration.


    Examples:
        # Create from file with materials
        grid = Grid.from_vti("file.vti", material_map={...})

        # Create programmatically
        grid = Grid.from_function(shape=(64,64,64), size=(1,1,1), marker_fn=fn)

        # Access fields
        E = grid.E  # or grid['E'] or grid.fields['E']

        # PyVista
        grid.plot(field='E')
        grid.save('results.vti')
    """

    # Geometry
    shape: Tuple[int, ...]
    size: Tuple[float, ...]
    origin: Tuple[float, ...] = (0.0, 0.0, 0.0)
    periodic: Tuple[bool, ...] = (True, True, True)

    # Data
    fields: Dict[str, jax.Array]
    metadata: Dict[str, any] = eqx.field(default_factory=dict, static=True)

    def __init__(
        self,
        shape: Tuple[int, ...],
        size: Tuple[float, ...] = (1.0, 1.0, 1.0),
        fields: Optional[Dict[str, jax.Array]] = None,
        origin: Tuple[float, ...] = (0.0, 0.0, 0.0),
        periodic: Tuple[bool, ...] = None,
        **field_kwargs,
    ):
        """
        Create grid directly.

        Args:
            shape: Grid shape (N1, N2, N3)
            size: Physical size (L1, L2, L3)
            fields: Dictionary of fields
            origin: Origin point
            periodic: Periodicity per dimension
            **field_kwargs: Additional fields as keyword arguments
        """
        self.shape = shape
        self.size = size
        self.origin = origin
        self.periodic = periodic if periodic else tuple([True] * len(shape))
        self.fields = {"markers": 0}
        self.fields.update(fields or {})
        self.metadata = {}

        # Add fields from kwargs
        for key, value in field_kwargs.items():
            self.fields[key] = self._validate_field(value)

    def _validate_field(self, data: jax.Array) -> jax.Array:
        """Validate field matches grid shape"""
        data = jnp.asarray(data)
        if data.shape[: self.ndim] != self.shape:
            raise ValueError(
                f"Field shape {data.shape} incompatible with grid {self.shape}"
            )
        return data

    # ========================================================================
    # Geometry properties
    # ========================================================================

    @property
    def ndim(self) -> int:
        """Number of dimensions"""
        return len(self.shape)

    @property
    def dim(self) -> int:
        """Alias for ndim (backward compatible)"""
        return self.ndim

    @property
    def spacing(self) -> Tuple[float, ...]:
        """Grid spacing (cell size)"""
        return tuple(L / N for L, N in zip(self.size, self.shape))

    @property
    def num_cells(self) -> int:
        """Total number of cells"""
        return int(np.prod(self.shape))

    @property
    def volume_fractions(self) -> dict:
        return {int(v): jnp.mean(self.markers == v) for v in jnp.unique(self.markers)}

    def cell_centers(self) -> jnp.ndarray:
        """Cell center coordinates"""
        h = self.spacing
        ranges = [
            jnp.arange(N) * h[i] + h[i] / 2 + self.origin[i]
            for i, N in enumerate(self.shape)
        ]
        return jnp.stack(jnp.meshgrid(*ranges, indexing="ij"), axis=-1)

    # ========================================================================
    # Field access (multiple ways for convenience)
    # ========================================================================

    def __getitem__(self, key: str) -> jax.Array:
        """Access field like grid['E']"""
        return self.fields[key]

    def __setitem__(self, key: str, value: jax.Array):
        """Set field like grid['E'] = value (creates new grid)"""
        self.fields[key] = self._validate_field(value)

    def add_field(self, name: str, data: jax.Array) -> "Grid":
        """Add or update field (returns new Grid)"""
        new_fields = {**self.fields, name: self._validate_field(data)}
        return Grid(self.shape, self.size, new_fields, self.origin, self.periodic)

    def get_field(self, name: str, default=None) -> Optional[jax.Array]:
        """Get field with optional default"""
        return self.fields.get(name, default)

    def has_field(self, name: str) -> bool:
        """Check if field exists"""
        return name in self.fields

    # ========================================================================
    # Material property shortcuts (just convenience - they're still fields!)
    # ========================================================================

    @property
    def markers(self) -> Optional[jax.Array]:
        """Marker field (backward compatible)"""
        return self.get_field("markers")

    def get_material_params(self) -> Tuple[jax.Array, jax.Array]:
        """Get (E, nu) for solver compatibility"""
        E = self.get_field("E")
        nu = self.get_field("nu")
        if E is None or nu is None:
            raise ValueError("Material properties (E, nu) not found")
        return E.ravel(), nu.ravel()

    def apply_material_map(self, material_map: Dict[int, Dict[str, float]]) -> "Grid":
        """
        Apply material properties from marker map.

        Args:
            material_map: {marker_id: {'E': value, 'nu': value, ...}}

        Example:
            materials = {
                0: {'E': 1.0, 'nu': 0.3},
                1: {'E': 1e-6, 'nu': 0.4},
            }
            grid = grid.apply_material_map(materials)
        """
        if "markers" not in self.fields:
            raise ValueError("Grid must have 'markers' field")

        markers = self.fields["markers"]
        new_fields = {**self.fields}

        # Get all property names
        prop_names = set()
        for props in material_map.values():
            prop_names.update(props.keys())
        prop_names.discard("name")  # Metadata, not field

        # Create field for each property
        for prop_name in prop_names:
            field = jnp.zeros(self.shape)
            for marker_id, props in material_map.items():
                if prop_name in props:
                    field = jnp.where(markers == marker_id, props[prop_name], field)
            new_fields[prop_name] = field

        return Grid(self.shape, self.size, new_fields, self.origin, self.periodic)

    # ========================================================================
    # Grid operations
    # ========================================================================

    def broadcast_to_grid(self, tensor):
        """Broadcast tensor to grid shape (backward compatible)"""
        from jaxmat.tensors import Tensor

        if isinstance(tensor, Tensor):
            return tensor.__class__(
                array=jnp.broadcast_to(
                    tensor.array,
                    self.shape + tensor.array.shape,
                )
            )
        else:
            return jnp.broadcast_to(tensor, self.shape)

    # ========================================================================
    # PyVista integration (built-in, no separate adapter needed)
    # ========================================================================

    def to_pyvista(self) -> pv.ImageData:
        """Convert to PyVista ImageData"""
        pv_grid = pv.ImageData()
        pv_grid.dimensions = tuple(n + 1 for n in self.shape)
        pv_grid.origin = self.origin
        pv_grid.spacing = self.spacing

        # Add all fields
        for name, field in self.fields.items():
            if field.ndim == self.ndim:
                # Scalar field
                pv_grid.cell_data[name] = np.array(field).flatten(order="F")
            else:
                # Tensor field
                n_cells = self.num_cells
                extra_shape = field.shape[self.ndim :]
                reshaped = np.array(field).reshape((n_cells,) + extra_shape, order="F")
                pv_grid.cell_data[name] = reshaped

        return pv_grid

    def save(self, filename: str):
        """Save to VTI file"""
        pv_grid = self.to_pyvista()
        pv_grid.save(filename)

    def plot(self, field: str = "markers", **kwargs):
        """Quick plot of a field"""
        pv_grid = self.to_pyvista()
        pv_grid.plot(scalars=field, **kwargs)

    # ========================================================================
    # Factory methods (class methods for creation)
    # ========================================================================

    @classmethod
    def from_vti(
        cls,
        filename: str,
        marker_field: Optional[str] = None,
        material_map: Optional[Dict] = None,
    ) -> "Grid":
        """
        Load from VTI file.

        Args:
            filename: Path to VTI file
            marker_field: Field to use as markers (None = first field)
            material_map: Optional material property mapping
        """
        pv_grid = pv.read(filename)

        # Extract geometry
        nx, ny, nz = tuple(d - 1 for d in pv_grid.dimensions)
        shape = (nx, ny, nz)
        size = tuple(pv_grid.spacing[i] * n for i, n in enumerate((nx, ny, nz)))
        origin = tuple(pv_grid.origin)

        # Extract fields
        fields = {}
        for name, data in pv_grid.cell_data.items():
            fields[name] = jnp.asarray(np.array(data).reshape(shape, order="F"))

        # Handle markers
        if marker_field is None and len(fields) > 0:
            first_key = list(fields.keys())[0]
            if "markers" not in fields:
                fields["markers"] = fields[first_key]
        elif marker_field and marker_field != "markers":
            if marker_field in fields:
                fields["markers"] = fields[marker_field]

        # Create grid
        grid = cls(shape, size, fields, origin)

        # Apply materials if provided
        if material_map:
            grid = grid.apply_material_map(material_map)

        return grid

    @classmethod
    def from_function(
        cls,
        shape: Tuple[int, ...],
        marker_fn: Callable,
        size: Tuple[float, ...] = (1.0, 1.0, 1.0),
        **field_fns,
    ) -> "Grid":
        """
        Create from function.

        Args:
            shape: Grid shape
            marker_fn: Function (x, y, z) -> markers
            size: Physical size
            **field_fns: Additional field functions
        """
        origin = tuple(0.0 for _ in shape)

        # Create coordinate arrays
        h = tuple(L / N for L, N in zip(size, shape))
        ranges = [jnp.arange(N) * h[i] + h[i] / 2 for i, N in enumerate(shape)]
        coords = jnp.stack(jnp.meshgrid(*ranges, indexing="ij"), axis=-1)

        # Evaluate marker function
        if len(shape) == 3:
            x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
            markers = marker_fn(x, y, z)
        elif len(shape) == 2:
            x, y = coords[..., 0], coords[..., 1]
            markers = marker_fn(x, y, jnp.zeros_like(x))
        else:
            raise NotImplementedError("Only 2D/3D supported")

        fields = {"markers": markers}

        # Additional fields
        for name, fn in field_fns.items():
            if len(shape) == 3:
                fields[name] = fn(x, y, z)
            else:
                fields[name] = fn(x, y, jnp.zeros_like(x))

        return cls(shape, size, fields, origin)

    def __repr__(self) -> str:
        return f"Grid(shape={self.shape}, fields={list(self.fields.keys())})"


class OldGrid(eqx.Module):
    size: jax.Array
    shape: tuple = eqx.field(static=True)
    markers: jax.Array
    dim: int = 3

    def __init__(self, markers, size=None):
        self.dim = markers.ndim
        self.shape = markers.shape
        if size is None:
            self.size = jnp.array(self.shape, dtype=jnp.float64)
        else:
            self.size = jnp.asarray(size)
        self.markers = markers

    def broadcast_to_grid(self, tensor):
        if isinstance(tensor, Tensor):
            return tensor.__class__(
                array=jnp.broadcast_to(
                    tensor.array,
                    self.shape + tensor.array.shape,
                )
            )


def load_vtk_grid(filename, marker_name=None):
    pv_grid = pv.read(filename)

    # Extract phase markers
    if marker_name is None:
        phase = pv_grid.cell_data.values()[0]
    else:
        phase = pv_grid.cell_data.get(marker_name)

    # Recover dimensions
    nx, ny, nz = jnp.array(pv_grid.dimensions) - 1
    phase = phase.reshape((nx, ny, nz), order="F")

    # Recover physical size
    Lx = pv_grid.spacing[0] * nx
    Ly = pv_grid.spacing[1] * ny
    Lz = pv_grid.spacing[2] * nz

    grid = Grid(
        markers=jnp.asarray(phase),
        shape=(int(nx), int(ny), int(nz)),
        size=jnp.array([Lx, Ly, Lz], dtype=jnp.float64),
    )

    return grid, pv_grid
