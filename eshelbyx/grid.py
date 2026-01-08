import jax
import jax.numpy as jnp
import equinox as eqx
import pyvista as pv


class Grid(eqx.Module):
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
        size=jnp.array([Lx, Ly, Lz], dtype=jnp.float64),
    )

    return grid, pv_grid
