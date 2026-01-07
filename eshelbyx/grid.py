import jax
import jax.numpy as jnp
import equinox as eqx


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
