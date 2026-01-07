import jax
import jax.numpy as jnp
import equinox as eqx


class DiscretizationScheme(eqx.Module):
    def D(self, xi: jax.Array, grid: eqx.Module) -> jax.Array:
        """Return differential vector D(xi)."""

    def singular_mask(self, xi: jax.Array, grid) -> jax.Array:
        """Return True where Γ₀ must be zeroed."""
        return jnp.all(xi == 0, axis=-1)


class MoulinecSuquetDiscretization(DiscretizationScheme):
    def D(self, xi, grid):
        return 1j * xi

    def singular_mask(self, xi, grid):
        # only zero mode
        return jnp.all(xi == 0, axis=-1)


class WillotDiscretization(DiscretizationScheme):
    tol: float = 1e-12

    def _q(self, xi, grid):
        # q_i = 2π n_i / N_i = xi_i * (L_i / N_i)
        return jnp.stack(
            [xi[..., i] * (grid.size[i] / grid.shape[i]) for i in range(3)], axis=-1
        )

    def D(self, xi, grid):
        q = self._q(xi, grid) / 2
        return (
            2j
            * jnp.stack(
                [
                    jnp.sin(q[..., 0]) * jnp.cos(q[..., 1]) * jnp.cos(q[..., 2]),
                    jnp.cos(q[..., 0]) * jnp.sin(q[..., 1]) * jnp.cos(q[..., 2]),
                    jnp.cos(q[..., 0]) * jnp.cos(q[..., 1]) * jnp.sin(q[..., 2]),
                ],
                axis=-1,
            )
            / (grid.size[0] / grid.shape[0])
        )

    def singular_mask(self, xi, grid):
        xi_abs = jnp.abs(xi)
        zero_mode = jnp.all(xi == 0, axis=-1)

        flags = []
        for i in range(3):
            is_even = grid.shape[i] % 2 == 0
            qi = xi_abs[..., i] * (grid.size[i] / grid.shape[i])
            flags.append(is_even & (jnp.abs(qi - jnp.pi) < self.tol))

        nyquist_pairs = (
            (flags[0] & flags[1]) | (flags[0] & flags[2]) | (flags[1] & flags[2])
        )

        return zero_mode | nyquist_pairs
