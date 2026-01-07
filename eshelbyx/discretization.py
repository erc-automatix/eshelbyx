import jax
import jax.numpy as jnp
import equinox as eqx


def xi_array_1D(N, L):
    """
    Fourier angular frequencies for a domain of size L with N points.
    """
    freq = jnp.fft.fftfreq(N, d=L / N)
    return 2.0 * jnp.pi * freq


def xi_array_3D(grid):
    """
    Returns the grid of Fourier angular frequencies.

    Output shape: (N1, N2, N3, 3)
    """
    N1, N2, N3 = grid.shape
    L1, L2, L3 = grid.size

    xi1 = xi_array_1D(N1, L1)
    xi2 = xi_array_1D(N2, L2)
    xi3 = xi_array_1D(N3, L3)

    return jnp.stack(
        jnp.meshgrid(xi1, xi2, xi3, indexing="ij"),
        axis=-1,
    )


class DiscretizationScheme(eqx.Module):
    xi: jax.Array
    D_plus: jax.Array
    D_minus: jax.Array
    singular_mask: jax.Array

    def __init__(self, grid):
        self.xi = xi_array_3D(grid)
        self.D_plus = self._D_plus(self.xi, grid)
        self.D_minus = self._D_minus(self.xi, grid)
        self.singular_mask = self._singular_mask(self.xi, grid)

    def _D(self, xi: jax.Array, grid: eqx.Module) -> jax.Array:
        """Return differential vector D(xi)."""

    def _singular_mask(self, xi: jax.Array, grid) -> jax.Array:
        """Return True where Γ₀ must be zeroed."""
        return jnp.all(xi == 0, axis=-1)

    def _D_minus(self, xi: jax.Array, grid: eqx.Module) -> jax.Array:
        """Return differential vector D(xi)."""
        return jnp.conj(self._D(xi, grid))

    def _D_plus(self, xi: jax.Array, grid: eqx.Module) -> jax.Array:
        """Return differential vector D(xi)."""
        return self._D(xi, grid)

    def apply_divergence(self, tau_hat):
        return jnp.einsum(
            "...ij,...j->...i",
            tau_hat,
            self.D_minus,
        )

    def apply_symmetric_gradient(self, u_hat):
        return 0.5 * (
            jnp.einsum("...i,...j->...ij", u_hat, self.D_plus)
            + jnp.einsum("...j,...i->...ij", u_hat, self.D_plus)
        )


class MoulinecSuquetDiscretization(DiscretizationScheme):
    def _D(self, xi, grid):
        return 1j * xi

    def _singular_mask(self, xi, grid):
        # only zero mode
        return jnp.all(xi == 0, axis=-1)


class WillotDiscretization(DiscretizationScheme):
    tol: float = 1e-12

    def _q(self, xi, grid):
        # q_i = 2π n_i / N_i = xi_i * (L_i / N_i)
        return jnp.stack(
            [xi[..., i] * (grid.size[i] / grid.shape[i]) for i in range(3)], axis=-1
        )

    def _D(self, xi, grid):
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

    def _singular_mask(self, xi, grid):
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


class StaggeredDiscretizationScheme(DiscretizationScheme):
    """Staggered schemes: forward/backward operators (e.g. Schneider)."""

    tol: float = 1e-12

    def _singular_mask(self, xi, grid) -> jax.Array:
        Dp = self.D_plus
        Dm = self.D_minus
        nDp = jnp.sum(jnp.real(Dp * jnp.conj(Dp)), axis=-1)
        nDm = jnp.sum(jnp.real(Dm * jnp.conj(Dm)), axis=-1)
        return (nDp < self.tol) | (nDm < self.tol)


class SchneiderDiscretization(StaggeredDiscretizationScheme):
    def _h(self, grid):
        return jnp.array([L / N for L, N in zip(grid.size, grid.shape)])

    def _D_plus(self, xi, grid):
        h = self._h(grid)
        return (jnp.exp(1j * xi * h) - 1.0) / h

    def _D_minus(self, xi, grid):
        h = self._h(grid)
        return (1.0 - jnp.exp(-1j * xi * h)) / h

    def apply_divergence(self, tau_hat):
        D_m = self.D_minus
        D_p = self.D_plus

        Div = jnp.stack(
            [
                jnp.stack([D_m[..., 0], D_p[..., 1], D_p[..., 2]], axis=-1),
                jnp.stack([D_p[..., 0], D_m[..., 1], D_p[..., 2]], axis=-1),
                jnp.stack([D_p[..., 0], D_p[..., 1], D_m[..., 2]], axis=-1),
            ],
            axis=-2,
        )
        return jnp.sum(jnp.multiply(Div, tau_hat), axis=-1)

    def apply_symmetric_gradient(self, u_hat):
        """
        Symmetric gradient in Fourier space (Schneider staggered grid)

        Parameters
        ----------
        u_hat : (..., 3)
            displacement Fourier field
        xi : (..., 3)
            frequencies
        grid : grid object

        Returns
        -------
        eps_hat : (..., 3, 3)
            symmetric strain tensor
        """

        D_p = self.D_plus  # (..., 3)
        D_m = self.D_minus  # (..., 3)

        # Diagonal terms
        e11 = D_p[..., 0] * u_hat[..., 0]
        e22 = D_p[..., 1] * u_hat[..., 1]
        e33 = D_p[..., 2] * u_hat[..., 2]

        # Off-diagonals (symmetric)
        e12 = 0.5 * (D_m[..., 1] * u_hat[..., 0] + D_m[..., 0] * u_hat[..., 1])
        e13 = 0.5 * (D_m[..., 2] * u_hat[..., 0] + D_m[..., 0] * u_hat[..., 2])
        e23 = 0.5 * (D_m[..., 2] * u_hat[..., 1] + D_m[..., 1] * u_hat[..., 2])

        row1 = jnp.stack([e11, e12, e13], axis=-1)
        row2 = jnp.stack([e12, e22, e23], axis=-1)
        row3 = jnp.stack([e13, e23, e33], axis=-1)

        return jnp.stack([row1, row2, row3], axis=-2)
