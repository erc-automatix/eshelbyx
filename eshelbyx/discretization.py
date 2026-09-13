"""Discrete differential operators in Fourier space.

Half-spectrum convention
------------------------
All strain, stress and polarization fields are real, so their Fourier
transforms are Hermitian: ``F(-k) = conj(F(k))``. The library therefore works
on the *half spectrum* produced by ``jnp.fft.rfftn``, in which the last spatial
axis is truncated to ``N // 2 + 1`` modes. Every array held by a
``DiscretizationScheme`` (``xi``, ``D_plus``, ``D_minus``, ``singular_mask``),
and the Green kernel built from them, has this reduced shape.

This is exact rather than an approximation, because every scheme satisfies
``D(-xi) = conj(D(xi))``. The Green kernel is assembled from ``D`` with real
coefficients, so the whole operator obeys ``M(-xi) = conj(M(xi))`` and maps
Hermitian input to Hermitian output. The discarded half is recovered exactly by
``irfftn``.

The practical consequence is that anything passed to ``apply_divergence`` or
``apply_symmetric_gradient`` must be a half-spectrum field from ``rfftn``, not a
full ``fftn`` spectrum.

Two- and three-dimensional grids
--------------------------------
A grid may be two-dimensional. Fields keep their full 3x3 tensor structure and
``D`` keeps three components; the out-of-plane frequency is identically zero, so
the out-of-plane component of ``D`` vanishes. This is plane strain: the strain
may have a non-zero out-of-plane component imposed macroscopically, but nothing
varies through the thickness. The out-of-plane displacement then decouples from
the in-plane ones in the acoustic tensor, which is anti-plane shear, and is
solved alongside at no extra cost.

The acoustic tensor stays invertible in 2D: with ``D = (D1, D2, 0)`` it is block
diagonal, a 2x2 in-plane block plus the scalar ``mu * D^2``, so the isotropic
kernel formula needs no special case.
"""

import equinox as eqx
import jax
import jax.numpy as jnp


def xi_array_1D(N, L, real=False):
    """
    Fourier angular frequencies for a domain of size L with N points.

    Parameters
    ----------
    N : int
        Number of points.
    L : float
        Domain size.
    real : bool
        If True, return only the non-negative frequencies kept by ``rfftn``,
        i.e. ``N // 2 + 1`` values.
    """
    freq = jnp.fft.rfftfreq(N, d=L / N) if real else jnp.fft.fftfreq(N, d=L / N)
    return 2.0 * jnp.pi * freq


def spectral_shape(grid):
    """Shape of the half spectrum produced by ``rfftn`` on this grid."""
    return (*grid.shape[:-1], grid.shape[-1] // 2 + 1)


def spacing(grid, pad=1.0):
    """Voxel spacing per direction, padded to three entries.

    On a 2D grid the third entry is a placeholder: the out-of-plane frequency is
    zero, so the corresponding component of ``D`` vanishes whatever the value,
    and it only exists so the schemes can index ``h[2]`` unconditionally.
    """
    h = [L / N for L, N in zip(grid.size, grid.shape, strict=True)]
    return h + [pad] * (3 - len(h))


def xi_array(grid):
    """
    Grid of Fourier angular frequencies on the half spectrum.

    Output shape: ``spectral_shape(grid) + (3,)``. The last spatial axis is
    truncated because ``rfftn`` transforms it as a real axis. On a 2D grid the
    third frequency component is identically zero; see the module docstring.
    """
    ndim = grid.ndim
    axes = [
        xi_array_1D(N, L, real=(i == ndim - 1))
        for i, (N, L) in enumerate(zip(grid.shape, grid.size, strict=True))
    ]
    xi = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
    if ndim < 3:
        pad = jnp.zeros((*xi.shape[:-1], 3 - ndim))
        xi = jnp.concatenate([xi, pad], axis=-1)
    return xi


# Retained under its old name; 2D grids are supported, so the "3D" is a misnomer.
xi_array_3D = xi_array


class DiscretizationScheme(eqx.Module):
    """Base class for discrete differential operators.

    A scheme supplies ``D_plus(xi)`` and ``D_minus(xi)``; the divergence and
    symmetric gradient are derived from them so that one is the negative adjoint
    of the other, which makes the Green operator self-adjoint. All arrays live
    on the half spectrum; see the module docstring.
    """

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

    def acoustic_tensor(self, C0):
        """Discrete acoustic tensor A(xi) = div_h . C0 . sym_grad_h.

        Assembled by applying the scheme's *own* divergence and symmetric
        gradient to the three unit displacement directions, so it is correct for
        any scheme, including those that override those operators with a
        different index convention. The Green kernel is its inverse.

        Returns
        -------
        array of shape ``spectral_shape + (3, 3)``
        """
        C0_full = jnp.asarray(C0.to_symmetric().tensor)
        columns = []
        for j in range(3):
            u_hat = jnp.zeros((*self.D_plus.shape[:-1], 3), dtype=self.D_plus.dtype)
            u_hat = u_hat.at[..., j].set(1.0)
            eps_hat = self.apply_symmetric_gradient(u_hat)
            sig_hat = jnp.einsum("ijkl,...kl->...ij", C0_full, eps_hat)
            columns.append(self.apply_divergence(sig_hat))
        return jnp.stack(columns, axis=-1)

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
    """Rotated finite differences on the voxel corners (Willot).

    .. math::
        D_i = \\frac{2i}{h_i} \\sin(q_i/2) \\prod_{j \\neq i} \\cos(q_j/2),
        \\qquad q_i = \\xi_i h_i

    The usual choice at high contrast. Each component carries its own voxel
    spacing ``h_i``; dividing them all by ``h_x`` instead leaves the operator
    correct on cubic voxels and wrong by ``h_i / h_x`` otherwise.
    """

    tol: float = 1e-12

    def _h(self, grid):
        return spacing(grid)

    def _q(self, xi, grid):
        # q_i = 2 pi n_i / N_i = xi_i * h_i
        h = self._h(grid)
        return jnp.stack([xi[..., i] * h[i] for i in range(3)], axis=-1)

    def _D(self, xi, grid):
        q = self._q(xi, grid) / 2
        h = self._h(grid)
        sin, cos = jnp.sin(q), jnp.cos(q)
        # Component i is scaled by its own spacing h_i, not by h_x. Gamma_0 is
        # homogeneous of degree zero in D, so a uniform mis-scaling of all three
        # components cancels exactly; only the relative weighting matters, which
        # is why this is invisible on cubic voxels and wrong on any other grid.
        return 2j * jnp.stack(
            [
                sin[..., 0] * cos[..., 1] * cos[..., 2] / h[0],
                cos[..., 0] * sin[..., 1] * cos[..., 2] / h[1],
                cos[..., 0] * cos[..., 1] * sin[..., 2] / h[2],
            ],
            axis=-1,
        )

    def _singular_mask(self, xi, grid):
        xi_abs = jnp.abs(xi)
        zero_mode = jnp.all(xi == 0, axis=-1)

        h = spacing(grid)
        flags = []
        for i in range(3):
            # A direction the grid does not have carries xi_i = 0 and so has no
            # Nyquist mode.
            is_even = i < grid.ndim and grid.shape[i] % 2 == 0
            qi = xi_abs[..., i] * h[i]
            flags.append(is_even & (jnp.abs(qi - jnp.pi) < self.tol))

        nyquist_pairs = (flags[0] & flags[1]) | (flags[0] & flags[2]) | (flags[1] & flags[2])

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
        return jnp.array(spacing(grid))

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
