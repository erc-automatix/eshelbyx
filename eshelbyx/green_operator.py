import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
from eshelbyx.tensors import (
    Tensor,
    SymmetricTensor2,
    SymmetricTensor4,
    IsotropicTensor4,
)
from eshelbyx.discretization import DiscretizationScheme
from eshelbyx.spectral_kernels import compute_G0_hat, compute_isotropic_G0_hat


class GreenOperator(lx.AbstractLinearOperator):
    C0: Tensor
    grid: eqx.Module
    discretization: DiscretizationScheme

    xi: jax.Array = eqx.field(init=False)
    D: jax.Array = eqx.field(init=False)
    G0_hat: jax.Array = eqx.field(init=False)

    def __post_init__(self):
        self.xi = xi_array_3D(self.grid)
        self.D = self.discretization.D(self.xi, self.grid)
        D = self.D.reshape(-1, 3)
        singular_mask = self.discretization.singular_mask(self.xi, self.grid)
        if isinstance(self.C0, IsotropicTensor4):
            self.G0_hat = jax.vmap(compute_isotropic_G0_hat, in_axes=(0, 0, None))(
                D, singular_mask.ravel(), self.C0
            )[0].reshape(self.grid.shape + (3, 3))
        else:
            self.G0_hat = jax.vmap(compute_G0_hat, in_axes=(0, None))(
                D, singular_mask, self.C0
            ).reshape(self.grid.shape + (3, 3))

    def Gamma_0_func(self, Y):
        Y_hat = jnp.fft.fftn(Y, axes=(0, 1, 2))

        v_hat = jnp.einsum(
            "...ij,...j->...i",
            Y_hat,
            jnp.conj(self.D),
        )

        u_hat = jnp.einsum(
            "...ij,...j->...i",
            self.G0_hat,
            v_hat,
        )

        eps_hat = 0.5 * (
            jnp.einsum("...i,...j->...ij", u_hat, self.D)
            + jnp.einsum("...j,...i->...ij", u_hat, self.D)
        )

        return jnp.fft.ifftn(eps_hat, axes=(0, 1, 2)).real

    def mv(self, vector, out=None):
        return SymmetricTensor2(tensor=self.Gamma_0_func(vector))

    @eqx.filter_jit
    def apply_C0(self, eps):
        return jnp.einsum("ijkl,...kl->...ij", self.C0, eps)

    def as_matrix(self):
        return lx.materialise(self).as_matrix()

    def transpose(self):
        return self  # FIXME

    def in_structure(self):
        return jax.ShapeDtypeStruct(shape=(self.grid.shape, 3, 3), dtype=jnp.float64)

    def out_structure(self):
        return self.in_structure


@lx.is_symmetric.register(GreenOperator)
def _(operator):
    return False


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


# def compute_Willot_Green_operator(grid, C0):
#     Lx, Ly, Lz = grid.size
#     m, n, p = grid.shape
#     xi_array = xi_array_3D(grid)

#     G0 = C0.mu
#     lambda_0 = C0.kappa - 2 / 3 * G0
#     c1, c2 = 1 / 4 / G0, (lambda_0 + G0) / G0 / (
#         lambda_0 + 2 * G0
#     )  # parameters for Linear combination for full field Green's operator

#     # Discrete frequencies (no FFT shift)
#     em = (
#         2
#         * jnp.pi
#         * jnp.concatenate([jnp.arange(0, m // 2), jnp.arange(-m // 2, 0)])
#         / Lx
#     )
#     en = (
#         2
#         * jnp.pi
#         * jnp.concatenate([jnp.arange(0, n // 2), jnp.arange(-n // 2, 0)])
#         / Ly
#     )
#     ep = (
#         2
#         * jnp.pi
#         * jnp.concatenate([jnp.arange(0, p // 2), jnp.arange(-p // 2, 0)])
#         / Lz
#     )
#     # Vectorized Green's operator computation, here we use the Willot's Green's operator form with finite difference scheme
#     em_grid, en_grid, ep_grid = jnp.meshgrid(em, en, ep, indexing="ij")  # (m, n, p)

#     e1 = 2 * m / Lx * jnp.sin(em_grid / 2) * jnp.cos(en_grid / 2) * jnp.cos(ep_grid / 2)
#     e2 = 2 * n / Ly * jnp.cos(em_grid / 2) * jnp.sin(en_grid / 2) * jnp.cos(ep_grid / 2)
#     e3 = 2 * p / Lz * jnp.cos(em_grid / 2) * jnp.cos(en_grid / 2) * jnp.sin(ep_grid / 2)
#     e1_t, e2_t, e3_t, E_sq = e1, e2, e3, e1**2 + e2**2 + e3**2
#     ## Components of the Green's operator in Voigt notation
#     T11 = c1 / E_sq * (4 * e1**2) - c2 * e1**4 / E_sq**2
#     T12 = -c2 * e1**2 * e2**2 / E_sq**2
#     T13 = -c2 * e1**2 * e3**2 / E_sq**2
#     T14 = c1 / E_sq * (2 * e1 * e2) - c2 * e1**3 * e2 / E_sq**2
#     T15 = c1 / E_sq * (2 * e1 * e3) - c2 * e1**3 * e3 / E_sq**2
#     T16 = -c2 * e1**2 * e2 * e3 / E_sq**2

#     T22 = c1 / E_sq * (4 * e2**2) - c2 * e2**4 / E_sq**2
#     T23 = -c2 * e2**2 * e3**2 / E_sq**2
#     T24 = c1 / E_sq * (2 * e1 * e2) - c2 * e2**3 * e1 / E_sq**2
#     T25 = -c2 * e2**2 * e1 * e3 / E_sq**2
#     T26 = c1 / E_sq * (2 * e2 * e3) - c2 * e2**3 * e3 / E_sq**2

#     T33 = c1 / E_sq * (4 * e3**2) - c2 * e3**4 / E_sq**2
#     T34 = -c2 * e3**2 * e1 * e2 / E_sq**2
#     T35 = c1 / E_sq * (2 * e3 * e1) - c2 * e3**3 * e1 / E_sq**2
#     T36 = c1 / E_sq * (2 * e3 * e2) - c2 * e3**3 * e2 / E_sq**2

#     T44 = c1 / E_sq * (e1**2 + e2**2) - c2 * e1**2 * e2**2 / E_sq**2
#     T45 = c1 / E_sq * (e2 * e3) - c2 * e1**2 * e2 * e3 / E_sq**2
#     T46 = c1 / E_sq * (e1 * e3) - c2 * e2**2 * e1 * e3 / E_sq**2

#     T55 = c1 / E_sq * (e1**2 + e3**2) - c2 * e1**2 * e3**2 / E_sq**2
#     T56 = c1 / E_sq * (e1 * e2) - c2 * e3**2 * e1 * e2 / E_sq**2

#     T66 = c1 / E_sq * (e2**2 + e3**2) - c2 * e2**2 * e3**2 / E_sq**2

#     # modify the Green function tensor according to voigt notation
#     T14 = T14 * 2
#     T15 = T15 * 2
#     T16 = T16 * 2
#     T24 = T24 * 2
#     T25 = T25 * 2
#     T26 = T26 * 2
#     T34 = T34 * 2
#     T35 = T35 * 2
#     T36 = T36 * 2
#     T44 = T44 * 4
#     T45 = T45 * 4
#     T46 = T46 * 4
#     T55 = T55 * 4
#     T56 = T56 * 4
#     T66 = T66 * 4
#     # Complete the symmetric part
#     T21 = T12
#     T31 = T13
#     T41 = T14
#     T51 = T15
#     T61 = T16
#     T32 = T23
#     T42 = T24
#     T52 = T25
#     T62 = T26
#     T43 = T34
#     T53 = T35
#     T63 = T36
#     T54 = T45
#     T64 = T46
#     T65 = T56
#     # Zero frequency modification
#     zero_mask = jnp.zeros((m, n, p), dtype=bool)
#     zero_mask = zero_mask.at[0, 0, 0].set(True)
#     T11 = jnp.where(zero_mask, 0, T11)
#     T12 = jnp.where(zero_mask, 0, T12)
#     T13 = jnp.where(zero_mask, 0, T13)
#     T14 = jnp.where(zero_mask, 0, T14)
#     T15 = jnp.where(zero_mask, 0, T15)
#     T16 = jnp.where(zero_mask, 0, T16)
#     T21 = jnp.where(zero_mask, 0, T21)
#     T22 = jnp.where(zero_mask, 0, T22)
#     T23 = jnp.where(zero_mask, 0, T23)
#     T24 = jnp.where(zero_mask, 0, T24)
#     T25 = jnp.where(zero_mask, 0, T25)
#     T26 = jnp.where(zero_mask, 0, T26)
#     T31 = jnp.where(zero_mask, 0, T31)
#     T32 = jnp.where(zero_mask, 0, T32)
#     T33 = jnp.where(zero_mask, 0, T33)
#     T34 = jnp.where(zero_mask, 0, T34)
#     T35 = jnp.where(zero_mask, 0, T35)
#     T36 = jnp.where(zero_mask, 0, T36)
#     T41 = jnp.where(zero_mask, 0, T41)
#     T42 = jnp.where(zero_mask, 0, T42)
#     T43 = jnp.where(zero_mask, 0, T43)
#     T44 = jnp.where(zero_mask, 0, T44)
#     T45 = jnp.where(zero_mask, 0, T45)
#     T46 = jnp.where(zero_mask, 0, T46)
#     T51 = jnp.where(zero_mask, 0, T51)
#     T52 = jnp.where(zero_mask, 0, T52)
#     T53 = jnp.where(zero_mask, 0, T53)
#     T54 = jnp.where(zero_mask, 0, T54)
#     T55 = jnp.where(zero_mask, 0, T55)
#     T56 = jnp.where(zero_mask, 0, T56)
#     T61 = jnp.where(zero_mask, 0, T61)
#     T62 = jnp.where(zero_mask, 0, T62)
#     T63 = jnp.where(zero_mask, 0, T63)
#     T64 = jnp.where(zero_mask, 0, T64)
#     T65 = jnp.where(zero_mask, 0, T65)
#     T66 = jnp.where(zero_mask, 0, T66)
#     # Stack Green's operator into a 6x6 matrix for each frequency if you would like to use 5D array version
#     T = jnp.stack(
#         [
#             jnp.stack([T11, T12, T13, T14, T15, T16], axis=0),
#             jnp.stack([T21, T22, T23, T24, T25, T26], axis=0),
#             jnp.stack([T31, T32, T33, T34, T35, T36], axis=0),
#             jnp.stack([T41, T42, T43, T44, T45, T46], axis=0),
#             jnp.stack([T51, T52, T53, T54, T55, T56], axis=0),
#             jnp.stack([T61, T62, T63, T64, T65, T66], axis=0),
#         ],
#         axis=0,
#     )  # Shape: (6,6, m, n, p)
#     return (e1_t, e2_t, e3_t), lx.MatrixLinearOperator(T)
