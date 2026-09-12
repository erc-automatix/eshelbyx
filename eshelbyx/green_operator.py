from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxmat.tensors import (
    IsotropicTensor4,
    Tensor,
)

from eshelbyx.discretization import DiscretizationScheme, spectral_shape
from eshelbyx.spectral_kernels import compute_G0_hat, compute_isotropic_G0_hat


def _lame(C0):
    """Lame coefficients of an isotropic reference medium."""
    mu = C0.mu
    lmbda = C0.kappa - 2.0 / 3.0 * mu
    return lmbda, mu


class GreenOperator(lx.AbstractLinearOperator):
    C0: Tensor
    grid: eqx.Module
    discretization: DiscretizationScheme

    def __init__(self, grid, C0, discretization):
        self.grid = grid
        self.C0 = C0
        self.discretization = discretization

    @property
    def G0_hat(self):
        """Fourier kernel on the half spectrum, recomputed on access.

        Exposed for inspection and testing. The hot path does not go through
        this property; see _apply_G0.
        """
        return self._compute_G0_hat()

    def _compute_G0_hat(self):
        if isinstance(self.C0, IsotropicTensor4):
            return jax.vmap(compute_isotropic_G0_hat, in_axes=(0, 0, 0, None))(
                self.discretization.D_plus.reshape(-1, 3),
                self.discretization.D_minus.reshape(-1, 3),
                self.discretization.singular_mask.ravel(),
                self.C0,
            )[0].reshape((*spectral_shape(self.grid), 3, 3))
        return jax.vmap(compute_G0_hat, in_axes=(0, 0, 0, None))(
            self.discretization.D_plus.reshape(-1, 3),
            self.discretization.D_minus.reshape(-1, 3),
            self.discretization.singular_mask.ravel(),
            self.C0,
        ).reshape((*spectral_shape(self.grid), 3, 3))

    def _apply_G0(self, v_hat):
        """Apply the kernel to a spectral vector field without materialising it.

        For an isotropic reference medium the kernel has the closed form

            G0 = Id / (mu D2)  -  c (D_minus (x) D_plus) / D2**2

        so its action is a scaled copy of v_hat plus a rank-one correction along
        D_minus. Evaluating it this way keeps only vector-sized temporaries,
        never the (..., 3, 3) kernel.
        """
        if not isinstance(self.C0, IsotropicTensor4):
            return jnp.einsum("...ij,...j->...i", self._compute_G0_hat(), v_hat)

        D_p = self.discretization.D_plus
        D_m = self.discretization.D_minus
        mask = self.discretization.singular_mask[..., None]

        lmbda, mu = _lame(self.C0)
        D2 = jnp.sum(jnp.real(D_m * D_p), axis=-1)[..., None]
        D2 = jnp.where(mask, 1.0, D2)  # guard the singular modes before dividing

        c = (lmbda + mu) / (mu * (lmbda + 2.0 * mu))
        proj = jnp.sum(D_p * v_hat, axis=-1)[..., None]
        out = v_hat / (mu * D2) - c * D_m * proj / D2**2
        return jnp.where(mask, 0.0, out)

    @partial(jax.remat, prevent_cse=False)
    def _Gamma_0_func_impl(self, Y):
        # Y is real, so its spectrum is Hermitian and only half of it is
        # independent. rfftn/irfftn work on that half throughout, which is exact
        # (see the discretization module docstring) and roughly halves both the
        # transform cost and the size of every intermediate spectral array.
        with jax.named_scope("Gamma0/fft"):
            Y_hat = jnp.fft.rfftn(Y, axes=(0, 1, 2))

        with jax.named_scope("Gamma0/div"):
            v_hat = self.discretization.apply_divergence(Y_hat)

        with jax.named_scope("Gamma0/G0_hat"):
            u_hat = jnp.einsum(
                "...ij,...j->...i",
                self.G0_hat,
                v_hat,
            )
        with jax.named_scope("Gamma0/sym_grad"):
            eps_hat = self.discretization.apply_symmetric_gradient(u_hat)

        with jax.named_scope("Gamma0/ifft"):
            return jnp.fft.irfftn(eps_hat, axes=(0, 1, 2), s=self.grid.shape)

    def Gamma_0_func(self, Y):
        return self._Gamma_0_func_impl(Y)

    @eqx.filter_jit
    def mv(self, vector, out=None):
        return self.Gamma_0_func(vector)
        # return SymmetricTensor2(tensor=self.Gamma_0_func(vector))

    @eqx.filter_jit
    def apply_C0(self, eps):
        return jnp.einsum("ijkl,...kl->...ij", self.C0.to_symmetric(), eps)

    @eqx.filter_jit
    def apply_C0_inv(self, eps):
        return jnp.einsum("ijkl,...kl->...ij", self.C0.inv.to_symmetric(), eps)

    def as_matrix(self):
        raise NotImplementedError("Matrix-free operator")

    def transpose(self):
        return self  # FIXME

    def in_structure(self):
        return jax.ShapeDtypeStruct(shape=(*self.grid.shape, 3, 3), dtype=jnp.float64)

    def out_structure(self):
        return self.in_structure()


@lx.is_symmetric.register(GreenOperator)
def _(operator):
    return False


# lineax queries every operator tag during __check_init__ and calls linearise
# from linear_solve, so all four must be registered or the operator cannot be
# constructed at all. Definiteness is declared conservatively here; Gamma_0 is
# in fact self-adjoint for a consistent discretization, but asserting that
# changes which solvers lineax selects and is handled separately.
@lx.is_positive_semidefinite.register(GreenOperator)
def _(operator):
    return False


@lx.is_negative_semidefinite.register(GreenOperator)
def _(operator):
    return False


@lx.linearise.register(GreenOperator)
def _(operator):
    return operator
