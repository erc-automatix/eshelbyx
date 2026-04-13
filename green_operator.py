from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
from eshelbyx.tensors import (
    Tensor,
    SymmetricTensor2,
    IsotropicTensor4,
)
from eshelbyx.discretization import DiscretizationScheme
from eshelbyx.spectral_kernels import compute_G0_hat, compute_isotropic_G0_hat


class GreenOperator(lx.AbstractLinearOperator):
    C0: Tensor
    grid: eqx.Module
    discretization: DiscretizationScheme
    G0_hat: jax.Array

    def __init__(self, grid, C0, discretization):
        self.grid = grid
        self.C0 = C0
        self.discretization = discretization
        self._compute_G0_hat()

    def _compute_G0_hat(self):
        if isinstance(self.C0, IsotropicTensor4):
            self.G0_hat = jax.vmap(compute_isotropic_G0_hat, in_axes=(0, 0, 0, None))(
                self.discretization.D_plus.reshape(-1, 3),
                self.discretization.D_minus.reshape(-1, 3),
                self.discretization.singular_mask.ravel(),
                self.C0,
            )[0].reshape(self.grid.shape + (3, 3))
        else:
            self.G0_hat = jax.vmap(compute_G0_hat, in_axes=(0, 0, 0, None))(
                self.discretization.D_plus.reshape(-1, 3),
                self.discretization.D_minus.reshape(-1, 3),
                self.discretization.singular_mask,
                self.C0,
            ).reshape(self.grid.shape + (3, 3))

    @partial(jax.remat, prevent_cse=False)
    def _Gamma_0_func_impl(self, Y):
        with jax.named_scope("Gamma0/fft"):
            Y_hat = jnp.fft.fftn(Y, axes=(0, 1, 2))

        with jax.named_scope("Gamma0/div"):
            v_hat = self.discretization.apply_divergence(Y_hat)

        with jax.named_scope("Gamma0/G0_hat"):
            u_hat = jnp.einsum(
                "...ij,...i->...j",
                self.G0_hat,
                v_hat,
            )
        with jax.named_scope("Gamma0/sym_grad"):
            eps_hat = self.discretization.apply_symmetric_gradient(u_hat)

        with jax.named_scope("Gamma0/ifft"):
            out = jnp.fft.ifftn(eps_hat, axes=(0, 1, 2)).real
            return out

    def Gamma_0_func(self, Y):
        return self._Gamma_0_func_impl(Y)

    @eqx.filter_jit
    def mv(self, vector, out=None):
        return self.Gamma_0_func(vector)
        #return SymmetricTensor2(tensor=self.Gamma_0_func(vector))

    @eqx.filter_jit
    def apply_C0(self, eps):
        return jnp.einsum("ijkl,...kl->...ij", self.C0, eps)

    @eqx.filter_jit
    def apply_C0_inv(self, eps):
        return jnp.einsum("ijkl,...kl->...ij", self.C0.inv, eps)

    def as_matrix(self):
        raise NotImplementedError("Matrix-free operator")

    def transpose(self):
        return self  # FIXME

    def in_structure(self):
        return jax.ShapeDtypeStruct(shape=self.grid.shape + (3, 3), dtype=jnp.float64)

    def out_structure(self):
        return self.in_structure()


@lx.is_symmetric.register(GreenOperator)
def _(operator):
    return False