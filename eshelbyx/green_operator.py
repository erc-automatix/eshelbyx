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
from eshelbyx.spectral_kernels import compute_isotropic_G0_hat


def _lame(C0):
    """Lame coefficients of an isotropic reference medium."""
    mu = C0.mu
    lmbda = C0.kappa - 2.0 / 3.0 * mu
    return lmbda, mu


# Kelvin (Mandel) packing of a symmetric second-order tensor, in jaxmat's
# convention: (xx, yy, zz, sqrt2*xy, sqrt2*xz, sqrt2*yz).
#
# These duplicate SymmetricTensor2(tensor=...).array and
# SymmetricTensor2(array=...).tensor, and tests/test_green_operator.py asserts
# they agree with them to the last bit. They exist only because jaxmat's
# versions are built from a gather and a scatter-add plus a symmetrisation
# pass, which do not fuse inside the jitted transform pair. Measured at 96^3,
# Gamma_0.mv costs 262 ms with these, 327 ms with jaxmat's packing and 362 ms
# with jaxmat's packing and unpacking -- the last being slower than doing no
# six-component packing at all (326 ms), i.e. jaxmat's conversions cost more
# than the FFT work they save.
#
# Delete these in favour of the properties once jaxmat has a fast path; the
# equivalence test makes that a one-line change.
_KELVIN_OFF = jnp.sqrt(2.0)


def _to_kelvin(T):
    """(..., 3, 3) tensor -> (..., 6) Kelvin array, symmetrising as it goes."""
    T = 0.5 * (T + jnp.swapaxes(T, -1, -2))
    return jnp.stack(
        [
            T[..., 0, 0],
            T[..., 1, 1],
            T[..., 2, 2],
            _KELVIN_OFF * T[..., 0, 1],
            _KELVIN_OFF * T[..., 0, 2],
            _KELVIN_OFF * T[..., 1, 2],
        ],
        axis=-1,
    )


def _from_kelvin(array):
    """(..., 6) Kelvin array -> (..., 3, 3) tensor."""
    xx, yy, zz, xy, xz, yz = (array[..., k] for k in range(6))
    xy, xz, yz = xy / _KELVIN_OFF, xz / _KELVIN_OFF, yz / _KELVIN_OFF
    return jnp.stack(
        [
            jnp.stack([xx, xy, xz], axis=-1),
            jnp.stack([xy, yy, yz], axis=-1),
            jnp.stack([xz, yz, zz], axis=-1),
        ],
        axis=-2,
    )


class GreenOperator(lx.AbstractLinearOperator):
    """Periodic Green operator of the reference medium, applied matrix-free.

    The Fourier kernel is *not* cached. It is a pure function of D_plus,
    D_minus and the singular mask, costing a handful of flops per mode, whereas
    storing it needs a full (N1, N2, N3//2+1, 3, 3) complex array: 119 MB at
    96^3 and several GB at 256^3, which made the kernel rather than the FFT the
    binding constraint on resolution. Recomputing it inside the transform pair
    is cheap next to the FFTs and frees that memory entirely.
    """

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
        # General reference medium: invert the discrete acoustic tensor mode by
        # mode. It is assembled from the scheme's own operators rather than from
        # D_minus . C0 . D_plus, because a scheme may override the divergence
        # and symmetric gradient with a different index convention (Schneider
        # does), in which case the latter is not its acoustic tensor.
        A = self.discretization.acoustic_tensor(self.C0)
        mask = self.discretization.singular_mask[..., None, None]
        safe = jnp.where(mask, jnp.eye(3, dtype=A.dtype), A)
        return jnp.where(mask, 0.0, jnp.linalg.inv(safe))

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
        # Y is symmetric, so only six of its nine components are independent,
        # and it is real, so its spectrum is Hermitian. Transforming the six
        # Kelvin components on the half spectrum rather than all nine on the
        # full one is the cheapest form of the same operator: measured at 96^3,
        # the transform pair costs 129 ms on nine components and 51 ms on six.
        # The Kelvin sqrt(2) weights pass through the transform unchanged, since
        # the FFT is linear and the packing is undone on the other side.
        #
        # _to_kelvin symmetrises as it packs, unlike jaxmat's constructor, which
        # reads the upper triangle; a non-symmetric input is therefore averaged
        # rather than silently truncated.
        with jax.named_scope("Gamma0/fft"):
            axes = tuple(range(self.grid.ndim))
            Y_hat = _from_kelvin(jnp.fft.rfftn(_to_kelvin(Y), axes=axes))

        with jax.named_scope("Gamma0/div"):
            v_hat = self.discretization.apply_divergence(Y_hat)

        with jax.named_scope("Gamma0/G0_hat"):
            u_hat = self._apply_G0(v_hat)
        with jax.named_scope("Gamma0/sym_grad"):
            eps_hat = self.discretization.apply_symmetric_gradient(u_hat)

        # eps_hat is a symmetric gradient, hence symmetric by construction, so
        # the same six-component packing applies on the way back with no
        # symmetrisation needed.
        with jax.named_scope("Gamma0/ifft"):
            return _from_kelvin(
                jnp.fft.irfftn(
                    _to_kelvin(eps_hat),
                    axes=tuple(range(self.grid.ndim)),
                    s=self.grid.shape,
                )
            )

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
