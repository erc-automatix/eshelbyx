"""
Plane-wave / patch test for the discrete Green operator Gamma_0, for every
DiscretizationScheme (Moulinec-Suquet, Willot, Schneider).

Theory
------
For any strain field  eps*  that is *discretely compatible* -- i.e.
eps* = sym_grad_h(u) for some periodic displacement u, built with the SAME
discrete symmetric gradient the scheme uses internally -- the Green operator
must reproduce it exactly on the non-singular Fourier modes:

        Gamma_0 ( C0 : eps* )  =  eps*                         (patch test)

This holds to machine precision iff the discrete kernel  G0_hat(k)  is the true
inverse of the discrete acoustic operator  div_h . C0 . sym_grad_h  of that
scheme.  It is therefore a sharp consistency check between the kernel builder
(spectral_kernels) and the div / sym_grad operators (discretization) for each
scheme.

A strictly stronger property is that  P := Gamma_0 . C0  is a *projector* onto
discretely-compatible strains, i.e.  P^2 = P  on arbitrary fields.

Empirically observed with this code base (double precision):
  * patch test: MS, Willot, Schneider all pass to ~1e-15 (incl. shear modes).
  * idempotency: Willot & Schneider pass to ~1e-15 on any grid; MS passes on
    ODD grids but FAILS (~1e-1) on EVEN grids, because MS masks only the k=0
    mode and never the Nyquist mode -> Gamma_0.C0 is not an exact discrete
    projector when a Nyquist mode is present.  That MS/even case is marked xfail.

Run:
    pytest test_green_operator_planewave.py -v
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import pytest

from eshelbyx.grid import Grid
from eshelbyx.green_operator import GreenOperator
from eshelbyx.discretization import (
    MoulinecSuquetDiscretization,
    WillotDiscretization,
    SchneiderDiscretization,
)
from jaxmat.tensors import IsotropicTensor4


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
SCHEMES = {
    "MS": MoulinecSuquetDiscretization,
    "Willot": WillotDiscretization,
    "Schneider": SchneiderDiscretization,
}

# Low-frequency modes; the shear-coupled ones (multiple non-zero components)
# are the discriminating cases for staggered schemes.
MODES = [(1, 0, 0), (2, 0, 0), (1, 1, 0), (1, 2, 3), (3, 1, 2)]

ATOL = 1e-10  # machine-precision-ish threshold for the reproduced field


def make_C0(E0=1.0, nu0=0.3):
    lam = E0 * nu0 / (1 + nu0) / (1 - 2 * nu0)
    mu = E0 / 2 / (1 + nu0)
    return IsotropicTensor4(kappa=lam + 2.0 / 3.0 * mu, mu=mu)


def make_grid(N):
    return Grid(
        shape=(N, N, N), size=(1.0, 1.0, 1.0), fields={"markers": jnp.zeros((N, N, N))}
    )


def build(scheme_name, N):
    grid = make_grid(N)
    disc = SCHEMES[scheme_name](grid)
    Gamma_0 = GreenOperator(grid=grid, C0=make_C0(), discretization=disc)
    return grid, disc, Gamma_0


def single_mode_displacement(shape, mode, amp):
    """Real periodic displacement u(x) built from a single Fourier mode
    (mode and its Hermitian conjugate), so eps* excites exactly that mode."""
    u_hat = jnp.zeros(shape + (3,), dtype=jnp.complex128)
    m = tuple(int(mi) % shape[i] for i, mi in enumerate(mode))
    negm = tuple((-int(mi)) % shape[i] for i, mi in enumerate(mode))
    u_hat = u_hat.at[m].set(amp)
    u_hat = u_hat.at[negm].set(jnp.conj(amp))
    return jnp.fft.ifftn(u_hat, axes=(0, 1, 2)).real


def compatible_strain(disc, u):
    """Discrete compatible strain eps* = sym_grad_h(u), using the scheme's own
    symmetric-gradient operator (the same one Gamma_0 uses internally)."""
    u_hat = jnp.fft.fftn(u, axes=(0, 1, 2))
    eps_hat = disc.apply_symmetric_gradient(u_hat)
    return jnp.fft.ifftn(eps_hat, axes=(0, 1, 2)).real


def mode_is_singular(disc, shape, mode):
    idx = tuple(int(mi) % shape[i] for i, mi in enumerate(mode))
    return bool(disc.singular_mask[idx])


def rel(a, b):
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-300))


# fixed complex amplitude that breaks directional symmetry
AMP = jnp.array([1.0 + 0.0j, 0.0 + 0.5j, -0.4 + 0.0j])


# --------------------------------------------------------------------------- #
# 1. Patch test:  Gamma_0 (C0 eps*) == eps*   for compatible eps*
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scheme_name", list(SCHEMES))
@pytest.mark.parametrize("mode", MODES)
def test_compatible_strain_reproduced(scheme_name, mode):
    grid, disc, Gamma_0 = build(scheme_name, N=16)

    if mode_is_singular(disc, grid.shape, mode):
        pytest.skip(f"mode {mode} is in the singular mask for {scheme_name}")

    u = single_mode_displacement(grid.shape, mode, AMP)
    eps_star = compatible_strain(disc, u)

    # sanity: the constructed compatible strain is non-trivial and zero-mean
    assert jnp.linalg.norm(eps_star) > 1e-8
    assert float(jnp.abs(jnp.mean(eps_star, axis=(0, 1, 2))).max()) < 1e-10

    lhs = Gamma_0.mv(Gamma_0.apply_C0(eps_star))
    assert rel(lhs, eps_star) < ATOL, (
        f"{scheme_name} mode {mode}: G0_hat is not the inverse of "
        f"div_h.C0.sym_grad_h (residual {rel(lhs, eps_star):.2e})"
    )


# --------------------------------------------------------------------------- #
# 2. Negative controls: the test must NOT pass vacuously
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scheme_name", list(SCHEMES))
def test_incompatible_strain_is_not_reproduced(scheme_name):
    """A generic (incompatible) symmetric, zero-mean field must be altered by
    Gamma_0.C0 -- otherwise the operator would be the identity and the patch
    test would be meaningless."""
    grid, disc, Gamma_0 = build(scheme_name, N=16)
    raw = jr.normal(jr.PRNGKey(0), grid.shape + (3, 3))
    eps = 0.5 * (raw + jnp.swapaxes(raw, -1, -2))
    eps = eps - jnp.mean(eps, axis=(0, 1, 2))
    lhs = Gamma_0.mv(Gamma_0.apply_C0(eps))
    assert rel(lhs, eps) > 1e-2, "Gamma_0.C0 behaves like identity -> test has no teeth"


@pytest.mark.parametrize("scheme_name", list(SCHEMES))
def test_singular_modes_are_annihilated(scheme_name):
    """Gamma_0 must return zero content on the masked (k=0 / Nyquist) modes."""
    grid, disc, Gamma_0 = build(scheme_name, N=16)
    # a uniform strain is pure k=0 -> Gamma_0 C0 must map it to ~0
    uniform = jnp.broadcast_to(jnp.eye(3), grid.shape + (3, 3))
    out = Gamma_0.mv(Gamma_0.apply_C0(uniform))
    assert float(jnp.linalg.norm(out)) < 1e-8


# --------------------------------------------------------------------------- #
# 3. Projector idempotency:  (Gamma_0 . C0)^2 == (Gamma_0 . C0)
#    The stronger property; discriminates MS/even (Nyquist) from the rest.
# --------------------------------------------------------------------------- #
def _idempotency_cases():
    cases = []
    for scheme_name in SCHEMES:
        for N in (15, 16):  # odd and even
            marks = ()
            if scheme_name == "MS" and N % 2 == 0:
                marks = pytest.mark.xfail(
                    strict=True,
                    reason="MS masks only k=0, not the Nyquist mode; "
                    "Gamma_0.C0 is not an exact discrete projector on even grids.",
                )
            cases.append(pytest.param(scheme_name, N, marks=marks))
    return cases


@pytest.mark.parametrize("scheme_name, N", _idempotency_cases())
def test_projector_idempotent(scheme_name, N):
    grid, disc, Gamma_0 = build(scheme_name, N)

    def P(x):
        return Gamma_0.mv(Gamma_0.apply_C0(x))

    raw = jr.normal(jr.PRNGKey(1), grid.shape + (3, 3))
    x = 0.5 * (raw + jnp.swapaxes(raw, -1, -2))
    Px = P(x)
    PPx = P(Px)
    assert rel(PPx, Px) < ATOL, (
        f"{scheme_name} N={N}: Gamma_0.C0 is not idempotent "
        f"(residual {rel(PPx, Px):.2e})"
    )


# --------------------------------------------------------------------------- #
# Standalone summary table (no pytest needed):  python test_..._planewave.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Patch test  ||Gamma0 C0 eps* - eps*|| / ||eps*||\n")
    hdr = "mode        " + "".join(f"{k:>13}" for k in SCHEMES)
    print(hdr + "\n" + "-" * len(hdr))
    for mode in MODES:
        row = f"{str(mode):<12}"
        for name in SCHEMES:
            grid, disc, Gamma_0 = build(name, 16)
            if mode_is_singular(disc, grid.shape, mode):
                row += f"{'(masked)':>13}"
                continue
            u = single_mode_displacement(grid.shape, mode, AMP)
            eps_star = compatible_strain(disc, u)
            lhs = Gamma_0.mv(Gamma_0.apply_C0(eps_star))
            row += f"{rel(lhs, eps_star):>13.2e}"
        print(row)

    print("\nProjector idempotency ||P^2 x - P x|| / ||P x||\n")
    hdr = "grid        " + "".join(f"{k:>13}" for k in SCHEMES)
    print(hdr + "\n" + "-" * len(hdr))
    for N in (15, 16):
        row = f"N={N:<10}"
        for name in SCHEMES:
            grid, disc, Gamma_0 = build(name, N)
            P = lambda x: Gamma_0.mv(Gamma_0.apply_C0(x))
            raw = jr.normal(jr.PRNGKey(1), grid.shape + (3, 3))
            x = 0.5 * (raw + jnp.swapaxes(raw, -1, -2))
            Px = P(x)
            row += f"{rel(P(Px), Px):>13.2e}"
        print(row)
