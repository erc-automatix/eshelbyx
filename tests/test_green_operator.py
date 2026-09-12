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
    pytest tests/test_green_operator.py -v
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxmat.tensors import IsotropicTensor4

from eshelbyx.discretization import (
    MoulinecSuquetDiscretization,
    SchneiderDiscretization,
    WillotDiscretization,
)
from eshelbyx.green_operator import GreenOperator
from eshelbyx.grid import Grid

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
    return Grid(shape=(N, N, N), size=(1.0, 1.0, 1.0), fields={"markers": jnp.zeros((N, N, N))})


def build(scheme_name, N):
    grid = make_grid(N)
    disc = SCHEMES[scheme_name](grid)
    Gamma_0 = GreenOperator(grid=grid, C0=make_C0(), discretization=disc)
    return grid, disc, Gamma_0


def single_mode_displacement(shape, mode, amp):
    """Real periodic displacement u(x) built from a single Fourier mode
    (mode and its Hermitian conjugate), so eps* excites exactly that mode."""
    u_hat = jnp.zeros((*shape, 3), dtype=jnp.complex128)
    m = tuple(int(mi) % shape[i] for i, mi in enumerate(mode))
    negm = tuple((-int(mi)) % shape[i] for i, mi in enumerate(mode))
    u_hat = u_hat.at[m].set(amp)
    u_hat = u_hat.at[negm].set(jnp.conj(amp))
    return jnp.fft.ifftn(u_hat, axes=(0, 1, 2)).real


def compatible_strain(disc, u):
    """Discrete compatible strain eps* = sym_grad_h(u), using the scheme's own
    symmetric-gradient operator (the same one Gamma_0 uses internally).

    The scheme's operators live on the half spectrum, so u must go through
    rfftn rather than fftn."""
    u_hat = jnp.fft.rfftn(u, axes=(0, 1, 2))
    eps_hat = disc.apply_symmetric_gradient(u_hat)
    return jnp.fft.irfftn(eps_hat, axes=(0, 1, 2), s=u.shape[:3])


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
    grid, _disc, Gamma_0 = build(scheme_name, N=16)
    raw = jr.normal(jr.PRNGKey(0), (*grid.shape, 3, 3))
    eps = 0.5 * (raw + jnp.swapaxes(raw, -1, -2))
    eps = eps - jnp.mean(eps, axis=(0, 1, 2))
    lhs = Gamma_0.mv(Gamma_0.apply_C0(eps))
    assert rel(lhs, eps) > 1e-2, "Gamma_0.C0 behaves like identity -> test has no teeth"


@pytest.mark.parametrize("scheme_name", list(SCHEMES))
def test_singular_modes_are_annihilated(scheme_name):
    """Gamma_0 must return zero content on the masked (k=0 / Nyquist) modes."""
    grid, _disc, Gamma_0 = build(scheme_name, N=16)
    # a uniform strain is pure k=0 -> Gamma_0 C0 must map it to ~0
    uniform = jnp.broadcast_to(jnp.eye(3), (*grid.shape, 3, 3))
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
    grid, _disc, Gamma_0 = build(scheme_name, N)

    def P(x):
        return Gamma_0.mv(Gamma_0.apply_C0(x))

    raw = jr.normal(jr.PRNGKey(1), (*grid.shape, 3, 3))
    x = 0.5 * (raw + jnp.swapaxes(raw, -1, -2))
    Px = P(x)
    PPx = P(Px)
    assert rel(PPx, Px) < ATOL, (
        f"{scheme_name} N={N}: Gamma_0.C0 is not idempotent (residual {rel(PPx, Px):.2e})"
    )


# --------------------------------------------------------------------------- #
# Standalone summary table (no pytest needed):  python test_..._planewave.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Patch test  ||Gamma0 C0 eps* - eps*|| / ||eps*||\n")
    hdr = "mode        " + "".join(f"{k:>13}" for k in SCHEMES)
    print(hdr + "\n" + "-" * len(hdr))
    for mode in MODES:
        row = f"{mode!s:<12}"
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

            def P(x):
                return Gamma_0.mv(Gamma_0.apply_C0(x))

            raw = jr.normal(jr.PRNGKey(1), (*grid.shape, 3, 3))
            x = 0.5 * (raw + jnp.swapaxes(raw, -1, -2))
            Px = P(x)
            row += f"{rel(P(Px), Px):>13.2e}"
        print(row)


# --------------------------------------------------------------------------- #
# 5. Half-spectrum equivalence
# --------------------------------------------------------------------------- #
def _gamma_0_full_spectrum(grid, disc, C0, Y):
    """Reference implementation of Gamma_0 on the FULL fftn spectrum.

    Kept as a test-only oracle: the library uses rfftn throughout, and this
    pins that optimisation to the mathematically transparent version.
    """
    from eshelbyx.discretization import xi_array_1D
    from eshelbyx.spectral_kernels import compute_isotropic_G0_hat

    N1, N2, N3 = grid.shape
    L1, L2, L3 = grid.size
    xi = jnp.stack(
        jnp.meshgrid(xi_array_1D(N1, L1), xi_array_1D(N2, L2), xi_array_1D(N3, L3), indexing="ij"),
        axis=-1,
    )
    # Staggered schemes derive their singular mask from D_plus/D_minus rather
    # than from xi, so the differential vectors have to be swapped in before the
    # mask is recomputed.
    full = eqx.tree_at(
        lambda d: (d.xi, d.D_plus, d.D_minus),
        disc,
        (xi, disc._D_plus(xi, grid), disc._D_minus(xi, grid)),
    )
    full = eqx.tree_at(lambda d: d.singular_mask, full, full._singular_mask(xi, grid))
    G0_hat = jax.vmap(compute_isotropic_G0_hat, in_axes=(0, 0, 0, None))(
        full.D_plus.reshape(-1, 3),
        full.D_minus.reshape(-1, 3),
        full.singular_mask.ravel(),
        C0,
    )[0].reshape((*grid.shape, 3, 3))

    Y_hat = jnp.fft.fftn(Y, axes=(0, 1, 2))
    v_hat = full.apply_divergence(Y_hat)
    u_hat = jnp.einsum("...ij,...j->...i", G0_hat, v_hat)
    eps_hat = full.apply_symmetric_gradient(u_hat)
    return jnp.fft.ifftn(eps_hat, axes=(0, 1, 2)).real


@pytest.mark.parametrize("scheme_name", list(SCHEMES))
@pytest.mark.parametrize("N", [8, 9])
def test_half_spectrum_matches_full_spectrum(scheme_name, N, request):
    if scheme_name == "MS" and N % 2 == 0:
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="MS is ambiguous at the Nyquist plane on even grids. "
                "rfftfreq places it at xi3=+pi/h and fftfreq at -pi/h, and since "
                "D=i*xi is odd the two conventions disagree there. Willot and "
                "Schneider are insensitive (their D is even at Nyquist, or the "
                "sign cancels between divergence and symmetric gradient). Same "
                "root cause as the MS/even idempotency xfail: MS masks only k=0 "
                "and never Nyquist. Retire this when MS gains a Nyquist mask.",
            )
        )
    """rfftn is exact, not an approximation: every scheme satisfies
    D(-xi) = conj(D(xi)), so the discarded half of the spectrum carries no
    independent information. Checked on both an even and an odd grid, since the
    Nyquist plane only exists for even N."""
    grid, disc, Gamma_0 = build(scheme_name, N=N)

    key = jr.PRNGKey(0)
    Y = jr.normal(key, (*grid.shape, 3, 3))
    Y = 0.5 * (Y + jnp.swapaxes(Y, -1, -2))  # polarization fields are symmetric

    got = Gamma_0.mv(Y)
    expected = _gamma_0_full_spectrum(grid, disc, make_C0(), Y)

    assert rel(got, expected) < 1e-12, (
        f"{scheme_name} N={N}: half-spectrum Gamma_0 differs from the full-spectrum "
        f"reference (relative difference {rel(got, expected):.2e})"
    )


def test_output_is_real_and_symmetric():
    """irfftn would silently discard an imaginary part if the operator did not
    preserve Hermitian symmetry, so check the symmetry of the result instead."""
    grid, _, Gamma_0 = build("Willot", N=8)
    Y = jr.normal(jr.PRNGKey(1), (*grid.shape, 3, 3))
    Y = 0.5 * (Y + jnp.swapaxes(Y, -1, -2))
    out = Gamma_0.mv(Y)
    assert out.shape == Y.shape
    assert jnp.isrealobj(out)
    assert float(jnp.abs(out - jnp.swapaxes(out, -1, -2)).max()) < 1e-14
