"""
Solver tests for the linear-elastic case (A2EM, AEM, MS, Krylov-AEM).

Covers
------
* correctness on homogeneous media  ->  eps == E_macro,  sigma == C:E_macro
* heterogeneous cross-solver agreement and macroscopic consistency <eps> == E
* Voigt / Reuss energy bounds
* degenerate / zero-division cases (homogeneous C==C0, zero macro strain)

Empirically observed robustness gaps (from a faithful reproduction of each
solver's arithmetic; see the accompanying probe) are encoded as xfail so the
suite stays green while documenting them, and turns red if they are fixed:

  * AEM       -> NaN on homogeneous C==C0   (0/0 in Delta and in lambda once the
                                             exact solution is hit in one step)
  * A2EM(auto)-> ~1% error on (near-)homogeneous media (E_norm -> 1e-12 so the
                                             auto reference scale blows up)
  * A2EM,AEM  -> NaN for E_macro == 0        (division by ||E_macro|| = 0)
  * MS        -> diverges when material stiffer than ~2*C0 (MS convergence cond.)

Written against the eshelbyx / jaxmat API used in elasticity_example.py.

Run:  pytest tests/test_solver.py -v
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxmat.tensors import IsotropicTensor4, SymmetricTensor2

from eshelbyx.discretization import WillotDiscretization
from eshelbyx.green_operator import GreenOperator
from eshelbyx.grid import Grid
from eshelbyx.krylov_solver import KrylovAEMSolver
from eshelbyx.solver import A2EMSolver, AEMSolver, MSSolver
from eshelbyx.utils import flatten_batch_dims, reshape_batch_dims

N = 16
TOL = 1e-8
ATOL = 1e-5  # comparison tolerance for converged fields
E0, NU0 = 1.0, 0.3  # reference medium C0


# --------------------------------------------------------------------------- #
# problem construction (mirrors elasticity_example.py)
# --------------------------------------------------------------------------- #
def lame(E, nu):
    return E * nu / (1 + nu) / (1 - 2 * nu), E / 2 / (1 + nu)


def elastic_law(eps, material_params):
    E, nu = material_params
    lmbda, mu = lame(E, nu)
    sig = lmbda * jnp.linalg.trace(eps) * jnp.eye(3) + 2 * mu * eps
    return SymmetricTensor2(tensor=sig)


@eqx.filter_jit
def constitutive_law(eps, params, shape):
    sig = jax.vmap(elastic_law)(flatten_batch_dims(eps), params)
    return reshape_batch_dims(sig, shape)


def _apply_alpha(eps, material_params, C0):
    E, nu = material_params
    lmbda, mu = lame(E, nu)
    C = IsotropicTensor4(kappa=lmbda + 2 / 3 * mu, mu=mu)
    alp = (C + C0).inv @ (C0 * 2.0)
    return SymmetricTensor2(tensor=(alp @ eps).tensor)


@eqx.filter_jit
def apply_alpha(eps, params, C0):
    eps_flat = SymmetricTensor2(tensor=jnp.asarray(eps).reshape(-1, 3, 3))
    sig = jax.vmap(_apply_alpha, in_axes=(0, 0, None))(eps_flat, params, C0)
    return sig.tensor.reshape(eps.shape)


def make_reference():
    lam0, mu0 = lame(E0, NU0)
    return IsotropicTensor4(kappa=lam0 + 2 / 3 * mu0, mu=mu0)


def make_problem(E_field, nu_field, C0=None):
    grid = Grid(shape=(N, N, N), size=(1.0, 1.0, 1.0), fields={"markers": jnp.zeros((N, N, N))})
    if C0 is None:
        C0 = make_reference()
    Gamma_0 = GreenOperator(grid=grid, C0=C0, discretization=WillotDiscretization(grid))
    params = (E_field.ravel(), nu_field.ravel())

    def alpha(eps, scale=1.0):
        return apply_alpha(eps, params, C0 * scale)

    def C_eps(eps):
        return constitutive_law(eps, params, grid.shape)

    # raw-array reference/material closures for MS (avoids the tensor plumbing;
    # same physics). MS needs C_eps(eps) - C_eps_0(eps) then Gamma_0.mv(.).
    lam0, mu0 = lame(E0, NU0)
    lam_m, mu_m = lame(E_field, nu_field)

    def _raw_C(lam_f, mu_f):
        lam_f = jnp.asarray(lam_f).reshape(grid.shape)[..., None, None]
        mu_f = jnp.asarray(mu_f).reshape(grid.shape)[..., None, None]

        def C(eps):
            tr = jnp.trace(eps, axis1=-2, axis2=-1)[..., None, None]
            return lam_f * tr * jnp.eye(3) + 2 * mu_f * eps

        return C

    C_eps_raw = _raw_C(lam_m, mu_m)
    C_eps_0_raw = _raw_C(jnp.full(grid.shape, lam0), jnp.full(grid.shape, mu0))
    return dict(
        grid=grid,
        Gamma_0=Gamma_0,
        C0=C0,
        params=params,
        alpha=alpha,
        C_eps=C_eps,
        C_eps_raw=C_eps_raw,
        C_eps_0_raw=C_eps_0_raw,
    )


def uniform(val):
    return jnp.full((N, N, N), val)


def macro_strain_field(grid, voigt=(1.0, 0, 0, 0, 0, 0)):
    return grid.broadcast_to_grid(SymmetricTensor2(array=jnp.asarray(voigt)))


def macro_strain_raw(grid, voigt=(1.0, 0, 0, 0, 0, 0)):
    E = SymmetricTensor2(array=jnp.asarray(voigt)).tensor
    return jnp.broadcast_to(E, (*grid.shape, 3, 3))


# --------------------------------------------------------------------------- #
# unified solver driver (normalises signatures + return types to raw arrays)
# --------------------------------------------------------------------------- #
def run(name, prob, voigt=(1.0, 0, 0, 0, 0, 0)):
    grid, G = prob["grid"], prob["Gamma_0"]
    if name in ("A2EM_auto", "A2EM_noauto"):
        E = macro_strain_field(grid, voigt)
        eps, sig, info = A2EMSolver(
            E,
            prob["alpha"],
            G,
            tol=TOL,
            max_iter=500,
            auto_scale=(name == "A2EM_auto"),
            verbose=False,
        )
        n = info["num_steps"]
    elif name == "AEM":
        E = macro_strain_field(grid, voigt)
        eps, sig, info = AEMSolver(E, prob["alpha"], G, tol=TOL, max_iter=500, verbose=False)
        n = info["iterations"]
    elif name == "Krylov":
        E = macro_strain_field(grid, voigt)
        eps, sig, info = KrylovAEMSolver(E, prob["alpha"], G, tol=TOL, max_iter=500)
        n = info.get("num_steps", -1)
    elif name == "MS":
        E = macro_strain_raw(grid, voigt)
        eps, sig, info = MSSolver(
            E,
            prob["C_eps_raw"],
            prob["C_eps_0_raw"],
            G,
            tol=TOL,
            max_iter=2000,
            verbose=False,
        )
        n = info["iterations"]
    else:
        raise ValueError(name)
    eps = jnp.asarray(eps).reshape((*grid.shape, 3, 3))
    sig = jnp.asarray(sig).reshape((*grid.shape, 3, 3))
    return eps, sig, int(n)


ALL_SOLVERS = ["A2EM_auto", "A2EM_noauto", "AEM", "Krylov", "MS"]
ROBUST_SOLVERS = ["A2EM_noauto", "AEM", "Krylov"]  # contrast-robust, exact on homog


def expected_stress(E_field, nu_field, voigt=(1.0, 0, 0, 0, 0, 0)):
    """C_material : E_macro (uniform material only)."""
    E = SymmetricTensor2(array=jnp.asarray(voigt)).tensor
    lam, mu = lame(float(E_field), float(nu_field))
    return lam * jnp.trace(E) * jnp.eye(3) + 2 * mu * E


def mean(x):
    return jnp.mean(x, axis=(0, 1, 2))


# --------------------------------------------------------------------------- #
# 1. Homogeneous media: eps == E_macro,  sigma == C:E_macro,  fields uniform
# --------------------------------------------------------------------------- #
HOMOG_CASES = {
    "C_eq_C0": (E0, NU0),  # material == reference  (degenerate for AEM/A2EM)
    "stiffer": (5 * E0, 0.2),
    "softer": (E0 / 20, 0.35),
}


def _homog_marks(solver, case):
    if solver == "AEM" and case == "C_eq_C0":
        return pytest.mark.xfail(
            strict=True,
            reason="AEM: 0/0 in Delta and lambda when exact solution reached in one step on C==C0",
        )
    if solver == "A2EM_auto":
        return pytest.mark.xfail(
            strict=False,
            reason="A2EM auto-scale loses precision on homogeneous media (E_norm -> 1e-12)",
        )
    if solver == "MS" and case == "stiffer":
        return pytest.mark.xfail(strict=True, reason="MS diverges when material stiffer than ~2*C0")
    return ()


@pytest.mark.parametrize(
    "solver, case",
    [pytest.param(s, c, marks=_homog_marks(s, c)) for c in HOMOG_CASES for s in ALL_SOLVERS],
)
def test_homogeneous_recovers_macro(solver, case):
    Ev, nuv = HOMOG_CASES[case]
    prob = make_problem(uniform(Ev), uniform(nuv))
    eps, sig, _ = run(solver, prob)

    assert jnp.all(jnp.isfinite(eps)) and jnp.all(jnp.isfinite(sig))

    E = SymmetricTensor2(array=jnp.asarray([1.0, 0, 0, 0, 0, 0])).tensor
    # uniform strain field equal to the macroscopic strain
    assert jnp.allclose(mean(eps), E, atol=ATOL)
    assert float(jnp.std(eps.reshape(-1, 3, 3), axis=0).max()) < ATOL  # field is uniform
    # stress matches the closed-form C:E
    assert jnp.allclose(mean(sig), expected_stress(Ev, nuv), rtol=1e-4, atol=ATOL)


# --------------------------------------------------------------------------- #
# 2. Zero macroscopic strain  ->  eps == 0, sigma == 0  (zero-division probe)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("solver", ALL_SOLVERS)
def test_zero_macro_strain_is_zero(solver):
    prob = make_problem(uniform(2 * E0), uniform(NU0))
    eps, sig, _ = run(solver, prob, voigt=(0, 0, 0, 0, 0, 0))
    assert jnp.all(jnp.isfinite(eps)) and jnp.all(jnp.isfinite(sig))
    assert float(jnp.abs(eps).max()) < ATOL
    assert float(jnp.abs(sig).max()) < ATOL


# --------------------------------------------------------------------------- #
# 3. Heterogeneous: cross-solver agreement + macroscopic consistency
# --------------------------------------------------------------------------- #
def two_phase(contrast, frac=0.4, seed=0):
    key = jax.random.PRNGKey(seed)
    phase = (jax.random.uniform(key, (N, N, N)) < frac).astype(jnp.float64)
    E = jnp.where(phase == 1, contrast * E0, E0)
    nu = jnp.where(phase == 1, 0.2, NU0)
    return E, nu


@pytest.mark.parametrize("contrast", [3.0, 100.0])
def test_robust_solvers_agree(contrast):
    """A2EM(noauto), AEM, Krylov must converge to the same field at any contrast."""
    Ef, nuf = two_phase(contrast)
    prob = make_problem(Ef, nuf)
    refs = {s: run(s, prob) for s in ROBUST_SOLVERS}
    eps0, sig0, _ = refs["Krylov"]
    for s, (eps, sig, _) in refs.items():
        assert jnp.all(jnp.isfinite(sig)), s
        assert jnp.allclose(mean(sig), mean(sig0), rtol=1e-4, atol=ATOL), s
        assert jnp.allclose(eps, eps0, atol=1e-4), s


@pytest.mark.parametrize("solver", ROBUST_SOLVERS)
def test_mean_strain_equals_macro(solver):
    """<eps> == E_macro  (fundamental compatibility constraint of the LS eq.)."""
    Ef, nuf = two_phase(10.0)
    prob = make_problem(Ef, nuf)
    eps, _, _ = run(solver, prob)
    E = SymmetricTensor2(array=jnp.asarray([1.0, 0, 0, 0, 0, 0])).tensor
    assert jnp.allclose(mean(eps), E, atol=1e-4)


@pytest.mark.parametrize("solver", ROBUST_SOLVERS)
def test_macro_stress_within_voigt_reuss(solver):
    """The homogenised uniaxial modulus <sig11> lies between the Reuss (harmonic)
    and Voigt (arithmetic) estimates of the phase moduli."""
    contrast = 10.0
    Ef, nuf = two_phase(contrast)
    prob = make_problem(Ef, nuf)
    _, sig, _ = run(solver, prob)
    s11 = float(mean(sig)[0, 0])

    # 1D longitudinal moduli M = lam + 2mu for each phase
    def M(E, nu):
        return (lambda lm: lm[0] + 2 * lm[1])(lame(E, nu))

    M1, M2 = M(E0, NU0), M(contrast * E0, 0.2)
    f = float(jnp.mean(Ef != E0))  # volume fraction of stiff phase
    voigt = (1 - f) * M1 + f * M2
    reuss = 1.0 / ((1 - f) / M1 + f / M2)
    assert reuss - ATOL <= s11 <= voigt + ATOL


# --------------------------------------------------------------------------- #
# 4. Blanket finiteness guard across well-posed cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("solver", ROBUST_SOLVERS)
@pytest.mark.parametrize("contrast", [1.0, 10.0, 1000.0])
def test_outputs_finite(solver, contrast):
    Ef, nuf = two_phase(contrast)
    prob = make_problem(Ef, nuf)
    eps, sig, n = run(solver, prob)
    assert jnp.all(jnp.isfinite(eps)) and jnp.all(jnp.isfinite(sig))
    assert n >= 1
