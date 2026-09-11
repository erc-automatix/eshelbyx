"""
Nonlinear FFT homogenisation with the (hardened) AEM / A2EM solvers.

Method: Newton--AEM
-------------------
Plugging a *nonlinear* local ``alpha`` straight into AEM/A2EM does NOT work --
those solvers apply ``alpha`` to search directions, valid only when ``alpha`` is
linear (empirically the Lippmann--Schwinger residual then plateaus ~5e-3).

Instead we Newton-linearise the global residual
    R(eps) = eps + Gamma0 : (sigma(eps) - C0 : eps) - E_macro = 0 ,
and solve each linearised cell problem R'(eps).deps = -R with the *linear* AEM,
using the consistent tangent C_t = d sigma / d eps through
    alpha_tangent(deps) = 2 (C_t + scale*C0)^{-1} (scale*C0) deps .
C_t is a full 6x6 (Mandel) tensor per voxel -- anisotropic in general -- and
enters only this local solve; C0 and the Green operator stay isotropic.

Inner solver: AEMSolver. Do NOT use A2EMSolver(auto_scale=True) as the inner
solver (its adaptive reference targets the physical problem, not a correction,
and diverges).

Spatially-varying properties (see ``plain_material`` / ``jaxmat_material``)
--------------------------------------------------------------------------
State-batching (``init_state(Nbatch)``) batches only internal variables, NOT the
material parameters. For heterogeneous properties the *behaviour itself* must be
batched: build it with array-valued parameters (leading axis = n_voxels) so its
pytree leaves are per-voxel, then vmap ``constitutive_update`` over the behaviour
too, via ``eqx.filter_vmap``. ``jax.vmap`` with the behaviour as a closure, and
jaxmat's ``batched_constitutive_update`` (which uses ``in_axes=(0,0,None)`` on
``self.constitutive_update``), both share one parameter set and cannot express
spatial variation.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple, Callable

from jaxmat.tensors import SymmetricTensor2


# FIXME: use jaxmat's natural tensorial/array representation to avoid to/from Mandel conversions

SQ2 = jnp.sqrt(2.0)


# --------------------------------------------------------------------------- #
# Mandel <-> 3x3  (orthonormal: to_mandel(A):to_mandel(B) == A:B)
# --------------------------------------------------------------------------- #
def to_mandel(A):
    return jnp.stack(
        [
            A[..., 0, 0],
            A[..., 1, 1],
            A[..., 2, 2],
            SQ2 * A[..., 0, 1],
            SQ2 * A[..., 0, 2],
            SQ2 * A[..., 1, 2],
        ],
        axis=-1,
    )


def from_mandel(v):
    a, b, c = v[..., 3] / SQ2, v[..., 4] / SQ2, v[..., 5] / SQ2
    r0 = jnp.stack([v[..., 0], a, b], axis=-1)
    r1 = jnp.stack([a, v[..., 1], c], axis=-1)
    r2 = jnp.stack([b, c, v[..., 2]], axis=-1)
    return jnp.stack([r0, r1, r2], axis=-2)


def isotropic_mandel(lam, mu):
    """6x6 Mandel matrix of the isotropic reference C0."""
    M = jnp.diag(jnp.array([0.0, 0, 0, 2 * mu, 2 * mu, 2 * mu]))
    return M.at[:3, :3].set(lam * jnp.ones((3, 3)) + 2 * mu * jnp.eye(3))


# --------------------------------------------------------------------------- #
# Material = field-level stress+tangent and a commit step.
# FIXME: no plain material anymore => do we still need a Material data structure
# --------------------------------------------------------------------------- #
class Material(NamedTuple):
    stress_tangent: Callable  # (eps_field, state) -> (sig_field, Ct[n_voxel,6,6])
    commit: Callable  # (eps_field, state) -> (sig_field, new_state)


# FIXME: per voxel params should alway be true, remove it
def jaxmat_material(behavior, dt=0.0, per_voxel_params=True):
    """Material from a jaxmat ``SmallStrainBehavior``.

    For spatially-varying properties (``per_voxel_params=True``) the behaviour
    must be built with array-valued parameters whose leading axis is the number
    of voxels, e.g.::

        beh = M.vonMisesIsotropicHardening(
                  elasticity=M.LinearElasticIsotropic(E=E_field, nu=nu_field),
                  yield_stress=M.VoceHardening(sig0=s0_field, sigu=su_field, b=b_field))

    where ``E_field`` etc. are shape ``(n_voxels,)``. ``eqx.filter_vmap`` then
    maps the behaviour's parameter leaves per voxel. With ``per_voxel_params=
    False`` the parameters are shared (homogeneous medium).
    """
    beh_axes = eqx.if_array(0) if per_voxel_params else None

    def _one(b, eps6, st):
        sig, ns = b.constitutive_update(SymmetricTensor2(array=eps6), st, dt)
        return sig.array, ns

    stress_v = eqx.filter_vmap(_one, in_axes=(beh_axes, 0, 0))
    tangent_v = eqx.filter_vmap(
        lambda b, e, s: jax.jacfwd(lambda x: _one(b, x, s)[0])(e),
        in_axes=(beh_axes, 0, 0),
    )

    def stress_tangent(eps_field, state):
        eps6 = to_mandel(eps_field.reshape(-1, 3, 3))
        sig6, _ = stress_v(behavior, eps6, state)
        Ct = tangent_v(behavior, eps6, state)
        return from_mandel(sig6).reshape(eps_field.shape), Ct

    def commit(eps_field, state):
        eps6 = to_mandel(eps_field.reshape(-1, 3, 3))
        sig6, ns = stress_v(behavior, eps6, state)
        return from_mandel(sig6).reshape(eps_field.shape), ns

    return Material(stress_tangent, commit)


# --------------------------------------------------------------------------- #
# Linear tangent-alpha (frozen, possibly anisotropic C_t)
# FIXME: should be able to use Tensor4 jaxmat algebra with inv method
# --------------------------------------------------------------------------- #
def make_alpha_tangent(Ct, C0_mandel):
    """Linear ``alpha(deps, scale)`` = 2 (C_t + scale C0)^{-1} (scale C0) deps,
    solved as a 6x6 system per voxel (handles anisotropic C_t)."""

    def alpha(deps_field, scale=1.0):
        shape = deps_field.shape
        C0s = scale * C0_mandel
        d6 = to_mandel(deps_field.reshape(-1, 3, 3))
        out6 = jax.vmap(lambda Cv, dv: 2.0 * jnp.linalg.solve(Cv + C0s, C0s @ dv))(
            Ct, d6
        )
        return from_mandel(out6).reshape(shape)

    return alpha


# --------------------------------------------------------------------------- #
# Newton--AEM (one load step, state frozen) and the incremental driver
# --------------------------------------------------------------------------- #
# FIXME: REmove in favor of optimistix solver, extensible with trust region etc.
# @eqx.filter_jit
def newton_aem(
    inner_solver,
    E_field,
    material,
    C0_mandel,
    state,
    Gamma_0,
    newton_tol=1e-8,
    newton_maxit=20,
    inner_tol=1e-10,
    inner_maxit=1000,
    inner_kw=None,
    eps0=None,
    verbose=False,
):
    """Solve the nonlinear cell problem for one load step (frozen ``state``).

    ``inner_solver`` should be ``AEMSolver``. ``material`` is a ``Material``.
    Returns eps, sigma (grid arrays) and an info dict with the Newton residuals.
    """
    inner_kw = inner_kw or {}
    eps = jnp.zeros_like(E_field) if eps0 is None else eps0
    Enorm = jnp.linalg.norm(E_field) + 1e-30
    residuals = []
    k = 0
    while k < newton_maxit:
        sig, Ct = material.stress_tangent(eps, state)
        R = eps + Gamma_0.mv(sig - Gamma_0.apply_C0(eps)) - E_field
        rnorm = jnp.linalg.norm(R) / Enorm
        residuals.append(rnorm)
        if verbose:
            jax.debug.print("  newton {}: ||R||={}", k, rnorm)
        if rnorm < newton_tol:
            break
        alpha_t = make_alpha_tangent(Ct, C0_mandel)
        deps, _, _ = inner_solver(
            -R, alpha_t, Gamma_0, tol=inner_tol, max_iter=inner_maxit, **inner_kw
        )
        eps = eps + deps
        k += 1
    sig, _ = material.stress_tangent(eps, state)
    return eps, sig, {"newton_steps": k, "newton_residuals": residuals}


# FIXME: should have better user callback facility to compute stress averages, output fields to filed, etc. not just the final state
# FIXME: also use a jax scan and checkpointing (using equinox maybe ?) for path-dependent adjoints
def solve_incremental(
    inner_solver,
    E_path,
    material,
    C0_mandel,
    Gamma_0,
    state,
    broadcast,
    warm_start=True,
    **newton_kw,
):
    """Load-step through ``E_path`` (list of 3x3 macroscopic strains).

    State is frozen inside each Newton solve and committed (via ``material.commit``
    at the converged strain) only after convergence -- correct for path-dependent
    behaviours. Returns a list of (eps, sigma, info) and the final state.
    """
    out, eps_prev = [], None
    for i, E in enumerate(E_path):
        print(f"Step {i}: strain {E}")
        eps, sig, info = newton_aem(
            inner_solver,
            broadcast(E),
            material,
            C0_mandel,
            state,
            Gamma_0,
            eps0=(eps_prev if warm_start else None),
            **newton_kw,
        )
        sig, state = material.commit(eps, state)  # commit state after convergence
        out.append((eps, sig, info))
        eps_prev = eps
    return out, state
