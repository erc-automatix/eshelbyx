import jax
import jax.numpy as jnp
import equinox as eqx


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _to_array(x):
    """Coerce a jaxmat tensor (or array-like) to a raw ``(..., 3, 3)`` array.

    The solvers operate internally on raw arrays; this lets callers pass either
    a ``SymmetricTensor2`` field or a plain array for ``E_macro`` and for the
    constitutive callables.
    """
    t = getattr(x, "tensor", None)
    return t if t is not None else jnp.asarray(x)


def _safe_div(num, den, floor=1e-300):
    """``num / den`` that returns 0 where ``|den| <= floor``.

    This turns the degenerate ``0/0`` ratios that appear when the exact solution
    is reached (homogeneous media, zero macroscopic strain) into 0 instead of
    ``NaN``, so the fixed-point iterations terminate cleanly.
    """
    ok = jnp.abs(den) > floor
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), 0.0)


def _operators(Gamma_0):
    """Build the ``C0`` inner product, its induced norm, and the projector onto
    compatible strains for a given reference Green operator.

    The inner product uses a conjugate so it is valid for complex fields and
    reduces to the usual real energy product for real fields.
    """

    def inner(X, Y):
        return jnp.sum(
            jnp.einsum("...ij,...ij->...", jnp.conjugate(X), Gamma_0.apply_C0(Y))
        )

    def norm(X):
        return jnp.sqrt(jnp.real(inner(X, X)))

    def project(X):
        # P = Gamma_0 . C0  : projection onto compatible, zero-mean strains
        return Gamma_0.mv(Gamma_0.apply_C0(X))

    return inner, norm, project


# --------------------------------------------------------------------------- #
# Adaptive Eyre–Milton with reference-medium auto-scaling (A2EM)
# --------------------------------------------------------------------------- #
@eqx.filter_jit
def A2EMSolver(
    E_macro,
    alpha,
    Gamma_0,
    max_iter=1000,
    tol=1e-10,
    auto_scale=True,
    step_adaptation=True,
    verbose=False,
):
    """Adaptive Eyre–Milton solver with reference-medium auto-scaling.

    Parameters
    ----------
    E_macro : array or SymmetricTensor2
        Imposed macroscopic strain, broadcast to the grid.
    alpha : callable
        ``alpha(eps_aux, scale)`` returning the true strain from the auxiliary
        strain; ``scale`` rescales the reference medium (``scale * C0``).
    Gamma_0 : GreenOperator
        Reference-medium Green operator (provides ``apply_C0``, ``apply_C0_inv``
        and ``mv``).

    Returns
    -------
    eps, sigma : arrays of shape ``grid.shape + (3, 3)``
    info : dict with ``num_steps``, ``Delta_final``, ``scale_final``.

    Notes
    -----
    Robust to homogeneous media and to ``E_macro == 0``: all ratios are guarded
    and the auto-scale is *frozen* once the strain fluctuation drops below a
    relative floor, which previously drove ``scale`` to ~1e12 and destroyed the
    accuracy of the reconstructed stress on (near-)homogeneous problems.
    """
    E_macro = _to_array(E_macro)
    inner, norm, project = _operators(Gamma_0)

    E_ref = norm(E_macro)
    floor = jnp.maximum(1e-30, 1e-14 * E_ref)  # relative denominator floor

    def step(carry):
        it, X, eps_aux, W, scale, Delta = carry

        aX = alpha(X, scale)
        Y = project(aX)
        Z = aX - 2.0 * Y + 2.0 * W / scale

        if step_adaptation:
            lam = _safe_div(inner(X, Z), inner(Z, Z))
        else:
            lam = 1.0

        eps_aux_new = eps_aux + lam * X
        eps_new = alpha(eps_aux_new, scale)
        sig_new = scale * Gamma_0.apply_C0(2.0 * eps_aux_new - eps_new)
        W_new = (1.0 - 2.0 * lam) * W + lam * scale * Y

        eps0 = Gamma_0.apply_C0_inv(sig_new)
        E_norm = norm(E_macro - eps_new)

        # Adapt the reference scale only while the fluctuation is significant;
        # freeze it otherwise so the final reconstruction stays well-conditioned.
        k = _safe_div(norm(eps0), E_norm)
        adapt = jnp.logical_and(auto_scale, E_norm > floor)
        new_scale = jnp.where(adapt, k, scale)

        eps_aux_new = 0.5 * (eps0 / new_scale + eps_new)
        X_new = X - lam * Z + (1.0 / new_scale - 1.0 / scale) * W_new
        Delta_new = _safe_div(norm(X_new), jnp.maximum(E_norm, floor))

        if verbose:
            jax.debug.print(
                "A2EM it {} ||X||={:.3e} Delta={:.3e} lambda={:.3e} scale={:.3e}",
                it,
                _safe_div(norm(X), jnp.maximum(E_ref, floor)),
                Delta_new,
                lam,
                new_scale,
            )
        return it + 1, X_new, eps_aux_new, W_new, new_scale, Delta_new

    def cond(carry):
        it, _, _, _, _, Delta = carry
        return jnp.logical_and(it < max_iter, Delta > tol)

    init = (
        0,
        E_macro,
        jnp.zeros_like(E_macro),
        jnp.zeros_like(E_macro),
        1.0,
        jnp.inf,
    )
    itf, _, eps_aux_f, _, scale_f, Delta_f = jax.lax.while_loop(cond, step, init)

    eps = alpha(eps_aux_f, scale_f)
    sigma = scale_f * Gamma_0.apply_C0(2.0 * eps_aux_f - eps)
    info = {"num_steps": itf, "Delta_final": Delta_f, "scale_final": scale_f}
    return eps, sigma, info


# --------------------------------------------------------------------------- #
# Adaptive Eyre–Milton (AEM)
# --------------------------------------------------------------------------- #
@eqx.filter_jit
def AEMSolver(E_macro, alpha, Gamma_0, max_iter=1000, tol=1e-10, verbose=False):
    """Adaptive Eyre–Milton solver (fixed reference medium).

    Returns raw ``(..., 3, 3)`` arrays and an ``info`` dict with ``num_steps``,
    ``Delta_final`` and a per-iteration ``Delta`` history.

    Robust to homogeneous media: the previous version produced ``NaN`` when the
    exact solution was reached in a single step (``0/0`` in both ``Delta`` and
    the step length ``lambda``); those ratios are now guarded.
    """
    E_macro = _to_array(E_macro)
    inner, norm, project = _operators(Gamma_0)

    E_ref = norm(E_macro)
    floor = jnp.maximum(1e-30, 1e-14 * E_ref)

    history = {"Delta": jnp.zeros((max_iter,)), "lambda": jnp.zeros((max_iter,))}

    def cond(state):
        i, _, _, _, Delta, _ = state
        return jnp.logical_and(i < max_iter, Delta > tol)

    def body(state):
        i, X, eps_aux, _, _, hist = state

        eps = alpha(eps_aux)
        Delta = _safe_div(norm(X), jnp.maximum(norm(E_macro - eps), floor))

        aX = alpha(X)
        Z = aX + 2.0 * project(X - aX)
        lam = _safe_div(inner(X, Z), inner(Z, Z))

        eps_aux_new = eps_aux + lam * X
        X_new = X - lam * Z

        hist = {
            "Delta": hist["Delta"].at[i].set(Delta),
            "lambda": hist["lambda"].at[i].set(jnp.real(lam)),
        }
        if verbose:
            jax.debug.print("AEM it {} Delta={:.5e} lambda={:.5e}", i, Delta, lam)
        return i + 1, X_new, eps_aux_new, lam, Delta, hist

    init = (0, E_macro, jnp.zeros_like(E_macro), 0.0, jnp.inf, history)
    i_f, _, eps_aux_f, _, Delta_f, history = jax.lax.while_loop(cond, body, init)

    eps = alpha(eps_aux_f)
    sigma = Gamma_0.apply_C0(2.0 * eps_aux_f - eps)
    info = {
        "num_steps": i_f,
        "iterations": i_f,  # backward-compatible alias
        "Delta_final": Delta_f,
        "history": history,
    }
    return eps, sigma, info


# --------------------------------------------------------------------------- #
# Basic Moulinec–Suquet scheme
# --------------------------------------------------------------------------- #
@eqx.filter_jit
def MSSolver(E_macro, C_eps, C_eps_0, Gamma_0, max_iter=100, tol=1e-10, verbose=False):
    """Moulinec–Suquet fixed-point solver.

    Parameters
    ----------
    C_eps, C_eps_0 : callables
        Raw-array constitutive maps, ``eps -> sigma`` with both of shape
        ``grid.shape + (3, 3)``. ``C_eps`` is the true material law and
        ``C_eps_0`` applies the reference stiffness ``C0``. (Passing raw-array
        callables — rather than tensor-valued ones — keeps the solver decoupled
        from the tensor backend and matrix-free.)

    Returns
    -------
    eps, sigma : arrays of shape ``grid.shape + (3, 3)``
    info : dict with ``num_steps`` and a per-iteration ``Delta`` history.

    Notes
    -----
    The basic MS scheme converges only when the reference medium is stiff enough
    (roughly ``C < 2 C0``); with a softer reference it diverges. The iteration is
    now guarded against ``E_macro == 0`` (division by ``||E_macro|| = 0``).
    """
    E_macro = _to_array(E_macro)
    _, norm, _ = _operators(Gamma_0)

    E_ref = norm(E_macro)
    floor = jnp.maximum(1e-30, 1e-14 * E_ref)

    history = {"Delta": jnp.zeros((max_iter,))}

    def cond(state):
        i, _, Delta, _, _ = state
        return jnp.logical_and(i < max_iter, Delta > tol)

    def body(state):
        i, X, Delta, eps, hist = state

        tau = _to_array(C_eps(eps)) - _to_array(C_eps_0(eps))
        X = E_macro - eps - Gamma_0.mv(tau)
        Delta = _safe_div(norm(X), jnp.maximum(E_ref, floor))
        eps = eps + X

        hist = {"Delta": hist["Delta"].at[i].set(Delta)}
        if verbose:
            jax.debug.print("MS it {} Delta={:.5e}", i, Delta)
        return i + 1, X, Delta, eps, hist

    init = (0, E_macro, jnp.inf, jnp.zeros_like(E_macro), history)
    i_f, _, Delta_f, eps, history = jax.lax.while_loop(cond, body, init)

    sigma = _to_array(C_eps(eps))
    info = {
        "num_steps": i_f,
        "iterations": i_f,  # backward-compatible alias
        "Delta_final": Delta_f,
        "history": history,
    }
    return eps, sigma, info


@eqx.filter_jit
def DRSolver_(
    E_macro,
    alpha,
    Gamma_0,
    max_iter=2000,
    tol=1e-10,
    theta=1.0,
    z0_scale=2.0,
    verbose=False,
):
    """Douglas--Rachford solver.

    Parameters
    ----------
    alpha : the AEM resolvent (alpha(w) solves sigma(e)+C0:e = 2 C0:w);
            DR uses J_B(z) = alpha(z/2).
    theta : relaxation in (0, 2]; theta=1 is standard DR, theta>1 over-relaxes.
    z0_scale : initial z = z0_scale * E_macro. The natural polarisation guess
            (eps=E, sigma=C0:E) is z0_scale=2 and makes a homogeneous medium
            converge in one step.

    Returns eps, sigma (grid arrays) and info with ``num_steps``/``Delta_final``.
    """
    E_macro = _to_array(E_macro)
    inner, norm, project = _operators(Gamma_0)
    Enorm = norm(E_macro) + 1e-300

    def cond(state):
        k, z, delta = state
        return jnp.logical_and(k < max_iter, delta > tol)

    def body(state):
        k, z, _ = state
        e = alpha(z / 2.0)  # J_B(z)
        gap = (E_macro + project(2.0 * e - z)) - e  # J_A(R_B z) - J_B z
        z_new = z + theta * gap
        delta = norm(gap) / Enorm  # DR feasibility residual
        if verbose:
            jax.debug.print("DR it {} Delta={:.5e}", k, delta)
        return k + 1, z_new, delta

    z0 = z0_scale * E_macro
    k, z, delta = jax.lax.while_loop(cond, body, (0, z0, jnp.inf))

    eps = alpha(z / 2.0)
    sigma = Gamma_0.apply_C0(z - eps)  # sigma = C0:(z - e)
    return eps, sigma, {"num_steps": k, "Delta_final": delta}


@eqx.filter_jit
def DRSolver(
    E_macro,
    alpha,
    Gamma_0,
    scale_init=1.0,
    max_iter=4000,
    tol=1e-10,
    theta=1.99,
    z0_scale=2.0,
    scale_update=True,
    scale_clip=(1e-8, 1e8),
    verbose=False,
):
    """Douglas-Rachford with adaptive reference medium.

    z = e + (scale*C0)^{-1} sigma. Per iteration:
      1. e = J_B(z; scale) = alpha(z/2, scale)
      2. (optional) s_new = scale * <z-e, e>_C0 / <e, e>_C0  ; transport
         z <- e + (scale/s_new)(z - e)   [keeps e = J_B(z; s_new) invariant]
      3. DR step:  z <- z + theta*( [E + P(2e - z)] - e ) ,  P = Gamma0 C0
    Recover eps = J_B(z; scale), sigma = scale*C0:(z - eps).
    """
    E_macro = _to_array(E_macro)
    inner, norm, project = _operators(Gamma_0)
    Enorm = norm(E_macro) + 1e-300
    floor = 1e-14 * Enorm

    def cond(state):
        k, z, scale, delta = state
        return jnp.logical_and(k < max_iter, delta > tol)

    def body(state):
        k, z, scale, _ = state
        e = alpha(z / 2.0, scale)  # J_B(z; scale)

        if scale_update:
            fluct = norm(E_macro - e)
            stress = scale * norm(z - e)  # ||sigma||_{C0^-1}
            s_new = jnp.where(fluct > floor, _safe_div(stress, fluct), scale)
            s_new = jnp.clip(s_new, scale_clip[0], scale_clip[1])
            z = e + (scale / s_new) * (z - e)  # transport (e invariant)
            scale = s_new

        gap = (E_macro + project(2.0 * e - z)) - e  # DR gap at current scale
        z_new = z + theta * gap
        delta = norm(gap) / Enorm
        if verbose:
            jax.debug.print("DR it {} d={:.3e} scale={:.4e}", k, delta, scale)
        return k + 1, z_new, scale, delta

    z0 = z0_scale * E_macro
    k, z, scale, delta = jax.lax.while_loop(cond, body, (0, z0, scale_init, jnp.inf))
    eps = alpha(z / 2.0, scale)
    sigma = scale * Gamma_0.apply_C0(z - eps)
    return eps, sigma, {"num_steps": k, "Delta_final": delta, "scale_final": scale}
