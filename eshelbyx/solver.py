import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
from jaxmat.tensors import SymmetricTensor2


# @eqx.debug.assert_max_traces(max_traces=1)
@eqx.filter_jit
def A2EMSolver(
    E_macro,
    alpha,
    Gamma_0,
    max_iter=1000,
    tol=1e-10,
    auto_scale=True,
    step_adaptation=True,
    verbose=True,
):

    # ------------------------
    # Helpers
    # ------------------------
    def C0_inner_product(X, Y):
        return jnp.einsum("...ij,...ij->", X, Gamma_0.apply_C0(Y))

    def ref_norm(X):
        return jnp.sqrt(C0_inner_product(X, X))

    def projection_on_D(X):
        z = Gamma_0.apply_C0(X)
        return Gamma_0.mv(z)

    # ------------------------
    # One AEM iteration
    # ------------------------
    def step(carry):
        it, X, eps_aux, W, scale, Delta = carry
        aX = alpha(X, scale)
        Y = projection_on_D(aX)

        Z = aX - 2.0 * Y + 2 * W / scale

        if step_adaptation:
            lam_new = C0_inner_product(X, Z) / C0_inner_product(Z, Z)
        else:
            lam_new = 1.0

        eps_aux_new = eps_aux + lam_new * X
        eps_new = alpha(eps_aux_new, scale)

        sig_new = scale * Gamma_0.apply_C0(2.0 * eps_aux_new - eps_new)

        W_new = (1 - 2 * lam_new) * W + lam_new * scale * Y

        eps0 = Gamma_0.apply_C0_inv(sig_new)
        E_norm = jnp.maximum(ref_norm(E_macro - eps_new), 1e-16)
        k_A2EM = ref_norm(eps0) / E_norm
        new_scale = jax.lax.select(auto_scale, k_A2EM, 1.0)

        eps_aux_new = 0.5 * (eps0 / new_scale + eps_new)

        X_new = X - lam_new * Z + (1 / new_scale - 1 / scale) * W_new
        Delta_new = ref_norm(X_new) / E_norm
        if verbose:
            jax.debug.print(
                "Iteration {} ||X|| = {:.5e} Delta = {:.5e} ||sig|| = {:.5e} k_A2EM = {:.5e} lambda = {:.5e} scale = {:.5e}",
                it,
                ref_norm(X) / ref_norm(E_macro),
                Delta_new,
                jnp.mean(jnp.linalg.matrix_norm(sig_new)),
                k_A2EM,
                lam_new,
                new_scale,
            )
        return (
            it + 1,
            X_new,
            eps_aux_new,
            W_new,
            new_scale,
            Delta_new,
        )

    # ------------------------
    # While condition
    # ------------------------
    def cond(carry):
        it, _, _, _, _, Delta = carry
        return jnp.logical_and(it < max_iter, Delta > tol)

    # ------------------------
    # Initial state
    # ------------------------
    init_carry = (
        0,
        jnp.asarray(E_macro),
        jnp.zeros_like(E_macro),
        jnp.zeros_like(E_macro),
        1.0,
        jnp.inf,
    )

    # ------------------------
    # Run while-loop (true early stop)
    # ------------------------
    itf, Xf, eps_aux_f, W_f, scale_f, Delta_f = jax.lax.while_loop(
        cond, step, init_carry
    )

    # ------------------------
    # Final fields
    # ------------------------
    eps = alpha(eps_aux_f, scale_f)
    sigma = scale_f * Gamma_0.apply_C0(2.0 * eps_aux_f - eps)

    convergence_history = {
        "num_steps": itf,
        "Delta_final": Delta_f,
        "scale_final": scale_f,
    }

    return eps, sigma, convergence_history


def AEMSolver(E_macro, alpha, Gamma_0, max_iter=1000, tol=1e-10, verbose=True):
    """
    Adaptive Eyre–Milton (AEM) solver – minimal version

    Returns
    -------
    eps : final strain field
    sigma : final stress field
    info : dict with convergence info
    """

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def C0_inner_product(X, Y):
        return jnp.sum(
            jnp.einsum("...ij,...ij->...", jnp.conjugate(X), Gamma_0.apply_C0(Y))
        )
        # return jnp.sum(jnp.einsum("...ij,...kl,ijkl", jnp.conjugate(X), Y, C0))

    def ref_norm(X):
        return jnp.sqrt(jnp.real(C0_inner_product(X, X)))

    def projection_on_D(X):
        return Gamma_0.mv(Gamma_0.apply_C0(X))

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    X = jnp.asarray(E_macro)
    eps_aux = jnp.zeros_like(E_macro)
    lam = 0.0

    # E_norm = ref_norm(E_macro)

    convergence_history = {
        # "residual": jnp.zeros((max_iter,)),
        "Delta": jnp.zeros((max_iter,)),
        "lambda": jnp.zeros((max_iter,)),
        "energy": jnp.zeros((max_iter,)),
    }

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def cond_fun(state):
        i, _, _, _, Delta, _ = state
        return jnp.logical_and(i < max_iter, Delta >= tol)

    def body_fun(state):
        i, X, eps_aux, _, _, convergence_history = state

        # Cached quantities
        eps = alpha(eps_aux)
        Delta = ref_norm(X) / ref_norm(E_macro - eps)

        # AEM update
        Z = alpha(X) + 2.0 * projection_on_D(X - alpha(X))

        lam_new = C0_inner_product(X, Z) / C0_inner_product(Z, Z)

        eps_aux_new = eps_aux + lam_new * X
        sig = Gamma_0.apply_C0(2.0 * eps_aux_new - eps)
        X = X - lam_new * Z

        # convergence_history["residual"] = (
        #     convergence_history["residual"].at[i].set(ref_norm(X) / E_norm)
        # )
        convergence_history["Delta"] = convergence_history["Delta"].at[i].set(Delta)
        convergence_history["lambda"] = convergence_history["lambda"].at[i].set(lam_new)
        convergence_history["energy"] = (
            convergence_history["energy"]
            .at[i]
            .set(0.5 * jnp.mean(jnp.einsum("...ij,...ij->...", sig, eps)))
        )
        if verbose:
            jax.debug.print(
                "Iteration {} ||X|| = {:.5e} Delta = {:.5e} ||sig|| = {:.5e} lambda = {:.5e}",
                i,
                ref_norm(X) / ref_norm(E_macro),
                Delta,
                jnp.mean(jnp.linalg.matrix_norm(sig)),
                lam_new,
            )
        return i + 1, X, eps_aux_new, lam_new, Delta, convergence_history

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    init_state = (0, X, eps_aux, lam, jnp.inf, convergence_history)
    i_final, _, eps_aux_f, _, _, convergence_history = jax.lax.while_loop(
        cond_fun, body_fun, init_state
    )

    # ------------------------------------------------------------------
    # Final fields
    # ------------------------------------------------------------------
    eps = alpha(eps_aux_f)
    sigma = Gamma_0.apply_C0(2.0 * eps_aux_f - eps)

    convergence_history["iterations"] = i_final

    return eps, sigma, convergence_history


def MSSolver(E_macro, C_eps, C_eps_0, Gamma_0, max_iter=100, tol=1e-10, verbose=True):
    """
    Moulinec and Suquet's solver
    """

    # ----------------
    # Helper functions
    # ----------------
    def C0_inner_product(X, Y):
        return jnp.sum(
            jnp.einsum("...ij,...ij->...", jnp.conjugate(X), Gamma_0.apply_C0(Y))
        )
        # return jnp.sum(jnp.einsum("...ij,...kl,ijkl", jnp.conjugate(X), Y, C0))

    def ref_norm(X):
        return jnp.sqrt(jnp.real(C0_inner_product(X, X)))

    def projection_on_D(X):
        return Gamma_0.mv(Gamma_0.apply_C0(X))

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    X = E_macro
    eps = jnp.zeros_like(E_macro)

    convergence_history = {
        # "residual": jnp.zeros((max_iter,)),
        "Delta": jnp.zeros((max_iter,)),
        "energy": jnp.zeros((max_iter,)),
    }

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def cond_fun(state):
        i, _, Delta, _, _ = state
        return jnp.logical_and(i < max_iter, Delta >= tol)

    def body_fun(state):
        i, X, Delta, eps, convergence_history = state
        tau = C_eps(eps) - C_eps_0(eps)
        X = E_macro - eps - Gamma_0.mv(tau)
        Delta = ref_norm(X) / ref_norm(E_macro)
        eps = eps + X
        sig = C_eps(eps)

        # convergence_history["residual"] = (
        #     convergence_history["residual"].at[i].set(ref_norm(X) / E_norm)
        # )
        convergence_history["Delta"] = convergence_history["Delta"].at[i].set(Delta)
        convergence_history["energy"] = (
            convergence_history["energy"]
            .at[i]
            .set(0.5 * jnp.mean(jnp.einsum("...ij,...ij->...", sig, eps)))
        )
        if verbose:
            jax.debug.print(
                "Iteration {}  Delta = {:.5e} ||sig|| = {:.5e}",
                i,
                Delta,
                jnp.mean(jnp.linalg.matrix_norm(sig)),
            )
        return i + 1, X, Delta, eps, convergence_history

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    init_state = (0, X, jnp.inf, eps, convergence_history)
    i_final, _, _, eps, convergence_history = jax.lax.while_loop(
        cond_fun, body_fun, init_state
    )

    # ------------------------------------------------------------------
    # Final fields
    # ------------------------------------------------------------------
    sig = C_eps(eps)
    eps = eps.tensor.reshape((-1, 3, 3))
    sig = sig.tensor.reshape((-1, 3, 3))

    convergence_history["iterations"] = i_final

    return eps, sig, convergence_history
