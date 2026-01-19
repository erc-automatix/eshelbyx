import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
from eshelbyx.tensors import SymmetricTensor2


# @eqx.debug.assert_max_traces(max_traces=1)
@eqx.filter_jit
def A2EM_solver(E_macro, alpha, Gamma_0, max_iter=1000, tol=1e-10, auto_scale=True):

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

        lam_new = C0_inner_product(X, Z) / C0_inner_product(Z, Z)

        eps_aux_new = eps_aux + lam_new * X
        eps_new = alpha(eps_aux_new, scale)

        sig_new = scale * Gamma_0.apply_C0(2.0 * eps_aux_new - eps_new)

        W_new = (1 - 2 * lam_new) * W + lam_new * scale * Y

        eps0 = Gamma_0.apply_C0_inv(sig_new)
        k_A2EM = ref_norm(eps0) / ref_norm(E_macro - eps_new)
        new_scale = jax.lax.select(auto_scale, k_A2EM, 1.0)

        eps_aux_new = 0.5 * (eps0 / new_scale + eps_new)

        X_new = X - lam_new * Z + (1 / new_scale - 1 / scale) * W_new
        Delta_new = ref_norm(X_new) / ref_norm(E_macro - eps_new)
        # jax.debug.print(
        #     "Iteration {} ||X|| = {:.5e} Delta = {:.5e} ||sig|| = {:.5e} k_A2EM = {:.5e} lambda = {:.5e} scale = {:.5e}",
        #     it,
        #     ref_norm(X) / ref_norm(E_macro),
        #     Delta_new,
        #     jnp.mean(sig_new[..., 0, 0]),
        #     k_A2EM,
        #     lam_new,
        #     new_scale,
        # )
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
        "iterations": itf,
        "Delta_final": Delta_f,
        "scale_final": scale_f,
    }

    return eps, sigma, convergence_history


@eqx.filter_jit
def polarization_solver(
    E_macro, alpha, Gamma_0, max_iter=1000, tol=1e-10, auto_scale=True
):

    # ------------------------
    # Helpers
    # ------------------------
    def C0_inner_product(X, Y):
        return jnp.sum(jnp.einsum("...ij,...ij->...", X, Gamma_0.apply_C0(Y)))

    def ref_norm(X):
        return jnp.sqrt(jnp.real(C0_inner_product(X, X)))

    def projection_on_D(X):
        z = Gamma_0.apply_C0(X)
        return Gamma_0.mv(z)

    # ------------------------
    # One AEM iteration
    # ------------------------
    def step(carry):
        it, X, eps_aux, scale, Delta = carry

        eps_aux_new = eps_aux + X
        eps_new = alpha(eps_aux_new, scale)

        sig_new = scale * Gamma_0.apply_C0(2.0 * eps_aux_new - eps_new)

        eps0 = Gamma_0.apply_C0_inv(sig_new)
        k_A2EM = ref_norm(eps0) / ref_norm(E_macro - eps_new) / 2.0
        new_scale = jax.lax.select(auto_scale, k_A2EM, 1.0)

        eps_aux_new = 0.5 * (eps0 / new_scale + eps_new)
        eps_new = alpha(eps_aux_new, new_scale)

        X_new = jnp.asarray(
            E_macro - eps_new - 2 * projection_on_D(eps_aux_new - eps_new)
        )
        Delta_new = ref_norm(X_new) / ref_norm(E_macro - eps_new)
        # jax.debug.print(
        #     "Iteration {} ||X|| = {:.5e} Delta = {:.5e} ||sig|| = {:.5e} k_A2EM = {:.5e} scale = {:.5e}",
        #     it,
        #     ref_norm(X) / ref_norm(E_macro),
        #     Delta_new,
        #     jnp.mean(sig_new[..., 0, 0]),
        #     k_A2EM,
        #     new_scale,
        # )
        return (
            it + 1,
            X_new,
            eps_aux_new,
            new_scale,
            Delta_new,
        )

    # ------------------------
    # While condition
    # ------------------------
    def cond(carry):
        it, _, _, _, Delta = carry
        return jnp.logical_and(it < max_iter, Delta > tol)

    # ------------------------
    # Initial state
    # ------------------------
    init_carry = (
        0,
        jnp.asarray(E_macro),
        jnp.zeros_like(E_macro),
        1.0,
        jnp.inf,
    )

    # ------------------------
    # Run while-loop (true early stop)
    # ------------------------
    itf, Xf, eps_aux_f, scale_f, Delta_f = jax.lax.while_loop(cond, step, init_carry)

    # ------------------------
    # Final fields
    # ------------------------
    eps = alpha(eps_aux_f, scale_f)
    sigma = scale_f * Gamma_0.apply_C0(2.0 * eps_aux_f - eps)

    convergence_history = {
        "iterations": itf,
        "Delta_final": Delta_f,
        "scale_final": scale_f,
    }

    return eps, sigma, convergence_history


@eqx.filter_jit
def AEM_solver(E_macro, alpha, Gamma_0, max_iter=1000, tol=1e-10):

    # ------------------------
    # Helpers
    # ------------------------
    def C0_inner_product(X, Y):
        return jnp.sum(jnp.einsum("...ij,...ij->...", X, Gamma_0.apply_C0(Y)))

    def ref_norm(X):
        return jnp.sqrt(jnp.real(C0_inner_product(X, X)))

    def projection_on_D(X):
        return Gamma_0.mv(Gamma_0.apply_C0(X))

    # ------------------------
    # One AEM iteration
    # ------------------------
    def step(carry, _):
        X, eps_aux, lam, Delta = carry

        eps = alpha(eps_aux)
        aX = alpha(X)

        Delta_new = ref_norm(X) / ref_norm(E_macro - eps)

        Z = aX + 2.0 * projection_on_D(X - aX)
        lam_new = C0_inner_product(X, Z) / C0_inner_product(Z, Z)

        eps_aux_new = eps_aux + lam_new * X
        sig = Gamma_0.apply_C0(2.0 * eps_aux_new - eps)
        X_new = X - lam_new * Z

        energy = 0.5 * jnp.mean(jnp.einsum("...ij,...ij->...", sig, eps))
        jax.debug.print(
            "   ||X|| = {:.5e} Delta = {:.5e} ||sig|| = {:.5e}",
            jnp.linalg.norm(X_new),
            Delta_new,
            jnp.mean(sig[..., 0, 0]),
        )
        new_carry = (X_new.sym, eps_aux_new.sym, lam_new, Delta_new)
        info = (Delta_new, lam_new, energy)

        return new_carry, info

    # ------------------------
    # Initial state
    # ------------------------
    X0 = E_macro
    eps_aux0 = SymmetricTensor2(tensor=jnp.zeros_like(E_macro))
    lam0 = 0.0
    Delta0 = jnp.inf

    init_carry = (X0, eps_aux0, lam0, Delta0)

    # ------------------------
    # Run scan
    # ------------------------
    (Xf, eps_aux_f, lam_f, Delta_f), history = jax.lax.scan(
        step, init_carry, xs=None, length=max_iter
    )

    Delta_hist, lambda_hist, energy_hist = history

    # ------------------------
    # Optional: early-stop mask
    # (keeps memory fixed but ignores tail)
    # ------------------------
    converged = Delta_hist < tol
    first_conv = jnp.argmax(converged)
    iterations = jnp.where(converged.any(), first_conv + 1, max_iter)

    # ------------------------
    # Final fields
    # ------------------------
    eps = alpha(eps_aux_f)
    sigma = SymmetricTensor2(tensor=Gamma_0.apply_C0(2.0 * eps_aux_f - eps))

    convergence_history = {
        "Delta": Delta_hist,
        "lambda": lambda_hist,
        "energy": energy_hist,
        "iterations": iterations,
    }

    return eps, sigma, convergence_history


def AEM_solver_while(E_macro, C0, alpha, Gamma_0, max_iter=1000, tol=1e-10):
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
        with jax.profiler.TraceAnnotation("C0_inner_product"):
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
    print(jax.profiler.device_memory_profile())
    X = E_macro
    eps_aux = SymmetricTensor2(tensor=jnp.zeros_like(E_macro))
    lam = 0.0
    print(jax.profiler.device_memory_profile())

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
        # XD = projection_on_D(X)
        # X_minus_eps = 2.0 * eps_aux - eps

        # Residuals
        # Delta_D = ref_norm(XD) / jnp.clip(ref_norm(X_minus_eps), 1e-16)
        # Delta_S = ref_norm(X - XD) / jnp.clip(ref_norm(eps - E_macro), 1e-16)
        # Delta = jnp.sqrt(Delta_D**2 + Delta_S**2)
        Delta = ref_norm(X) / ref_norm(E_macro - eps)

        # AEM update
        with jax.profiler.TraceAnnotation("AEM: AEM update"):
            Z = alpha(X) + 2.0 * projection_on_D(X - alpha(X))

        with jax.profiler.TraceAnnotation("AEM: lambda calc"):
            lam_new = C0_inner_product(X, Z) / C0_inner_product(Z, Z)

        eps_aux = eps_aux + lam_new * X
        with jax.profiler.TraceAnnotation("AEM: stress computation"):
            sig = Gamma_0.apply_C0(2.0 * eps_aux - eps)
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
        return i + 1, X.sym, eps_aux.sym, lam_new, Delta, convergence_history

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    init_state = (0, X, eps_aux, lam, jnp.inf, convergence_history)
    i_final, _, eps_aux, _, _, convergence_history = jax.lax.while_loop(
        cond_fun, body_fun, init_state
    )

    # ------------------------------------------------------------------
    # Final fields
    # ------------------------------------------------------------------
    eps = alpha(eps_aux)
    sigma = SymmetricTensor2(tensor=Gamma_0.apply_C0(2.0 * eps_aux - eps))

    convergence_history["iterations"] = i_final

    return eps, sigma, convergence_history
