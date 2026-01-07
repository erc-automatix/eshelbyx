import jax
import jax.numpy as jnp
from functools import partial
from eshelbyx.tensors import SymmetricTensor2


def AEM_solver(
    E_macro,
    C0,
    alpha,
    Gamma_0,
    max_iter=1000,
    tol=1e-10,
    lambda_value=None,
):
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
        return jnp.sum(jnp.einsum("...ij,...kl,ijkl", jnp.conjugate(X), Y, C0))

    def ref_norm(X):
        return jnp.sqrt(jnp.real(C0_inner_product(X, X)))

    def projection_on_D(X):
        return Gamma_0.mv(Gamma_0.apply_C0(X))

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    X = E_macro
    eps_aux = SymmetricTensor2(tensor=jnp.zeros_like(E_macro))
    lam = 0.0

    E_norm = ref_norm(E_macro)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def cond_fun(state):
        i, _, _, _, Delta = state
        return jnp.logical_and(i < max_iter, Delta >= tol)

    def body_fun(state):
        i, X, eps_aux, lam, _ = state

        # Cached quantities
        eps = alpha(eps_aux)
        XD = projection_on_D(X)
        X_minus_eps = 2.0 * eps_aux - eps

        # Residuals
        Delta_D = ref_norm(XD) / ref_norm(X_minus_eps)
        Delta_S = ref_norm(X - XD) / ref_norm(eps - E_macro)
        Delta = jnp.sqrt(Delta_D**2 + Delta_S**2)

        # AEM update
        Z = alpha(X) + 2.0 * projection_on_D(X - alpha(X))

        jax.debug.print(
            "Iteration {}: ||X||={}   Delta = {}", i + 1, C0_inner_product(X, X), Delta
        )

        lam_new = C0_inner_product(X, Z) / C0_inner_product(Z, Z)

        eps_aux = eps_aux + lam_new * X
        X = X - lam_new * Z

        return i + 1, X.sym, eps_aux.sym, lam_new, Delta

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    init_state = (0, X, eps_aux, lam, jnp.inf)
    i_final, X, eps_aux, lam, Delta = jax.lax.while_loop(cond_fun, body_fun, init_state)

    # ------------------------------------------------------------------
    # Final fields
    # ------------------------------------------------------------------
    eps = alpha(eps_aux)
    sigma = SymmetricTensor2(tensor=Gamma_0.apply_C0(2.0 * eps_aux - eps))

    info = {
        "iterations": i_final,
        "Delta": Delta,
        "lambda": lam,
        "resX": ref_norm(X) / E_norm,
    }

    return eps, sigma, info
