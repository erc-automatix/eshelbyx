import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
from jaxmat.tensors import SymmetricTensor2, IsotropicTensor4


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


    fact_safe = 1e-6     # used for homogenous solutions (eps = E_macro => k_A2EM explodes)
      
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

        k_A2EM = ref_norm(eps0) / (ref_norm(E_macro - eps_new)+fact_safe)

        new_scale = jax.lax.select(auto_scale, k_A2EM, 1.0)

        eps_aux_new = 0.5 * (eps0 / new_scale + eps_new)

        X_new = X - lam_new * Z + (1 / new_scale - 1 / scale) * W_new

        Delta_new = ref_norm(X_new) / (ref_norm(E_macro - eps_new)+fact_safe)

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

    return eps, sigma, convergence_history, scale_f, eps_aux_f, itf, Delta_f


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




def damage_Feugueur_solver(damage_init, A, B, A0, Gamma_0, d_old, max_iter = 1000, tol=1e-6, verbose=True):
    """
    Damage solver from Gerald Feugueur's thesis
    """
    
    # --------------------------
    # Helper functions
    # --------------------------
    
    # FFT operator : X/(A0 + ξ.ξ)
    def OperatorFFT(X):
        return Gamma_0.mv(X)
    
    """
    # 2nd FFT operator : ΔX
    def OperatorFFT_err(X):
        return Gamma_0_err.mv(X)
    """
    
    # L2 norm of the field X
    def norm2(X):
        return jnp.sqrt(jnp.sum(X**2))
    
    # Projector : d -> [d_old,1] 
    def projector_damage(d, d_old):
        return jnp.clip(d, d_old, 1.0)
    
    def laplacianFFT(X):
        return Gamma_0.laplacian(X)
    
    
    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    xi = B - ((A-A0)*damage_init)
    
    convergence_history = {
        "err": jnp.zeros((max_iter,)),
    }
    
    
    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def cond_fun(state):
        i, _, _, err, _ = state
        return jnp.logical_and(i < max_iter, err >= tol)
    
    
    def body_fun(state):
        i, d, xi, err, convergence_history = state
        
        d_new = projector_damage(OperatorFFT(xi),d_old)
        #d_new = OperatorFFT(xi)
        
        xi_new = B - ((A-A0)*d_new)
        
        
        # -------------
        # Error
        # -------------
        #err = norm2(xi_new-xi)/norm2(xi_new)
        err = norm2(A*d_new-laplacianFFT(d_new)-B)/jnp.maximum(norm2(A*d_new),1e-12)
        
        
        
        convergence_history["err"] = convergence_history["err"].at[i].set(err)
        if verbose:
            jax.debug.print(
                "Iteration {}  err = {:.5e} max(d) = {:.5e}" ,
                i,
                err,
                jnp.max(d_new)
            )
        return i + 1, d_new, xi_new, err, convergence_history
    
    
    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    init_state = (0, damage_init, xi, jnp.inf, convergence_history)
    i_final, d_final, _, err_final, convergence_history = jax.lax.while_loop(
        cond_fun, body_fun, init_state
    )
    
    return d_final, i_final, err_final
  





def damage_EAEM_solver(damage_init, A, B, A0, Gamma_0, d_old, max_iter = 1000, tol=1e-6, verbose=True):
    """
    Damage solver with EAEM methodology
    """
    
    # --------------------------
    # Helper functions
    # --------------------------
    
    # FFT operator : X/(A0 + ξ.ξ)
    def OperatorFFT(X,Gamma_0):
        return Gamma_0.mv(X)
    
    # L2 norm of the field X
    def norm2(X):
        return jnp.sqrt(jnp.sum(X**2))
    
    # Projector : d -> [d_old,1] 
    def projector_damage(d,d_old):
        return jnp.clip(d, d_old, 1.0)
    
    def laplacianFFT(X,Gamma_0):
        return Gamma_0.laplacian(X)
    
    
    
    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    alpha = 2*A0/(A+A0)
    d_aux = damage_init/alpha
    lmbda = 1.0     # Scalar parameter used to minimize the residual at each iteration
    
    convergence_history = {
        "err": jnp.zeros((max_iter,)),
    }
    
    
    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def cond_fun(state):
        i, _, _, err, _,_ = state
        return jnp.logical_and(i < max_iter, err >= tol)
    
    
    def body_fun(state):
    
        i, d_aux, _, err, convergence_history, A0 = state  
        
        alpha = 2*A0/(A+A0)     # Auxiliar parameter in [0,2]
        X = OperatorFFT(B-(2*A0*(1-alpha)*d_aux), Gamma_0(A0)) - alpha*d_aux
        Z = alpha*X + OperatorFFT(2*A0*(1-alpha)*X, Gamma_0(A0))
        lmbda = jnp.sum(X*Z)/jnp.sum(Z**2)
        d_aux_new = d_aux + lmbda*X
        
        
        
        
        # -----------------------------------------
        # New A0
        # -----------------------------------------
        
        
        """
        # ----------------
        # Same prescribed A0
        # --------------------
        A0_new = A0
        """
        
        
        """
        # --------------------------
        # Method without gradients
        # --------------------------
        A0_new = norm2(B-((2-alpha)*A0*d_aux_new))/norm2(alpha*d_aux_new)
        k0 = A0_new/A0
        alpha_new = 2*A0_new/(A+A0_new) 
        d_aux_new = (2*d_aux_new) / ( ((2-alpha_new)*k0) +alpha_new)
        """
        
        
        
        
        # ---------------------
        # eps_hat tq sig SA
        # ---------------------
        
        #A0_new = jnp.sqrt((-A0*jnp.mean(OperatorFFT(B, Gamma_0(A0))*B) + jnp.mean(A**2*d_new**2))/jnp.maximum(jnp.mean(d_new**2),1e-12))
        
        #A0_new = (jnp.mean(A**2*d_new**2) - (A0**2)*jnp.mean(d_new**2))/jnp.maximum(jnp.mean(OperatorFFT(B, Gamma_0(A0))*B),1e-12)
        
        a = jnp.mean((alpha*d_aux_new)**2)
        b = jnp.mean(B*OperatorFFT(B, Gamma_0(A0)))
        c = -jnp.mean((A0*(2-alpha)*d_aux_new)**2)
        delta = b**2 - 4*a*c
        A0_new = (-b+jnp.sqrt(delta))/(2*a)
        
        
        
        """
        # eps_hat CA
        
        B_n = A0*OperatorFFT(B, Gamma_0(A0))-B
        
        #A0_new = jnp.sqrt((jnp.mean(B_n*(B-2*A*d_new) + (B-A*d_new)**2) + 2*A0*jnp.mean(B_n*d_new)) / jnp.maximum(jnp.mean(d_new**2),1e-12))
        
        #A0_new = ((A0**2)*jnp.mean(d_new**2) - jnp.mean(B_n*(B-(2*A*d_new)) + (B-A*d_new)**2)) / jnp.maximum((2*jnp.mean(B_n*d_new)),1e-12)
        
        
        a = jnp.mean(d_new**2)
        b = -2*jnp.mean(B_n*d_new)
        c = -jnp.mean(B_n*(B-(2*A*d_new)) + (B-A*d_new)**2)
        delta = b**2 - 4*a*c
        A0_new = (-b+jnp.sqrt(delta))/(2*a)
        """
        
        
        
        # ----------------------
        # Update of d_aux_new
        # ----------------------
        k0 = A0_new/A0
        alpha_new = 2*A0_new/(A+A0_new) 
        d_aux_new = (2*d_aux_new) / ( ((2-alpha_new)*k0) + alpha_new)
        

        
        # ----------------------------
        # Projection on [d_old,1]
        # ----------------------------
        d_new = projector_damage(alpha_new*d_aux_new,d_old)
        d_aux_new = d_new / alpha_new
        
        
        
        # -------------------
        # Error
        # ----------------------
        sig_new = (A0*(2-alpha_new)*d_aux_new)
        err = norm2(sig_new-laplacianFFT(alpha_new*d_aux_new, Gamma_0(A0_new))-B)/jnp.maximum(norm2(sig_new),1e-12)
        

        convergence_history["err"] = convergence_history["err"].at[i].set(err)
        if verbose:
            jax.debug.print(
                "Iteration {}  err = {:.5e} lmbda = {:.5e} A0 = {:.5e}",
                i,
                err,
                lmbda,
                A0_new,
            )
            
            
        
        return i + 1, d_aux_new, lmbda, err, convergence_history, A0_new
    
    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    init_state = (0, d_aux, lmbda, jnp.inf, convergence_history, A0)
    i_final, d_aux_final, _, err_final, convergence_history, A0_final = jax.lax.while_loop(
        cond_fun, body_fun, init_state
    )
    
    alpha_final = 2*A0_final/(A+A0_final)     

    d_final = projector_damage(alpha_final*d_aux_final, d_old)
    
    return d_final, i_final, err_final







def AEM_expansionSolver(sigma_load, alpha, Gamma_0, max_iter=1000, tol=1e-10, verbose=True):
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
    
    def mean_field(X):
        spatial_axes = tuple(range(X.ndim - 2))
        X_mean = jnp.mean(X, axis=spatial_axes)
        X_mean_field = jnp.broadcast_to(X_mean, X.shape)
        return X_mean_field
    
    
    def apply_C0_inv(eps):
        return Gamma_0.apply_C0_inv(eps)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    #X = mean_field(eps_load) - eps_load + projection_on_D(eps_load) 
    X = apply_C0_inv(mean_field(sigma_load)) + projection_on_D(apply_C0_inv(sigma_load))
    eps_aux = jnp.zeros_like(sigma_load)
    lam = 0.0


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
        eps = alpha(eps_aux) #+ eps_load
        #Delta = ref_norm(X) / ref_norm(2*mean_field(alpha(eps_aux)-eps_aux) + mean_field(eps_load) - eps)
        Delta = ref_norm(X) / ref_norm(2*mean_field(alpha(eps_aux)-eps_aux) + apply_C0_inv(mean_field(sigma_load)) - eps)

        # AEM update
        Z = alpha(X) + 2.0 * projection_on_D(X - alpha(X)) + 2*mean_field(X-alpha(X))

        lam_new = C0_inner_product(X, Z) / C0_inner_product(Z, Z)

        eps_aux_new = eps_aux + lam_new * X
        sig = Gamma_0.apply_C0(2*eps_aux_new-alpha(eps_aux_new))-sigma_load
        #sig = Gamma_0.apply_C0(2*eps_aux_new-alpha(eps_aux_new))-sigma_load
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
                ref_norm(X) / ref_norm(sigma_load),
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
    sigma = Gamma_0.apply_C0(2.0 * eps_aux_f - alpha(eps_aux_f)) - sigma_load

    convergence_history["iterations"] = i_final

    return eps, sigma, convergence_history




# @eqx.debug.assert_max_traces(max_traces=1)
@eqx.filter_jit
def A2EM_expansionSolver(
    eps_load,
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
    
    def mean_field(X):
        spatial_axes = tuple(range(X.ndim - 2))
        X_mean = jnp.mean(X, axis=spatial_axes)
        X_mean_field = jnp.broadcast_to(X_mean, X.shape)
        return X_mean_field

    fact_safe = 1e-6     # used for homogenous solutions (eps = E_macro => k_A2EM explodes)
      
    # ------------------------
    # One AEM iteration
    # ------------------------
    def step(carry):
        it, X, eps_aux, W, scale, Delta = carry
        aX = alpha(X, scale)
        Y = projection_on_D(aX)

        Z = aX - 2.0 * Y + 2 * W / scale + 2*mean_field(X-aX)

        if step_adaptation:
            lam_new = C0_inner_product(X, Z) / C0_inner_product(Z, Z)
        else:
            lam_new = 1.0

        eps_aux_new = eps_aux + lam_new * X
        eps_new = alpha(eps_aux_new, scale) + eps_load

        sig_new = scale * Gamma_0.apply_C0(2.0 * eps_aux_new - (eps_new-eps_load))

        W_new = (1 - 2 * lam_new) * W + lam_new * scale * Y

        eps0 = Gamma_0.apply_C0_inv(sig_new)

        k_A2EM = ref_norm(eps0) / (ref_norm(2*mean_field(eps_new-eps_load-eps_aux) + mean_field(eps_load) - eps_new)+fact_safe)

        new_scale = jax.lax.select(auto_scale, k_A2EM, 1.0)

        eps_aux_new = 0.5 * (eps0 / new_scale + eps_new - eps_load)

        X_new = X - lam_new * Z + (1 / new_scale - 1 / scale) * W_new + (1 / scale - 1 / new_scale) * Gamma_0.apply_C0_inv(mean_field(sig_new))

        Delta_new = ref_norm(X_new) / (ref_norm(2*mean_field(eps_new-eps_load-eps_aux) + mean_field(eps_load) - eps_new)+fact_safe)

        if verbose:
            jax.debug.print(
                "Iteration {} ||X|| = {:.5e} Delta = {:.5e} ||sig|| = {:.5e} k_A2EM = {:.5e} lambda = {:.5e} scale = {:.5e}",
                it,
                ref_norm(X) / ref_norm(2*mean_field(eps_new-eps_load-eps_aux) + mean_field(eps_load)),
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
        mean_field(eps_load) - eps_load + projection_on_D(eps_load),
        jnp.zeros_like(eps_load),
        jnp.zeros_like(eps_load),
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
    eps = alpha(eps_aux_f, scale_f) + eps_load
    sigma = scale_f * Gamma_0.apply_C0(2.0 * eps_aux_f - (eps-eps_load))

    convergence_history = {
        "num_steps": itf,
        "Delta_final": Delta_f,
        "scale_final": scale_f,
    }

    return eps, sigma, convergence_history




# @eqx.debug.assert_max_traces(max_traces=1)
@eqx.filter_jit
def A2EM_iterativSolver(
    eps_aux_init,
    X_init,
    scale_init,
    W_init,
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


    fact_safe = 1e-6     # used for homogenous solutions (eps = E_macro => k_A2EM explodes)
      
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

        k_A2EM = ref_norm(eps0) / (ref_norm(E_macro - eps_new)+fact_safe)

        new_scale = jax.lax.select(auto_scale, k_A2EM, 1.0)

        eps_aux_new = 0.5 * (eps0 / new_scale + eps_new)

        X_new = X - lam_new * Z + (1 / new_scale - 1 / scale) * W_new

        Delta_new = ref_norm(X_new) / (ref_norm(E_macro - eps_new)+fact_safe)

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
        X_init,
        eps_aux_init,
        W_init,
        scale_init,
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

    return eps, sigma, convergence_history, scale_f, eps_aux_f, itf, Delta_f





# @eqx.debug.assert_max_traces(max_traces=1)
@eqx.filter_jit
def A2EM_iterativ_expansionSolver(
    eps_aux_init,
    X_init,
    scale_init,
    W_init,
    eps_load,
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
    
    def mean_field(X):
        spatial_axes = tuple(range(X.ndim - 2))
        X_mean = jnp.mean(X, axis=spatial_axes)
        X_mean_field = jnp.broadcast_to(X_mean, X.shape)
        return X_mean_field

    fact_safe = 1e-6     # used for homogenous solutions (eps = E_macro => k_A2EM explodes)
      
    # ------------------------
    # One AEM iteration
    # ------------------------
    def step(carry):
        it, X, eps_aux, W, scale, Delta = carry
        aX = alpha(X, scale)
        Y = projection_on_D(aX)

        Z = aX - 2.0 * Y + 2 * W / scale + 2*mean_field(X-aX)

        if step_adaptation:
            lam_new = C0_inner_product(X, Z) / C0_inner_product(Z, Z)
        else:
            lam_new = 1.0

        eps_aux_new = eps_aux + lam_new * X
        eps_new = alpha(eps_aux_new, scale) + eps_load

        sig_new = scale * Gamma_0.apply_C0(2.0 * eps_aux_new - (eps_new-eps_load))

        W_new = (1 - 2 * lam_new) * W + lam_new * scale * Y

        eps0 = Gamma_0.apply_C0_inv(sig_new)

        k_A2EM = ref_norm(eps0) / (ref_norm(2*mean_field(eps_new-eps_load-eps_aux) + mean_field(eps_load) - eps_new)+fact_safe)

        new_scale = jax.lax.select(auto_scale, k_A2EM, 1.0)

        eps_aux_new = 0.5 * (eps0 / new_scale + eps_new - eps_load)

        X_new = X - lam_new * Z + (1 / new_scale - 1 / scale) * W_new + (1 / scale - 1 / new_scale) * Gamma_0.apply_C0_inv(mean_field(sig_new))

        Delta_new = ref_norm(X_new) / (ref_norm(2*mean_field(eps_new-eps_load-eps_aux) + mean_field(eps_load) - eps_new)+fact_safe)

        if verbose:
            jax.debug.print(
                "Iteration {} ||X|| = {:.5e} Delta = {:.5e} ||sig|| = {:.5e} k_A2EM = {:.5e} lambda = {:.5e} scale = {:.5e}",
                it,
                ref_norm(X) / ref_norm(2*mean_field(eps_new-eps_load-eps_aux) + mean_field(eps_load)),
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
        X_init,
        eps_aux_init,
        W_init,
        scale_init,
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
    eps = alpha(eps_aux_f, scale_f) + eps_load
    sigma = scale_f * Gamma_0.apply_C0(2.0 * eps_aux_f - (eps-eps_load))

    convergence_history = {
        "num_steps": itf,
        "Delta_final": Delta_f,
        "scale_final": scale_f,
    }

    return eps, sigma, scale_f, eps_aux_f, convergence_history, itf, Delta_f