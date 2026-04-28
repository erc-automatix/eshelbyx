import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx


class AbstractLSOperator(lx.AbstractLinearOperator):
    def as_matrix(self):
        raise NotImplementedError("Matrix-free operator")

    def in_structure(self):
        return jax.ShapeDtypeStruct(
            shape=self.Gamma0.grid.shape + (3, 3), dtype=jnp.float64
        )

    def out_structure(self):
        return self.in_structure()

    def transpose(self):
        return self


class AEMAuxOperator(AbstractLSOperator):
    """Auxiliary Lippmann-Schwinger linear operator:
    E_macro = alpha @ eps_aux + 2 * Gamma_0 @ [C0 @ (I - alpha) @ eps_aux] = Z @ eps_aux
    """

    C0: any
    alpha: any
    Gamma0: any

    def mv(self, eps_aux):
        aeps = self.alpha(eps_aux)  # .tensor
        z = eps_aux - aeps
        z = self.Gamma0.apply_C0(z)
        z = self.Gamma0.mv(z)
        return aeps + 2.0 * z

    def transpose(self):
        raise NotImplementedError("Matrix-free operator")


@lx.is_symmetric.register(AEMAuxOperator)
def _(operator):
    return False


@eqx.filter_jit
def KrylovAEMSolver(E_macro, alpha, Gamma_0, max_iter=100, tol=1e-6, options=None):
    """A GMRES Solver of the auxiliary Lippmann-Schwinger equation:

    Z(eps*) = E

    where the `Z` operator is given by `AEMAuxOperator`

    Parameters
    ----------
    E_macro : jax.Array
        The imposed macroscopic strain (broadcast to the grid size)
    alpha : callable
        A function alpha(eps*, scale) which returns the true strain `epsilon` as a function of the auxiliary
        strain. The scale argument allows to change the amplitude of the reference medium by considering scale*C0.
    Gamma_0 : eshelbyx.green_operator.GreenOperator
        The Green operator associated with the reference medium.
    max_iter : int, optional
        Number of maximum steps of the GMRES solver, by default 100
    tol : float, optional
        Convergence tolerance, by default 1e-6
    options : dict
        Additional solver-specific dictionary e.g. `options = {"restart": 20, "stagnation_iters": 20}`

    Returns
    -------
    eps : jax.Array
        The solution strain field epsilon
    sigma : jax.Array
        The solution stress field epsilon
    info : dict
        The solver statistics dictionary
    """
    if options is None:
        options = {}
    solver = lx.GMRES(rtol=tol, atol=tol, max_steps=max_iter, **options)

    Z = AEMAuxOperator(Gamma_0.C0, alpha, Gamma_0)

    sol = lx.linear_solve(
        Z,
        E_macro.tensor,
        solver=solver,
    )

    eps_aux = sol.value
    eps = alpha(eps_aux)

    sigma = Gamma_0.apply_C0(2.0 * eps_aux - eps)

    return eps, sigma, sol.stats
