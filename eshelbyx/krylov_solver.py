import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
from eshelbyx.tensors import SymmetricTensor2


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


class LSOperator(AbstractLSOperator):
    """Lippmann-Schwinger linear operator:
    E_macro = eps + Gamma_0 @ [(C - C0) @ eps] = L @ eps
    """

    C0: any
    C: any
    Gamma0: any

    def mv(self, eps):
        sig = self.C(SymmetricTensor2(tensor=eps)).tensor
        z = sig - self.Gamma0.apply_C0(eps)
        z = self.Gamma0.mv(z)
        return eps + z


class DeltaC(AbstractLSOperator):
    C: any
    Gamma0: any

    def mv(self, eps):
        sig = self.C(SymmetricTensor2(tensor=eps)).tensor
        return sig - self.Gamma0.apply_C0(eps)


class CGOperator(AbstractLSOperator):
    """Lippmann-Schwinger linear operator:
    E_macro = (Delta C)^{-1}*tau + Gamma_0 @ [tau] = L @ tau
    """

    C0: any
    ideltaC: any
    Gamma0: any

    def mv(self, tau):
        z = self.Gamma0.mv(tau)
        z = self.ideltaC(SymmetricTensor2(tensor=tau)).tensor + z
        return z


class AEMAuxOperator(AbstractLSOperator):
    """Auxiliary Lippmann-Schwinger linear operator:
    E_macro = alpha @ eps_aux + 2 * Gamma_0 @ [C0 @ (I - alpha) @ eps_aux] = Z @ eps_aux
    """

    C0: any
    alpha: any
    Gamma0: any

    def mv(self, eps_aux):
        aeps = self.alpha(SymmetricTensor2(tensor=eps_aux)).tensor
        z = eps_aux - aeps
        z = self.Gamma0.apply_C0(z)
        z = self.Gamma0.mv(z)
        return aeps + 2.0 * z

    def transpose(self):
        raise NotImplementedError("Matrix-free operator")


@lx.is_symmetric.register(LSOperator)
def _(operator):
    return False


@lx.is_symmetric.register(AEMAuxOperator)
def _(operator):
    return False


@lx.is_symmetric.register(DeltaC)
def _(operator):
    return True


@lx.is_positive_semidefinite.register(DeltaC)
def _(operator):
    return True


@lx.is_symmetric.register(CGOperator)
def _(operator):
    return True


@lx.is_negative_semidefinite.register(CGOperator)
def _(operator):
    return False


@lx.is_positive_semidefinite.register(CGOperator)
def _(operator):
    return True


@eqx.filter_jit
def KrylovLSSolver(
    E_macro, C0, constitutive_law, alpha, Gamma_0, max_iter=100, tol=1e-6
):
    L = LSOperator(C0, constitutive_law, Gamma_0)
    solver = lx.GMRES(rtol=tol, atol=tol, max_steps=max_iter)

    sol = lx.linear_solve(
        L,
        E_macro.tensor,
        solver=solver,
        # options=lx.SolveOptions(throw=False),
    )

    eps = SymmetricTensor2(tensor=sol.value)

    X = E_macro - L.mv(eps)

    # Delta = ref_norm(X) / ref_norm(E_macro)

    sig = constitutive_law(eps)
    # energy = 0.5 * jnp.mean(jnp.einsum("...ij,...ij->...", sig, eps))

    sigma = SymmetricTensor2(tensor=sig)
    eps = SymmetricTensor2(tensor=eps)

    return eps, sigma, sol.stats


@eqx.filter_jit
def KrylovAEMSolver(
    E_macro, C0, constitutive_law, alpha, Gamma_0, max_iter=100, tol=1e-6
):
    Z = AEMAuxOperator(C0, alpha, Gamma_0)
    solver = lx.GMRES(rtol=tol, atol=tol, max_steps=max_iter)
    sol = lx.linear_solve(
        Z,
        E_macro.tensor,
        solver=solver,
        # options=lx.SolveOptions(throw=False),
    )

    eps_aux = SymmetricTensor2(tensor=sol.value)
    eps = alpha(eps_aux)
    # X = E_macro - Z.mv(eps_aux)

    # Delta = ref_norm(X) / ref_norm(E_macro)

    sig = Gamma_0.apply_C0(2.0 * eps_aux - eps)

    sigma = SymmetricTensor2(tensor=sig)
    eps = SymmetricTensor2(tensor=eps)

    return eps, sigma, sol.stats


@eqx.filter_jit
def CGLSSolver(E_macro, C0, constitutive_law, ideltaC, Gamma_0, max_iter=100, tol=1e-6):
    L = CGOperator(C0, ideltaC, Gamma_0)
    solver = lx.CG(rtol=tol, atol=tol, max_steps=max_iter)

    sol = lx.linear_solve(
        L,
        E_macro.tensor,
        solver=solver,
        options={
            "preconditioner": DeltaC(constitutive_law, Gamma_0),
            # "y0": Gamma_0.apply_C0(E_macro),
        },
        throw=False,
        # options=lx.SolveOptions(throw=False),
    )

    tau = SymmetricTensor2(tensor=sol.value)
    eps = ideltaC(tau)
    sig = tau + Gamma_0.apply_C0(eps)

    sigma = SymmetricTensor2(tensor=sig)
    eps = SymmetricTensor2(tensor=eps)

    return eps, sigma, sol.stats
