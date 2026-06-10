from typing import Callable, Any
import jax
from jax import lax, tree_util
import jax.numpy as jnp
import lineax as lx


def newton_solve_jittable(
    f: Callable,
    x0: Any,
    args: Any,
    solver: lx.AbstractLinearSolver,
    *,
    tol: float = 1e-6,
    maxiter: int = 20,
    damping: float = 1.0,
    has_aux: bool = False,
    jac="fwd",
):
    flat_leaves, treedef = tree_util.tree_flatten(x0)
    leaf_shapes = [leaf.shape for leaf in flat_leaves]
    leaf_sizes = [leaf.size for leaf in flat_leaves]

    def vec_to_pytree(vec):
        parts = []
        idx = 0
        for shape, size in zip(leaf_shapes, leaf_sizes):
            part = vec[idx : idx + size].reshape(shape)
            parts.append(part)
            idx += size
        return tree_util.tree_unflatten(treedef, parts)

    def flat_f(vec):
        x = vec_to_pytree(vec)
        if has_aux:
            fx, aux = f(x, args)
        else:
            fx = f(x, args)
            aux = None
        flat_fx, _ = tree_util.tree_flatten(fx)
        return jnp.concatenate([jnp.ravel(leaf) for leaf in flat_fx]), aux

    fx0, aux0 = flat_f(jnp.concatenate([jnp.ravel(leaf) for leaf in flat_leaves]))

    def cond_fun(state):
        i, x, fx, _, _ = state
        jax.debug.print("{}", jnp.linalg.norm(fx))
        return jnp.logical_and(i < maxiter, jnp.linalg.norm(fx) > tol)

    def body_fun(state):
        i, x, fx, _, _ = state

        def f_no_aux(vec):
            return flat_f(vec)[0]

        if jac == "fwd":
            J = jax.jacfwd(f_no_aux)(x)
        elif jac == "bwd":
            J = jax.jacrev(f_no_aux)(x)
        if isinstance(jac, lx.AbstractLinearOperator):
            operator = jac
        else:
            if jac == "fwd":
                J = jax.jacfwd(f_no_aux)(x)
            elif jac == "bwd":
                J = jax.jacrev(f_no_aux)(x)
            operator = lx.MatrixLinearOperator(J)
        solution = lx.linear_solve(
            operator,
            -jnp.reshape(fx, operator.in_structure().shape),
            solver=solver,
        )
        dx_aux = SymmetricTensor2(tensor=solution.value)
        dx = solution.state.alpha(dx_aux)
        jax.debug.print("dx={}", jnp.mean(dx, axis=[0, 1, 2]))
        dx = flat_f(dx)[0]
        x_new = x + damping * dx
        fx_new, aux = flat_f(x_new)
        return (i + 1, x_new, fx_new, dx, aux)

    x0_vec = jnp.concatenate([jnp.ravel(leaf) for leaf in flat_leaves])
    init_state = (0, x0_vec, fx0, jnp.zeros_like(x0_vec), aux0)
    num_it, x_final, _, _, aux_final = lax.while_loop(cond_fun, body_fun, init_state)

    result = vec_to_pytree(x_final)
    return ((result, aux_final) if has_aux else result), {"num_steps", num_it}


import equinox as eqx
from eshelbyx.krylov_solver import AEMAuxOperator
from jaxmat.tensors import SymmetricTensor2


# FIXME: not working, need to rethink interaction with tensors
def NewtonKrylovAEMSolver(
    E_macro, C0, constitutive_law, alpha_tang, Gamma_0, max_iter=100, tol=1e-6
):
    solver = lx.GMRES(rtol=tol, atol=tol, max_steps=max_iter, restart=50)
    base_shape = E_macro.shape

    Z = AEMAuxOperator(C0, alpha_tang, Gamma_0)

    def residual(eps, E_macro):
        eps_ = SymmetricTensor2(tensor=eps.reshape(base_shape))
        sig = constitutive_law(eps_)
        z = sig - Gamma_0.apply_C0(eps_)
        z_ = Gamma_0.mv(z)
        # (
        #     SymmetricTensor2(tensor=Gamma_0.mv(z))
        #     .flatten_batch_dims()
        #     .tensor.reshape((-1, 9))
        # )
        X = eps_ + z_ - E_macro
        return X, sig

    @eqx.filter_jit
    def newton_solve(eps0, E_macro, Z, tol, max_iter):

        def cond_fun(state):
            it, _, res, _, _ = state
            return jnp.logical_and(res > tol, it < max_iter)

        def body_fun(state):
            it, eps, _, _, stats = state

            X, sig = residual(eps.tensor, E_macro)  # X is tensor
            sol = lx.linear_solve(
                Z,
                -X.tensor,
                solver=solver,
            )
            deps_aux = SymmetricTensor2(tensor=sol.value)
            eps_new = eps + alpha_tang(deps_aux)

            R, sig = residual(eps_new.tensor, E_macro)
            res_new = jnp.linalg.norm(R)
            stats["linear_solver_iterations"] = (
                stats["linear_solver_iterations"].at[it].set(sol.stats["num_steps"])
            )
            stats["num_steps"] = it + 1
            stats["residual"] = stats["residual"].at[it].set(res_new)
            # jax.debug.print("Iteration {} ||res||={}", it, res_new)
            return it + 1, eps_new, res_new, sig, stats

        it0 = 0
        res0 = 1.0
        stats = {
            "num_steps": 0,
            "linear_solver_iterations": jnp.zeros((max_iter,)),
            "residual": jnp.zeros((max_iter,)),
        }
        state0 = (it0, eps0, res0, eps0, stats)

        it_f, eps_f, res_f, sig_f, stats = lax.while_loop(cond_fun, body_fun, state0)

        return eps_f, sig_f, stats

    eps = E_macro * 0.0

    eps_f, sig_f, stats = newton_solve(eps, E_macro, Z, tol, max_iter)
    stats["linear_solver_iterations"] = stats["linear_solver_iterations"][
        : stats["num_steps"]
    ]
    stats["residual"] = stats["residual"][: stats["num_steps"]]
    return eps_f, sig_f, stats
