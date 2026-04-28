# %%
import os
from pathlib import Path
import pyvista as pv
import jax, h5py
from time import time

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
# jax.config.update("jax_disable_jit", True)

import jax.numpy as jnp
import equinox as eqx
from eshelbyx.green_operator import (
    GreenOperator,
)
from eshelbyx.discretization import (
    MoulinecSuquetDiscretization,
    WillotDiscretization,
    SchneiderDiscretization,
)
from eshelbyx.solver import AEMSolver

from eshelbyx.grid import Grid, load_vtk_grid
from eshelbyx.utils import flatten_batch_dims, reshape_batch_dims
from jaxmat.tensors import SymmetricTensor2, IsotropicTensor4, SymmetricTensor4

current_folder = Path(__file__).parent.resolve()


# with h5py.File(current_folder / "RVE3D-80.mat", "r") as f:
#     rd_fiber = f["mesh3D"][()].T
# grid = Grid(rd_fiber, size=jnp.array([1.0, 1.0, 1.0]))

file_name = "3_phase_microstructure_32.vti"
grid, pv_grid = load_vtk_grid(file_name, marker_name="phase")

E0 = 1.0
nu0 = 0.3
lambda_0, mu0 = E0 * nu0 / (1 + nu0) / (1 - 2 * nu0), E0 / 2 / (1 + nu0)

C0 = IsotropicTensor4(kappa=lambda_0 + 2 / 3 * mu0, mu=mu0)

E_macro = SymmetricTensor2(array=jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def broadcast_to_grid(tensor, grid):
    return tensor.__class__(
        array=jnp.broadcast_to(
            tensor.array,
            grid.shape + tensor.array.shape,
        )
    )


def piecewise_field(field, value_map, grid):
    for index, value in value_map.items():
        field = field.at[grid.markers == index, ...].set(value)
    return field


chi = 1e6
Eref = 1.0
E_matrix, E_incl, E_pore = Eref, Eref * chi, Eref / chi
nu_matrix, nu_incl, nu_pore = 0.1, 0.2, 0.35
E_map = {0: E_matrix, 1: E_pore, 2: E_incl}
nu_map = {0: nu_matrix, 1: nu_pore, 2: nu_incl}
E = piecewise_field(jnp.zeros(grid.shape), E_map, grid)
nu = piecewise_field(jnp.zeros(grid.shape), nu_map, grid)


def elastic_law(eps, material_params):
    E, nu = material_params
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    e = eps + eps @ eps / 2
    sig = lmbda * jnp.linalg.trace(e) * jnp.eye(3) + 2 * mu * e
    return SymmetricTensor2(tensor=sig)


@eqx.filter_jit
def constitutive_law(eps, params):
    eps_flatten = flatten_batch_dims(eps)
    sig_flatten = jax.vmap(elastic_law)(eps_flatten, params)
    return reshape_batch_dims(sig_flatten, grid.shape)


def _apply_alpha(eps, material_params, C0):
    # E, nu = material_params
    # lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    # mu = E / 2 / (1 + nu)
    # C_tang = IsotropicTensor4(lmbda + 2 / 3 * mu, mu)

    def elastic_law_v(eps_v, material_params):
        eps = SymmetricTensor2(array=eps_v)
        return elastic_law(eps, material_params).array

    C_ = SymmetricTensor4(
        array=eqx.filter_jacfwd(elastic_law_v)(eps.array, material_params)
    )
    # kappa_tang = jnp.einsum("ijkl,klij", C_, SymmetricTensor4.J()) / 3.0
    # mu_tang = jnp.einsum("ijkl,klij", C_, SymmetricTensor4.K()) / 2.0
    # C_tang = IsotropicTensor4(kappa_tang, mu_tang)
    C0_ = C0.to_symmetric()
    C_tang = C_
    # jax.debug.print("Ct = {}", C_.shape)
    alpha = ((C_tang + C0_).inv @ C0_) * 2.0
    sig = alpha @ eps
    return SymmetricTensor2(tensor=sig)


@eqx.filter_jit
def apply_alpha(eps, params, C0):
    eps_flatten = SymmetricTensor2(tensor=jnp.asarray(eps).reshape(-1, 3, 3))
    sig_tensor = jax.vmap(_apply_alpha, in_axes=(0, 0, None))(eps_flatten, params, C0)
    return sig_tensor.tensor.reshape(eps.shape)


Gamma_0 = GreenOperator(grid=grid, C0=C0, discretization=WillotDiscretization(grid))

material_params = (E.ravel(), nu.ravel())


E_macro_list = [
    SymmetricTensor2(array=jnp.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])),
    #     SymmetricTensor2(array=jnp.asarray([1.0, -0.5, -0.5, 0.0, 0.0, 0.0])),
    #     SymmetricTensor2(array=jnp.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])),
]
alpha = lambda eps: apply_alpha(eps, material_params, C0)
C_eps = lambda eps: constitutive_law(eps, material_params)

from jax import profiler

options = jax.profiler.ProfileOptions()
# options.python_tracer_level = 1
# options.host_tracer_level = 1
profiler.start_trace("/tmp/jax_trace")
from eshelbyx.krylov_solver import KrylovAEMSolver
from eshelbyx.nonlinear_krylov_solver import NewtonKrylovAEMSolver

for E_macro in E_macro_list:
    for solver in [KrylovAEMSolver]:
        tic = time()
        eps, sig, info = solver(
            broadcast_to_grid(E_macro, grid), alpha, Gamma_0, tol=1e-6
        )
        print("Iterations", info["num_steps"])
        print(f"{solver.__name__} solver time", time() - tic)
        print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))

    for solver in [NewtonKrylovAEMSolver]:
        tic = time()
        eps, sig, info = solver(
            broadcast_to_grid(E_macro, grid), C0, C_eps, alpha, Gamma_0, tol=1e-6
        )
        it = info["num_steps"]
        print("Iterations", it)
        print("Linear solver iterations", info["linear_solver_iterations"])
        print("Convergence", info["residual"])
        print(f"{solver.__name__} solver time", time() - tic)
        print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))

    # for solver in [NewtonKrylovAEMSolver]:
    #     tic = time()
    #     eps, sig, info = solver(
    #         broadcast_to_grid(E_macro, grid), C0, C_eps, alpha, Gamma_0, tol=1e-6
    #     )
    #     print("Iterations", info["num_steps"])
    #     print(f"{solver.__name__} solver time", time() - tic)
    #     print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))
    print("\n")
profiler.stop_trace()
# print("Mean alpha_strain", jnp.mean(alp_eps, axis=[0, 1, 2]))
# print("Mean stress", jnp.mean(sig, axis=0))  # jnp.mean(sig, axis=[0, 1, 2]))

pv_grid.cell_data["strain"] = eps.array.reshape((-1, 6))
pv_grid.cell_data["stress"] = sig.array.reshape((-1, 6))
pv_grid.save(file_name.removesuffix(".vti") + "_results.vtk")
