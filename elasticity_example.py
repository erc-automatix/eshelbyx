# %%
import os
from pathlib import Path
import pyvista as pv
import jax
from time import time

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

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

from eshelbyx.grid import load_vtk_grid
from eshelbyx.tensors import SymmetricTensor2, IsotropicTensor4
from eshelbyx.krylov_solver import KrylovAEMSolver
from eshelbyx.solver import AEM_solver, A2EM_solver

current_folder = Path(__file__).parent.resolve()

file_name = "3_phase_microstructure_32.vti"
grid, pv_grid = load_vtk_grid(file_name, marker_name="phase")

# Reference medium
E0 = 1.0
nu0 = 0.3
lambda_0, mu0 = E0 * nu0 / (1 + nu0) / (1 - 2 * nu0), E0 / 2 / (1 + nu0)
C0 = IsotropicTensor4(lambda_0 + 2 / 3 * mu0, mu0)


# Define phase-wise properties based on grid.markers
def piecewise_field(field, value_map, grid):
    for index, value in value_map.items():
        field = field.at[grid.markers == index, ...].set(value)
    return field


chi = 10000.0
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
    sig = lmbda * jnp.linalg.trace(eps) * jnp.eye(3) + 2 * mu * eps
    return SymmetricTensor2(tensor=sig)


@eqx.filter_jit
def constitutive_law(eps, params):
    eps_flatten = eps.flatten_batch_dims()
    sig_flatten = jax.vmap(elastic_law)(eps_flatten, params)
    return SymmetricTensor2(tensor=sig_flatten.tensor.reshape(eps.tensor.shape))


def _apply_alpha(eps, material_params, C0):
    E, nu = material_params
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    C = IsotropicTensor4(lmbda + 2 / 3 * mu, mu)

    iC = (C + C0).inv
    sig = jnp.einsum("ijkl,klmn,mn->ij", iC, 2 * C0, eps)
    return SymmetricTensor2(tensor=sig)


@eqx.filter_jit
def apply_alpha(eps, params, C0):
    eps_flatten = eps.flatten_batch_dims()
    sig_flatten = jax.vmap(_apply_alpha, in_axes=(0, 0, None))(eps_flatten, params, C0)
    return SymmetricTensor2(tensor=sig_flatten.tensor.reshape(eps.tensor.shape))


material_params = (E.ravel(), nu.ravel())


alpha = lambda eps, scale=1.0: apply_alpha(eps, material_params, C0 * scale)
C_eps = lambda eps: constitutive_law(eps, material_params)


E_macro = SymmetricTensor2(array=jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
E_macro = grid.broadcast_to_grid(E_macro)  # create a field on the grid

Gamma_0 = GreenOperator(grid=grid, C0=C0, discretization=WillotDiscretization(grid))


for solver in [A2EM_solver] * 2:
    tic = time()
    eps, sig, info = solver(E_macro, alpha, Gamma_0, tol=1e-6, max_iter=200)
    print("Iterations", info["num_steps"])
    print(f"{solver.__name__} solver time", time() - tic)
    print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))

    print("\n")

# store results
pv_grid.cell_data["strain"] = eps.array.reshape((-1, 6))
pv_grid.cell_data["stress"] = sig.array.reshape((-1, 6))
pv_grid.save(file_name.removesuffix(".vti") + "_results.vtk")
