# %%
import os
from pathlib import Path
import pyvista as pv
import jax, h5py
from time import time

jax.config.update("jax_enable_x64", True)
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
from eshelbyx.solver import AEM_solver

from eshelbyx.grid import Grid
from eshelbyx.tensors import (
    SymmetricTensor2,
    IsotropicTensor4,
)

current_folder = Path(__file__).parent.resolve()


with h5py.File(current_folder / "RVE3D-80.mat", "r") as f:
    rd_fiber = f["mesh3D"][()].T


grid = Grid(rd_fiber, size=jnp.array([1.0, 1.0, 1.0]))

lambda_0, mu0 = 1.0, 1.0

C0 = IsotropicTensor4(lambda_0 + 2 / 3 * mu0, mu0)

E_macro = SymmetricTensor2(array=jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
# E_macro = SymmetricTensor2(array=jnp.eye(6))


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


field = jnp.zeros(grid.shape + (3, 3))
tau = SymmetricTensor2(
    tensor=piecewise_field(field, {1: jnp.eye(3), 0: jnp.zeros((3, 3))}, grid)
)

E_matrix, E_incl = 1.0, jnp.inf
nu_matrix, nu_incl = 0.2, 0.3
E_map = {0: E_matrix, 1: E_incl}
nu_map = {0: nu_matrix, 1: nu_incl}
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
    return sig_flatten


def _apply_alpha(eps, material_params, C0):
    E, nu = material_params
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    C = IsotropicTensor4(lmbda + 2 / 3 * mu, mu)
    alpha = C0 @ (C + C0).inv  # C0 @ (C + C0).inv * 2.0
    sig = alpha.eval_jvp(eps)
    return SymmetricTensor2(tensor=sig)


@eqx.filter_jit
def apply_alpha(eps, params, C0):
    eps_flatten = eps.flatten_batch_dims()
    sig_flatten = jax.vmap(_apply_alpha, in_axes=(0, 0, None))(eps_flatten, params, C0)
    return SymmetricTensor2(tensor=sig_flatten.tensor.reshape(eps.tensor.shape))


Gamma_0 = GreenOperator(grid=grid, C0=C0, discretization=WillotDiscretization(grid))

material_params = (E.ravel(), nu.ravel())
eps, sig, info = AEM_solver(
    broadcast_to_grid(E_macro, grid),
    C0,
    lambda eps: apply_alpha(eps, material_params, C0),
    Gamma_0,
    tol=1e-6,
)

# tic = time()
# print("Gamma_0 construction:", time() - tic)
# material_params = (E.ravel(), nu.ravel())
# eps = -Gamma_0.mv(tau)
# for i in range(5):
#     tic = time()
#     eps = -Gamma_0.mv(tau)
#     print(f"Gamma_0 vp {i}:", time() - tic)

#     tic = time()
#     # sig = constitutive_law(eps, material_params)
#     # sig = Gamma_0.apply_C0(eps)
#     sig = apply_alpha(eps, material_params, C0)
#     print(f"Constitutive update {i}:", time() - tic)
# eps2 = -jnp.tensordot(tau.array, T, axes=(-1, 0))
# eps2 = -jnp.einsum("ijklm,klmj->klmi", T, tau.array)
# print(tau.array)

image = pv.ImageData()
# Set the grid dimensions: shape + 1 because we want to inject our values on
#   the CELL data
image.dimensions = (s + 1 for s in grid.shape)
# Edit the spatial reference
image.origin = (0, 0, 0)  # The bottom left corner of the data set
image.spacing = tuple(
    grid.size[i] / grid.shape[i] for i in range(3)
)  # These are the cell sizes along each axis
image.cell_data["strain"] = eps.array.reshape((-1, 6))
image.cell_data["stress"] = sig.array.reshape((-1, 6))
# C = IsotropicTensor4()
image.save("RVE_DATA.vtk")
