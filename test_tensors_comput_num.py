import pyvista as pv
import numpy as np
import os
from pathlib import Path
import pyvista as pv
import jax
from time import time

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

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
from jaxmat.tensors import SymmetricTensor2, IsotropicTensor4, SymmetricTensor4
from eshelbyx.krylov_solver import KrylovAEMSolver
from eshelbyx.solver import A2EMSolver, AEMSolver, MSSolver

current_folder = Path(__file__).parent.resolve()


# Contrast
chi = 10

# Solver
solver_used = A2EMSolver

# Discretization
discretization_used = WillotDiscretization


# ----------------------------------------------------------------------------
# Creation of the micro-structure : laminate
# ----------------------------------------------------------------------------


def voxelize_on_grid(mesh: pv.PolyData, grid: pv.ImageData):
    """
    Project a surface mesh onto an existing grid.
    Returns:
        inside_cells : (nx-1, ny-1, nz-1) boolean array
    """
    centers = grid.cell_centers()

    enclosed = centers.select_enclosed_points(mesh, tolerance=1e-16)
    inside = enclosed["SelectedPoints"].view(np.bool_)

    nx, ny, nz = grid.dimensions
    inside_cells = inside.reshape((nx - 1, ny - 1, nz - 1), order="F")

    return inside_cells


def make_grid(bounds, dims):
    nx, ny, nz = dims
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.origin = (xmin, ymin, zmin)
    grid.spacing = (
        (xmax - xmin) / (nx - 1),
        (ymax - ymin) / (ny - 1),
        (zmax - zmin) / (nz - 1),
    )
    return grid


N = 2  # Number of voxels in each direction
bounds = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)  # Boundaries of the domain
grid = make_grid(bounds, (N + 1, N + 1, N + 1))

box = pv.Box(bounds=bounds)
inside_box = voxelize_on_grid(box, grid)

# ------------------
# Phase labeling
# ------------------
centers = grid.cell_centers().points
phase = np.ones(centers.shape[0], dtype=np.uint8)

# Condition : z < 0 -> 1, otherwise -> 2
phase[centers[:, 2] >= 0] = 2
grid.cell_data["phase"] = phase

grid.save(
    f"test_box_{N}.vti"
)  # Laminate with normal between the materials in the z direction


# -----------------------------------------------------------------------------
# Mechanical computation using the solver
# -----------------------------------------------------------------------------

file_name = f"test_box_{N}.vti"
grid, pv_grid = load_vtk_grid(file_name, marker_name="phase")


# Define phase-wise properties based on grid.markers
def piecewise_field(field, value_map, grid):
    for index, value in value_map.items():
        field = field.at[grid.markers == index, ...].set(value)
    return field


# Material parameters
E1 = 1.0
nu1 = 0.3

E2 = E1 * chi
nu2 = 0.0


# Reference medium depending on the solver being used
if (
    solver_used == AEMSolver
    or solver_used == A2EMSolver
    or solver_used == KrylovAEMSolver
):
    # Reference medium for the AEM and A2EM solvers
    E0 = 1.0
    nu0 = 0.3

elif solver_used == MSSolver:
    # Reference medium for MS solver
    K1 = E1 / (3 * (1 - 2 * nu1))
    K2 = E2 / (3 * (1 - 2 * nu2))
    K0 = (K1 + K2) / 2

    mu1 = E1 / (2 * (1 + nu1))
    mu2 = E2 / (2 * (1 + nu2))
    mu0 = (mu1 + mu2) / 2

    E0 = (9 * K0 * mu0) / (3 * K0 + mu0)
    nu0 = (3 * K0 - 2 * mu0) / 2 / (3 * K0 + mu0)


# Reference medium stiffness tensor written using the IsotropicTensor4 class
lambda_0, mu0 = E0 * nu0 / (1 + nu0) / (1 - 2 * nu0), E0 / 2 / (1 + nu0)
C0 = IsotropicTensor4(kappa=lambda_0 + (2 / 3) * mu0, mu=mu0)


# Material parameters mapping
E_map = {1: E1, 2: E2}
nu_map = {1: nu1, 2: nu2}
E = piecewise_field(jnp.zeros(grid.shape), E_map, grid)
nu = piecewise_field(jnp.zeros(grid.shape), nu_map, grid)


def elastic_law(eps, material_params):
    # Elastic law at the material point level. Takes a `SymmetricTensor2` as strain input and outputs stress as a `SymmetricTensor2`.
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


# Material parameters mapping of reference medium for Moulinec and Suquet's solver
E0_map = {1: E0, 2: E0}
nu0_map = {1: nu0, 2: nu0}
E0 = piecewise_field(jnp.zeros(grid.shape), E0_map, grid)
nu0 = piecewise_field(jnp.zeros(grid.shape), nu0_map, grid)

material_params_0 = (E0.ravel(), nu0.ravel())

C_eps_0 = lambda eps: constitutive_law(eps, material_params_0)


# Computation and application of the alpha 4th-order tensor
def _apply_alpha(eps, material_params, C0):

    E, nu = material_params

    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    C = IsotropicTensor4(kappa=lmbda + 2 / 3 * mu, mu=mu)

    iC = (C + C0).inv
    alp = iC @ (C0 * 2.0)
    return alp @ eps  # here alp is a IsotropicTensor4


@eqx.filter_jit
def apply_alpha(eps, params, C0):
    eps_flatten = SymmetricTensor2(tensor=jnp.asarray(eps).reshape(-1, 3, 3))
    sig_tensor = jax.vmap(_apply_alpha, in_axes=(0, 0, None))(eps_flatten, params, C0)
    return sig_tensor.tensor.reshape(eps.shape)


material_params = (E.ravel(), nu.ravel())


alpha = lambda eps, scale=1.0: apply_alpha(eps, material_params, C0 * scale)
C_eps = lambda eps: constitutive_law(eps, material_params)

# Green operator
Gamma_0 = GreenOperator(grid=grid, C0=C0, discretization=discretization_used(grid))


def comput_num(E_array):
    E_macro = SymmetricTensor2(array=jnp.asarray(E_array))
    E_macro = grid.broadcast_to_grid(E_macro)  # create a field on the grid

    # Numerical computation depending on the solver being used
    if solver_used == MSSolver:
        # Moulinec and Suquet's solver
        for solver in [MSSolver]:
            tic = time()
            eps, sig, info = solver(
                E_macro, C_eps, C_eps_0, Gamma_0, tol=1e-6, max_iter=50
            )
            print("Solver info:", info)  # ["num_steps"])
            print(f"{solver.__name__} solver time", time() - tic)
            print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))
            print("\n")

    elif solver_used == AEMSolver:
        # AEM solver
        for solver in [AEMSolver]:
            tic = time()
            eps_aux, eps, sig, info = solver(
                E_macro, alpha, Gamma_0, tol=1e-6, max_iter=50
            )
            print("Solver info:", info)  # ["num_steps"])
            print(f"{solver.__name__} solver time", time() - tic)
            print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))
            print("\n")
            eps_aux_jnp = jnp.array(eps_aux)
            pv_grid.cell_data["strain_aux"] = eps_aux_jnp.reshape((-1, 3, 3))
            scale_f = 1.0

    elif solver_used == A2EMSolver:
        # A2EM solver
        for solver in [A2EMSolver]:
            tic = time()
            eps_aux, eps, sig, info, scale_f = solver(
                E_macro,
                alpha,
                Gamma_0,
                tol=1e-16,
                max_iter=200,
                auto_scale=True,
                verbose=True,
            )

            print("Solver info:", info)  # ["num_steps"])
            print(f"{solver.__name__} solver time", time() - tic)
            print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))
            print("\n")
            pv_grid.cell_data["strain_aux"] = eps_aux.reshape((-1, 3, 3))

    elif solver_used == KrylovAEMSolver:
        # Krylov-AEM solver
        for solver in [KrylovAEMSolver]:
            tic = time()
            eps, sig, info = solver(E_macro, alpha, Gamma_0, tol=1e-6, max_iter=50)
            print("Solver info:", info)  # ["num_steps"])
            print(f"{solver.__name__} solver time", time() - tic)
            print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))
            print("\n")

    sig_mean_num = jnp.mean(sig, axis=[0, 1, 2])

    return sig_mean_num


# -----------------------------------------------------------------
# Analytical solution
# -----------------------------------------------------------------
def P_func(n):
    n = jnp.asarray(n)
    delta = jnp.eye(3)

    # Construction du tenseur P_ijkl (3x3x3x3)
    term1 = jnp.einsum("i,jl,m->ijlm", n, delta, n)
    term2 = jnp.einsum("i,jm,l->ijlm", n, delta, n)
    term3 = jnp.einsum("j,il,m->ijlm", n, delta, n)
    term4 = jnp.einsum("j,im,l->ijlm", n, delta, n)

    P4 = 0.5 * (term1 + term2 + term3 + term4) - jnp.einsum("i,j,l,m->ijlm", n, n, n, n)

    # Order (6x6)
    order = jnp.array([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]])

    # P matrix 6*6
    P_array = P4[
        order[:, 0][:, None],
        order[:, 1][:, None],
        order[:, 0][None, :],
        order[:, 1][None, :],
    ]

    return P_array


def C_sym_func(lmbda, mu):
    C_sym_array = jnp.array(
        [
            [lmbda + 2 * mu, lmbda, lmbda, 0, 0, 0],
            [lmbda, lmbda + 2 * mu, lmbda, 0, 0, 0],
            [lmbda, lmbda, lmbda + 2 * mu, 0, 0, 0],
            [0, 0, 0, 2 * mu, 0, 0],
            [0, 0, 0, 0, 2 * mu, 0],
            [0, 0, 0, 0, 0, 2 * mu],
        ]
    )
    return C_sym_array


K1 = E1 / (3 * (1 - 2 * nu1))
K2 = E2 / (3 * (1 - 2 * nu2))


lmbda1 = E1 * nu1 / (1 + nu1) / (1 - 2 * nu1)
mu1 = E1 / 2 / (1 + nu1)
C1 = C_sym_func(lmbda1, mu1)

lmbda2 = E2 * nu2 / (1 + nu2) / (1 - 2 * nu2)
mu2 = E2 / 2 / (1 + nu2)
C2 = C_sym_func(lmbda2, mu2)

secu = 2  # Safety parameter
eigenvalues = jnp.array([3 * K1, 3 * K2, 2 * mu1, 2 * mu2])
lambda_param = jnp.max(eigenvalues) * secu

n = [0, 0, 1]
P = P_func(n)

I = jnp.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)


A1 = jnp.linalg.inv(P + (lambda_param * (jnp.linalg.inv(C1 - lambda_param * I))))
A2 = jnp.linalg.inv(P + (lambda_param * (jnp.linalg.inv(C2 - lambda_param * I))))
A = A1 * 0.5 + A2 * 0.5

C_lam = (I + (jnp.linalg.inv(jnp.linalg.inv(A) - P))) * lambda_param


def to_3d(sig_array):
    sig = jnp.array(
        [
            [sig_array[0], sig_array[3] / jnp.sqrt(2), sig_array[4] / jnp.sqrt(2)],
            [sig_array[3] / jnp.sqrt(2), sig_array[1], sig_array[5] / jnp.sqrt(2)],
            [sig_array[4] / jnp.sqrt(2), sig_array[5] / jnp.sqrt(2), sig_array[2]],
        ]
    )
    return sig


def comput_an(E_array):
    sig_array = jnp.einsum("ij,j->i", C_lam, jnp.array(E_array))
    sig = to_3d(sig_array)
    return sig


# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------

# Macroscopic strain E applied
E_array_xx = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

E_array_yy = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

E_array_zz = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

E_array_xy = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

E_array_xz = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

E_array_yz = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


# Numerical stress solutions
sig_num_xx = comput_num(E_array_xx)
sig_num_yy = comput_num(E_array_yy)
sig_num_zz = comput_num(E_array_zz)
sig_num_xy = comput_num(E_array_xy)
sig_num_xz = comput_num(E_array_xz)
sig_num_yz = comput_num(E_array_yz)


# Analytical stress solutions
sig_an_xx = comput_an(E_array_xx)
sig_an_yy = comput_an(E_array_yy)
sig_an_zz = comput_an(E_array_zz)
sig_an_xy = comput_an(E_array_xy)
sig_an_xz = comput_an(E_array_xz)
sig_an_yz = comput_an(E_array_yz)


# Errors
def err(sig_num, sig_an):
    diff = jnp.abs(sig_num - sig_an)
    norm_diff = jnp.sqrt(jnp.einsum("ij,ij->", diff, diff))
    norm_an = jnp.sqrt(jnp.einsum("ij,ij->", sig_an, sig_an))
    return (norm_diff / norm_an) * 100


err_xx = err(sig_num_xx, sig_an_xx)
err_yy = err(sig_num_yy, sig_an_yy)
err_zz = err(sig_num_zz, sig_an_zz)
err_xy = err(sig_num_xy, sig_an_xy)
err_xz = err(sig_num_xz, sig_an_xz)
err_yz = err(sig_num_yz, sig_an_yz)

print(f"err_xx : {err_xx:.2f} %")
print(f"err_yy : {err_yy:.2f} %")
print(f"err_zz : {err_zz:.2f} %")
print(f"err_xy : {err_xy:.2f} %")
print(f"err_xz : {err_xz:.2f} %")
print(f"err_yz : {err_yz:.2f} %")
