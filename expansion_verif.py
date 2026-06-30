# -------------------------------------------------------------------------------------------------------------------
# Verification of the expansion.py results : Hashin coated sphere with sig = P*Id in inclusion and sig = 0 in matrix
# -------------------------------------------------------------------------------------------------------------------

import os
from pathlib import Path
import pyvista as pv
import jax
from time import time

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jaxmat.materials as jm

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
from eshelbyx.solver import AEM_expansionSolver, A2EM_expansionSolver

current_folder = Path(__file__).parent.resolve()


# Discretization
discretization_used = WillotDiscretization

# Solver
solver_used = A2EM_expansionSolver

# Hybrid voxels treatment depending on the strategy
strat_hybrid = "None"


# Micro-structure created with microgen.py"
file_name = "eshelbyx/results/Hashin_num_64.vti"
grid, pv_grid = load_vtk_grid(file_name, marker_name="phase")



# --------------------
# Material parameters
# --------------------
P = 20
Rint = 0.2
Rext = 0.4

chi = 10 # contrast

lmbda_matrix = 0.0
mu_matrix = 2
E_matrix = mu_matrix*(3*lmbda_matrix + 2*mu_matrix)/(lmbda_matrix+mu_matrix)
nu_matrix = lmbda_matrix/(2*(lmbda_matrix + mu_matrix))

lmbda_inc = lmbda_matrix/chi
mu_inc = mu_matrix/chi
E_inc = mu_inc*(3*lmbda_inc + 2*mu_inc)/(lmbda_inc + mu_inc)
nu_inc = lmbda_inc/(2*(lmbda_inc+mu_inc))

lmbda_coat = lmbda_matrix*chi
mu_coat = mu_matrix*chi
E_coat = mu_coat*(3*lmbda_coat + 2*mu_coat)/(lmbda_coat + mu_coat)
nu_coat = lmbda_coat/(2*(lmbda_coat + mu_coat))


# ------------------------------------------------------------
# Jean Salençon solution for a spherical shell under pressure 
# ------------------------------------------------------------
A = P*(Rint**3)/((Rext**3)-(Rint**3))
B = 0.5*P*(Rint**3)*(Rext**3)/((Rext**3)-(Rint**3))

a = A/(3*lmbda_coat+2*mu_coat)
b = B/(2*mu_coat)

eps_int_value = a + (b/(Rint**3))
eps_ext_value = a + (b/(Rext**3))

eps_int = SymmetricTensor2(array=jnp.asarray([eps_int_value, eps_int_value, eps_int_value, 0.0, 0.0, 0.0]))
eps_ext = SymmetricTensor2(array=jnp.asarray([eps_ext_value, eps_ext_value, eps_ext_value, 0.0, 0.0, 0.0]))





# -----------------------------------------------------------------------------------------------
# Loading : eps_load = eps_int + P C_inc.inv:Id in inclusion, 0 in coating, eps_ext in matrix
# -----------------------------------------------------------------------------------------------

Id = SymmetricTensor2(array=jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
Id = grid.broadcast_to_grid(Id)
Id_flatten = SymmetricTensor2(tensor=jnp.asarray(Id).reshape((-1, 3, 3),order="F"))

C_inc = IsotropicTensor4(kappa=lmbda_inc + (2 / 3) * mu_inc, mu=mu_inc)


eps_load_inc = eps_int + (jax.vmap(lambda e: C_inc.inv @ e)(Id_flatten))*P
eps_load_matrix = eps_ext

eps_load_inc_array = jnp.asarray(eps_load_inc)
eps_load_matrix_array = jnp.asarray(eps_load_matrix)

mask_inc = (grid.markers == 1)
mask_matrix = (grid.markers == 0)
mask_coat = (grid.markers == 2)

mask_inc_flat = mask_inc.reshape(-1,order="F")
mask_matrix_flat = mask_matrix.reshape(-1,order="F")
mask_coat_flat = mask_coat.reshape(-1,order="F")


eps_load_array = jnp.zeros_like(Id_flatten)
eps_load_array = jnp.where(mask_inc_flat[:, None, None], eps_load_inc_array, eps_load_array)
eps_load_array = jnp.where(mask_matrix_flat[:, None, None], eps_load_matrix_array, eps_load_array)


N = pv_grid.dimensions[0] - 1
eps_load = eps_load_array.reshape((N, N, N, 3, 3),order="F")
pv_grid.cell_data["strain_load"] = eps_load.reshape((-1, 3, 3), order="F")




# Define phase-wise properties based on grid.markers
def piecewise_field(field, value_map, grid):
    for index, value in value_map.items():
        field = field.at[grid.markers == index, ...].set(value)
    return field

# Equivalent of "piecewise_field" function but for phases couples (concerning hybrid voxels)
def piecewise_field_couple(field, value_map, markers):
    # markers : (n_cells, 2)

    for (a, b), value in value_map.items():
        mask = (markers[:, 0] == a) & (markers[:, 1] == b)
        field = field.at[mask].set(value)

    return field




# -------------------------------------------------------
# Reference medium depending on the solver being used
# -------------------------------------------------------

K_inc = E_inc / (3 * (1 - 2 * nu_inc))
K_matrix = E_matrix / (3 * (1 - 2 * nu_matrix))
K_coat = E_coat / (3 * (1 - 2 * nu_coat))
K_min = jnp.min(jnp.array([K_inc, K_matrix, K_coat]))
K_max = jnp.max(jnp.array([K_inc, K_matrix, K_coat]))
K0 = jnp.sqrt(K_min * K_max)

mu_inc = E_inc / (2 * (1 + nu_inc))
mu_matrix = E_matrix / (2 * (1 + nu_matrix))
mu_coat = E_coat / (2 * (1 + nu_coat))
mu_min = jnp.min(jnp.array([mu_inc, mu_matrix, mu_coat]))
mu_max = jnp.max(jnp.array([mu_inc, mu_matrix, mu_coat]))
mu0 = jnp.sqrt(mu_min * mu_max)

E0 = (9 * K0 * mu0) / (3 * K0 + mu0)
nu0 = (3 * K0 - 2 * mu0) / 2 / (3 * K0 + mu0)

lambda_0, mu0 = E0 * nu0 / (1 + nu0) / (1 - 2 * nu0), E0 / 2 / (1 + nu0)
C0 = IsotropicTensor4(kappa=lambda_0 + (2 / 3) * mu0, mu=mu0)



# Material parameters mapping
E_map = {0: E_matrix, 1: E_inc, 2: E_coat}
nu_map = {0: nu_matrix, 1: nu_inc, 2 : nu_coat}
E = piecewise_field(jnp.zeros(grid.shape), E_map, grid)
nu = piecewise_field(jnp.zeros(grid.shape), nu_map, grid)


# Data extraction : hybrid booleans (voxel hybrid or not)
bool_hybrid_data = pv_grid.cell_data["bool_hybrid"]
bool_hybrid = jnp.array(bool_hybrid_data)


# Data extraction : volumic fraction (of the internal phase for hybrid voxels)
volumic_fraction_data = pv_grid.cell_data["volumic_fraction"]
volumic_fraction = jnp.array(volumic_fraction_data)


# Data extraction : phases couple (internal and external phases for hybrid voxels)
phases_couple_data = pv_grid.cell_data["phases_couple"]
phases_couple = jnp.array(phases_couple_data)


# Internal and external material parameters mapping. For hybrid voxels, internal part of the voxel is the one closer to (0,0,0)
n_cells = phases_couple.shape[0]

E_int_map = {(0, 0): E_matrix, (1, 2): E_inc, (2, 0): E_coat}
nu_int_map = {(0, 0): nu_matrix, (1, 2): nu_inc, (2, 0): nu_coat}
E_int = piecewise_field_couple(jnp.zeros(n_cells), E_int_map, phases_couple)
nu_int = piecewise_field_couple(jnp.zeros(n_cells), nu_int_map, phases_couple)

E_ext_map = {(0, 0): E_matrix, (1, 2): E_coat, (2, 0): E_matrix}
nu_ext_map = {(0, 0): nu_matrix, (1, 2): nu_coat, (2, 0): nu_matrix}
E_ext = piecewise_field_couple(jnp.zeros(n_cells), E_ext_map, phases_couple)
nu_ext = piecewise_field_couple(jnp.zeros(n_cells), nu_ext_map, phases_couple)


pv_grid.cell_data["E_int"] = E_int.flatten()
pv_grid.cell_data["E_ext"] = E_ext.flatten()



# Computation and application of the alpha 4th-order tensor depending on the hybridation strategy for hybrid voxels
def _apply_alpha(
    eps,
    material_params,
    bool_hybrid_list,
    material_params_int,
    material_params_ext,
    volumic_fraction_list,
    C0,

):

    def normal_branch(_):
        # Usual computation of alpha for non-hybrid voxels

        E, nu = material_params

        lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
        mu = E / 2 / (1 + nu)
        C = IsotropicTensor4(kappa=lmbda + 2 / 3 * mu, mu=mu)

        iC = (C + C0).inv
        alp = iC @ (C0 * 2.0)
        sig = alp @ eps  # here alp is a IsotropicTensor4
        return SymmetricTensor2(tensor=sig)

    def mean_alpha_branch(_):
        # Takes the mean of alpha for hybrid voxels

        E_int, nu_int = material_params_int
        E_ext, nu_ext = material_params_ext
        phi_int = volumic_fraction_list

        lmbda_int = E_int * nu_int / (1 + nu_int) / (1 - 2 * nu_int)
        mu_int = E_int / 2 / (1 + nu_int)
        C_int = IsotropicTensor4(kappa=lmbda_int + 2 / 3 * mu_int, mu=mu_int)
        iC_int = (C_int + C0).inv
        alp_int = iC_int @ (C0 * 2.0)

        lmbda_ext = E_ext * nu_ext / (1 + nu_ext) / (1 - 2 * nu_ext)
        mu_ext = E_ext / 2 / (1 + nu_ext)
        C_ext = IsotropicTensor4(kappa=lmbda_ext + 2 / 3 * mu_ext, mu=mu_ext)
        iC_ext = (C_ext + C0).inv
        alp_ext = iC_ext @ (C0 * 2.0)

        alp = IsotropicTensor4(
            kappa=phi_int * alp_int.kappa + (1 - phi_int) * alp_ext.kappa,
            mu=phi_int * alp_int.mu + (1 - phi_int) * alp_ext.mu,
        )
        sig = alp @ eps  # here alp is a IsotropicTensor4
        return SymmetricTensor2(tensor=sig)

    def Voigt_branch(_):
        # Takes the mean of stiffness tensor C for hybrid voxels

        E_int, nu_int = material_params_int
        E_ext, nu_ext = material_params_ext
        phi_int = volumic_fraction_list

        lmbda_int = E_int * nu_int / (1 + nu_int) / (1 - 2 * nu_int)
        mu_int = E_int / 2 / (1 + nu_int)

        lmbda_ext = E_ext * nu_ext / (1 + nu_ext) / (1 - 2 * nu_ext)
        mu_ext = E_ext / 2 / (1 + nu_ext)

        C_int = IsotropicTensor4(kappa=lmbda_int + 2 / 3 * mu_int, mu=mu_int)
        C_ext = IsotropicTensor4(kappa=lmbda_ext + 2 / 3 * mu_ext, mu=mu_ext)

        C = IsotropicTensor4(
            kappa=phi_int * C_int.kappa + (1 - phi_int) * C_ext.kappa,
            mu=phi_int * C_int.mu + (1 - phi_int) * C_ext.mu,
        )

        iC = (C + C0).inv
        alp = iC @ (C0 * 2.0)
        sig = alp @ eps  # here alp is a IsotropicTensor4
        return SymmetricTensor2(tensor=sig)

    def Reuss_branch(_):
        # Take the mean of compliance tensor S for hybrid voxels

        E_int, nu_int = material_params_int
        E_ext, nu_ext = material_params_ext
        phi_int = volumic_fraction_list

        lmbda_int = E_int * nu_int / (1 + nu_int) / (1 - 2 * nu_int)
        mu_int = E_int / 2 / (1 + nu_int)

        lmbda_ext = E_ext * nu_ext / (1 + nu_ext) / (1 - 2 * nu_ext)
        mu_ext = E_ext / 2 / (1 + nu_ext)

        C_int = IsotropicTensor4(kappa=lmbda_int + 2 / 3 * mu_int, mu=mu_int)
        S_int = C_int.inv

        C_ext = IsotropicTensor4(kappa=lmbda_ext + 2 / 3 * mu_ext, mu=mu_ext)
        S_ext = C_ext.inv

        S = IsotropicTensor4(
            kappa=phi_int * S_int.kappa + (1 - phi_int) * S_ext.kappa,
            mu=phi_int * S_int.mu + (1 - phi_int) * S_ext.mu,
        )
        C = S.inv

        iC = (C + C0).inv
        alp = iC @ (C0 * 2.0)
        sig = alp @ eps  # here alp is a IsotropicTensor4
        return SymmetricTensor2(tensor=sig)




    branch_map = {
        "alpha": mean_alpha_branch,
        "Voigt": Voigt_branch,
        "Reuss": Reuss_branch,
        "None": normal_branch,
    }

    selected_branch = branch_map[strat_hybrid]

    return jax.lax.cond(bool_hybrid_list, selected_branch, normal_branch, operand=None)


@eqx.filter_jit
def apply_alpha(
    eps, params, bool_hybrid, params_int, params_ext, phi, C0
):
    eps_flatten = SymmetricTensor2(tensor=jnp.asarray(eps).reshape((-1, 3, 3),order="F"))
    sig_tensor = jax.vmap(_apply_alpha, in_axes=(0, 0, 0, 0, 0, 0, None))(
        eps_flatten,
        params,
        bool_hybrid,
        params_int,
        params_ext,
        phi,
        C0,
    )
    return sig_tensor.tensor.reshape(eps.shape,order="F")


material_params = (E.flatten(order="F"), nu.flatten(order="F"))
bool_hybrid_list = bool_hybrid.flatten(order="F")
material_params_int = (E_int.flatten(order="F"), nu_int.flatten(order="F"))
material_params_ext = (E_ext.flatten(order="F"), nu_ext.flatten(order="F"))
volumic_fraction_list = volumic_fraction.flatten(order="F")


alpha = lambda eps, scale=1.0: apply_alpha(
    eps,
    material_params,
    bool_hybrid_list,
    material_params_int,
    material_params_ext,
    volumic_fraction_list,
    C0 * scale,
)





# Green operator
Gamma_0 = GreenOperator(grid=grid, C0=C0, discretization=discretization_used(grid))




# Numerical computation depending on the solver being used


        
if solver_used == A2EM_expansionSolver:
    # A2EM solver
    for solver in [A2EM_expansionSolver]:
        tic = time()
        eps, sig, info = solver(
            eps_load,
            alpha,
            Gamma_0,
            tol=1e-6,
            max_iter=1000,
            auto_scale=True,
            verbose=True,
        )

        print("Solver info:", info)  # ["num_steps"])
        print(f"{solver.__name__} solver time", time() - tic)
        print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))
        print("\n")


# store results
pv_grid.cell_data["strain"] = eps.reshape((-1, 3, 3),order="F")
pv_grid.cell_data["stress"] = sig.reshape((-1, 3, 3),order="F")


# Extraction of the elastic energy
W_elast_num = 0.5 * jnp.einsum("...ij,...ij->...", sig, eps)




# -----------------------------------------------------------------
# Analytical solution
# -----------------------------------------------------------------

N = pv_grid.dimensions[0]-1

indices = jnp.argwhere(
    (grid.markers == 0) |
    (grid.markers == 1) |
    (grid.markers == 2)
)

origin = jnp.array(pv_grid.origin)
spacing = jnp.array(pv_grid.spacing)

voxel_min = origin + indices * spacing

# Voxels centers
centers = voxel_min + spacing / 2

x = centers[:, 0]
y = centers[:, 1]
z = centers[:, 2]
r = jnp.maximum(jnp.sqrt(x**2 + y**2 + z**2),1e-12)
r2 = r**2

sig_coat_rr = A - (2*B/r**3)
sig_coat_tt = A + B/r**3
sig_coat_pp = sig_coat_tt

eps_coat_rr = a - (2*b/r**3)
eps_coat_tt = a + (b/r**3)
eps_coat_pp = eps_coat_tt


# Spherical -> cartesian
sig_coat_xx = sig_coat_tt + (sig_coat_rr - sig_coat_tt) * x*x/r2
sig_coat_yy = sig_coat_tt + (sig_coat_rr - sig_coat_tt) * y*y/r2
sig_coat_zz = sig_coat_tt + (sig_coat_rr - sig_coat_tt) * z*z/r2
sig_coat_xy = (sig_coat_rr - sig_coat_tt) * x*y/r2
sig_coat_xz = (sig_coat_rr - sig_coat_tt) * x*z/r2
sig_coat_yz = (sig_coat_rr - sig_coat_tt) * y*z/r2


sig_coat = SymmetricTensor2(array=jnp.stack([sig_coat_xx,
                                             sig_coat_yy,
                                             sig_coat_zz,
                                             jnp.sqrt(2) * sig_coat_xy,
                                             jnp.sqrt(2) * sig_coat_xz,
                                             jnp.sqrt(2) * sig_coat_yz,], axis=1))


eps_coat_xx = eps_coat_tt + (eps_coat_rr - eps_coat_tt) * x*x/r2
eps_coat_yy = eps_coat_tt + (eps_coat_rr - eps_coat_tt) * y*y/r2
eps_coat_zz = eps_coat_tt + (eps_coat_rr - eps_coat_tt) * z*z/r2
eps_coat_xy = (eps_coat_rr - eps_coat_tt) * x*y/r2
eps_coat_xz = (eps_coat_rr - eps_coat_tt) * x*z/r2
eps_coat_yz = (eps_coat_rr - eps_coat_tt) * y*z/r2


eps_coat = SymmetricTensor2(array=jnp.stack([eps_coat_xx,
                                             eps_coat_yy,
                                             eps_coat_zz,
                                             jnp.sqrt(2) * eps_coat_xy,
                                             jnp.sqrt(2) * eps_coat_xz,
                                             jnp.sqrt(2) * eps_coat_yz,], axis=1))




sig_an = jnp.zeros_like(Id_flatten)
sig_an = jnp.where(mask_inc_flat[:, None, None], -P*Id_flatten, sig_an)
sig_an = jnp.where(mask_matrix_flat[:, None, None], 0, sig_an)
sig_an = jnp.where(mask_coat_flat[:, None, None], sig_coat, sig_an)

sig_an = sig_an.reshape((N, N, N, 3, 3),order="F")

pv_grid.cell_data["sig_an"] = sig_an.reshape((-1, 3, 3),order="F")


eps_an = jnp.zeros_like(Id_flatten)
eps_an = jnp.where(mask_inc_flat[:, None, None], Id_flatten*eps_int_value, eps_an)
eps_an = jnp.where(mask_matrix_flat[:, None, None], eps_load_matrix, eps_an)
eps_an = jnp.where(mask_coat_flat[:, None, None], eps_coat, eps_an)

eps_an = eps_an.reshape((N, N, N, 3, 3),order="F")

pv_grid.cell_data["sig_an"] = sig_an.reshape((-1, 3, 3),order="F")
pv_grid.cell_data["eps_an"] = eps_an.reshape((-1, 3, 3),order="F")

W_elast_an = 0.5 * jnp.einsum("...ij,...ij->...", sig_an, eps_an)

pv_grid.cell_data["W_elast_num"] = W_elast_num.flatten(order="F")
pv_grid.cell_data["W_elast_an"] = W_elast_an.flatten(order="F")



err = 100*jnp.abs(W_elast_num-W_elast_an)/jnp.max(W_elast_an)
pv_grid.cell_data["err"] = err.flatten(order="F")


pv_grid.save(file_name.removesuffix(".vti") + "_results.vtk")
