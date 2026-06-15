# --------------------------------------------------------------------
# Isotropic expansion of a spherical inclusion within a cubic matrix
# --------------------------------------------------------------------

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
from eshelbyx.solver import AEM_expansionSolver, A2EM_expansionSolver, A2EM_iterativ_expansionSolver

current_folder = Path(__file__).parent.resolve()


# Discretization
discretization_used = WillotDiscretization

# Solver
solver_used = A2EM_iterativ_expansionSolver

# Hybrid voxels treatment depending on the strategy
strat_hybrid = "None"


# Micro-structure created with microgen.py
file_name = "eshelbyx/results/2phases_expansion_num_2D_25.vti"
grid, pv_grid = load_vtk_grid(file_name, marker_name="phase")



# number of increments for the loading
n_incr = 10


# --------------------
# Material parameters
# --------------------

chi = 2 # contrast

lmbda_matrix = 0.0
mu_matrix = 2
E_matrix = mu_matrix*(3*lmbda_matrix + 2*mu_matrix)/(lmbda_matrix+mu_matrix)
nu_matrix = lmbda_matrix/(2*(lmbda_matrix + mu_matrix))

lmbda_inc = lmbda_matrix*chi
mu_inc = mu_matrix*chi
E_inc = mu_inc*(3*lmbda_inc + 2*mu_inc)/(lmbda_inc + mu_inc)
nu_inc = lmbda_inc/(2*(lmbda_inc+mu_inc))


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





# ------------------------------------------------------
# Loading : eps_load = beta*Id in inclusion, 0 in the matrix
# ------------------------------------------------------

bool_load = (grid.markers == 1) # |(grid.markers == 0)

beta = 0.1
Id = SymmetricTensor2(array=jnp.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))

eps_load_final = grid.broadcast_to_grid(Id * beta)
eps_load_final = eps_load_final * bool_load[..., None] 
eps_load_final = jnp.asarray(eps_load_final)

pv_grid.cell_data["strain_load_final"] = eps_load_final.reshape((-1, 3, 3),order="F")

Delta_eps_load = eps_load_final/n_incr


# -----------------------------------------------------
# Reference medium depending on the solver being used
# -----------------------------------------------------

K_inc = E_inc / (3 * (1 - 2 * nu_inc))
K_matrix = E_matrix / (3 * (1 - 2 * nu_matrix))
K0 = jnp.sqrt(K_inc * K_matrix)

mu_inc = E_inc / (2 * (1 + nu_inc))
mu_matrix = E_matrix / (2 * (1 + nu_matrix))
mu0 = jnp.sqrt(mu_inc * mu_matrix)

E0 = (9 * K0 * mu0) / (3 * K0 + mu0)
nu0 = (3 * K0 - 2 * mu0) / 2 / (3 * K0 + mu0)

lambda_0, mu0 = E0 * nu0 / (1 + nu0) / (1 - 2 * nu0), E0 / 2 / (1 + nu0)
C0 = IsotropicTensor4(kappa=lambda_0 + (2 / 3) * mu0, mu=mu0)



# Material parameters mapping
E_map = {0: E_matrix, 1: E_inc}
nu_map = {0: nu_matrix, 1: nu_inc}
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

E_int_map = {(0, 0): E_matrix, (1, 0): E_inc}
nu_int_map = {(0, 0): nu_matrix, (1, 0): nu_inc}
E_int = piecewise_field_couple(jnp.zeros(n_cells), E_int_map, phases_couple)
nu_int = piecewise_field_couple(jnp.zeros(n_cells), nu_int_map, phases_couple)

E_ext_map = {(0, 0): E_matrix, (1, 0): E_matrix}
nu_ext_map = {(0, 0): nu_matrix, (1, 0): nu_matrix}
E_ext = piecewise_field_couple(jnp.zeros(n_cells), E_ext_map, phases_couple)
nu_ext = piecewise_field_couple(jnp.zeros(n_cells), nu_ext_map, phases_couple)


pv_grid.cell_data["E_int"] = E_int.flatten(order="F")
pv_grid.cell_data["E_ext"] = E_ext.flatten(order="F")




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
    return sig_tensor.tensor.reshape((eps.shape),order="F")


alpha = lambda eps, scale=1.0: apply_alpha(
    eps,
    material_params,
    bool_hybrid_list,
    material_params_int,
    material_params_ext,
    volumic_fraction_list,
    C0 * scale,
)


material_params = (E.flatten(order="F"), nu.flatten(order="F"))
bool_hybrid_list = bool_hybrid.flatten(order="F")
material_params_int = (E_int.flatten(order="F"), nu_int.flatten(order="F"))
material_params_ext = (E_ext.flatten(order="F"), nu_ext.flatten(order="F"))
volumic_fraction_list = volumic_fraction.flatten(order="F")




# Green operator
Gamma_0 = GreenOperator(grid=grid, C0=C0, discretization=discretization_used(grid))




# ---------------------------------------------------
# Definition of the increment loop
# ---------------------------------------------------

def mean_field(X):
    spatial_axes = tuple(range(X.ndim - 2))
    X_mean = jnp.mean(X, axis=spatial_axes)
    X_mean_field = jnp.broadcast_to(X_mean, X.shape)
    return X_mean_field

# Convergence condition
def cond_fun_incr(state):
    incr, eps, sig, eps_aux, scale, eps_load = state
    return incr<n_incr



# Body function of the increment loop
def body_fun_incr(state):
    incr, eps, sig, eps_aux, scale, eps_load = state
    

    # Incremental loading
    eps_load = eps_load + Delta_eps_load
    eps = eps + Delta_eps_load
    
    X = -(1/scale)*Gamma_0.apply_C0_inv(mean_field(sig)) + mean_field(eps) - eps - (1/scale)*Gamma_0.mv(sig) + Gamma_0.mv(Gamma_0.apply_C0(eps))
    W = -Gamma_0.mv(sig)
    

    # Numerical computation depending on the solver being used
    
    # AEM expansion solver
    if solver_used == AEM_expansionSolver:
        for solver in [AEM_expansionSolver]:
            tic = time()
            eps, sig, info = solver(
                eps_load, alpha, Gamma_0, tol=1e-6, max_iter=1000
            )
            print("Solver info:", info)  # ["num_steps"])
            print(f"{solver.__name__} solver time", time() - tic)
            print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))
            print("\n")
           
            
    elif solver_used == A2EM_iterativ_expansionSolver:
        # A2EM solver
        for solver in [A2EM_iterativ_expansionSolver]:
            tic = time()
            eps, sig, scale, eps_aux, info = solver(
                eps_aux,
                X,
                scale,
                W,
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
            
    # Print increment number
    jax.debug.print("\nIncrement {}",incr+1,)
            
    return incr+1, eps, sig, eps_aux, scale, eps_load
    


# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------
incr = 0
eps_load = jnp.zeros_like(eps_load_final)
eps = jnp.zeros_like(eps_load_final)
sig = jnp.zeros_like(eps_load_final)
scale = 1.0
eps_aux = jnp.zeros_like(eps_load_final)



# ----------------------------------------------------------------
# Run increment loop
# ----------------------------------------------------------------

init_state_incr = incr, eps, sig, eps_aux, scale, eps_load

incr_f, eps_f, sig_f, eps_aux_f, scale_f, eps_load_f = jax.lax.while_loop(
    cond_fun_incr, body_fun_incr, init_state_incr
)




# store results
pv_grid.cell_data["strain"] = eps_f.reshape((-1, 3, 3),order="F")
pv_grid.cell_data["stress"] = sig_f.reshape((-1, 3, 3),order="F")


# Extraction of the xx component of the stress
sigma_xx = sig_f[..., 0, 0].flatten(order="F")
pv_grid.cell_data["stress_xx"] = sigma_xx




pv_grid.save(file_name.removesuffix(".vti") + "_results.vtk")
