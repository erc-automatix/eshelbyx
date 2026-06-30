# %%
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
from eshelbyx.solver import A2EMSolver, AEMSolver, MSSolver, A2EM_iterativSolver

current_folder = Path(__file__).parent.resolve()


# Hybrid voxels treatment depending on the strategy
strat_hybrid = "None"

# Solver
solver_used = A2EM_iterativSolver

# Discretization
discretization_used = WillotDiscretization


# Micro-structure created with microgen.py"
file_name = "eshelbyx/results/homogenization_shear_64.vti"
grid, pv_grid = load_vtk_grid(file_name, marker_name="phase")


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




# -----------------------------------------------------
# Material parameters
# -----------------------------------------------------
mu_matrix = 1.0

chi = 1e8
mu_NE = chi
mu_NO = 1/chi
mu_SO = chi
mu_SE = 1/chi

lmbda = 0.0

E_matrix = mu_matrix*(3*lmbda + 2*mu_matrix)/(lmbda+mu_matrix)
nu_matrix = lmbda/(2*(lmbda + mu_matrix))

E_NE = mu_NE*(3*lmbda + 2*mu_NE)/(lmbda+mu_NE)
nu_NE = lmbda/(2*(lmbda + mu_NE))

E_NO = mu_NO*(3*lmbda + 2*mu_NO)/(lmbda+mu_NO)
nu_NO = lmbda/(2*(lmbda + mu_NO))

E_SO = mu_SO*(3*lmbda + 2*mu_SO)/(lmbda+mu_SO)
nu_SO = lmbda/(2*(lmbda + mu_SO))

E_SE = mu_SE*(3*lmbda + 2*mu_SE)/(lmbda+mu_SE)
nu_SE = lmbda/(2*(lmbda + mu_SE))



# Reference medium 
E0 = 1.0
nu0 = 0.3

lambda_0, mu0 = E0 * nu0 / (1 + nu0) / (1 - 2 * nu0), E0 / 2 / (1 + nu0)
C0 = IsotropicTensor4(kappa=lambda_0 + (2 / 3) * mu0, mu=mu0)




# Material parameters mapping
E_map = {0: E_matrix, 1: E_NE, 2: E_NO, 3: E_SO, 4: E_SE}
nu_map = {0: nu_matrix, 1: nu_NE, 2: nu_NO, 3: nu_SO, 4: nu_SE}
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

E_int_map = {(0, 0): E_matrix, (1,0): E_NE, (2,0): E_NO, (3,0): E_SO, (4,0): E_SE}
nu_int_map = {(0, 0): nu_matrix, (1,0): nu_NE, (2,0): nu_NO, (3,0): nu_SO, (4,0): nu_SE}
E_int = piecewise_field_couple(jnp.zeros(n_cells), E_int_map, phases_couple)
nu_int = piecewise_field_couple(jnp.zeros(n_cells), nu_int_map, phases_couple)

E_ext_map = {(0, 0): E_matrix, (1,0): E_matrix, (2,0): E_matrix, (3,0): E_matrix, (4,0): E_matrix}
nu_ext_map = {(0, 0): nu_matrix, (1,0): nu_matrix, (2,0): nu_matrix, (3,0): nu_matrix, (4,0): nu_matrix}
E_ext = piecewise_field_couple(jnp.zeros(n_cells), E_int_map, phases_couple)
nu_ext = piecewise_field_couple(jnp.zeros(n_cells), nu_int_map, phases_couple)


pv_grid.cell_data["E_int"] = E_int.flatten()
pv_grid.cell_data["E_ext"] = E_ext.flatten()


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
    return SymmetricTensor2(tensor=sig_flatten.tensor.reshape(eps.tensor.shape,order="F"))




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
C_eps = lambda eps: constitutive_law(eps, material_params)



n_incr = 1

# Final macroscopic strain E applied : E = (Exx, Eyy, Ezz, sqrt(2)Exy, sqrt(2)Exz, sqrt(2)Eyz)
E_macro_final = SymmetricTensor2(array=jnp.asarray([0.0, 0.0, 0.0, 0.0, jnp.sqrt(2), 0.0]))
E_macro_final = grid.broadcast_to_grid(E_macro_final)  # create a field on the grid

Delta_E_macro = E_macro_final/n_incr


# Green operator
Gamma_0 = GreenOperator(grid=grid, C0=C0, discretization=discretization_used(grid))





# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------
incr = 0
E_macro = jnp.zeros_like(E_macro_final)
eps = jnp.zeros_like(E_macro_final)
sig = jnp.zeros_like(E_macro_final)


scale_old = 1.0
eps_aux_old = jnp.zeros_like(E_macro_final)

#eps0 = Gamma_0.apply_C0_inv(sig)
#eps_aux_old = 0.5 * (eps0 / scale_old + eps) 


while incr< n_incr:

    
    # Incremental loading
    E_macro = E_macro + Delta_E_macro
    X_old = E_macro - eps - (1/scale_old)*Gamma_0.mv(sig - scale_old * Gamma_0.apply_C0(eps))
    W_old = -Gamma_0.mv(sig)
    
    
    if solver_used == AEMSolver:
        # AEM solver
        for solver in [AEMSolver]:
            tic = time()
            eps_aux, eps, sig, info = solver(
                E_macro, alpha, Gamma_0, tol=1.0e-6, max_iter=500
            )
            print("Solver info:", info)  # ["num_steps"])
            print(f"{solver.__name__} solver time", time() - tic)
            print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))
            print("\n")
        
    
    elif solver_used == A2EMSolver:
        # A2EM solver
        for solver in [A2EMSolver]:
            tic = time()
            eps, sig, info = solver(
                E_macro,
                alpha,
                Gamma_0,
                tol=1e-6,
                max_iter=500,
                auto_scale=True,
                verbose=True,
            )
    
            print("Solver info:", info)  # ["num_steps"])
            print(f"{solver.__name__} solver time", time() - tic)
            print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))
            print("\n")
    
    elif solver_used == KrylovAEMSolver:
        # Krylov-AEM solver
        for solver in [KrylovAEMSolver]:
            tic = time()
            eps, sig, info = solver(E_macro, alpha, Gamma_0, tol=1e-6, max_iter=300)
            print("Solver info:", info)  # ["num_steps"])
            print(f"{solver.__name__} solver time", time() - tic)
            print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))
            print("\n")
            
            
    elif solver_used == A2EM_iterativSolver:
        # A2EM solver
        for solver in [A2EM_iterativSolver]:
            tic = time()
            eps, sig, info, scale_old, eps_aux_old, it_elast, Delta_elast = solver(
                eps_aux_old,
                X_old,
                scale_old,
                W_old,
                E_macro,
                alpha,
                Gamma_0,
                tol=1e-6,
                max_iter=500,
                auto_scale=True,
                verbose=True,
                )
    
            print("Solver info:", info)  # ["num_steps"])
            print(f"{solver.__name__} solver time", time() - tic)
            print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))
            print("\n")
            
    incr+=1





# store results
pv_grid.cell_data["strain"] = eps.reshape((-1, 3, 3),order="F")
pv_grid.cell_data["stress"] = sig.reshape((-1, 3, 3),order="F")


# Extraction of the zz component of the stress
sigma_xz = sig[..., 0, 2].flatten(order="F")
pv_grid.cell_data["stress_xz"] = sigma_xz


# Extraction of the zz component of the strain
eps_xz = eps[..., 0, 2].flatten(order="F")
pv_grid.cell_data["eps_xz"] = eps_xz

# Extraction of the elastic energy
W_elast = 0.5 * jnp.einsum("...ij,...ij->...", sig, eps)
pv_grid.cell_data["W_elast"] = W_elast.flatten(order="F")


mu_eff = 0.5*jnp.mean(sig, axis=[0, 1, 2])[0][2]
print("err sur mu:", jnp.abs(mu_eff-1))

pv_grid.save(file_name.removesuffix(".vti") + "_results.vtk")
