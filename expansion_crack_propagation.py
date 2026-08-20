# --------------------------------------------------------------------------
# Expansion of a 2D sphere within a square matrix with crack propagation
# --------------------------------------------------------------------------


# --------------------------------
# Imports
# --------------------------------

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
import matplotlib.pyplot as plt

from eshelbyx.green_operator import (
    GreenOperator,
)
from eshelbyx.green_operator import (
    DamageFeugueurGreenOperator,
)


from eshelbyx.discretization import (
    WillotDiscretization,
)


from eshelbyx.grid import load_vtk_grid
from jaxmat.tensors import SymmetricTensor2, IsotropicTensor4, SymmetricTensor4
from jaxmat.tensors.tensor_utils import eigenvalues
from jaxmat.tensors.linear_algebra import eig33


from eshelbyx.solver import A2EM_generalSolver, damage_Feugueur_solver, damage_EAEM_solver

current_folder = Path(__file__).parent.resolve()




# --------------------------------------
# Solvers parameters
# --------------------------------------

# Solver for elasticity problem
solver_elasticity = A2EM_generalSolver

# Solver for damage problem
solver_damage = damage_EAEM_solver

# Discretization (FFT)
discretization_used = WillotDiscretization

# Damage model
damage_model = "AT1"

# Hybrid voxels treatment depending on the strategy
strat_hybrid = "alpha"

# True : plot the macroscopic stress-strain curve
plot_stress_strain = True


n_incr = 300   # number of increments for the loading
max_iter_elasticity = 2000  # maximum number of iterations per step for the elastic solver
max_iter_damage = 5000  # maximum number of iterations per step for the damage solver
tol_elasticity = 1e-6   # tolerance for the elastic solver (||error|| < tol => stop)
tol_damage = 1e-6   # tolerance for the damage solver
n_step_max = 100    # maximum number of steps per increment
tol_step = 2e-4    # step tolerance (||d_end_step - d_beginning_step|| < tol_step => next step)


freq_save = 10     # save results in .vtk file every freq_save increments


# Array used to plot the A0 evolution on the 20 first increments
les_A0_save = jnp.zeros(20)



# Micro-structure created with microgen.py
file_name = "eshelbyx/results/2phases_expansion_num_2D_100.vti"
grid, pv_grid = load_vtk_grid(file_name, marker_name="phase")
Nx = pv_grid.dimensions[0]-1    # number of voxels in x direction
Ny = pv_grid.dimensions[1]-1    # number of voxels in y direction
# Nz = 1 (2D grid)




# -------------------------------------------------------------
# Usefull functions
# -------------------------------------------------------------

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


# Spatial mean of a field over the entire domain
@eqx.filter_jit
def mean_field(X):
    spatial_axes = tuple(range(X.ndim - 2))
    X_mean = jnp.mean(X, axis=spatial_axes)
    X_mean_field = jnp.broadcast_to(X_mean, X.shape)
    return X_mean_field





# --------------------
# Material parameters
# --------------------

chi = 1.0001 # contrast

lmbda_matrix = 0.0
mu_matrix = 34e-3
E_matrix = mu_matrix*(3*lmbda_matrix + 2*mu_matrix)/(lmbda_matrix+mu_matrix)
nu_matrix = lmbda_matrix/(2*(lmbda_matrix + mu_matrix))

lmbda_inc = lmbda_matrix*chi
mu_inc = mu_matrix*chi
E_inc = mu_inc*(3*lmbda_inc + 2*mu_inc)/(lmbda_inc + mu_inc)
nu_inc = lmbda_inc/(2*(lmbda_inc+mu_inc))


l0 = 3  # caracteristic length for damage field regularization

# Toughness
Gc_matrix = 8e-6
Gc_inc = Gc_matrix*1e12





# -----------------------------------------------------
# Reference medium 
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



# ----------------------------------------
# Data extraction
# ----------------------------------------

# Data extraction : damage init d=0
damage_init = pv_grid.cell_data["damage_init"]

# Data extraction : inclusion voxels with strain loading
bool_load = (grid.markers == 1) 

# Data extraction : hybrid booleans (voxel hybrid or not)
bool_hybrid_data = pv_grid.cell_data["bool_hybrid"]
bool_hybrid = jnp.array(bool_hybrid_data)


# Data extraction : volumic fraction (of the internal phase for hybrid voxels)
volumic_fraction_data = pv_grid.cell_data["volumic_fraction"]
volumic_fraction = jnp.array(volumic_fraction_data)

# Data extraction : phases couple (internal and external phases for hybrid voxels)
phases_couple_data = pv_grid.cell_data["phases_couple"]
phases_couple = jnp.array(phases_couple_data)





# -----------------------------------------
# Material parameters mapping
# -----------------------------------------

# Material parameters at the center of the voxels
E_map = {0: E_matrix, 1: E_inc}
nu_map = {0: nu_matrix, 1: nu_inc}
E = piecewise_field(jnp.zeros(grid.shape), E_map, grid)
nu = piecewise_field(jnp.zeros(grid.shape), nu_map, grid)



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



# -----------------------------------------
# Toughness mapping
# -----------------------------------------

# Toughness at the center of the voxels
Gc_map = {0: Gc_matrix, 1: Gc_inc}
Gc_value = piecewise_field(jnp.zeros(grid.shape), Gc_map, grid)


Gc_int_map = {(0, 0): Gc_matrix, (1, 0): Gc_inc}
Gc_ext_map = {(0, 0): Gc_matrix, (1, 0): Gc_matrix}
Gc_int_value = piecewise_field_couple(jnp.zeros(n_cells), Gc_int_map, phases_couple)
Gc_ext_value = piecewise_field_couple(jnp.zeros(n_cells), Gc_ext_map, phases_couple)


pv_grid.cell_data["Gc_int"] = Gc_int_value.flatten(order="F")
pv_grid.cell_data["Gc_ext"] = Gc_ext_value.flatten(order="F")


# Averaging the toughness for hybrid voxels at the interface
volumic_fraction_3d = volumic_fraction.reshape(grid.shape, order="F")
Gc_int_value_3d = Gc_int_value.reshape(grid.shape, order="F")
Gc_ext_value_3d = Gc_ext_value.reshape(grid.shape, order="F")
bool_hybrid_3d = bool_hybrid.reshape(grid.shape, order="F")
Gc_value = jnp.where(
    bool_hybrid_3d,
    volumic_fraction_3d * Gc_int_value_3d
    + (1 - volumic_fraction_3d) * Gc_ext_value_3d,
    Gc_value,
)

pv_grid.cell_data["Gc"] = Gc_value.flatten(order="F")




# ------------------------------------------------------------------------
# Loading : eps_load = factor_eps_load*Id in inclusion, 0 in the matrix
# ------------------------------------------------------------------------

# Pre-strain set at voxel center 
factor_eps_load = 0.015
Id = SymmetricTensor2(array=jnp.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))

eps_load_final = grid.broadcast_to_grid(Id * factor_eps_load)

eps_load_final_None = eps_load_final * bool_load[..., None]
eps_load_final_None = jnp.asarray(eps_load_final_None)


pv_grid.cell_data["strain_load_final"] = eps_load_final_None.reshape((-1, 3, 3),order="F")


Delta_eps_load = eps_load_final_None/n_incr # Incremental loading







# Internal and external pre-strains for hybrid voxels
eps_load_flat = jnp.asarray(eps_load_final).reshape((n_cells, -1), order="F")

factor_eps_load_int = 1.0
factor_eps_load_ext = 0.0

mask_load = jnp.all(phases_couple == jnp.array([1, 0]), axis=1)

eps_load_final_int = jnp.where(mask_load[:, None],factor_eps_load_int * eps_load_flat,0.0,).reshape((Nx, Ny, 1, 3, 3),order="F")
eps_load_final_ext = jnp.where(mask_load[:, None],factor_eps_load_ext * eps_load_flat,0.0,).reshape((Nx, Ny, 1, 3, 3),order="F")


pv_grid.cell_data["strain_load_final_int"] = eps_load_final_int.reshape((-1, 3, 3),order="F")
pv_grid.cell_data["strain_load_final_ext"] = eps_load_final_ext.reshape((-1, 3, 3),order="F")


# Incremental loading
Delta_eps_load_int = eps_load_final_int/n_incr
Delta_eps_load_ext = eps_load_final_ext/n_incr



# -----------------------------------------------------------------
# Macroscopic strain and stress applied (must be complementary)
# -----------------------------------------------------------------

# Macroscopic strain
E_macro_final_array = jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
E_macro_final = SymmetricTensor2(array = E_macro_final_array)
E_macro_final = grid.broadcast_to_grid(E_macro_final)  # create a field on the grid

Delta_E_macro = E_macro_final/n_incr


# Macroscopic stress
S_macro_final_array = jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
S_macro_final = SymmetricTensor2(array = S_macro_final_array)
S_macro_final = grid.broadcast_to_grid(S_macro_final)  # create a field on the grid

Delta_S_macro = S_macro_final/n_incr


# Projector
P_projector_array = (E_macro_final_array > 0).astype(jnp.float64)
P = SymmetricTensor4(array=jnp.diag(P_projector_array))
I = SymmetricTensor4.identity()

def _apply_P_tilde(X):
    sig = (I-P) @ X
    return SymmetricTensor2(tensor=sig)


@eqx.filter_jit
def apply_P_tilde(
    X
):
    X_flatten = SymmetricTensor2(tensor=jnp.asarray(X).reshape((-1, 3, 3),order="F"))
    sig_tensor = jax.vmap(_apply_P_tilde, in_axes=(0))(
        X_flatten,
    )
    return sig_tensor.tensor.reshape(X.shape,order="F")


P_tilde = lambda X: apply_P_tilde(
    X,
)






# ----------------------------------------------------------------------
# Spectral decomposition (damage nucleating only in traction areas)
# ----------------------------------------------------------------------

@eqx.filter_jit
def spectral_decomposition(E):
    # return eigenvalues and eigenvectors of E

    def _spectral_decomposition(E):
        return eig33(E)

    shape = E.shape[:-2]

    vals_prop, vects_prop = jax.vmap(
        _spectral_decomposition
    )(jnp.asarray(E).reshape((-1, 3, 3),order="F"))

    vals_prop = vals_prop.reshape((*shape, 3),order="F")
    vects_prop = vects_prop.reshape((*shape, 3, 3, 3),order="F")

    return vals_prop, vects_prop


 
def energy_decomposition(eps_aux, alpha, scale, Gamma_0_elasticity):
    # return the positive and negative parts of the elastic energy
    
    def positive_part(x):
        return (x+jnp.abs(x))/2

    def negative_part(x):
        return (x-jnp.abs(x))/2
    
    # Eigenvalues and eigenvectors of the auxiliar strain
    vals_prop, vects_prop = spectral_decomposition(eps_aux)
    
    lambda1 = vals_prop[..., 0]
    lambda2 = vals_prop[..., 1]
    lambda3 = vals_prop[..., 2]
    
    P1 = vects_prop[..., 0, :, :]
    P2 = vects_prop[..., 1, :, :]
    P3 = vects_prop[..., 2, :, :]

    
    # Positive and negative parts of the auxiliar strain
    eps_aux_plus = (
        positive_part(lambda1[..., None, None]) * P1
        + positive_part(lambda2[..., None, None]) * P2
        + positive_part(lambda3[..., None, None]) * P3
    )

    eps_aux_minus = (
        negative_part(lambda1[..., None, None]) * P1
        + negative_part(lambda2[..., None, None]) * P2
        + negative_part(lambda3[..., None, None]) * P3
    )
    
    
    # Positive and negative parts of the elastic energy
    eps_plus = alpha(eps_aux_plus, scale)
    sig_plus = scale * Gamma_0_elasticity.apply_C0(2.0 * eps_aux_plus - eps_plus)
    W_elast_plus = 0.5 * jnp.einsum("...ij,...ij->...", sig_plus, eps_plus)
    
    eps_minus = alpha(eps_aux_minus, scale)
    sig_minus = scale * Gamma_0_elasticity.apply_C0(2.0 * eps_aux_minus - eps_minus)
    W_elast_minus = 0.5 * jnp.einsum("...ij,...ij->...", sig_minus, eps_minus)
    
    
    return W_elast_plus, W_elast_minus





material_params = (E.flatten(order="F"), nu.flatten(order="F"))
bool_hybrid_list = bool_hybrid.flatten(order="F")
material_params_int = (E_int.flatten(order="F"), nu_int.flatten(order="F"))
material_params_ext = (E_ext.flatten(order="F"), nu_ext.flatten(order="F"))
volumic_fraction_list = volumic_fraction.flatten(order="F")



# ------------------------------------------------------------------------------------------------------------------
# Computation and application of the alpha 4th-order tensor depending on the hybridation strategy for hybrid voxels
# ------------------------------------------------------------------------------------------------------------------

def _apply_alpha(
    eps,
    material_params,
    bool_hybrid_list,
    material_params_int,
    material_params_ext,
    volumic_fraction_list,
    C0,
    damage,

):

    def normal_branch(_):
        # Usual computation of alpha for non-hybrid voxels

        E, nu = material_params

        lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
        mu = E / 2 / (1 + nu)
        C_intact = IsotropicTensor4(kappa=lmbda + 2 / 3 * mu, mu=mu)
        
        k = 1e-12
        degrad_func = k + (1-damage)**2
        C = degrad_func * C_intact
        

        iC = (C + C0).inv
        alp = iC @ (C0 * 2.0)
        sig = alp @ eps  # here alp is a IsotropicTensor4
        return SymmetricTensor2(tensor=sig)

    def mean_alpha_branch(_):
        # Takes the mean of alpha for hybrid voxels
        
        k = 1e-12
        degrad_func = k + (1-damage)**2

        E_int, nu_int = material_params_int
        E_ext, nu_ext = material_params_ext
        phi_int = volumic_fraction_list

        lmbda_int = E_int * nu_int / (1 + nu_int) / (1 - 2 * nu_int)
        mu_int = E_int / 2 / (1 + nu_int)
        C_int_intact = IsotropicTensor4(kappa=lmbda_int + 2 / 3 * mu_int, mu=mu_int)
        C_int = degrad_func * C_int_intact
        
        iC_int = (C_int + C0).inv
        alp_int = iC_int @ (C0 * 2.0)

        lmbda_ext = E_ext * nu_ext / (1 + nu_ext) / (1 - 2 * nu_ext)
        mu_ext = E_ext / 2 / (1 + nu_ext)
        C_ext_intact = IsotropicTensor4(kappa=lmbda_ext + 2 / 3 * mu_ext, mu=mu_ext)
        C_ext = degrad_func * C_ext_intact
        
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

        C_intact = IsotropicTensor4(
            kappa=phi_int * C_int.kappa + (1 - phi_int) * C_ext.kappa,
            mu=phi_int * C_int.mu + (1 - phi_int) * C_ext.mu,
        )
        
        k = 1e-12
        degrad_func = k + (1-damage)**2
        C = degrad_func * C_intact

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
        C_intact = S.inv
        
        k = 1e-12
        degrad_func = k + (1-damage)**2
        C = degrad_func * C_intact

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
    eps, params, bool_hybrid, params_int, params_ext, phi, C0, damage
):
    eps_flatten = SymmetricTensor2(tensor=jnp.asarray(eps).reshape((-1, 3, 3),order="F"))
    sig_tensor = jax.vmap(_apply_alpha, in_axes=(0, 0, 0, 0, 0, 0, None, 0))(
        eps_flatten,
        params,
        bool_hybrid,
        params_int,
        params_ext,
        phi,
        C0,
        damage
    )
    return sig_tensor.tensor.reshape(eps.shape,order="F")




# ------------------------------------------------------------------------------------------
# Apply loading strain using the "beta" function : Cvox : eps_load_vox = < C : eps_load >
# ------------------------------------------------------------------------------------------

def _apply_beta(
    eps_load,
    eps_load_int,
    eps_load_ext,
    material_params,
    bool_hybrid_list,
    material_params_int,
    material_params_ext,
    volumic_fraction_list,
    C0,
    damage,
):

    def normal_branch(_):
        # Usual computation of alpha for non-hybrid voxels
        return SymmetricTensor2(tensor=eps_load)


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

        C_intact = IsotropicTensor4(
            kappa=phi_int * C_int.kappa + (1 - phi_int) * C_ext.kappa,
            mu=phi_int * C_int.mu + (1 - phi_int) * C_ext.mu,
        )
        
        
        
        sigma_load_int = C_int @ eps_load_int
        sigma_load_ext = C_ext @ eps_load_ext
        
        eps_load_bis = C_intact.inv @ (sigma_load_int*phi_int + sigma_load_ext*(1-phi_int))
        
        return SymmetricTensor2(tensor=eps_load_bis)

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
        C_intact = S.inv
        


        sigma_load_int = C_int @ eps_load_int
        sigma_load_ext = C_ext @ eps_load_ext
        
        eps_load_bis = C_intact.inv @ (sigma_load_int*phi_int + sigma_load_ext*(1-phi_int))
        
        
        return SymmetricTensor2(tensor=eps_load_bis)
    
    
    def mean_alpha_branch(_):
        # Takes the mean of stiffness tensor C for hybrid voxels

        E_int, nu_int = material_params_int
        E_ext, nu_ext = material_params_ext
        phi_int = volumic_fraction_list

        lmbda_int = E_int * nu_int / (1 + nu_int) / (1 - 2 * nu_int)
        mu_int = E_int / 2 / (1 + nu_int)

        lmbda_ext = E_ext * nu_ext / (1 + nu_ext) / (1 - 2 * nu_ext)
        mu_ext = E_ext / 2 / (1 + nu_ext)

        C_int_intact = IsotropicTensor4(kappa=lmbda_int + 2 / 3 * mu_int, mu=mu_int)
        C_ext_intact = IsotropicTensor4(kappa=lmbda_ext + 2 / 3 * mu_ext, mu=mu_ext)
        
        C_intact = (phi_int*((C_int_intact+C0).inv) + (1-phi_int)*((C_ext_intact+C0).inv)).inv - C0
        

        sigma_load_int = C_int_intact @ eps_load_int
        sigma_load_ext = C_ext_intact @ eps_load_ext
        
        eps_load_bis = C_intact.inv @ (sigma_load_int*phi_int + sigma_load_ext*(1-phi_int))
        
        
        return SymmetricTensor2(tensor=eps_load_bis)


    branch_map = {
        "Voigt": Voigt_branch,
        "Reuss": Reuss_branch,
        "None": normal_branch,
        "alpha": mean_alpha_branch,
    }

    selected_branch = branch_map[strat_hybrid]

    return jax.lax.cond(bool_hybrid_list, selected_branch, normal_branch, operand=None)


@eqx.filter_jit
def apply_beta(
    eps_load, eps_load_int, eps_load_ext, params, bool_hybrid, params_int, params_ext, phi, C0, damage
):
    eps_load_flatten = SymmetricTensor2(tensor=jnp.asarray(eps_load).reshape((-1, 3, 3),order="F"))
    eps_load_int_flatten = SymmetricTensor2(tensor=jnp.asarray(eps_load_int).reshape((-1, 3, 3),order="F"))
    eps_load_ext_flatten = SymmetricTensor2(tensor=jnp.asarray(eps_load_ext).reshape((-1, 3, 3),order="F"))
    eps_load_bis_tensor = jax.vmap(_apply_beta, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, 0))(
        eps_load_flatten,
        eps_load_int_flatten,
        eps_load_ext_flatten,
        params,
        bool_hybrid,
        params_int,
        params_ext,
        phi,
        C0,
        damage
    )
    return eps_load_bis_tensor.tensor.reshape(eps_load.shape,order="F")






# --------------------------------
# Elasticity Green operator
# --------------------------------
Gamma_0_elasticity = GreenOperator(grid=grid, C0=C0, discretization=discretization_used(grid))

# ----------------------------------------------------------------
# Damage Green Operator : function of A0 for the EAEM solver
# ----------------------------------------------------------------
@eqx.filter_jit
def Gamma_0_damage_func(A0):
    return DamageFeugueurGreenOperator(grid=grid, discretization=discretization_used(grid), A0 = A0)





# ------------------------------------------------
# Definition of the step loop
# ------------------------------------------------

# Convergence condition
def cond_fun_step(state):
    step, eps, sig, d, eps_aux, scale, A0, err_step, eps_load, eps_load_int, eps_load_ext, E_macro, S_macro, it_damage, d_old, les_A0_save = state
    return jnp.logical_and(step < n_step_max, err_step > tol_step)


# Body function of the step loop
def body_fun_step(state):
    step, eps, sig, d, eps_aux, scale, A0, err_step, eps_load, eps_load_int, eps_load_ext, E_macro, S_macro, it_damage, d_old, les_A0_save = state
    
    
    d_beginning_step = d    # used for convergence condition
    
    

    # Definition of the alpha and beta functions inside loop (because d is an argument)
    alpha = lambda eps, scale=1.0: apply_alpha(
        eps,
        material_params,
        bool_hybrid_list,
        material_params_int,
        material_params_ext,
        volumic_fraction_list,
        C0 * scale,
        d
    )
    
    beta = lambda eps_load, eps_load_int, eps_load_ext, scale=1.0 : apply_beta(
        eps_load,
        eps_load_int,
        eps_load_ext,
        material_params,
        bool_hybrid_list,
        material_params_int,
        material_params_ext,
        volumic_fraction_list,
        C0*scale,
        d
    )
    
    
    # Previous residual X and other variable W for initialization of the elastic solver
    X = E_macro + (1/scale)*Gamma_0_elasticity.apply_C0_inv(S_macro) - Gamma_0_elasticity.apply_C0_inv(P_tilde(mean_field(sig))) + P_tilde(mean_field(eps))  - eps - (1/scale)*Gamma_0_elasticity.mv(sig) + Gamma_0_elasticity.mv(Gamma_0_elasticity.apply_C0(eps))
    W = -Gamma_0_elasticity.mv(sig)
    
    
    
    
    # Solving the elastic problem (knowing the previous damage field)
    if solver_elasticity == A2EM_generalSolver:
        for solver in [A2EM_generalSolver]:
            eps, sig, scale, eps_aux, info, it_max_elasticity, delta_elasticity = solver(
                eps_aux,
                X,
                scale,
                W,
                eps_load,
                eps_load_int,
                eps_load_ext,
                alpha,
                beta,
                E_macro,
                S_macro,
                P_tilde,
                Gamma_0_elasticity,
                tol=tol_elasticity,
                max_iter=max_iter_elasticity,
                auto_scale=True,
                verbose=False,
            )
    
    
    
    # Degradation function g(d)
    k = 1e-12
    degrad_func = k + (1-d)**2
    degrad_func = degrad_func.reshape((Nx,Ny,1),order="F")
    
    
    # Hybrid decomposition : degradation function only affects positive part of the elastic energy (traction)
    W_elast_plus, W_elast_minus = energy_decomposition(eps_aux, alpha, scale, Gamma_0_elasticity)
    W_elast_0 = W_elast_plus/degrad_func
    
    
    
    
    # Parameters for solving the damage problem
    if damage_model == "AT1":
        A = (8*W_elast_0)/(3*Gc_value*l0)
        B = (8*W_elast_0)/(3*Gc_value*l0) - (1/(2*(l0**2)))
    
    elif damage_model == "AT2":
        A = (1/l0**2) + (2*W_elast_0/(Gc_value*l0))
        B = 2*W_elast_0/(Gc_value*l0)
        
        
        
    # Reshaping d
    d = jnp.array(d).reshape((grid.shape),order="F")
    d_old = jnp.array(d_old).reshape((grid.shape),order="F")
    
    
    # Solving the damage problem (knowing the previous strain and stress fields)
    if solver_damage == damage_Feugueur_solver:
        A0 = (jnp.max(A)+jnp.min(A))/2
        
        # Green operator
        Gamma_0_damage = DamageFeugueurGreenOperator(grid=grid, discretization=discretization_used(grid), A0 = A0)
        
        # Solver
        for solver in [damage_Feugueur_solver]:
            d, it_max_damage, delta_damage = solver(d, A, B, A0, Gamma_0_damage, d_old, max_iter = max_iter_damage, tol=tol_damage, verbose=False)
        
        A0_new = A0

    elif solver_damage == damage_EAEM_solver:
        # Green operator
        Gamma_0_damage = lambda A0: Gamma_0_damage_func(A0)
        
        # Solver
        for solver in [damage_EAEM_solver]:
            d, it_max_damage, delta_damage, A0_new, les_A0_save = solver(d, A, B, A0, Gamma_0_damage, d_old, les_A0_save, max_iter = max_iter_damage, tol=tol_damage, verbose=False)



    it_damage = it_damage + it_max_damage   # used to count the total number of damage iterations per increment



    # Reshaping d
    d = d.flatten(order="F")
    d_old = d_old.flatten(order="F")
    
    
    
    
    err_step = jnp.sqrt(jnp.sum((d-d_beginning_step)**2))   # convergence error to go to another increment
    
    # Print step infos
    jax.debug.print("step {} err_step {:.2e} imax_elast {} imax_damage {} err_elast {:.2e} err_damage {:.2e} A0 {:.2e}",
                    step+1,
                    err_step,
                    it_max_elasticity,
                    it_max_damage,
                    delta_elasticity,
                    delta_damage,
                    A0_new
                    )
    
    
    return step+1, eps, sig, d, eps_aux, scale, A0_new, err_step, eps_load, eps_load_int, eps_load_ext, E_macro, S_macro, it_damage, d_old, les_A0_save






# ---------------------------------------------------
# Definition of the increment loop
# ---------------------------------------------------

# Convergence condition
def cond_fun_incr(state):
    incr, eps, sig, d, eps_aux, scale, A0, les_eps_mean, les_sig_mean, eps_load, eps_load_int, eps_load_ext, E_macro, S_macro, les_d, les_W_elast, les_it_damage, les_A0_save = state
    return incr<n_incr


# Body function of the increment loop
def body_fun_incr(state):
    incr, eps, sig, d, eps_aux, scale, A0, les_eps_mean, les_sig_mean, eps_load, eps_load_int, eps_load_ext, E_macro, S_macro, les_d, les_W_elast, les_it_damage, les_A0_save = state

    
    beta = lambda eps_load, eps_load_int, eps_load_ext, scale=1.0 : apply_beta(
        eps_load,
        eps_load_int,
        eps_load_ext,
        material_params,
        bool_hybrid_list,
        material_params_int,
        material_params_ext,
        volumic_fraction_list,
        C0*scale,
        d
    )
    
    # Incremental loading
    eps_load = eps_load + Delta_eps_load
    eps_load_int = eps_load_int + Delta_eps_load_int
    eps_load_ext = eps_load_ext + Delta_eps_load_ext
    
    
    Delta_eps_load_vox = beta(Delta_eps_load, Delta_eps_load_int, Delta_eps_load_ext, scale)
    
    E_macro = E_macro + Delta_E_macro
    S_macro = S_macro + Delta_S_macro
     
    
    eps = eps + Delta_eps_load_vox
    
    
    d_old = d
    
    d_verif_old = d_old
    
    
    # ------------------------------------------------------------------
    # Run step loop (until d converges)
    # ------------------------------------------------------------------
    init_state_step = (0, eps, sig, d, eps_aux, scale, A0, jnp.inf, eps_load, eps_load_int, eps_load_ext, E_macro, S_macro, 0, d_old, les_A0_save)
    
    step_f_step, eps_f_step, sig_f_step, d_f_step, eps_aux_f_step, scale_f_step, A0_f_step, err_step_f_step, eps_load_f_step, eps_load_int_f_step, eps_load_ext_f_step, E_macro_f_step, S_macro_f_step, it_damage_f, d_old_f, les_A0_save = jax.lax.while_loop(
        cond_fun_step, body_fun_step, init_state_step
    )
    
    
    # Test
    test1 = jnp.sum((d_verif_old-d_old_f)**2)
    test2 = jnp.sum(jnp.where(d_f_step-d_verif_old>0, 0, (d_verif_old-d_f_step)**2))
    
    # ---------------------------------------------------------------------
    # Post-treatment
    # ---------------------------------------------------------------------
    
    # Saving mean strain and stress fields to plot the degraded macroscopic strain-stress law
    sigma_yy_mean = jnp.mean(sig_f_step[..., 1, 1].flatten(order="F"))
    eps_yy_mean = jnp.mean(eps_f_step[..., 1, 1].flatten(order="F"))
    
    les_sig_mean = les_sig_mean.at[incr].set(sigma_yy_mean)
    les_eps_mean = les_eps_mean.at[incr].set(eps_yy_mean)
    
    
    # Saving total number of iterations for damage solver : comparison between Feugueur and EAEM damage models
    les_it_damage = les_it_damage.at[incr].set(it_damage_f)
    
    
    # Saving damage field : plot Paraview
    les_d = les_d.at[incr].set(d_f_step)
    
    
    # Saving elastic energy : plot Paraview
    W_elast_f = 0.5 * jnp.einsum("...ij,...ij->...", sig_f_step, eps_f_step)
    les_W_elast = les_W_elast.at[incr].set(W_elast_f.flatten(order="F"))
    
    
    
    
    # Print increment number
    jax.debug.print("\nIncrement {} test1 {} test2 {}",incr+1,test1,test2)


    return incr+1, eps_f_step, sig_f_step, d_f_step, eps_aux_f_step, scale_f_step, A0_f_step, les_eps_mean, les_sig_mean, eps_load_f_step, eps_load_int_f_step, eps_load_ext_f_step, E_macro_f_step, S_macro_f_step, les_d, les_W_elast, les_it_damage, les_A0_save




# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------
incr = 0

d = jnp.zeros_like(damage_init)
eps = jnp.zeros_like(E_macro_final)
sig = jnp.zeros_like(E_macro_final)
scale = 1.0
A0 = 1.0
eps_aux = jnp.zeros_like(E_macro_final)
les_eps_mean = jnp.zeros(n_incr)
les_sig_mean = jnp.zeros(n_incr)
les_d = jnp.zeros((n_incr, *d.shape))
les_W_elast = jnp.zeros((n_incr, *d.shape))
les_it_damage = jnp.zeros(n_incr)

eps_load = jnp.zeros_like(eps_load_final_None)
eps_load_int = jnp.zeros_like(eps_load_final_int)
eps_load_ext = jnp.zeros_like(eps_load_final_ext)
E_macro = jnp.zeros_like(eps_load_final_None)
S_macro = jnp.zeros_like(eps_load_final_None)




# ----------------------------------------------------------------
# Run increment loop
# ----------------------------------------------------------------

init_state_incr = incr, eps, sig, d, eps_aux, scale, A0, les_eps_mean, les_sig_mean, eps_load, eps_load_int, eps_load_ext, E_macro, S_macro, les_d, les_W_elast, les_it_damage, les_A0_save

incr_f, eps_f, sig_f, d_f, eps_aux_f, scale_f, A0_f, les_eps_mean, les_sig_mean, eps_load_f, eps_load_int_f, eps_load_ext_f, E_macro_f, S_macro_f, les_d, les_W_elast, les_it_damage, les_A0_save = jax.lax.while_loop(
    cond_fun_incr, body_fun_incr, init_state_incr
)

    

# ----------------------------------------------------
# Saving results for the convergence speed
# ----------------------------------------------------
import numpy as np

les_load_percentage = 100*jnp.arange(1,n_incr+1,1)/n_incr
data = np.column_stack((les_load_percentage, les_it_damage))
np.savetxt(
    "donnees.csv",
    data,
    delimiter=",",
    header="load_percentage,it_damage",
    comments="",
)



    

# -------------------------------------------------------------------------------
# Saving results of elastic energy and damage fields for Paraview visualization
#--------------------------------------------------------------------------------

for incr in range(n_incr):

    if (incr+1) % freq_save == 0:
        pv_grid.cell_data["damage"] = jnp.asarray(les_d[incr])
        pv_grid.cell_data["W_elast"] = jnp.asarray(les_W_elast[incr])

        pv_grid.save(
            file_name.removesuffix(".vti")
            + f"_results_{incr:04d}.vtk"
        )


"""
# --------------------------------------------------------------------------------
# Plot macroscopic strain evolution : detection of damage starting
# --------------------------------------------------------------------------------
les_i = jnp.arange(0,n_incr,1)
les_eps_load = jnp.array([(i+1)*factor_eps_load/n_incr for i in les_i])
les_y = les_eps_mean/les_eps_load
les_x = 100*les_eps_load
plt.plot(les_x,les_y)
plt.grid()
plt.xlabel(r"$\varepsilon_s\ (\%)$")
plt.ylabel(r"$\frac{E_{yy}}{\varepsilon_s}$")
plt.show()


# --------------------------------------------------------------------------------
# Plot number of damage solver iterations as a function of the loading
# --------------------------------------------------------------------------------
les_incr_percentage = jnp.array([100*incr/n_incr for incr in range(1,n_incr+1)])
plt.plot(les_incr_percentage,les_it_damage_Feugueur, label = "Feugueur")
plt.plot(les_incr_percentage, les_it_damage_EAEM, label = "EAEM")
plt.yscale("log")
plt.legend()
plt.grid(which="both", axis="both")
plt.xlabel("% loading")
plt.ylabel(r"$n_{iter}$")
plt.show()
"""