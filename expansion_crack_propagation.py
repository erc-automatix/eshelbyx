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


from eshelbyx.solver import A2EM_iterativ_expansionSolver, damage_Feugueur_solver, damage_EAEM_solver

current_folder = Path(__file__).parent.resolve()




# --------------------------------------
# Solvers parameters
# --------------------------------------

# Solver for elasticity problem
solver_elasticity = A2EM_iterativ_expansionSolver

# Solver for damage problem
solver_damage = damage_EAEM_solver

# Discretization (FFT)
discretization_used = WillotDiscretization


# Damage model
damage_model = "AT2"

# True : plot the macroscopic stress-strain curve
plot_stress_strain = True


n_incr = 300    # number of increments for the loading
max_iter_elasticity = 2000  # maximum number of iterations per step for the elastic solver
max_iter_damage = 5000  # maximum number of iterations per step for the damage solver
tol_elasticity = 1e-6   # tolerance for the elastic solver (||error|| < tol => stop)
tol_damage = 1e-6   # tolerance for the damage solver
n_step_max = 100    # maximum number of steps per increment
tol_step = 2e-4    # step tolerance (||dnew - dold|| < tol => stop)


freq_save = 10     # save results in .vtk file every freq_save increments



# Micro-structure created with microgen.py
file_name = "eshelbyx/results/2phases_expansion_num_2D_100.vti"
grid, pv_grid = load_vtk_grid(file_name, marker_name="phase")
Nx = pv_grid.dimensions[0]-1    # number of voxels in x direction
Ny = pv_grid.dimensions[1]-1    # number of voxels in y direction



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



# Data extraction : damage init d=0
damage_init = pv_grid.cell_data["damage_init"]



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


l0 = 3

Gc_matrix = 8e-6
Gc_inc = Gc_matrix*1e6



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




# ------------------------------------------------------
# Loading : eps_load = beta*Id in inclusion, 0 in the matrix
# ------------------------------------------------------

bool_load = (grid.markers == 1) # |(grid.markers == 0)

beta = 0.015
Id = SymmetricTensor2(array=jnp.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))

eps_load_final = grid.broadcast_to_grid(Id * beta)
eps_load_final = eps_load_final * bool_load[..., None] 
eps_load_final = jnp.asarray(eps_load_final)

pv_grid.cell_data["strain_load_final"] = eps_load_final.reshape((-1, 3, 3),order="F")

Delta_eps_load = eps_load_final/n_incr



# Material parameters mapping
E_map = {0: E_matrix, 1: E_inc}
nu_map = {0: nu_matrix, 1: nu_inc}
E = piecewise_field(jnp.zeros(grid.shape), E_map, grid)
nu = piecewise_field(jnp.zeros(grid.shape), nu_map, grid)


# Material parameters mapping
Gc_map = {0: Gc_matrix, 1: Gc_inc}
Gc_value = piecewise_field(jnp.zeros(grid.shape), Gc_map, grid)


"""
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
"""


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




# --------------------------------------------------------
# alpha function for regularizing infinite contrasts
# --------------------------------------------------------


# Computation and application of the alpha 4th-order tensor depending on the hybridation strategy for hybrid voxels
def _apply_alpha(
    eps,
    material_params,
    C0,
    damage,
):

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



@eqx.filter_jit
def apply_alpha(
    eps, params, C0, damage
):
    eps_flatten = SymmetricTensor2(tensor=jnp.asarray(eps).reshape((-1, 3, 3),order="F"))
    sig_tensor = jax.vmap(_apply_alpha, in_axes=(0, 0, None, 0))(
        eps_flatten,
        params,
        C0,
        damage,
    )
    return sig_tensor.tensor.reshape((eps.shape),order="F")






material_params = (E.flatten(order="F"), nu.flatten(order="F"))


# Elasticity Green operator
Gamma_0_elasticity = GreenOperator(grid=grid, C0=C0, discretization=discretization_used(grid))




def mean_field(X):
    spatial_axes = tuple(range(X.ndim - 2))
    X_mean = jnp.mean(X, axis=spatial_axes)
    X_mean_field = jnp.broadcast_to(X_mean, X.shape)
    return X_mean_field



# ------------------------------------------------
# Definition of the step loop
# ------------------------------------------------

# Convergence condition
def cond_fun_step(state):
    step, eps, sig, d, eps_aux, scale, err_step, eps_load, it_damage, d_old = state
    return jnp.logical_and(step < n_step_max, err_step > tol_step)


# Body function of the step loop
def body_fun_step(state):
    step, eps, sig, d, eps_aux, scale, err_step, eps_load, it_damage, d_old = state
    
    
    d_beginning_step = d    # used for convergence condition

    # Definition of the alpha function inside loop (because d is an argument)
    alpha = lambda eps, scale=1.0: apply_alpha(
        eps,
        material_params,
        C0 * scale,
        d,
    )
    
    
    # Previous residual X and other variable W for initialization of the elastic solver
    X = -(1/scale)*Gamma_0_elasticity.apply_C0_inv(mean_field(sig)) + mean_field(eps) - eps - (1/scale)*Gamma_0_elasticity.mv(sig) + Gamma_0_elasticity.mv(Gamma_0_elasticity.apply_C0(eps))
    W = -Gamma_0_elasticity.mv(sig)
    
    
    
    
    # Solving the elastic problem (knowing the previous damage field)
    if solver_elasticity == A2EM_iterativ_expansionSolver:
        # A2EM solver
        for solver in [A2EM_iterativ_expansionSolver]:
            eps, sig, scale, eps_aux, info, it_max_elasticity, delta_elasticity = solver(
                eps_aux,
                X,
                scale,
                W,
                eps_load,
                alpha,
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
        
    
    
    # Reference A parameter depending on the damage solver used
    if solver_damage == damage_Feugueur_solver:
        A0 = (jnp.max(A)+jnp.min(A))/2
    elif solver_damage == damage_EAEM_solver:
        A0 = jnp.sqrt(jnp.max(A)*jnp.min(A))
        
        
        
    # Reshaping d
    d = jnp.array(d).reshape((grid.shape),order="F")
    
    
    # Solving the damage problem (knowing the previous strain and stress fields)
    if solver_damage == damage_Feugueur_solver:
        # Green operator
        Gamma_0_damage = DamageFeugueurGreenOperator(grid=grid, discretization=discretization_used(grid), A0 = A0)
        
        # Solver
        for solver in [damage_Feugueur_solver]:
            d, it_max_damage, delta_damage = solver(d, A, B, A0, Gamma_0_damage, max_iter = max_iter_damage, tol=tol_damage, verbose=False)


    elif solver_damage == damage_EAEM_solver:
        # Green operator
        Gamma_0_damage = DamageFeugueurGreenOperator(grid=grid, discretization=discretization_used(grid), A0 = A0)
        
        # Solver
        for solver in [damage_EAEM_solver]:
            d, it_max_damage, delta_damage = solver(d, A, B, A0, Gamma_0_damage, max_iter = max_iter_damage, tol=tol_damage, verbose=False)



    it_damage = it_damage + it_max_damage   # used to count the total number of damage iterations per increment



    # Reshaping d
    d = d.flatten(order="F")
    
    
    
    # Irreversibility
    d = jnp.maximum(d,d_old)    # Every damage fields computed in the steps must be greater than the damage field at the previous increment
    
    err_step = jnp.sqrt(jnp.sum((d-d_beginning_step)**2))   # convergence error to go to another increment
    
    # Print step infos
    jax.debug.print("step {} err_step {:.2e} imax_elast {} imax_damage {} err_elast {:.2e} err_damage {:.2e}",
                    step+1,
                    err_step,
                    it_max_elasticity,
                    it_max_damage,
                    delta_elasticity,
                    delta_damage,)
    
    
    return step+1, eps, sig, d, eps_aux, scale, err_step, eps_load, it_damage, d_old





# ---------------------------------------------------
# Definition of the increment loop
# ---------------------------------------------------

# Convergence condition
def cond_fun_incr(state):
    incr, eps, sig, d, eps_aux, scale, les_eps_mean, les_sig_mean, eps_load, les_d, les_W_elast, les_it_damage = state
    return incr<n_incr


# Body function of the increment loop
def body_fun_incr(state):
    incr, eps, sig, d, eps_aux, scale, les_eps_mean, les_sig_mean, eps_load, les_d, les_W_elast, les_it_damage = state

    
    
    # Incremental loading
    eps_load = eps_load + Delta_eps_load
    eps = eps + Delta_eps_load
    
    
    d_old = d
    
    
    # ------------------------------------------------------------------
    # Run step loop (until d converges)
    # ------------------------------------------------------------------
    init_state_step = (0, eps, sig, d, eps_aux, scale, jnp.inf, eps_load, 0, d_old)
    
    step_f_step, eps_f_step, sig_f_step, d_f_step, eps_aux_f_step, scale_f_step, err_step_f_step, eps_load_f_step, it_damage_f, d_old_f= jax.lax.while_loop(
        cond_fun_step, body_fun_step, init_state_step
    )
    
    
    # ---------------------------------------------------------------------
    # Post-treatment
    # ---------------------------------------------------------------------
    
    # Saving mean strain and stress fields to plot the degraded macroscopic strain-stress law
    sigma_xx_mean = jnp.mean(sig_f_step[..., 1, 1].flatten(order="F"))
    eps_xx_mean = jnp.mean(eps_f_step[..., 1, 1].flatten(order="F"))
    
    les_sig_mean = les_sig_mean.at[incr].set(sigma_xx_mean)
    les_eps_mean = les_eps_mean.at[incr].set(eps_xx_mean)
    
    
    # Saving total number of iterations for damage solver : comparison between Feugueur and EAEM damage models
    les_it_damage = les_it_damage.at[incr].set(it_damage_f)
    
    
    # Saving damage field : plot Paraview
    les_d = les_d.at[incr].set(d_f_step)
    
    
    # Saving elastic energy : plot Paraview
    W_elast_f = 0.5 * jnp.einsum("...ij,...ij->...", sig_f_step, eps_f_step)
    les_W_elast = les_W_elast.at[incr].set(W_elast_f.flatten(order="F"))
    
    
    
    
    # Print increment number
    jax.debug.print("\nIncrement {}",incr+1,)


    return incr+1, eps_f_step, sig_f_step, d_f_step, eps_aux_f_step, scale_f_step, les_eps_mean, les_sig_mean, eps_load_f_step, les_d, les_W_elast, les_it_damage




# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------
incr = 0
eps_load = jnp.zeros_like(eps_load_final)
d = damage_init
eps = jnp.zeros_like(eps_load_final)
sig = jnp.zeros_like(eps_load_final)
scale = 1.0
eps_aux = jnp.zeros_like(eps_load_final)
les_eps_mean = jnp.zeros(n_incr)
les_sig_mean = jnp.zeros(n_incr)
les_d = jnp.zeros((n_incr, *d.shape))
les_W_elast = jnp.zeros((n_incr, *d.shape))
les_it_damage = jnp.zeros(n_incr)




# ----------------------------------------------------------------
# Run increment loop
# ----------------------------------------------------------------

init_state_incr = incr, eps, sig, d, eps_aux, scale, les_eps_mean, les_sig_mean, eps_load, les_d, les_W_elast, les_it_damage

incr_f, eps_f, sig_f, d_f, eps_aux_f, scale_f, les_eps_mean, les_sig_mean, eps_load_f, les_d, les_W_elast, les_it_damage = jax.lax.while_loop(
    cond_fun_incr, body_fun_incr, init_state_incr
)
    
    


    
    
# -------------------------------------------------------------------------------
# Saving results of elastic energy and damage fields for Paraview visualization
#--------------------------------------------------------------------------------

for incr in range(n_incr):

    if incr % freq_save == 0:
        pv_grid.cell_data["damage"] = jnp.asarray(les_d[incr])
        pv_grid.cell_data["W_elast"] = jnp.asarray(les_W_elast[incr])

        pv_grid.save(
            file_name.removesuffix(".vti")
            + f"_results_{incr:04d}.vtk"
        )


