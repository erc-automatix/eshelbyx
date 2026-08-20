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
from eshelbyx.solver import A2EM_generalSolver, AEMSolver, MSSolver

current_folder = Path(__file__).parent.resolve()


# Hybrid voxels treatment depending on the strategy
strat_hybrid = "laminate"

# Solver
solver_used = A2EM_generalSolver

# Discretization
discretization_used = WillotDiscretization


# Number of iterations
n_incr = 1



# Micro-structure created with microgen.py"
file_name = "eshelbyx/results/Hashin_num_32.vti"
grid, pv_grid = load_vtk_grid(file_name, marker_name="phase")



# Size of the grid
Nx = pv_grid.dimensions[0]-1
Ny = pv_grid.dimensions[1]-1
Nz = pv_grid.dimensions[2]-1




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




# --------------------------------------------------------------------------------------------------------------------
# Material parameters from Matti Schneider's paper : M. Schneider et al. Computational homogenization of
# elasticity on a staggered grid, International Journal for Numerical Methods in Engineering, 105, 693–720. (2016)
# ---------------------------------------------------------------------------------------------------------------------
Rint = 0.2
Rext = 0.4
phi1 = (
    Rint / Rext
) ** 3  # Volumic fraction of the central sphere in the set (central sphere + coating)
phi2 = 1 - phi1  # Volumic fraction of the coating in the set (central sphere + coating)
soft_center = False  # True : pore softer than the coating
chi = 1e12  # Material contrast between central sphere and coating


Keff = 1.0
mueff = 0.6

if soft_center == True:
    K1 = (9 + (5 * ((1 / chi) - 1) * phi2)) * Keff / (9 - (4 * (1 - chi) * phi2))
    K2 = K1 * chi
else:
    K1 = (9 - (5 * (1 - chi) * phi2)) * Keff / (9 + (4 * ((1 / chi) - 1) * phi2))
    K2 = K1 / chi

mu1 = 3 * K1 / 5
mu2 = 3 * K2 / 5

E_matrix = (9 * Keff * mueff) / (3 * Keff + mueff)

E_pore = (9 * K1 * mu1) / (3 * K1 + mu1)
E_incl = (9 * K2 * mu2) / (3 * K2 + mu2)
nu_matrix = (3 * Keff - 2 * mueff) / (2 * (3 * Keff + mueff))
nu_pore = (3 * K1 - 2 * mu1) / (2 * (3 * K1 + mu1))
nu_incl = (3 * K2 - 2 * mu2) / (2 * (3 * K2 + mu2))




# ------------------------------------------------------------
# Reference medium depending on the solver being used
# ------------------------------------------------------------

if (
    solver_used == AEMSolver
    or solver_used == A2EM_generalSolver
):
    # Reference medium for the AEM and A2EM solvers
    E0 = 1.0
    nu0 = 0.3

elif solver_used == MSSolver:
    # Reference medium for MS solver
    K_incl = E_incl / (3 * (1 - 2 * nu_incl))
    K_pore = E_pore / (3 * (1 - 2 * nu_pore))
    K0 = (K_incl + K_pore) / 2

    mu_incl = E_incl / (2 * (1 + nu_incl))
    mu_pore = E_pore / (2 * (1 + nu_pore))
    mu0 = (mu_incl + mu_pore) / 2

    E0 = (9 * K0 * mu0) / (3 * K0 + mu0)
    nu0 = (3 * K0 - 2 * mu0) / 2 / (3 * K0 + mu0)
    
    
# Reference medium stiffness tensor written using the IsotropicTensor4 class
lambda_0, mu0 = E0 * nu0 / (1 + nu0) / (1 - 2 * nu0), E0 / 2 / (1 + nu0)

if strat_hybrid == "BD":
    K_max = jnp.maximum(K1,K2)
    K_min = jnp.minimum(K1,K2)
    if soft_center == True:
        scale_BD = K_max/(lambda_0 + (2 / 3) * mu0)*10
    else : 
        scale_BD = K_min/(lambda_0 + (2 / 3) * mu0)/10
else:
    scale_BD = 1.0

C0 = IsotropicTensor4(kappa=lambda_0 + (2 / 3) * mu0, mu=mu0)*scale_BD


# Reference medium stiffness tensor written using the SymmetricTensor4 class (for laminate hybrid voxels)
mat_0 = jm.LinearElasticIsotropic(E=E0, nu=nu0)
C0_aniso = mat_0.C.to_symmetric().array




# ----------------------------------------
# Data extraction
# ----------------------------------------

# Data extraction : hybrid booleans (voxel hybrid or not)
bool_hybrid_data = pv_grid.cell_data["bool_hybrid"]
bool_hybrid = jnp.array(bool_hybrid_data)


# Data extraction : local normal
local_normal_data = pv_grid.cell_data["local_cartesian_normal"]
local_normal = jnp.array(local_normal_data)

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
E_map = {0: E_matrix, 1: E_pore, 2: E_incl}
nu_map = {0: nu_matrix, 1: nu_pore, 2: nu_incl}
E = piecewise_field(jnp.zeros(grid.shape), E_map, grid)
nu = piecewise_field(jnp.zeros(grid.shape), nu_map, grid)


# Internal and external material parameters mapping. For hybrid voxels, internal part of the voxel is the one closer to (0,0,0)
n_cells = phases_couple.shape[0]

E_int_map = {(0, 0): E_matrix, (1, 2): E_pore, (2, 0): E_incl}
nu_int_map = {(0, 0): nu_matrix, (1, 2): nu_pore, (2, 0): nu_incl}
E_int = piecewise_field_couple(jnp.zeros(n_cells), E_int_map, phases_couple)
nu_int = piecewise_field_couple(jnp.zeros(n_cells), nu_int_map, phases_couple)

E_ext_map = {(0, 0): E_matrix, (1, 2): E_incl, (2, 0): E_matrix}
nu_ext_map = {(0, 0): nu_matrix, (1, 2): nu_incl, (2, 0): nu_matrix}
E_ext = piecewise_field_couple(jnp.zeros(n_cells), E_ext_map, phases_couple)
nu_ext = piecewise_field_couple(jnp.zeros(n_cells), nu_ext_map, phases_couple)


pv_grid.cell_data["E_int"] = E_int.flatten()
pv_grid.cell_data["E_ext"] = E_ext.flatten()






# ----------------------------------------------------------------------------------------------------------------------
# Eshelby 4th-order tensor computed with the local cartesian normal n and used for the laminate hybrid voxels strategy
# ----------------------------------------------------------------------------------------------------------------------

def P_func(n):
    n = jnp.asarray(n)
    delta = jnp.eye(3)

    # Construction du tenseur P_ijkl (3x3x3x3)
    term1 = jnp.einsum("i,jl,m->ijlm", n, delta, n)
    term2 = jnp.einsum("i,jm,l->ijlm", n, delta, n)
    term3 = jnp.einsum("j,il,m->ijlm", n, delta, n)
    term4 = jnp.einsum("j,im,l->ijlm", n, delta, n)
    
    P4 = SymmetricTensor4(
        tensor=0.5 * (term1 + term2 + term3 + term4)
        - jnp.einsum("i,j,l,m->ijlm", n, n, n, n)
    )

    return P4.array





# ----------------------------------------------------------------------------------
# Pre-strain loading : eps_load = 0
# ----------------------------------------------------------------------------------

# Pre-strain set at voxel center 
factor_eps_load = 0
Id = SymmetricTensor2(array=jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))

eps_load_final = grid.broadcast_to_grid(Id * factor_eps_load)
eps_load_final_None = jnp.asarray(eps_load_final)   

pv_grid.cell_data["strain_load_final"] = eps_load_final_None.reshape((-1, 3, 3),order="F")


# Incremental loading
Delta_eps_load = eps_load_final_None/n_incr



# Internal and external pre-strains for hybrid voxels
eps_load_final_int = eps_load_final_None.reshape((Nx, Ny, Nz, 3, 3),order="F")
eps_load_final_ext = eps_load_final_None.reshape((Nx, Ny, Nz, 3, 3),order="F")

pv_grid.cell_data["strain_load_final_int"] = (eps_load_final_int.reshape((-1, 3, 3), order="F"))

pv_grid.cell_data["strain_load_final_ext"] = (eps_load_final_ext.reshape((-1, 3, 3), order="F"))

# Incremental loading
Delta_eps_load_int = eps_load_final_int / n_incr
Delta_eps_load_ext = eps_load_final_ext / n_incr






# -----------------------------------------------------------------
# Macroscopic strain and stress applied (must be complementary)
# -----------------------------------------------------------------

# Macroscopic strain
E_macro_final_array = jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
E_macro_final = SymmetricTensor2(array = E_macro_final_array)
E_macro_final = grid.broadcast_to_grid(E_macro_final)  # create a field on the grid

Delta_E_macro = E_macro_final/n_incr


# Macroscopic stress
S_macro_final_array = jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
S_macro_final = SymmetricTensor2(array = S_macro_final_array)
S_macro_final = grid.broadcast_to_grid(S_macro_final)  # create a field on the grid

Delta_S_macro = S_macro_final/n_incr



# Projector for the components of applied macroscopic strain and stress
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


P_tilde = lambda X: apply_P_tilde(X,)






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
    local_normal,
    C0_aniso,
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

    def laminate_branch(_):
        # Considers the interface between 2 materials as a local bi-phase laminate

        E_int, nu_int = material_params_int
        E_ext, nu_ext = material_params_ext
        phi_int = volumic_fraction_list
        
        
        mat_int = jm.LinearElasticIsotropic(E=E_int, nu=nu_int)
        C_int = mat_int.C.to_symmetric().array
        
        mat_ext = jm.LinearElasticIsotropic(E=E_ext, nu=nu_ext)
        C_ext = mat_ext.C.to_symmetric().array
        

        
        n = local_normal
        P_laminate = P_func(n)
        Id = SymmetricTensor4.identity().array
        

        K_int = E_int / (3 * (1 - 2 * nu_int))
        K_ext = E_ext / (3 * (1 - 2 * nu_ext))
        mu_int = E_int / (2 * (1 + nu_int))
        mu_ext = E_ext / (2 * (1 + nu_ext))
        secu = 1.001  # Safety parameter (lambda_param > max(eigenvalues) but not too big)
        eigenvalues = jnp.array([3 * K_int, 3 * K_ext, 2 * mu_int, 2 * mu_ext])
        lambda_param = jnp.max(eigenvalues) * secu


        A_int = jnp.linalg.inv(P_laminate + (lambda_param * (jnp.linalg.inv(C_int - lambda_param * Id))))
        A_ext = jnp.linalg.inv(P_laminate + (lambda_param * (jnp.linalg.inv(C_ext - lambda_param * Id))))
        A = A_int * phi_int + A_ext * (1 - phi_int)
        

        C_lam = (Id + (jnp.linalg.inv(jnp.linalg.inv(A) - P_laminate))) * lambda_param
        
        
        iC_lam = jnp.linalg.inv(C_lam + C0_aniso)
        alp_array = iC_lam @ (C0_aniso * 2.0)
        
        alp = SymmetricTensor4(array = alp_array)
  
        
        sig = alp @ eps  # here alp is a IsotropicTensor4
        return SymmetricTensor2(tensor=sig)
    
    
    def BD_branch(_):
        # Brisard-Dormieux polarization rule
        
        E_int, nu_int = material_params_int
        E_ext, nu_ext = material_params_ext
        phi_int = volumic_fraction_list

        lmbda_int = E_int * nu_int / (1 + nu_int) / (1 - 2 * nu_int)
        mu_int = E_int / 2 / (1 + nu_int)

        lmbda_ext = E_ext * nu_ext / (1 + nu_ext) / (1 - 2 * nu_ext)
        mu_ext = E_ext / 2 / (1 + nu_ext)

        C_int = IsotropicTensor4(kappa=lmbda_int + 2 / 3 * mu_int, mu=mu_int)
        C_ext = IsotropicTensor4(kappa=lmbda_ext + 2 / 3 * mu_ext, mu=mu_ext)
        
        A_int = (C_int-C0).inv
        A_ext = (C_ext-C0).inv
        
        A = A_int * phi_int + A_ext * (1 - phi_int)
        
        C_BD = A.inv + C0
        
        iC = (C_BD + C0).inv
        alp = iC @ (C0 * 2.0)
        sig = alp @ eps  # here alp is a IsotropicTensor4
        return SymmetricTensor2(tensor=sig)


    branch_map = {
        "alpha": mean_alpha_branch,
        "Voigt": Voigt_branch,
        "Reuss": Reuss_branch,
        "None": normal_branch,
        "laminate": laminate_branch,
        "BD" : BD_branch,
    }

    selected_branch = branch_map[strat_hybrid]

    return jax.lax.cond(bool_hybrid_list, selected_branch, normal_branch, operand=None)


@eqx.filter_jit
def apply_alpha(
    eps, params, bool_hybrid, params_int, params_ext, phi, C0, local_normal, C0_aniso
):
    eps_flatten = SymmetricTensor2(tensor=jnp.asarray(eps).reshape((-1, 3, 3),order="F"))
    sig_tensor = jax.vmap(_apply_alpha, in_axes=(0, 0, 0, 0, 0, 0, None, 0, None))(
        eps_flatten,
        params,
        bool_hybrid,
        params_int,
        params_ext,
        phi,
        C0,
        local_normal,
        C0_aniso,
    )
    return sig_tensor.tensor.reshape(eps.shape,order="F")




alpha = lambda eps, scale=1.0: apply_alpha(
    eps,
    material_params,
    bool_hybrid_list,
    material_params_int,
    material_params_ext,
    volumic_fraction_list,
    C0 * scale,
    local_normal,
    C0_aniso * scale,
)





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
    local_normal,
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

        C = IsotropicTensor4(
            kappa=phi_int * C_int.kappa + (1 - phi_int) * C_ext.kappa,
            mu=phi_int * C_int.mu + (1 - phi_int) * C_ext.mu,
        )
        
        sigma_load_int = C_int @ eps_load_int
        sigma_load_ext = C_ext @ eps_load_ext
        
        eps_load_bis = C.inv @ (sigma_load_int*phi_int + sigma_load_ext*(1-phi_int))
        
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
        C = S.inv

        sigma_load_int = C_int @ eps_load_int
        sigma_load_ext = C_ext @ eps_load_ext
        
        eps_load_bis = C.inv @ (sigma_load_int*phi_int + sigma_load_ext*(1-phi_int))
        
        
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

        C_int = IsotropicTensor4(kappa=lmbda_int + 2 / 3 * mu_int, mu=mu_int)
        C_ext = IsotropicTensor4(kappa=lmbda_ext + 2 / 3 * mu_ext, mu=mu_ext)
        
        C = (phi_int*((C_int+C0).inv) + (1-phi_int)*((C_ext+C0).inv)).inv - C0
        
        sigma_load_int = C_int @ eps_load_int
        sigma_load_ext = C_ext @ eps_load_ext
        
        eps_load_bis = C.inv @ (sigma_load_int*phi_int + sigma_load_ext*(1-phi_int))
        
        
        return SymmetricTensor2(tensor=eps_load_bis)
    
    
    def laminate_branch(_):
        # Considers the interface between 2 materials as a local bi-phase laminate

        E_int, nu_int = material_params_int
        E_ext, nu_ext = material_params_ext
        phi_int = volumic_fraction_list
        
        
        mat_int = jm.LinearElasticIsotropic(E=E_int, nu=nu_int)
        C_int = mat_int.C.to_symmetric().array
        
        mat_ext = jm.LinearElasticIsotropic(E=E_ext, nu=nu_ext)
        C_ext = mat_ext.C.to_symmetric().array
        
        
        
        K_int = E_int / (3 * (1 - 2 * nu_int))
        K_ext = E_ext / (3 * (1 - 2 * nu_ext))
        mu_int = E_int / (2 * (1 + nu_int))
        mu_ext = E_ext / (2 * (1 + nu_ext))
        secu = 1.001  # Safety parameter (lambda_param > max(eigenvalues) but not too big)
        eigenvalues = jnp.array([3 * K_int, 3 * K_ext, 2 * mu_int, 2 * mu_ext])
        lambda_param = jnp.max(eigenvalues) * secu
        
        
        n = local_normal
        P_laminate = P_func(n)
        Id = SymmetricTensor4.identity().array

        A_int = jnp.linalg.inv(P_laminate + (lambda_param * (jnp.linalg.inv(C_int - lambda_param * Id))))
        A_ext = jnp.linalg.inv(P_laminate + (lambda_param * (jnp.linalg.inv(C_ext - lambda_param * Id))))
        A = A_int * phi_int + A_ext * (1 - phi_int)
        

        C = (Id + (jnp.linalg.inv(jnp.linalg.inv(A) - P_laminate))) * lambda_param
        
        C_int_sym = SymmetricTensor4(array = C_int)
        C_ext_sym = SymmetricTensor4(array = C_ext)
        sigma_load_int = C_int_sym @ eps_load_int
        sigma_load_ext = C_ext_sym @ eps_load_ext
        
        
        eps_load_bis = SymmetricTensor4(array = jnp.linalg.inv(C)) @ (sigma_load_int*phi_int + sigma_load_ext*(1-phi_int))
        
        return SymmetricTensor2(tensor=eps_load_bis)


    branch_map = {
        "Voigt": Voigt_branch,
        "Reuss": Reuss_branch,
        "None": normal_branch,
        "alpha": mean_alpha_branch,
        "laminate": laminate_branch,
    }

    selected_branch = branch_map[strat_hybrid]

    return jax.lax.cond(bool_hybrid_list, selected_branch, normal_branch, operand=None)


@eqx.filter_jit
def apply_beta(
    eps_load, eps_load_int, eps_load_ext, params, bool_hybrid, params_int, params_ext, phi, C0, local_normal
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
        local_normal,
    )
    return eps_load_bis_tensor.tensor.reshape(eps_load.shape,order="F")


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
    local_normal,
)




# ------------------------
# Green operator
# ------------------------
Gamma_0 = GreenOperator(grid=grid, C0=C0, discretization=discretization_used(grid))




# ---------------------------------------------------
# Definition of the increment loop
# ---------------------------------------------------

# Convergence condition
def cond_fun_incr(state):
    incr, eps, sig, eps_aux, scale, eps_load, eps_load_int, eps_load_ext, E_macro, S_macro = state
    return incr<n_incr



# Body function of the increment loop
def body_fun_incr(state):
    incr, eps, sig, eps_aux, scale, eps_load, eps_load_int, eps_load_ext, E_macro, S_macro = state
    

    # Incremental loading
    eps_load = eps_load + Delta_eps_load
    eps_load_int = eps_load_int + Delta_eps_load_int
    eps_load_ext = eps_load_ext + Delta_eps_load_ext
    
    Delta_eps_load_vox = beta(Delta_eps_load, Delta_eps_load_int, Delta_eps_load_ext, scale)
    
    E_macro = E_macro + Delta_E_macro
    S_macro = S_macro + Delta_S_macro
     
    
    eps = eps + Delta_eps_load_vox
    
    X = E_macro + (1/scale)*Gamma_0.apply_C0_inv(S_macro) - Gamma_0.apply_C0_inv(P_tilde(mean_field(sig))) + P_tilde(mean_field(eps))  - eps - (1/scale)*Gamma_0.mv(sig) + Gamma_0.mv(Gamma_0.apply_C0(eps))
    W = -Gamma_0.mv(sig)
    
           
            
    if solver_used == A2EM_generalSolver:
        # A2EM solver
        for solver in [A2EM_generalSolver]:
            tic = time()
            eps, sig, scale, eps_aux, info, itf, Deltaf = solver(
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
                Gamma_0,
                tol=1e-6,
                max_iter=500,
                auto_scale=True,
                step_adaptation=True,
                verbose=True,
            )
    
            print("Solver info:", info)  # ["num_steps"])
            print(f"{solver.__name__} solver time", time() - tic)
            print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))
            print("\n")
            
    # Print increment number
    jax.debug.print("\nIncrement {}",incr+1,)
            
    return incr+1, eps, sig, eps_aux, scale, eps_load, eps_load_int, eps_load_ext, E_macro, S_macro
    


# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------
incr = 0
eps_load = jnp.zeros_like(eps_load_final_None)
eps_load_int = jnp.zeros_like(eps_load_final_int)
eps_load_ext = jnp.zeros_like(eps_load_final_ext)
eps = jnp.zeros_like(eps_load_final_None)
sig = jnp.zeros_like(eps_load_final_None)
scale = 1.0
eps_aux = jnp.zeros_like(eps_load_final_None)
E_macro = jnp.zeros_like(eps_load_final_None)
S_macro = jnp.zeros_like(eps_load_final_None)




  
# ----------------------------------------------------------------
# Run increment loop
# ----------------------------------------------------------------

init_state_incr = incr, eps, sig, eps_aux, scale, eps_load, eps_load_int, eps_load_ext, E_macro, S_macro

incr_f, eps_f, sig_f, eps_aux_f, scale_f, eps_load_f, eps_load_int_f, eps_load_ext_f, E_macro_f, S_macro_f = jax.lax.while_loop(
    cond_fun_incr, body_fun_incr, init_state_incr
)



# store results
pv_grid.cell_data["strain"] = eps_f.reshape((-1, 3, 3),order="F")
pv_grid.cell_data["stress"] = sig_f.reshape((-1, 3, 3),order="F")


# Extraction of the zz component of the stress
sigma_zz = sig_f[..., 2, 2].flatten(order="F")
pv_grid.cell_data["stress_zz"] = sigma_zz




# Extraction of the elastic energy
W_elast = 0.5 * jnp.einsum("...ij,...ij->...", sig_f, eps_f)
pv_grid.cell_data["W_elast"] = W_elast.flatten(order="F")



# Save for paraview visualization
pv_grid.save(file_name.removesuffix(".vti") + "_results.vtk")






# --------------------------------------
# Verifications laminate hybridation
# --------------------------------------

E_int, nu_int = (E_incl, nu_incl)
E_ext, nu_ext = (E_matrix, nu_matrix)
phi_int = 0.5


mat_int = jm.LinearElasticIsotropic(E=E_int, nu=nu_int)
C_int = mat_int.C.to_symmetric().array

mat_ext = jm.LinearElasticIsotropic(E=E_ext, nu=nu_ext)
C_ext = mat_ext.C.to_symmetric().array



n = jnp.array([1,1,1])
n = n / jnp.linalg.norm(n)

P_laminate = P_func(n)
Id = SymmetricTensor4.identity().array





# verif cours homog
lmbda_int, mu_int = E_int * nu_int / (1 + nu_int) / (1 - 2 * nu_int), E_int / 2 / (1 + nu_int)
lmbda_ext, mu_ext = E_ext * nu_ext / (1 + nu_ext) / (1 - 2 * nu_ext), E_ext / 2 / (1 + nu_ext)

def mean_func(X_int, X_ext):
    return phi_int*X_int + (1-phi_int)*X_ext

C1111 = 2*mean_func(mu_int,mu_ext) + 2*mean_func(lmbda_int*mu_int/(lmbda_int+2*mu_int), lmbda_ext*mu_ext/(lmbda_ext+2*mu_ext)) + (1/mean_func(1/(lmbda_int+2*mu_int), 1/(lmbda_ext + 2*mu_ext)))*mean_func(lmbda_int/(lmbda_int+2*mu_int), lmbda_ext/(lmbda_ext+2*mu_ext))**2 
C1122 = 2*mean_func(lmbda_int*mu_int/(lmbda_int+2*mu_int), lmbda_ext*mu_ext/(lmbda_ext+2*mu_ext)) + (1/mean_func(1/(lmbda_int+2*mu_int), 1/(lmbda_ext + 2*mu_ext)))*mean_func(lmbda_int/(lmbda_int+2*mu_int), lmbda_ext/(lmbda_ext+2*mu_ext))**2 
C1133 = (1/mean_func(1/(lmbda_int+2*mu_int), 1/(lmbda_ext + 2*mu_ext)))*mean_func(lmbda_int/(lmbda_int+2*mu_int), lmbda_ext/(lmbda_ext+2*mu_ext))
C2222 = C1111
C2233 = C1133
C3333 = (1/mean_func(1/(lmbda_int+2*mu_int), 1/(lmbda_ext + 2*mu_ext)))
C2323 = 1/mean_func(1/mu_int, 1/mu_ext)
C3131 = C2323
C1212 = mean_func(mu_int, mu_ext)

C_verif_e3 = jnp.array([[C1111, C1122, C1133, 0, 0, 0],
                     [C1122, C2222, C2233, 0, 0, 0],
                     [C1133, C2233, C3333, 0, 0, 0],
                     [0, 0, 0, 2*C1212, 0, 0],
                     [0, 0, 0, 0, 2*C3131, 0],
                     [0, 0, 0, 0, 0, 2*C2323]])




# ============================================================
# Rotation of C_verif tensor
# ============================================================


# ------------------------------------------------------------
# 1. Matrice de rotation : e3 -> n
# ------------------------------------------------------------

def rotation_e3_to_n(n):
    e3 = jnp.array([0., 0., 1.])

    # Axe de rotation
    axis = jnp.cross(e3, n)
    sin_theta = jnp.linalg.norm(axis)
    cos_theta = jnp.dot(e3, n)

    # Cas particulier : e3 et n parallèles
    def rotation_general(_):
        axis_unit = axis / sin_theta

        # Matrice antisymétrique associée à axis_unit
        K = jnp.array([
            [0., -axis_unit[2], axis_unit[1]],
            [axis_unit[2], 0., -axis_unit[0]],
            [-axis_unit[1], axis_unit[0], 0.]
        ])

        # Rodrigues
        R = (
            jnp.eye(3)
            + sin_theta * K
            + (1. - cos_theta) * (K @ K)
        )

        return R

    def rotation_identity(_):
        return jnp.eye(3)

    return jax.lax.cond(
        sin_theta > 1e-12,
        rotation_general,
        rotation_identity,
        operand=None
    )


def voigt_to_tensor4(Cv):

    C4 = jnp.zeros((3, 3, 3, 3))

    # composantes normales
    C4 = C4.at[0,0,0,0].set(Cv[0,0])
    C4 = C4.at[1,1,1,1].set(Cv[1,1])
    C4 = C4.at[2,2,2,2].set(Cv[2,2])

    C4 = C4.at[0,0,1,1].set(Cv[0,1])
    C4 = C4.at[1,1,0,0].set(Cv[1,0])

    C4 = C4.at[0,0,2,2].set(Cv[0,2])
    C4 = C4.at[2,2,0,0].set(Cv[2,0])

    C4 = C4.at[1,1,2,2].set(Cv[1,2])
    C4 = C4.at[2,2,1,1].set(Cv[2,1])

    # cisaillements :
    # Cv[3,3] = 2*C2323, etc.
    C2323 = Cv[3,3] / 2
    C1313 = Cv[4,4] / 2
    C1212 = Cv[5,5] / 2

    for (i,j,k,l), value in [
        ((1,2,1,2), C2323),
        ((1,2,2,1), C2323),
        ((2,1,1,2), C2323),
        ((2,1,2,1), C2323),

        ((0,2,0,2), C1313),
        ((0,2,2,0), C1313),
        ((2,0,0,2), C1313),
        ((2,0,2,0), C1313),

        ((0,1,0,1), C1212),
        ((0,1,1,0), C1212),
        ((1,0,0,1), C1212),
        ((1,0,1,0), C1212),
    ]:
        C4 = C4.at[i,j,k,l].set(value)

    return C4

# ------------------------------------------------------------
# 3. Rotation du tenseur d'ordre 4
# ------------------------------------------------------------

R = rotation_e3_to_n(n)

C4 = voigt_to_tensor4(C_verif_e3)

C4_rot = jnp.einsum(
    'ia,jb,kc,ld,abcd->ijkl',
    R, R, R, R, C4
)

# ------------------------------------------------------------
# 4. Tenseur d'ordre 4 -> Voigt engineering
# ------------------------------------------------------------

def tensor4_to_voigt(C4):

    Cv = jnp.zeros((6, 6))

    pairs = [
        (0, 0),
        (1, 1),
        (2, 2),
        (1, 2),
        (0, 2),
        (0, 1),
    ]

    for I, (i,j) in enumerate(pairs):
        for J, (k,l) in enumerate(pairs):

            value = C4[i,j,k,l]

            # notation engineering :
            # les colonnes de cisaillement sont multipliées par 2
            if J >= 3:
                value = 2 * value

            Cv = Cv.at[I,J].set(value)

    return Cv


C_verif_rot = tensor4_to_voigt(C4_rot)




les_lambda_param = jnp.array([1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12,1e13,1e14,1e15,1e16])

def calcul_error(lambda_param):

    A_int = jnp.linalg.inv(P_laminate + (lambda_param * (jnp.linalg.inv(C_int - lambda_param * Id))))
    A_ext = jnp.linalg.inv(P_laminate + (lambda_param * (jnp.linalg.inv(C_ext - lambda_param * Id))))
    A = A_int * phi_int + A_ext * (1 - phi_int)
    
    C = (Id + (jnp.linalg.inv(jnp.linalg.inv(A) - P_laminate))) * lambda_param
    
    
    Delta_C = C-C_verif_rot
    
    mask_verif = C_verif_rot > 1e-6

    error_C = jnp.sqrt(jnp.sum(
        jnp.where(mask_verif, (Delta_C / C_verif_rot)**2, 0.0)
    ))
    
    return error_C


les_error_C = [calcul_error(lambda_param) for lambda_param in les_lambda_param]



import matplotlib.pyplot as plt
plt.loglog(les_lambda_param, les_error_C)
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$e_{\mathbb{C}}$")
plt.grid(True)
plt.show()
