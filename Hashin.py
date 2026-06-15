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
file_name = "eshelbyx/results/Hashin_num_64.vti"
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


# Data extraction : local normal
local_normal_data = pv_grid.cell_data["local_cartesian_normal"]
local_normal = jnp.array(local_normal_data)


# Material parameters from Matti Schneider's paper : M. Schneider et al. Computational homogenization of
# elasticity on a staggered grid, International Journal for Numerical Methods in Engineering, 105, 693–720. (2016)
Rint = 0.2
Rext = 0.4
phi1 = (
    Rint / Rext
) ** 3  # Volumic fraction of the central sphere in the set (central sphere + coating)
phi2 = 1 - phi1  # Volumic fraction of the coating in the set (central sphere + coating)
soft_center = True  # True : pore softer than the coating
chi = 100  # Material contrast between central sphere and coating



# Material parameters
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


# Reference medium depending on the solver being used
if (
    solver_used == AEMSolver
    or solver_used == A2EMSolver
    or solver_used == KrylovAEMSolver
    or solver_used == A2EM_iterativSolver
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


# Reference medium stiffness tensor written using the SymmetricTensor4 class
mat_0 = jm.LinearElasticIsotropic(E=E0, nu=nu0)
C0_aniso = mat_0.C.to_symmetric().array


# Material parameters mapping
E_map = {0: E_matrix, 1: E_pore, 2: E_incl}
nu_map = {0: nu_matrix, 1: nu_pore, 2: nu_incl}
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
E0_map = {0: E0, 1: E0, 2: E0}
nu0_map = {0: nu0, 1: nu0, 2: nu0}
E0 = piecewise_field(jnp.zeros(grid.shape), E0_map, grid)
nu0 = piecewise_field(jnp.zeros(grid.shape), nu0_map, grid)

material_params_0 = (E0.ravel(), nu0.ravel())

C_eps_0 = lambda eps: constitutive_law(eps, material_params_0)


# Maximum eigenvalue of the stiffness matrices used for the laminate hybrid voxels strategy
secu = 2  # Safety parameter
eigenvalues = jnp.array([3 * Keff, 3 * K1, 3 * K2, 2 * mueff, 2 * mu1, 2 * mu2])
lambda_param = jnp.max(eigenvalues) * secu



# Eshelby 4th-order tensor computed with the local cartesian normal n and used for the laminate hybrid voxels strategy
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


# Computation and application of the alpha 4th-order tensor depending on the hybridation strategy for hybrid voxels
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
        P = P_func(n)
        Id = SymmetricTensor4.identity().array

        A_int = jnp.linalg.inv(P + (lambda_param * (jnp.linalg.inv(C_int - lambda_param * Id))))
        A_ext = jnp.linalg.inv(P + (lambda_param * (jnp.linalg.inv(C_ext - lambda_param * Id))))
        A = A_int * phi_int + A_ext * (1 - phi_int)
        

        C_lam = (Id + (jnp.linalg.inv(jnp.linalg.inv(A) - P))) * lambda_param
        
        
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
    eps_flatten = SymmetricTensor2(tensor=jnp.asarray(eps).reshape(-1, 3, 3))
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
    return sig_tensor.tensor.reshape(eps.shape)


material_params = (E.ravel(), nu.ravel())
bool_hybrid_list = bool_hybrid.ravel()
material_params_int = (E_int.ravel(), nu_int.ravel())
material_params_ext = (E_ext.ravel(), nu_ext.ravel())
volumic_fraction_list = volumic_fraction.ravel()


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
C_eps = lambda eps: constitutive_law(eps, material_params)



n_incr = 10

# Macroscopic strain E applied
E_macro_final = SymmetricTensor2(array=jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
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
    
    # Numerical computation depending on the solver being used
    if solver_used == MSSolver:
        # Moulinec and Suquet's solver
        for solver in [MSSolver]:
            tic = time()
            eps, sig, info = solver(E_macro, C_eps, C_eps_0, Gamma_0, tol=1e-6, max_iter=5)
            print("Solver info:", info)  # ["num_steps"])
            print(f"{solver.__name__} solver time", time() - tic)
            print("Mean stress", jnp.mean(sig, axis=[0, 1, 2]))
            print("\n")
    
    elif solver_used == AEMSolver:
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
            eps, sig, info, scale_old, eps_aux_old = solver(
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
pv_grid.cell_data["strain"] = eps.reshape((-1, 3, 3))
pv_grid.cell_data["stress"] = sig.reshape((-1, 3, 3))


# Extraction of the zz component of the stress
sigma_zz = sig[..., 2, 2].ravel()
pv_grid.cell_data["stress_zz"] = sigma_zz


# Extraction of the zz component of the strain
eps_zz = eps[..., 2, 2].ravel()
pv_grid.cell_data["eps_zz"] = eps_zz

# Extraction of the elastic energy
W_elast = 0.5 * jnp.einsum("...ij,...ij->...", sig, eps)
pv_grid.cell_data["W_elast"] = W_elast.ravel()



pv_grid.save(file_name.removesuffix(".vti") + "_results.vtk")
