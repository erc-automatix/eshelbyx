import numpy as np
import matplotlib.pyplot as plt

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
from eshelbyx.tensors import SymmetricTensor2, IsotropicTensor4, SymmetricTensor4
from eshelbyx.krylov_solver import KrylovAEMSolver
from eshelbyx.solver import A2EMSolver, AEMSolver, MSSolver

current_folder = Path(__file__).parent.resolve()


file_name_None = "eshelbyx/results/Hashin_num_65_results_None.vtk"
#file_name_64 = "eshelbyx/results/Hashin_num_64_results_alpha.vtk"
#file_name_32 = "eshelbyx/results/Hashin_num_32_results_alpha.vtk"
#file_name_16 = "eshelbyx/results/Hashin_num_16_results_alpha.vtk"

grid, pv_grid = load_vtk_grid(file_name_None, marker_name="phase")
#grid_64, pv_grid_64 = load_vtk_grid(file_name_64, marker_name="phase")
#grid_32, pv_grid_32 = load_vtk_grid(file_name_32, marker_name="phase")
#grid_16, pv_grid_16 = load_vtk_grid(file_name_16, marker_name="phase")

Nx, Ny, Nz = pv_grid.dimensions[0]-1, pv_grid.dimensions[1]-1, pv_grid.dimensions[2]-1
#Nx_64, Ny_64, Nz_64 = pv_grid_64.dimensions[0]-1, pv_grid_64.dimensions[1]-1, pv_grid_64.dimensions[2]-1
#Nx_32, Ny_32, Nz_32 = pv_grid_32.dimensions[0]-1, pv_grid_32.dimensions[1]-1, pv_grid_32.dimensions[2]-1
#Nx_16, Ny_16, Nz_16 = pv_grid_16.dimensions[0]-1, pv_grid_16.dimensions[1]-1, pv_grid_16.dimensions[2]-1
bounds = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)

#Nloc = jnp.array([1,2,3])
Nloc = 1

# Extraction of the voxels at y=0 and z=0 over the x coordinate only + refinement
centers = pv_grid.cell_centers().points  



x_c = centers[:, 0]
y_c = centers[:, 1]
z_c = centers[:, 2]


tol = 1e-6
#shift = 1/64/2
shift = 0

mask_center = (jnp.abs(y_c-shift) < tol) & (jnp.abs(x_c-shift) < tol)
les_z = z_c[mask_center]





#-----------------------------------------------------------
# Analytical solution
#-----------------------------------------------------------

N = Nx*10    # Number of plotted points

Rint = 0.2
Rext = 0.4
coating_thickness = Rext-Rint

# Material parameters from Matti Schneider's paper : M. Schneider et al. Computational homogenization of 
# elasticity on a staggered grid, International Journal for Numerical Methods in Engineering, 105, 693–720. (2016)
phi1 = (Rint/Rext)**3
phi2 = 1 - phi1
soft_center = True     # True : pore softer than the coating
chi = 1e12
Keff=1.0
mueff=0.6

if soft_center==True:
    K1 = (9 + (5*((1/chi)-1)*phi2))*Keff / (9 - (4*(1-chi)*phi2))
    K2 = K1*chi
else:
    K1 = (9 - (5*(1-chi)*phi2))*Keff / (9 + (4*((1/chi)-1)*phi2))
    K2 = K1/chi

mu1 = 3*K1/5
mu2 = 3*K2/5

aeff = 1.0
a2 = (3*Keff + 4*mu2)/(3*K2 + 4*mu2)
b2 = (3*((Rint+coating_thickness)**3)*(K2-Keff))/(3*K2 + 4*mu2)
a1 = a2 + (b2/(Rint**3))



# Radius computation
x = np.linspace(bounds[0], bounds[1], N)
y = np.linspace(bounds[2], bounds[3], N)
z = np.linspace(bounds[4], bounds[5], N)



# Minimalist computation
y = 0
z = 0


def sigma(x):
    r = np.sqrt(x**2)
    r_safe = np.maximum(r,1e-12)
    
    if r<=Rint: # pore
        sigma_rr = 3*K1*a1
        sigma_tt = sigma_rr
        sigma_pp = sigma_rr
        
    elif r>Rint and r<=Rext: # coating
        sigma_rr = 3*K2*a2 - (4*mu2*b2/r_safe**3)
        sigma_tt = 3*K2*a2 + (2*mu2*b2/r_safe**3)
        sigma_pp = sigma_tt
        
    else :  # matrix
        sigma_rr = 3*Keff*aeff
        sigma_tt = sigma_rr
        sigma_pp = sigma_rr
    
    
    # Spherical -> cartesian
    theta = np.arccos(z/r_safe)
    phi = np.atan2(y,x)

    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)

    er = np.stack((st*cp, st*sp, ct), axis=-1)
    et = np.stack((ct*cp, ct*sp, -st), axis=-1)
    ep = np.stack((-sp, cp, np.zeros_like(r)), axis=-1)
    
    sigma_xx = sigma_rr*er[...,0]*er[...,0] + sigma_tt*et[...,0]*et[...,0] + sigma_pp*ep[...,0]*ep[...,0]
    sigma_yy = sigma_rr*er[...,1]*er[...,1] + sigma_tt*et[...,1]*et[...,1] + sigma_pp*ep[...,1]*ep[...,1]
    sigma_zz = sigma_rr*er[...,2]*er[...,2] + sigma_tt*et[...,2]*et[...,2] + sigma_pp*ep[...,2]*ep[...,2]
    
    sigma_xy = sigma_rr*er[...,0]*er[...,1] + sigma_tt*et[...,0]*et[...,1] + sigma_pp*ep[...,0]*ep[...,1]
    sigma_xz = sigma_rr*er[...,0]*er[...,2] + sigma_tt*et[...,0]*et[...,2] + sigma_pp*ep[...,0]*ep[...,2]
    sigma_yz = sigma_rr*er[...,1]*er[...,2] + sigma_tt*et[...,1]*et[...,2] + sigma_pp*ep[...,1]*ep[...,2]
    
    return sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz




#les_sigma_zz_an = np.array([sigma(x)[2] for x in les_x])






# ---------------------------------------------------
# Numerical solution (sigma zz over x coordinate)
# ---------------------------------------------------
file_map = {
        "None": "eshelbyx/results/Hashin_num_65_results_None.vtk",
        "Voigt": "eshelbyx/results/Hashin_num_65_results_Voigt.vtk",
        "Reuss": "eshelbyx/results/Hashin_num_65_results_Reuss.vtk",
        "alpha": "eshelbyx/results/Hashin_num_65_results_Reuss.vtk",
        "analytical" : "eshelbyx/results/Hashin_an_65.vti",
        "laminate" : "eshelbyx/results/Hashin_num_65_results_laminate.vtk"
    }
    

# final k0 obtained with the A2EM solver ( = 1.0 in other cases) depending on the hybridation strategy

"""
# chi = 1000, soft
scale_f_alpha = 5.07512
scale_f_None = 4.67975
scale_f_Voigt = 4.79981
scale_f_Reuss = 4.37367
scale_f_laminate = 4.43823


# chi = 1000, hard
scale_f_alpha = 5.11543
scale_f_None = 4.95389
scale_f_Voigt = 4.67618
scale_f_Reuss = 5.07995
scale_f_laminate = 4.77451


# chi = 100, soft
scale_f_alpha = 5.21360
scale_f_None = 4.83149
scale_f_Voigt = 5.03935
scale_f_Reuss = 4.73022
scale_f_laminate = 5.01731


# chi = 10, soft
scale_f_alpha = 6.28828
scale_f_None = 6.05559
scale_f_Voigt = 6.24404
scale_f_Reuss = 6.03526
scale_f_laminate = 6.27183

"""
# chi = 1e12, soft
scale_f_alpha = 5.03266
scale_f_None = 4.66207
scale_f_Voigt = 4.69569
scale_f_Reuss = 4.36596
scale_f_laminate = 4.01811
"""

# chi = 1e12, hard
scale_f_alpha = 5.22902
scale_f_None = 4.94377
scale_f_Voigt = 4.62139
scale_f_Reuss = 5.07826
scale_f_laminate = 4.71176


# chi = 10, hard
scale_f_alpha = 6.15663
scale_f_None = 6.05980
scale_f_Voigt = 5.97373
scale_f_Reuss = 6.20288
scale_f_laminate = 6.06679

# chi = 100, hard
scale_f_alpha = 5.13415
scale_f_None = 5.04561
scale_f_Voigt = 4.83578
scale_f_Reuss = 5.16626
scale_f_laminate = 4.95827
"""

scales_map = {
    "None": scale_f_None,
    "Voigt": scale_f_Voigt,
    "Reuss": scale_f_Reuss,
    "alpha": scale_f_alpha,
    "laminate": scale_f_laminate
}


E_matrix = (9*Keff*mueff)/(3*Keff + mueff)
E_pore = (9*K1*mu1)/(3*K1 + mu1)
E_incl = (9*K2*mu2)/(3*K2 + mu2)
nu_matrix = (3*Keff - 2*mueff)/(2*(3*Keff + mueff))
nu_pore = (3*K1 - 2*mu1)/(2*(3*K1 + mu1))
nu_incl = (3*K2 - 2*mu2)/(2*(3*K2 + mu2))

# Material parameters mapping
E_map = {0: E_matrix, 1: E_pore, 2: E_incl}
nu_map = {0: nu_matrix, 1: nu_pore, 2: nu_incl}


# Solver 
solver_used = A2EMSolver

# Discretization
discretization_used = WillotDiscretization


# Reference medium depending on the solver being used
if solver_used == AEMSolver or solver_used == A2EMSolver or solver_used == KrylovAEMSolver :
    # Reference medium for the AEM and A2EM solvers
    E0 = 1.0
    nu0 = 0.3
    
elif solver_used == MSSolver :
    # Reference medium for MS solver
    E0 = (E_incl+E_pore)/2
    nu0 = (nu_incl+nu_pore)/2

lambda_0, mu0 = E0 * nu0 / (1 + nu0) / (1 - 2 * nu0), E0 / 2 / (1 + nu0)
C0 = IsotropicTensor4(lambda_0 + 2 / 3 * mu0, mu0)

def piecewise_field(field, value_map, grid):
    for index, value in value_map.items():
        field = field.at[grid.markers == index, ...].set(value)
    return field


"""
def main_plot(strat_hybrid, scale_f):
    file_name = file_map.get(strat_hybrid)
    grid, pv_grid = load_vtk_grid(file_name, marker_name="phase")

    Gamma_0 = GreenOperator(grid=grid, C0=C0, discretization=discretization_used(grid))


    # Extraction of the voxels at y=0 and z=0 over the x coordinate only + refinement
    centers = pv_grid.cell_centers().points  
    
    #x_c = centers[:, 0]
    y_c = centers[:, 1]
    z_c = centers[:, 2]
    
    tol = 1e-6
    

    x_line_refined = jnp.linspace(bounds[0], bounds[1], Nx*Nloc)
    
    
    
    # Extraction of the corresponding phases + refinement
    conditions = [
        (x_line_refined > -Rint) & (x_line_refined < Rint),
        ((x_line_refined > -Rext) & (x_line_refined < -Rint)) |
        ((x_line_refined > Rint) & (x_line_refined < Rext)),
    ]
    
    
    
    # Association of the corresonding material parameters
    
    choices_E = [E_pore, E_incl]
    E_line_refined = jnp.select(conditions, choices_E, default=E_matrix)
    
    choices_nu = [nu_pore, nu_incl]
    nu_line_refined = jnp.select(conditions, choices_nu, default=nu_matrix)
    
    material_params_ref = (E_line_refined, nu_line_refined)
    
    
    
    # Extraction of the corresponding auxiliar strain + refinement
    coarse_strain_aux = pv_grid.cell_data["strain_aux"]
    strain_aux_line = coarse_strain_aux[mask_center]
    strain_aux_line_refined_3d = jnp.repeat(strain_aux_line, Nloc, axis=0)
    
    
    
    # Computation of the refined strain with alpha
    def _apply_alpha_ref(eps, material_params_ref, C0):
            
        E, nu = material_params_ref
                
        lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
        mu = E / 2 / (1 + nu)
        C = IsotropicTensor4(lmbda + 2 / 3 * mu, mu)
    
        iC = (C + C0).inv
        alp = iC @ (C0 * 2.0)
        sig = alp.eval_jvp(eps)  # here alp is a IsotropicTensor4
        return SymmetricTensor2(tensor=sig)
    
    @eqx.filter_jit
    def apply_alpha_ref(eps, params, C0):
        eps_flatten = jnp.asarray(eps).reshape(-1, 3, 3)
        sig_tensor = jax.vmap(_apply_alpha_ref, in_axes=(0, 0, None))(eps_flatten, params, C0)
        return sig_tensor.tensor.reshape(eps.shape)
    
    alpha_ref = lambda eps, scale=scale_f : apply_alpha_ref(eps, material_params_ref, C0 * scale)
    
    
    strain_line_refined_3d = alpha_ref(strain_aux_line_refined_3d).reshape((-1, 9), order="F")
    
    
    
    # Computation of the refined stress
    strain_aux_line_refined_bis = strain_aux_line_refined_3d.reshape((Nx*Nloc, 3, 3))
    strain_line_refined_bis = strain_line_refined_3d.reshape((Nx*Nloc, 3, 3))
    stress_line_refined_3d = scale_f*Gamma_0.apply_C0(2.0 * strain_aux_line_refined_bis - strain_line_refined_bis)
    
    
    # Amplitude of the auxiliar strain
    norm_strain_aux = jnp.linalg.norm(strain_aux_line_refined_bis, axis=(1,2))
    
    # Amplitude of the strain
    norm_strain = jnp.linalg.norm(strain_line_refined_bis, axis=(1,2))
    
    
    # Extraction of the zz component of the stress
    stress_ref_zz = stress_line_refined_3d[..., 0, 0]
    stress_ref_zz_flat = jnp.reshape(stress_ref_zz, (-1,))
    
    return x_line_refined, stress_ref_zz_flat, norm_strain_aux, norm_strain


les_x_None, stress_ref_zz_None, norm_strain_aux_None, norm_strain_None = main_plot("None",scale_f_None)
les_x_Voigt, stress_ref_zz_Voigt, norm_strain_aux_Voigt, norm_strain_Voigt = main_plot("Voigt",scale_f_Voigt)
les_x_Reuss, stress_ref_zz_Reuss, norm_strain_aux_Reuss, norm_strain_Reuss = main_plot("Reuss",scale_f_Reuss)
les_x_alpha, stress_ref_zz_alpha, norm_strain_aux_alpha, norm_strain_alpha = main_plot("alpha",scale_f_alpha)
les_x_laminate, stress_ref_zz_laminate, norm_strain_aux_laminate, norm_strain_laminate = main_plot("laminate",scale_f_laminate)




# Plot norm Frobenius strain aux
plt.figure(figsize=(6,4))
plt.plot(les_x, norm_strain_aux_None, label=r"$\|\varepsilon_{aux}^{None}\|$")
plt.plot(les_x, norm_strain_aux_Voigt, label=r"$\|\varepsilon_{aux}^{Voigt}\|$")
plt.plot(les_x, norm_strain_aux_Reuss, label=r"$\|\varepsilon_{aux}^{Reuss}\|$")
plt.plot(les_x, norm_strain_aux_alpha, label=r"$\|\varepsilon_{aux}^{\alpha}\|$")
plt.plot(les_x, norm_strain_aux_laminate, label=r"$\|\varepsilon_{aux}^{laminate}\|$")
plt.axvline(x=-0.4, color='grey', linestyle='--')
plt.axvline(x=-0.2, color='grey', linestyle='--')
plt.axvline(x=0.2, color='grey', linestyle='--')
plt.axvline(x=0.4, color='grey', linestyle='--')
plt.xlabel('x')
plt.ylabel(r"$\|\varepsilon_{aux}\|$")
plt.grid(True)
plt.legend()
plt.grid(True)
plt.legend()
plt.show()


# Plot norm Frobenius strain
plt.figure(figsize=(6,4))
plt.plot(les_x, norm_strain_None, label=r"$\|\varepsilon^{None}\|$")
plt.plot(les_x, norm_strain_Voigt, label=r"$\|\varepsilon^{Voigt}\|$")
plt.plot(les_x, norm_strain_Reuss, label=r"$\|\varepsilon^{Reuss}\|$")
plt.plot(les_x, norm_strain_alpha, label=r"$\|\varepsilon^{\alpha}\|$")
plt.plot(les_x, norm_strain_laminate, label=r"$\|\varepsilon^{laminate}\|$")
plt.axvline(x=-0.4, color='grey', linestyle='--')
plt.axvline(x=-0.2, color='grey', linestyle='--')
plt.axvline(x=0.2, color='grey', linestyle='--')
plt.axvline(x=0.4, color='grey', linestyle='--')
plt.xlabel('x')
plt.ylabel(r"$\|\varepsilon\|$")
plt.grid(True)
plt.legend()
plt.grid(True)
plt.legend()
plt.show()


# Plot stress zz general view
plt.figure(figsize=(6,4))
plt.plot(les_x, les_sigma_zz_an, label=r'$\sigma_{zz}^{an}$', color='black', linestyle='--', linewidth=1)
plt.xlabel('x')
plt.ylabel(r'$\sigma_{zz}$')
plt.grid(True)
plt.legend()
plt.plot(les_x, stress_ref_zz_None, label=r"$\sigma_{zz}^{None}$")
plt.plot(les_x, stress_ref_zz_Voigt, label=r"$\sigma_{zz}^{Voigt}$")
plt.plot(les_x, stress_ref_zz_Reuss, label=r"$\sigma_{zz}^{Reuss}$")
plt.plot(les_x, stress_ref_zz_alpha, label=r"$\sigma_{zz}^{\alpha}$")
plt.plot(les_x, stress_ref_zz_laminate, label=r"$\sigma_{zz}^{laminate}$")
plt.axvline(x=-0.4, color='grey', linestyle='--')
plt.axvline(x=-0.2, color='grey', linestyle='--')
plt.axvline(x=0.2, color='grey', linestyle='--')
plt.axvline(x=0.4, color='grey', linestyle='--')
plt.grid(True)
plt.legend()
plt.show()


# Plot stress zz zoomed view
plt.figure(figsize=(6,4))
plt.plot(les_x, les_sigma_zz_an, label=r'$\sigma_{zz}^{an}$', color='black', linestyle='--', linewidth=1)
plt.xlabel('x')
plt.ylabel(r'$\sigma_{zz}$')
plt.grid(True)
plt.legend()
plt.plot(les_x, stress_ref_zz_None, label=r"$\sigma_{zz}^{None}$")
plt.plot(les_x, stress_ref_zz_Voigt, label=r"$\sigma_{zz}^{Voigt}$")
plt.plot(les_x, stress_ref_zz_Reuss, label=r"$\sigma_{zz}^{Reuss}$")
plt.plot(les_x, stress_ref_zz_alpha, label=r"$\sigma_{zz}^{\alpha}$")
plt.plot(les_x, stress_ref_zz_laminate, label=r"$\sigma_{zz}^{laminate}$")
plt.axvline(x=-0.4, color='grey', linestyle='--')
plt.axvline(x=-0.2, color='grey', linestyle='--')
plt.axvline(x=0.2, color='grey', linestyle='--')
plt.axvline(x=0.4, color='grey', linestyle='--')
ymin, ymax = plt.ylim()
#plt.ylim(3, ymax)
plt.xlim(0.1,0.3)
plt.grid(True)
plt.legend()
plt.show()
"""





E0 = 1.0
nu0 = 0.3
K0 = E0 / (3 * (1 - 2*nu0))
mu0 = E0 / (2 * (1 + nu0))


def refine_pyvista_grid(grid, Nloc):
    nx, ny, nz = grid.dimensions
    sx, sy, sz = grid.spacing
    ox, oy, oz = grid.origin

    new_dims = (
        (nx - 1) * Nloc + 1,
        (ny - 1) * Nloc + 1,
        (nz - 1) * Nloc + 1,
    )

    new_spacing = (
        sx / Nloc,
        sy / Nloc,
        sz / Nloc,
    )
    refined = pv.ImageData()
    refined.dimensions = new_dims
    refined.origin = (ox, oy, oz)
    refined.spacing = new_spacing

    return refined

# ------------------------------------------------------------------------------------------
# L2 adimensional error on the whole domain (with paraview export on a single file)
# ------------------------------------------------------------------------------------------
file_types = ["laminate", "alpha", "None", "Voigt", "Reuss"]
#file_types = ["alpha", "None", "Voigt", "Reuss"]

file_name_an = "eshelbyx/results/Hashin_an_65.vti"


grid_an, pv_grid_an = load_vtk_grid(file_name_an, marker_name="phase")

values_sigma_zz_an = pv_grid_an['sigma_zz_an']
values_sigma_xx_an = pv_grid_an['sigma_xx_an']
values_sigma_yy_an = pv_grid_an['sigma_yy_an']

values_sigma_trace_an = values_sigma_xx_an + values_sigma_yy_an + values_sigma_zz_an

values_eps_zz_an = pv_grid_an['eps_zz_an']

values_W_elast_an = pv_grid_an['W_elast']


err_grid = pv_grid_an.copy()


plt.figure(figsize=(6,4))
# Plot sigma
for ftype in file_types:
    
    file_name_num = f"eshelbyx/results/Hashin_num_65_results_{ftype}.vtk"
    #file_name_num_64 = f"eshelbyx/results/Hashin_num_64_results_{ftype}.vtk"
    #file_name_num_32 = f"eshelbyx/results/Hashin_num_32_results_{ftype}.vtk"
    #file_name_num_16 = f"eshelbyx/results/Hashin_num_16_results_{ftype}.vtk"
    
    grid_num, pv_grid_num = load_vtk_grid(file_name_num, marker_name="phase")
    #grid_num_64, pv_grid_num_64 = load_vtk_grid(file_name_num_64, marker_name="phase")
    #grid_num_32, pv_grid_num_32 = load_vtk_grid(file_name_num_32, marker_name="phase")
    #grid_num_16, pv_grid_num_16 = load_vtk_grid(file_name_num_16, marker_name="phase")
    
    phase = pv_grid_num["phase"]
    #phase_64 = pv_grid_num_64["phase"]
    #phase_32 = pv_grid_num_32["phase"]
    #phase_16 = pv_grid_num_16["phase"]
    
    values_sigma_zz_num = pv_grid_num['stress_zz']
    #values_sigma_zz_num_64 = pv_grid_num_64['stress_zz']
    #values_sigma_zz_num_32 = pv_grid_num_32['stress_zz']
    #values_sigma_zz_num_16 = pv_grid_num_16['stress_zz']
    
    values_W_elast_num = pv_grid_num['W_elast']
    
    """
    # refinement
    refined_sigma_zz_num_32 = refine_pyvista_grid(pv_grid_num_32, 2)
    refined_sigma_zz_num_16 = refine_pyvista_grid(pv_grid_num_16, 4)
    
    
    sigma_zz_32_3d = values_sigma_zz_num_32.reshape((32, 32, 32))
    sigma_zz_16_3d = values_sigma_zz_num_16.reshape((16, 16, 16))

    fine_sigma_zz_32 = jnp.repeat(sigma_zz_32_3d, 2, axis=0)
    fine_sigma_zz_32 = jnp.repeat(fine_sigma_zz_32, 2, axis=1)
    fine_sigma_zz_32 = jnp.repeat(fine_sigma_zz_32, 2, axis=2)
    
    fine_sigma_zz_16 = jnp.repeat(sigma_zz_16_3d, 4, axis=0)
    fine_sigma_zz_16 = jnp.repeat(fine_sigma_zz_16, 4, axis=1)
    fine_sigma_zz_16 = jnp.repeat(fine_sigma_zz_16, 4, axis=2)
    
    
    fine_sigma_zz_32_array = fine_sigma_zz_32.flatten()
    fine_sigma_zz_16_array = fine_sigma_zz_16.flatten()
    """
    
    sigma_zz_max = jnp.max(values_sigma_zz_an)
    W_elast_max = jnp.max(values_W_elast_an)
    
    err_sigma = jnp.abs((values_sigma_zz_num - values_sigma_zz_an) / (sigma_zz_max))*100
    #err_sigma_64 = jnp.abs((values_sigma_zz_num_64 - values_sigma_zz_an) / (sigma_zz_max))*100
    #err_sigma_32 = jnp.abs((fine_sigma_zz_32_array - values_sigma_zz_an) / (sigma_zz_max))*100
    #err_sigma_16 = jnp.abs((fine_sigma_zz_16_array - values_sigma_zz_an) / (sigma_zz_max))*100
    err_W_elast = jnp.abs((values_W_elast_num - values_W_elast_an) / (W_elast_max))*100
    
    err_grid[f'err_{ftype}_sigma'] = err_sigma
    #err_grid[f'err_{ftype}_sigma_64'] = err_sigma_64
    #err_grid[f'err_{ftype}_sigma_32'] = err_sigma_32
    #err_grid[f'err_{ftype}_sigma_16'] = err_sigma_16
    err_grid[f'err_{ftype}_W_elast'] = err_W_elast
    
    
    # Plot error on sigma zz along x
    les_err_plot_sigma = err_sigma[mask_center]
    #les_err_plot_sigma_64 = err_sigma_64[mask_center]
    #les_err_plot_sigma_32 = err_sigma_32[mask_center]
    #les_err_plot_sigma_16 = err_sigma_16[mask_center]
    plt.plot(les_z, les_err_plot_sigma, label=fr'$err_{{L_2}}(\sigma_{{zz}}^{{{ftype}}})$')
    #plt.plot(les_z, les_err_plot_sigma_64, label='N=64')
    #plt.plot(les_z, les_err_plot_sigma_32, label='N=32')
    #plt.plot(les_z, les_err_plot_sigma_16, label='N=16')
    
plt.axvline(x=-0.4, color='grey', linestyle='--')
plt.axvline(x=-0.2, color='grey', linestyle='--')
plt.axvline(x=0.2, color='grey', linestyle='--')
plt.axvline(x=0.4, color='grey', linestyle='--')
plt.xlabel('z')
plt.ylabel("err [%]")
plt.xlim(0.15,0.25)
plt.grid(True)
plt.legend()
plt.show()  


# Plot eps  
plt.figure(figsize=(6,4))  
for ftype in file_types:
    
    file_name_num = f"eshelbyx/results/Hashin_num_65_results_{ftype}.vtk"
    
    grid_num, pv_grid_num = load_vtk_grid(file_name_num, marker_name="phase")
    
    phase = pv_grid_num["phase"]
    values_eps_zz_num = pv_grid_num['eps_zz']

    eps_zz_max = jnp.max(values_eps_zz_an)
    
    err_eps = jnp.abs((values_eps_zz_num - values_eps_zz_an) / (eps_zz_max))*100

    err_grid[f'err_{ftype}_eps'] = err_eps
    
    
    # Plot error on eps zz along x
    les_err_plot_eps = err_eps[mask_center]
    plt.plot(les_z, les_err_plot_eps, label=fr'$err_{{L_2}}(\varepsilon_{{zz}}^{{{ftype}}})$')

plt.axvline(x=-0.4, color='grey', linestyle='--')
plt.axvline(x=-0.2, color='grey', linestyle='--')
plt.axvline(x=0.2, color='grey', linestyle='--')
plt.axvline(x=0.4, color='grey', linestyle='--')
plt.xlabel('z')
plt.ylabel("err [%]")
plt.xlim(0.15,0.25)
plt.grid(True)
plt.legend()
plt.show()      
    

# Plot eps aux
plt.figure(figsize=(6,4))  
for ftype in file_types:
    K0_f = K0*scales_map[ftype]
    mu0_f = mu0*scales_map[ftype]
    E0_f = (9*K0_f*mu0_f) / (3*K0_f + mu0_f)
    nu0_f = (3*K0_f - 2*mu0_f) / 2 / (3*K0_f + mu0_f)
    values_eps_aux_zz_an = 0.5*( ((1+nu0_f)/E0_f)*values_sigma_zz_an - (nu0_f/E0_f)*values_sigma_trace_an + values_eps_zz_an)
    
    file_name_num = f"eshelbyx/results/Hashin_num_65_results_{ftype}.vtk"
    
    grid_num, pv_grid_num = load_vtk_grid(file_name_num, marker_name="phase")
    
    phase = pv_grid_num["phase"]
    values_eps_aux_zz_num = pv_grid_num['eps_aux_zz']
    
    eps_aux_zz_max = jnp.max(values_eps_aux_zz_an)
    
    err_eps_aux = jnp.abs((values_eps_aux_zz_num - values_eps_aux_zz_an) / (eps_aux_zz_max))*100

    err_grid[f'err_{ftype}_eps_aux'] = err_eps_aux
    
    
    # Plot error on eps aux zz along x
    les_err_plot_eps_aux = err_eps_aux[mask_center]
    plt.plot(les_z, les_err_plot_eps_aux, label=fr'$err_{{L_2}}(\varepsilon_{{\alpha,zz}}^{{{ftype}}})$')

plt.axvline(x=-0.4, color='grey', linestyle='--')
plt.axvline(x=-0.2, color='grey', linestyle='--')
plt.axvline(x=0.2, color='grey', linestyle='--')
plt.axvline(x=0.4, color='grey', linestyle='--')
plt.xlabel('z')
plt.ylabel("err [%]")
plt.xlim(0.15,0.25)
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(6,4))
# Plot Welast
for ftype in file_types:
    
    file_name_num = f"eshelbyx/results/Hashin_num_65_results_{ftype}.vtk"
    
    grid_num, pv_grid_num = load_vtk_grid(file_name_num, marker_name="phase")
    
    phase = pv_grid_num["phase"]
    
    values_W_elast_num = pv_grid_num['W_elast']
    
    W_elast_max = jnp.max(values_W_elast_an)
    
    err_W_elast = jnp.abs((values_W_elast_num - values_W_elast_an) / (W_elast_max))*100
    
    err_grid[f'err_{ftype}_W_elast'] = err_W_elast
    
    
    # Plot error on sigma zz along x
    les_err_plot_W_elast = err_W_elast[mask_center]
    plt.plot(les_z, les_err_plot_W_elast, label=fr'$err_{{L_2}}(W_{{elast}}^{{{ftype}}})$')
    #plt.plot(les_z, les_err_plot_sigma_64, label='N=64')
    #plt.plot(les_z, les_err_plot_sigma_32, label='N=32')
    #plt.plot(les_z, les_err_plot_sigma_16, label='N=16')
    
plt.axvline(x=-0.4, color='grey', linestyle='--')
plt.axvline(x=-0.2, color='grey', linestyle='--')
plt.axvline(x=0.2, color='grey', linestyle='--')
plt.axvline(x=0.4, color='grey', linestyle='--')
plt.xlabel('z')
plt.ylabel("err [%]")
plt.xlim(0.15,0.25)
plt.ylim(0,70)
plt.grid(True)
plt.legend()
plt.show() 




# Saving
output_name = "eshelbyx/results/err_grille_all.vti"
err_grid.save(output_name)


print(f"Saved: {output_name}")

