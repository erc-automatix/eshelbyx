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

bounds = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)  # Boundaries of the domain

#Nloc = jnp.array([1,2,3])
Nloc = 1



# --------------------------------------------------------------------------------------
# Extraction of the voxels at x=0 and y=0 over the z coordinate only + refinement
# --------------------------------------------------------------------------------------
centers = pv_grid.cell_centers().points  

x_c = centers[:, 0]
y_c = centers[:, 1]
z_c = centers[:, 2]

tol = 1e-6
#shift = 1/64/2
shift = 0

mask_center = (jnp.abs(y_c-shift) < tol) & (jnp.abs(x_c-shift) < tol)
les_z = z_c[mask_center]




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



# Material parameters of the reference medium 
E0 = 1.0
nu0 = 0.3
K0 = E0 / (3 * (1 - 2*nu0))
mu0 = E0 / (2 * (1 + nu0))


# Function used to refine the 32*32*32 or 16*16*16 grids in order to compare the results with the analytical 64*64*64 grid
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
# Data extraction : analytical solution
# ------------------------------------------------------------------------------------------
file_types = ["laminate", "alpha", "None", "Voigt", "Reuss"]

file_name_an = "eshelbyx/results/Hashin_an_65.vti"  # Analytical solution
grid_an, pv_grid_an = load_vtk_grid(file_name_an, marker_name="phase")

# Data extraction : analytical stress components
values_sigma_zz_an = pv_grid_an['sigma_zz_an']
values_sigma_xx_an = pv_grid_an['sigma_xx_an']
values_sigma_yy_an = pv_grid_an['sigma_yy_an']

values_sigma_trace_an = values_sigma_xx_an + values_sigma_yy_an + values_sigma_zz_an    # tr(sigma)

# Data extraction : analytical strain zz component
values_eps_zz_an = pv_grid_an['eps_zz_an']

# Data extraction : analytical elastic energy
values_W_elast_an = pv_grid_an['W_elast']



# err_grid : file to store the error results
err_grid = pv_grid_an.copy()


# Compute and plot err on sigma zz
plt.figure(figsize=(6,4))
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
    
    err_sigma = jnp.abs((values_sigma_zz_num - values_sigma_zz_an) / (sigma_zz_max))*100
    #err_sigma_64 = jnp.abs((values_sigma_zz_num_64 - values_sigma_zz_an) / (sigma_zz_max))*100
    #err_sigma_32 = jnp.abs((fine_sigma_zz_32_array - values_sigma_zz_an) / (sigma_zz_max))*100
    #err_sigma_16 = jnp.abs((fine_sigma_zz_16_array - values_sigma_zz_an) / (sigma_zz_max))*100
    
    err_grid[f'err_{ftype}_sigma'] = err_sigma
    #err_grid[f'err_{ftype}_sigma_64'] = err_sigma_64
    #err_grid[f'err_{ftype}_sigma_32'] = err_sigma_32
    #err_grid[f'err_{ftype}_sigma_16'] = err_sigma_16
    
    
    # Plot error on sigma zz along z
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




# Compute and plot err on eps zz 
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
    


# Compute and plot err on eps aux zz
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
    
    
    # Plot error on eps aux zz along z
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




# Compute and plot err on W_elast
plt.figure(figsize=(6,4))
for ftype in file_types:
    
    file_name_num = f"eshelbyx/results/Hashin_num_65_results_{ftype}.vtk"
    
    grid_num, pv_grid_num = load_vtk_grid(file_name_num, marker_name="phase")
    
    phase = pv_grid_num["phase"]
    
    values_W_elast_num = pv_grid_num['W_elast']
    
    W_elast_max = jnp.max(values_W_elast_an)
    
    err_W_elast = jnp.abs((values_W_elast_num - values_W_elast_an) / (W_elast_max))*100
    
    err_grid[f'err_{ftype}_W_elast'] = err_W_elast
    
    
    # Plot error on W_elast along z
    les_err_plot_W_elast = err_W_elast[mask_center]
    plt.plot(les_z, les_err_plot_W_elast, label=fr'$err_{{L_2}}(W_{{elast}}^{{{ftype}}})$')

    
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



# Saving the results on a single file
output_name = "eshelbyx/results/err_grille_all.vti"
err_grid.save(output_name)

print(f"Saved: {output_name}")

