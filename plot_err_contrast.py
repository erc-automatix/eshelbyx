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
from jaxmat.tensors import SymmetricTensor2, IsotropicTensor4, SymmetricTensor4
from eshelbyx.krylov_solver import KrylovAEMSolver
from eshelbyx.solver import A2EMSolver, AEMSolver, MSSolver

current_folder = Path(__file__).parent.resolve()


# Micro-structure file for extracting the hybrid voxels
file_name_num = "eshelbyx/results/hybrid_voxels/64*64*64/chi_1000_soft/Hashin_num_64_results_1000_soft_None.vtk"
grid_num, pv_grid_num = load_vtk_grid(file_name_num, marker_name="phase")
bool_hybrid_num = pv_grid_num["bool_hybrid_pore"]
mask_hybrid = bool_hybrid_num == True
#mask_hybrid = jnp.ones_like(bool_hybrid_num, dtype=bool)
n_true = jnp.sum(mask_hybrid)


file_types = [
    "1e6_soft",
    "1000_soft",
    "100_soft",
    "10_soft",
    "1_soft",
    "10_hard",
    "100_hard",
    "1000_hard",
    "1e6_hard",
]  # Error files for the different contrast values

hybrid_types = ["None", "Voigt", "Reuss", "alpha", "laminate"]


les_contrast_log = jnp.log10(
    jnp.array(([1e-6, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1e6]))
)  # Different contrast values


# Empty arrays to save the results
les_err_None_sigma = []
les_err_Voigt_sigma = []
les_err_Reuss_sigma = []
les_err_laminate_sigma = []
les_err_alpha_sigma = []
les_err_BD_sigma = []

les_err_None_eps = []
les_err_Voigt_eps = []
les_err_Reuss_eps = []
les_err_laminate_eps = []
les_err_alpha_eps = []
les_err_BD_eps = []

les_err_None_W_elast = []
les_err_Voigt_W_elast = []
les_err_Reuss_W_elast = []
les_err_laminate_W_elast = []
les_err_alpha_W_elast = []
les_err_BD_W_elast = []

les_err_None_W_elast_glob = []
les_err_Voigt_W_elast_glob = []
les_err_Reuss_W_elast_glob = []
les_err_laminate_W_elast_glob = []
les_err_alpha_W_elast_glob = []
les_err_BD_W_elast_glob = []



# Loop for each contrast value
for ftype in file_types:
    
    # error file
    file_name_err = f"eshelbyx/results/hybrid_voxels/64*64*64/chi_{ftype}/err_grille_all_{ftype}.vti"
    grid_err, pv_grid_err = load_vtk_grid(file_name_err, marker_name="phase")
    phase = pv_grid_err["phase"]
    
    for htype in hybrid_types : 
        file_name_results = f"eshelbyx/results/hybrid_voxels/64*64*64/chi_{ftype}/Hashin_num_64_results_{ftype}_{htype}.vtk"
        grid_results, pv_grid_results = load_vtk_grid(file_name_results, marker_name="phase")
        
        Nx, Ny, Nz = (
            pv_grid_results.dimensions[0] - 1,
            pv_grid_results.dimensions[1] - 1,
            pv_grid_results.dimensions[2] - 1,
        )
        bounds = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)  # Boundaries of the domain
        Lx = bounds[1]-bounds[0]
        Ly = bounds[3]-bounds[2]
        Lz = bounds[5]-bounds[4]
        
        W_elast = pv_grid_results["W_elast"]
        W_elast_glob = jnp.sum(W_elast)*Lx*Ly*Lz/(Nx*Ny*Nz)
        
        Keff = 1.0
        W_elast_glob_an = 9*Keff/2
        
        err = 100*jnp.abs(W_elast_glob-W_elast_glob_an)/jnp.abs(W_elast_glob_an)
        if htype == "None":
            les_err_None_W_elast_glob.append(err)
        elif htype == "Voigt":
            les_err_Voigt_W_elast_glob.append(err)
        elif htype == "Reuss":
            les_err_Reuss_W_elast_glob.append(err)
        elif htype == "alpha":
            les_err_alpha_W_elast_glob.append(err)
        else:
            les_err_laminate_W_elast_glob.append(err)
        
        
        
        
    # Data extraction

    err_None_eps = pv_grid_err["err_None_eps"]
    err_Voigt_eps = pv_grid_err["err_Voigt_eps"]
    err_Reuss_eps = pv_grid_err["err_Reuss_eps"]
    err_laminate_eps = pv_grid_err["err_laminate_eps"]
    err_alpha_eps = pv_grid_err["err_alpha_eps"]

    err_None_sigma = pv_grid_err["err_None_sigma"]
    err_Voigt_sigma = pv_grid_err["err_Voigt_sigma"]
    err_Reuss_sigma = pv_grid_err["err_Reuss_sigma"]
    err_laminate_sigma = pv_grid_err["err_laminate_sigma"]
    err_alpha_sigma = pv_grid_err["err_alpha_sigma"]

    err_None_W_elast = pv_grid_err["err_None_W_elast"]
    err_Voigt_W_elast = pv_grid_err["err_Voigt_W_elast"]
    err_Reuss_W_elast = pv_grid_err["err_Reuss_W_elast"]
    err_laminate_W_elast = pv_grid_err["err_laminate_W_elast"]
    err_alpha_W_elast = pv_grid_err["err_alpha_W_elast"]
    
    
    

    # Computation of the mean error on the hybrid voxels

    sum_err_None_eps = jnp.sum(err_None_eps[mask_hybrid]) / n_true
    sum_err_Voigt_eps = jnp.sum(err_Voigt_eps[mask_hybrid]) / n_true
    sum_err_Reuss_eps = jnp.sum(err_Reuss_eps[mask_hybrid]) / n_true
    sum_err_laminate_eps = jnp.sum(err_laminate_eps[mask_hybrid]) / n_true
    sum_err_alpha_eps = jnp.sum(err_alpha_eps[mask_hybrid]) / n_true

    sum_err_None_sigma = jnp.sum(err_None_sigma[mask_hybrid]) / n_true
    sum_err_Voigt_sigma = jnp.sum(err_Voigt_sigma[mask_hybrid]) / n_true
    sum_err_Reuss_sigma = jnp.sum(err_Reuss_sigma[mask_hybrid]) / n_true
    sum_err_laminate_sigma = jnp.sum(err_laminate_sigma[mask_hybrid]) / n_true
    sum_err_alpha_sigma = jnp.sum(err_alpha_sigma[mask_hybrid]) / n_true

    sum_err_None_W_elast = jnp.sum(err_None_W_elast[mask_hybrid]) / n_true
    sum_err_Voigt_W_elast = jnp.sum(err_Voigt_W_elast[mask_hybrid]) / n_true
    sum_err_Reuss_W_elast = jnp.sum(err_Reuss_W_elast[mask_hybrid]) / n_true
    sum_err_laminate_W_elast = jnp.sum(err_laminate_W_elast[mask_hybrid]) / n_true
    sum_err_alpha_W_elast = jnp.sum(err_alpha_W_elast[mask_hybrid]) / n_true

    # Adding the mean error to the error list for futur plot

    les_err_None_eps.append(sum_err_None_eps)
    les_err_Voigt_eps.append(sum_err_Voigt_eps)
    les_err_Reuss_eps.append(sum_err_Reuss_eps)
    les_err_laminate_eps.append(sum_err_laminate_eps)
    les_err_alpha_eps.append(sum_err_alpha_eps)

    les_err_None_sigma.append(sum_err_None_sigma)
    les_err_Voigt_sigma.append(sum_err_Voigt_sigma)
    les_err_Reuss_sigma.append(sum_err_Reuss_sigma)
    les_err_laminate_sigma.append(sum_err_laminate_sigma)
    les_err_alpha_sigma.append(sum_err_alpha_sigma)

    les_err_None_W_elast.append(sum_err_None_W_elast)
    les_err_Voigt_W_elast.append(sum_err_Voigt_W_elast)
    les_err_Reuss_W_elast.append(sum_err_Reuss_W_elast)
    les_err_laminate_W_elast.append(sum_err_laminate_W_elast)
    les_err_alpha_W_elast.append(sum_err_alpha_W_elast)




# --------------------------
# Figures
# --------------------------

# Plot error W_elast
plt.figure(figsize=(6, 4))

plt.plot(les_contrast_log, les_err_None_W_elast, "--o", label="None", color="#2ca02c", markersize = 4)
plt.plot(les_contrast_log, les_err_Voigt_W_elast, "--o", label="Voigt", color="#ff7f0e", markersize = 4)
plt.plot(les_contrast_log, les_err_Reuss_W_elast, "--o", label="Reuss", color="#9467bd", markersize = 4)
plt.plot(
    les_contrast_log, les_err_laminate_W_elast, "--o", label="laminate", color="#1f77b4", markersize = 4
)
#plt.plot(les_contrast_log, les_err_BD_W_elast, "--o", label="B-D", color="#8c564b", markersize = 4)
plt.plot(les_contrast_log, les_err_alpha_W_elast, "-o", label="P", color = "#d62728", markersize = 4)

plt.xlabel(r"$\log(c)$")
plt.ylabel(r"$\overline{e_{loc}}^{\mathbb{V}}$ [%]")
xmin, xmax = plt.xlim()
#plt.ylim(0, 2.5)
plt.grid(True)
plt.legend()
plt.show()



# Plot error W_elast_glob
plt.figure(figsize=(6, 4))

plt.plot(les_contrast_log, les_err_None_W_elast_glob, "--o", label="None", color="#2ca02c", markersize = 4)
#plt.plot(les_contrast_log, les_err_Voigt_W_elast_glob, "--o", label="Voigt", color="#ff7f0e", markersize = 4)
#plt.plot(les_contrast_log, les_err_Reuss_W_elast_glob, "--o", label="Reuss", color="#9467bd", markersize = 4)
#plt.plot(les_contrast_log, les_err_laminate_W_elast_glob, "--o", label="laminate", color="#1f77b4", markersize = 4)
#plt.plot(les_contrast_log, les_err_alpha_W_elast_glob, "-o", label="alpha", color = "#d62728", linewidth=3, markersize = 8)

plt.xlabel(r"$\log(c)$")
plt.ylabel(r"$e_{tot}$ [%]")
xmin, xmax = plt.xlim()
plt.ylim(0, 0.3)
plt.grid(True)
#plt.legend()
plt.show()


# ------------------------------------
# Plot err for None with slice x=0
# ------------------------------------
file_types = [
    "1e12_soft",
    "1000_soft",
    "100_soft",
    "10_soft",
    "1_soft",
    "10_hard",
    "100_hard",
    "1000_hard",
    "1e12_hard",
]  # Error files for the different contrast values

contrast_map = {
    "1e12_soft" : "-6",
    "1000_soft" : "-3",
    "100_soft" : "-2",
    "10_soft" : "-1",
    "1_soft" : "0",
    "10_hard" : "1",
    "100_hard" : "2",
    "1000_hard" : "3",
    "1e12_hard" : "6",
    }

# Loop for each contrast value
for ftype in file_types:
    
    # error file
    file_name_err = f"eshelbyx/results/hybrid_voxels/64*64*64/chi_{ftype}/err_grille_all_{ftype}.vti"
    grid_err, pv_grid_err = load_vtk_grid(file_name_err, marker_name="phase")
    phase = pv_grid_err["phase"]

    centers = pv_grid_err.cell_centers().points
    
    x_c = centers[:, 0]
    y_c = centers[:, 1]
    z_c = centers[:, 2]
    
    tol = 1e-6
    shift = 1/64/2
    #shift = 0
    
    mask_center = (jnp.abs(y_c - shift) < tol) & (jnp.abs(z_c - shift) < tol)
    les_x = x_c[mask_center]
    
    err = pv_grid_err["err_None_W_elast"]
    les_err_plot = err[mask_center]
    if ftype == "1e12_hard": 
        plt.plot(les_x,les_err_plot,label=r"$c=10^{12}$")
    elif ftype == "1e12_soft": 
        plt.plot(les_x,les_err_plot,label=r"$c=10^{-12}$")
    elif ftype == "1_soft": 
        plt.plot(les_x,les_err_plot,label=r"$c=1$")
    
    
plt.axvline(x=-0.4, color="black", linestyle="--")
plt.axvline(x=-0.2, color="black", linestyle="--")
plt.axvline(x=0.2, color="black", linestyle="--")
plt.axvline(x=0.4, color="black", linestyle="--")
plt.xlabel("r")
plt.ylabel(r"$e_{loc}^m$ [%]")
plt.xlim(0,0.5)
plt.grid(True)
plt.legend()
plt.show()



