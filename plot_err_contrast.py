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


# Micro-structure file for extracting the hybrid voxels
file_name_num = f"eshelbyx/results/Hashin_num_65.vti"
grid_num, pv_grid_num = load_vtk_grid(file_name_num, marker_name="phase")
bool_hybrid_num = pv_grid_num['bool_hybrid']
mask_hybrid = (bool_hybrid_num == True)
n_true = jnp.sum(mask_hybrid)



file_types = ["1e-12","0001", "001", "01", "10", "100", "1000","1e12"]  # Error files for the different contrast values
les_contrast_log = jnp.log10(jnp.array(([1e-12,0.001,0.01,0.1,1,10,100,1000, 1e12])))   # Different contrast values


# Empty arrays to save the results
les_err_None_eps_aux = []
les_err_Voigt_eps_aux = []
les_err_Reuss_eps_aux = []
les_err_laminate_eps_aux = []
les_err_alpha_eps_aux = []

les_err_None_sigma = []
les_err_Voigt_sigma = []
les_err_Reuss_sigma = []
les_err_laminate_sigma = []
les_err_alpha_sigma = []

les_err_None_eps = []
les_err_Voigt_eps = []
les_err_Reuss_eps = []
les_err_laminate_eps = []
les_err_alpha_eps = []
    
les_err_None_W_elast = []
les_err_Voigt_W_elast = []
les_err_Reuss_W_elast = []
les_err_laminate_W_elast = []
les_err_alpha_W_elast = []


# Loop for each contrast value
for ftype in file_types:
    
    file_name = f"eshelbyx/results/err_grille_all_{ftype}.vti"  # Error file
    grid, pv_grid = load_vtk_grid(file_name, marker_name="phase")
    phase = pv_grid["phase"]
    
    # Data extraction
    err_None_eps_aux = pv_grid['err_None_eps_aux']
    err_Voigt_eps_aux = pv_grid['err_Voigt_eps_aux']
    err_Reuss_eps_aux = pv_grid['err_Reuss_eps_aux']
    err_laminate_eps_aux = pv_grid['err_laminate_eps_aux']
    err_alpha_eps_aux = pv_grid['err_alpha_eps_aux']
    
    err_None_eps = pv_grid['err_None_eps']
    err_Voigt_eps = pv_grid['err_Voigt_eps']
    err_Reuss_eps = pv_grid['err_Reuss_eps']
    err_laminate_eps = pv_grid['err_laminate_eps']
    err_alpha_eps = pv_grid['err_alpha_eps']
    
    err_None_sigma = pv_grid['err_None_sigma']
    err_Voigt_sigma = pv_grid['err_Voigt_sigma']
    err_Reuss_sigma = pv_grid['err_Reuss_sigma']
    err_laminate_sigma = pv_grid['err_laminate_sigma']
    err_alpha_sigma = pv_grid['err_alpha_sigma']
    
    err_None_W_elast = pv_grid['err_None_W_elast']
    err_Voigt_W_elast = pv_grid['err_Voigt_W_elast']
    err_Reuss_W_elast = pv_grid['err_Reuss_W_elast']
    err_laminate_W_elast = pv_grid['err_laminate_W_elast']
    err_alpha_W_elast = pv_grid['err_alpha_W_elast']
    
    
    # Computation of the mean error on the hybrid voxels
    sum_err_None_eps_aux = jnp.sum(err_None_eps_aux[mask_hybrid])/n_true
    sum_err_Voigt_eps_aux = jnp.sum(err_Voigt_eps_aux[mask_hybrid])/n_true
    sum_err_Reuss_eps_aux = jnp.sum(err_Reuss_eps_aux[mask_hybrid])/n_true
    sum_err_laminate_eps_aux = jnp.sum(err_laminate_eps_aux[mask_hybrid])/n_true
    sum_err_alpha_eps_aux = jnp.sum(err_alpha_eps_aux[mask_hybrid])/n_true
    
    sum_err_None_eps = jnp.sum(err_None_eps[mask_hybrid])/n_true
    sum_err_Voigt_eps = jnp.sum(err_Voigt_eps[mask_hybrid])/n_true
    sum_err_Reuss_eps = jnp.sum(err_Reuss_eps[mask_hybrid])/n_true
    sum_err_laminate_eps = jnp.sum(err_laminate_eps[mask_hybrid])/n_true
    sum_err_alpha_eps = jnp.sum(err_alpha_eps[mask_hybrid])/n_true
    
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
    les_err_None_eps_aux.append(sum_err_None_eps_aux)
    les_err_Voigt_eps_aux.append(sum_err_Voigt_eps_aux)
    les_err_Reuss_eps_aux.append(sum_err_Reuss_eps_aux)
    les_err_laminate_eps_aux.append(sum_err_laminate_eps_aux)
    les_err_alpha_eps_aux.append(sum_err_alpha_eps_aux)
    
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
 
    
 
    
# ------------------------------------------------------------------
# Adding "0" value to the error list for contrast=1
# ------------------------------------------------------------------

def insert_zero(arr, idx):
    arr = jnp.array(arr)
    return jnp.concatenate([arr[:idx], jnp.array([0.0]), arr[idx:]])

les_err_None_eps_aux = insert_zero(les_err_None_eps_aux, 4)
les_err_Voigt_eps_aux = insert_zero(les_err_Voigt_eps_aux, 4)
les_err_Reuss_eps_aux = insert_zero(les_err_Reuss_eps_aux, 4)
les_err_laminate_eps_aux = insert_zero(les_err_laminate_eps_aux, 4)
les_err_alpha_eps_aux = insert_zero(les_err_alpha_eps_aux, 4)

les_err_None_eps = insert_zero(les_err_None_eps, 4)
les_err_Voigt_eps = insert_zero(les_err_Voigt_eps, 4)
les_err_Reuss_eps = insert_zero(les_err_Reuss_eps, 4)
les_err_laminate_eps = insert_zero(les_err_laminate_eps, 4)
les_err_alpha_eps = insert_zero(les_err_alpha_eps, 4)

les_err_None_sigma = insert_zero(les_err_None_sigma, 4)
les_err_Voigt_sigma = insert_zero(les_err_Voigt_sigma, 4)
les_err_Reuss_sigma = insert_zero(les_err_Reuss_sigma, 4)
les_err_laminate_sigma = insert_zero(les_err_laminate_sigma, 4)
les_err_alpha_sigma = insert_zero(les_err_alpha_sigma, 4)

les_err_None_W_elast = insert_zero(les_err_None_W_elast, 4)
les_err_Voigt_W_elast = insert_zero(les_err_Voigt_W_elast, 4)
les_err_Reuss_W_elast = insert_zero(les_err_Reuss_W_elast, 4)
les_err_laminate_W_elast = insert_zero(les_err_laminate_W_elast, 4)
les_err_alpha_W_elast = insert_zero(les_err_alpha_W_elast, 4)




# --------------------------
# Figures
# --------------------------

# Plot error eps aux
plt.figure(figsize=(6,4))  
plt.plot(les_contrast_log,les_err_None_eps_aux, "-x",label="None")
plt.plot(les_contrast_log,les_err_Voigt_eps_aux,"-x", label="Voigt")
plt.plot(les_contrast_log,les_err_Reuss_eps_aux,"-x", label="Reuss")
plt.plot(les_contrast_log,les_err_laminate_eps_aux,"-x", label="laminate")
plt.plot(les_contrast_log,les_err_alpha_eps_aux,"-x", label="alpha")

plt.xlabel('log(contrast)')
plt.ylabel("err [%]")
plt.title(r"$\text{err}(\varepsilon_{\alpha})$")
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax)
plt.grid(True)
plt.legend()
plt.show()


# Plot error eps
plt.figure(figsize=(6,4))  

plt.plot(les_contrast_log, les_err_None_eps, "-x", label="None")
plt.plot(les_contrast_log, les_err_Voigt_eps, "-x", label="Voigt")
plt.plot(les_contrast_log, les_err_Reuss_eps, "-x", label="Reuss")
plt.plot(les_contrast_log, les_err_laminate_eps, "-x", label="laminate")
plt.plot(les_contrast_log, les_err_alpha_eps, "-x", label="alpha")

plt.xlabel('log(contrast)')
plt.ylabel("err [%]")
plt.title(r"$\text{err}(\varepsilon)$")
xmin, xmax = plt.xlim()
plt.xlim(0, xmax)
#plt.ylim(0, 7)
plt.grid(True)
plt.legend()
plt.show()


# Plot error sigma
plt.figure(figsize=(6,4))  

plt.plot(les_contrast_log, les_err_None_sigma, "-x", label="None")
plt.plot(les_contrast_log, les_err_Voigt_sigma, "-x", label="Voigt")
plt.plot(les_contrast_log, les_err_Reuss_sigma, "-x", label="Reuss")
plt.plot(les_contrast_log, les_err_laminate_sigma, "-x", label="laminate")
plt.plot(les_contrast_log, les_err_alpha_sigma, "-x", label="alpha")

plt.xlabel('log(contrast)')
plt.ylabel("err [%]")
plt.title(r"$\text{err}(\sigma)$")
xmin, xmax = plt.xlim()
plt.xlim(xmin, 0)
#plt.ylim(0, 8)
plt.grid(True)
plt.legend()
plt.show()
 
    
# Plot error W_elast
plt.figure(figsize=(6,4))  

plt.plot(les_contrast_log, les_err_None_W_elast, "-x", label="None", color="#2ca02c")
plt.plot(les_contrast_log, les_err_Voigt_W_elast, "-x", label="Voigt", color= "#d62728")
plt.plot(les_contrast_log, les_err_Reuss_W_elast, "-x", label="Reuss", color = "#9467bd")
plt.plot(les_contrast_log, les_err_laminate_W_elast, "-x", label="laminate", color="#1f77b4")
plt.plot(les_contrast_log, les_err_alpha_W_elast, "-x", label="alpha", color="#ff7f0e")

plt.xlabel('log(contrast)')
plt.ylabel("err [%]")
plt.title(r"$\text{err}(W_{elast})$")
xmin, xmax = plt.xlim()
#plt.ylim(0, 8)
plt.grid(True)
plt.legend()
plt.show()
    
    
    
    
    
    

