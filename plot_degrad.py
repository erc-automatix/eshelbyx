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



size_types = [
    "16",
    "32",
    "64"
]  # Error files for the different contrast values

hybrid_types = ["None", "Voigt", "Reuss", "alpha", "laminate"]




les_size = jnp.array(([1,2,3]))


# Empty arrays to save the results
les_err_None = []
les_err_Voigt = []
les_err_Reuss = []
les_err_laminate = []
les_err_alpha = []




# analytical file
file_name_an = "eshelbyx/results/hybrid_voxels/degrad/1e6_hard/Hashin_an_64_1e6_hard.vti"
grid_an, pv_grid_an = load_vtk_grid(file_name_an, marker_name="phase")
Nx, Ny, Nz = (
    pv_grid_an.dimensions[0] - 1,
    pv_grid_an.dimensions[1] - 1,
    pv_grid_an.dimensions[2] - 1,
)
phase_an = pv_grid_an["phase"]
values_W_elast_an = pv_grid_an["W_elast"]
W_elast_an_glob = jnp.sum(values_W_elast_an)/(Nx*Ny*Nz)



# Loop for each grid size
for stype in size_types:
    
    
    for htype in hybrid_types : 
        file_name_results = f"eshelbyx/results/hybrid_voxels/degrad/1e6_hard/Hashin_num_{stype}_results_1e6_hard_{htype}.vtk"
        grid_results, pv_grid_results = load_vtk_grid(file_name_results, marker_name="phase")
        values_W_elast_num = pv_grid_results["W_elast"]
        bool_hybrid = pv_grid_results["bool_hybrid_pore"]
        
        if stype=="64":
            size=64
            fact=1
            
        elif stype=="32":
            size=32
            fact=2
        else:
            size=16
            fact=4
            
        
        # refinement
        W_elast_num_3d = values_W_elast_num.reshape((size, size, size))
        bool_hybrid_3d = bool_hybrid.reshape((size, size, size))

        fine_W_elast_num_3d = jnp.repeat(W_elast_num_3d, fact, axis=0)
        fine_W_elast_num_3d = jnp.repeat(fine_W_elast_num_3d, fact, axis=1)
        fine_W_elast_num_3d = jnp.repeat(fine_W_elast_num_3d, fact, axis=2)
        
        fine_bool_hybrid_3d = jnp.repeat(bool_hybrid_3d, fact, axis=0)
        fine_bool_hybrid_3d = jnp.repeat(fine_bool_hybrid_3d, fact, axis=1)
        fine_bool_hybrid_3d = jnp.repeat(fine_bool_hybrid_3d, fact, axis=2)

        fine_W_elast_num_array = fine_W_elast_num_3d.flatten()
        fine_bool_hybrid_array = fine_bool_hybrid_3d.flatten()
        
        

        mask_hybrid = (fine_bool_hybrid_array == True)
        n_true = jnp.sum(mask_hybrid)

        
        W_elast_max = jnp.max(values_W_elast_an)

        les_err = (jnp.abs((fine_W_elast_num_array - values_W_elast_an) / (W_elast_max)) * 100)
        W_elast_num_glob = jnp.sum(fine_W_elast_num_array)/(64**3)
        
        #err = jnp.sum(les_err[mask_hybrid])/n_true
        err = jnp.sum(les_err)/(64**3)
        #err = jnp.abs(W_elast_an_glob-W_elast_num_glob)/jnp.abs(W_elast_an_glob)
        
        
        if htype == "None":
            les_err_None.append(err)
        elif htype == "Voigt":
            les_err_Voigt.append(err)
        elif htype == "Reuss":
            les_err_Reuss.append(err)
        elif htype == "alpha":
            les_err_alpha.append(err)
        else:
            les_err_laminate.append(err)
        
        
        
        
    




# --------------------------
# Figures
# --------------------------


# Plot error W_elast
les_size = jnp.array(([1,2,3]))
plt.figure(figsize=(6, 4))

plt.plot(les_size, les_err_None, "--o", label="None", color="#2ca02c", markersize=4)
plt.plot(les_size, les_err_Voigt, "--o", label="Voigt", color="#ff7f0e", markersize=4)
plt.plot(les_size, les_err_Reuss, "--o", label="Reuss", color="#9467bd", markersize=4)
plt.plot(les_size, les_err_laminate, "--o", label="laminate", color="#1f77b4", markersize=4)
plt.plot(les_size, les_err_alpha, "-o", label="P", color="#d62728", markersize=4)

plt.xlabel("N")
plt.ylabel(r"$\overline{e_{loc}}^{\mathbb{V}}$ [%]")

# Afficher littéralement 16^3, 32^3, 64^3 sur l'axe x
xticks_labels = [r"$16^3$", r"$32^3$", r"$64^3$"]
plt.xticks(les_size, xticks_labels)
plt.ylim(0,12)
plt.grid(True)
plt.legend()
plt.show()


