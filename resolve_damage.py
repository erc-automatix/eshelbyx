# %%
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
    DamageGreenOperator,
)
from eshelbyx.discretization import (
    MoulinecSuquetDiscretization,
    WillotDiscretization,
    SchneiderDiscretization,
)


from eshelbyx.grid import load_vtk_grid
from eshelbyx.tensors import SymmetricTensor2, IsotropicTensor4, SymmetricTensor4
from eshelbyx.solver import damage_Feugueur_solver, damage_naive_solver

current_folder = Path(__file__).parent.resolve()


# Discretization
discretization_used = WillotDiscretization



# Micro-structure created with microgen.py"
file_name = "eshelbyx/results/damage_num_32.vti"
grid, pv_grid = load_vtk_grid(file_name, marker_name="damage")

N = 32
#Gc = 1.0
bounds = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
dx = (bounds[1] - bounds[0]) / N
#lc = dx*5


A_data = pv_grid.cell_data["A"]
A = jnp.array(A_data).reshape(grid.shape)

B_data = pv_grid.cell_data["B"]
B = jnp.array(B_data).reshape(grid.shape)


A0 = (jnp.max(A)+jnp.min(A))/2

# Data extraction : hybrid booleans (voxel hybrid or not)
damage_init_data = pv_grid.cell_data["damage"]
damage_init = jnp.array(damage_init_data).reshape(grid.shape)




# Green operator
Gamma_0 = DamageGreenOperator(grid=grid, discretization=discretization_used(grid), A0 = A0)

for solver in [damage_Feugueur_solver]:
    tic = time()
    d = solver(damage_init, A, B, A0, Gamma_0, max_iter = 1000, tol=1e-10, verbose=True)
    print(f"{solver.__name__} solver time", time() - tic)
    print("\n")



pv_grid.cell_data["damage_new"] = d.ravel()
pv_grid.save(file_name.removesuffix(".vti") + "_results.vtk")


