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
    DamageFeugueurGreenOperator,
    DamageNaiveGreenOperator,
)
from eshelbyx.discretization import (
    MoulinecSuquetDiscretization,
    WillotDiscretization,
    SchneiderDiscretization,
)


from eshelbyx.grid import load_vtk_grid
from eshelbyx.solver import damage_Feugueur_solver, damage_EAEM_solver

current_folder = Path(__file__).parent.resolve()


# Discretization
discretization_used = WillotDiscretization

# Solver
solver_used = damage_EAEM_solver


# Micro-structure created with microgen.py
file_name = "eshelbyx/results/2phases_damage_64.vti"
grid, pv_grid = load_vtk_grid(file_name, marker_name="phase")


# --------------------
# Material parameters
# --------------------

chi = 1e4 # contrast

A_matrix = 1
A_pore = A_matrix*chi

B_matrix = 1.0
B_pore = B_matrix



# Define phase-wise properties based on grid.markers
def piecewise_field(field, value_map, grid):
    for index, value in value_map.items():
        field = field.at[grid.markers == index, ...].set(value)
    return field


# Material parameters mapping
A_map = {0: A_matrix, 1: A_pore}
B_map = {0: B_matrix, 1: B_pore}
A = piecewise_field(jnp.zeros(grid.shape), A_map, grid)
B = piecewise_field(jnp.zeros(grid.shape), B_map, grid)


# Reference A
if solver_used == damage_Feugueur_solver:
    A0 = (jnp.max(A)+jnp.min(A))/2
elif solver_used == damage_EAEM_solver:
    A0 = jnp.sqrt(jnp.max(A)*jnp.min(A))
    

# Data extraction : damage init (=0)
damage_init_data = pv_grid.cell_data["damage_init"]
damage_init = jnp.array(damage_init_data).reshape(grid.shape)



if solver_used == damage_Feugueur_solver:
    # Green operator
    Gamma_0 = DamageFeugueurGreenOperator(grid=grid, discretization=discretization_used(grid), A0 = A0)
    #Gamma_0_err = DamageNaiveGreenOperator(grid=grid, discretization=discretization_used(grid))
    
    # Solver
    for solver in [damage_Feugueur_solver]:
        tic = time()
        d, it_max_damage, delta_damage = solver(damage_init, A, B, A0, Gamma_0, damage_init, max_iter = 5000, tol=1e-6, verbose=True)
        print(f"{solver.__name__} solver time", time() - tic)
        print("\n")


elif solver_used == damage_EAEM_solver:
    # Green operator
    Gamma_0 = DamageFeugueurGreenOperator(grid=grid, discretization=discretization_used(grid), A0 = A0)
    
    # Solver
    for solver in [damage_EAEM_solver]:
        tic = time()
        d, it_max_damage, delta_damage = solver(damage_init, A, B, A0, Gamma_0, damage_init, max_iter = 5000, tol=1e-6, verbose=True)
        print(f"{solver.__name__} solver time", time() - tic)
        print("\n")
        

        
        
        

pv_grid.cell_data["damage_new"] = d.flatten(order="F")
pv_grid.save(file_name.removesuffix(".vti") + "_results.vtk")


