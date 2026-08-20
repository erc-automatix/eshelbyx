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

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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
A_pore = 1e4

B_matrix = 1
B_pore = 1



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



# Initial A0
A0 = 1e12
    


# Data extraction : damage init (=0)
damage_init_data = pv_grid.cell_data["damage_init"]
damage_init = jnp.array(damage_init_data).reshape(grid.shape,order="F")




# Damage Green Operator : function of A0 (for EAEM because A0 evolves)
@eqx.filter_jit
def Gamma_0_func(A0):
    return DamageFeugueurGreenOperator(grid=grid, discretization=discretization_used(grid), A0 = A0)

Gamma_0 = lambda A0: Gamma_0_func(A0)


# Array to save the A0 evolution (here of size 20)
les_A0_save = jnp.zeros(20)



if solver_used == damage_Feugueur_solver:
    # Green operator
    Gamma_0 = DamageFeugueurGreenOperator(grid=grid, discretization=discretization_used(grid), A0 = A0)
    
    # Solver
    for solver in [damage_Feugueur_solver]:
        tic = time()
        d, it_max_damage, delta_damage = solver(damage_init, A, B, A0, Gamma_0, damage_init, max_iter = 500, tol=1e-6, verbose=True)
        print(f"{solver.__name__} solver time", time() - tic)
        print("\n")


elif solver_used == damage_EAEM_solver:
    # Solver
    for solver in [damage_EAEM_solver]:
        tic = time()
        d, it_max_damage, delta_damage, A0_new, les_A0_save = solver(damage_init, A, B, A0, Gamma_0, damage_init, les_A0_save, max_iter = 1000, tol=1e-6, verbose=True)
        print(f"{solver.__name__} solver time", time() - tic)
        print("\n")
        

        

pv_grid.cell_data["damage_new"] = d.flatten(order="F")
pv_grid.save(file_name.removesuffix(".vti") + "_results.vtk")



# ----------------------------------
# Plot evolution A0
# -----------------------------------
les_A0_save = jnp.concatenate((jnp.array([1e12]), les_A0_save))
les_it = jnp.array([it for it in range(0,jnp.size(les_A0_save))])
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(les_it, les_A0_save, "-x")
ax.set_xlabel("Iteration")
ax.set_ylabel(r"$A_0$")
ax.set_yscale("log")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(True, which="both", linestyle="--", alpha=0.7)
ax.axhline(
    A0_new,
    color="black",
    linestyle="--",
    label=r"$A_0^{EA2EM}$"
)
plt.legend()
plt.savefig("evolution_A0_1e4.pdf")
plt.show()





"""
# ------------------------
# Verifications
# ------------------------
A0_opt_EAEM = 50



# FFT operator : X/(A0 + ξ.ξ)
def OperatorFFT(X,Gamma_0):
    return Gamma_0.mv(X)

# L2 norm of the field X
def norm2(X):
    return jnp.sqrt(jnp.sum(X**2))


les_A0 = jnp.array([1,5,10,20,30,40,50,58,70,80,90,100])



def func_test(A0,d):
    Gamma_0 = DamageFeugueurGreenOperator(grid=grid, discretization=discretization_used(grid), A0 = A0)
    sigma_n = A*d
    err = (A0**2)*jnp.mean(d**2) + A0*jnp.mean(OperatorFFT(B,Gamma_0)*B)-jnp.mean(sigma_n**2)
    return err


les_err = [func_test(A0,d) for A0 in les_A0]

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(
    les_A0,
    les_err,
    "-o"
)

ax.set_xlabel(r"$A_0$")
ax.set_ylabel(r"$e_{A_0}$")

# Ligne verticale
ax.axvline(
    A0_opt_EAEM,
    color="black",
    linestyle="--",
    label=r"$A_0^{EA2EM}$"
)

# Légende
ax.legend(
)

ax.grid(True)

plt.savefig("err_EAEM_1e4.pdf", format="pdf", bbox_inches="tight")
plt.show()
"""



