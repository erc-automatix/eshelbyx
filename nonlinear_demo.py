# %%
# Nonlinear FFT homogenisation with Newton-AEM with spatially varying
# material properties for jaxmat behaviours.
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from jaxmat.tensors import IsotropicTensor4
import jaxmat.materials as jm


from eshelbyx.green_operator import GreenOperator
from eshelbyx.discretization import WillotDiscretization
from eshelbyx.grid import load_vtk_grid
from eshelbyx.solver import AEMSolver, A2EMSolver, DRSolver

import eshelbyx.nonlinear as NL
import eshelbyx.newton_optx as NLx

# ---------------------------------------------------------------------------
# Grid, reference medium, Green operator
# ---------------------------------------------------------------------------
file_name = "3_phase_microstructure_80.vti"
grid, pv_grid = load_vtk_grid(file_name, marker_name="phase")
ncell = int(jnp.prod(jnp.array(grid.shape)))
broadcast = lambda E33: jnp.broadcast_to(jnp.asarray(E33), tuple(grid.shape) + (3, 3))


def piecewise_flat(value_map):
    """Per-voxel field (flat, shape (n_voxels,)) from a {marker: value} map."""
    field = jnp.zeros(grid.shape)
    for idx, val in value_map.items():
        field = field.at[grid.markers == idx, ...].set(val)
    return field.ravel()


E0, nu0 = 70e3, 0.3
lambda_0, mu0 = E0 * nu0 / (1 + nu0) / (1 - 2 * nu0), E0 / 2 / (1 + nu0)
C0 = IsotropicTensor4(kappa=lambda_0 + 2 / 3 * mu0, mu=mu0)

Gamma_0 = GreenOperator(grid=grid, C0=C0, discretization=WillotDiscretization(grid))

E_f = piecewise_flat({0: 70e3, 1: 1000e3, 2: 50e3})
nu_f = jnp.full((ncell,), 0.3)
sig0_f = piecewise_flat({0: 250.0, 1: 100.0, 2: 500.0})  # phase-wise yield stress
sigu_f = piecewise_flat({0: 350.0, 1: 300.0, 2: 700.0})
b_f = jnp.full((ncell,), 100.0)

behavior = jm.vonMisesIsotropicHardening(
    elasticity=jm.LinearElasticIsotropic(E=E_f, nu=nu_f),  # array leaves
    yield_stress=jm.VoceHardening(sig0=sig0_f, sigu=sigu_f, b=b_f),  # array leaves
)
material = NL.jaxmat_material(behavior)
state = behavior.init_state(Nbatch=ncell)

E_path = [
    jnp.diag(jnp.array([e, -0.5 * e, -0.5 * e])) for e in jnp.linspace(0, 6e-3, 20)
]
results, state = NL.solve_incremental(
    AEMSolver,
    E_path,
    material,
    C0.array,
    Gamma_0,
    state,
    broadcast,
    verbose=True,
    newton_tol=1e-6,
    newton_maxit=15,
    inner_tol=1e-6,
    inner_maxit=1000,
)
for E33, (eps_k, sig_k, info_k) in zip(E_path, results):
    print(
        f"[vonMises hetero] E11={float(E33[0, 0]):.3f}  "
        f"newton_steps={info_k['newton_steps']}  "
        f"mean_sig11={float(jnp.mean(sig_k, axis=(0, 1, 2))[0, 0]):.2f}"
    )

pv_grid.cell_data["strain"] = eps_k.reshape((-1, 3, 3))
pv_grid.cell_data["stress"] = sig_k.reshape((-1, 3, 3))
pv_grid.save(file_name.removesuffix(".vti") + "_nonlinear_results.vtk")
