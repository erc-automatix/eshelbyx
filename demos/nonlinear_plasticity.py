"""Elastoplastic homogenization with a spatially varying jaxmat behaviour.

Von Mises plasticity with Voce isotropic hardening, phase-wise elastic moduli
and yield stresses, solved by Newton linearisation with AEM as the inner linear
solver.

    python demos/nonlinear_plasticity.py
"""

from pathlib import Path

import jax.numpy as jnp
import jaxmat.materials as jm
from jaxmat.tensors import IsotropicTensor4

import eshelbyx as ex  # enables float64 globally on import
import eshelbyx.nonlinear as NL

HERE = Path(__file__).parent
INPUT = HERE / "3_phase_microstructure_32.vti"

# --------------------------------------------------------------------------- #
# Microstructure and Green operator
# --------------------------------------------------------------------------- #
grid = ex.Grid.from_vti(INPUT, marker_field="phase")
n_cells = grid.num_cells


def broadcast(E33):
    return jnp.broadcast_to(jnp.asarray(E33), (*tuple(grid.shape), 3, 3))


def phase_field(value_map):
    """Flat per-voxel field from a ``{marker: value}`` map."""
    field = jnp.zeros(grid.shape)
    for marker, value in value_map.items():
        field = jnp.where(grid.markers == marker, value, field)
    return field.ravel()


E0, nu0 = 70e3, 0.3
lambda_0 = E0 * nu0 / (1 + nu0) / (1 - 2 * nu0)
mu0 = E0 / 2 / (1 + nu0)
C0 = IsotropicTensor4(kappa=lambda_0 + 2 / 3 * mu0, mu=mu0)

Gamma_0 = ex.GreenOperator(grid, C0, ex.WillotDiscretization(grid))

# --------------------------------------------------------------------------- #
# Behaviour with per-voxel parameters
#
# The parameter leaves are arrays of length n_cells, so eqx.filter_vmap maps the
# behaviour itself per voxel. A behaviour built with scalar parameters would
# share one parameter set across the whole grid.
# --------------------------------------------------------------------------- #
behavior = jm.vonMisesIsotropicHardening(
    elasticity=jm.LinearElasticIsotropic(
        E=phase_field({0: 70e3, 1: 1000e3, 2: 50e3}),
        nu=jnp.full((n_cells,), 0.3),
    ),
    yield_stress=jm.VoceHardening(
        sig0=phase_field({0: 250.0, 1: 100.0, 2: 500.0}),
        sigu=phase_field({0: 350.0, 1: 300.0, 2: 700.0}),
        b=jnp.full((n_cells,), 100.0),
    ),
)
material = NL.jaxmat_material(behavior)
state = behavior.init_state(Nbatch=n_cells)

# --------------------------------------------------------------------------- #
# Uniaxial load path
# --------------------------------------------------------------------------- #
E_path = [jnp.diag(jnp.array([e, -0.5 * e, -0.5 * e])) for e in jnp.linspace(0, 6e-3, 20)]

results, state = NL.solve_incremental(
    ex.AEMSolver,
    E_path,
    material,
    C0.array,
    Gamma_0,
    state,
    broadcast,
    newton_tol=1e-6,
    newton_maxit=15,
    inner_tol=1e-6,
    inner_maxit=1000,
)

print(f"\n{'E11':>10}  {'newton':>6}  {'mean sig11':>12}")
for E33, (_eps, sig, info) in zip(E_path, results, strict=True):
    print(
        f"{float(E33[0, 0]):10.4f}  {info['newton_steps']:6d}  "
        f"{float(jnp.mean(sig, axis=(0, 1, 2))[0, 0]):12.2f}"
    )

eps, sig, _ = results[-1]
grid.save(
    HERE / "3_phase_microstructure_32_plasticity_results.vti",
    fields={"strain": eps, "stress": sig},
)
