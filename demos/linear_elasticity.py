"""Linear elastic homogenization of a three-phase microstructure.

Runs the Eyre-Milton family of solvers on a matrix / inclusion / pore
microstructure at infinite contrast (rigid inclusions and voids), and writes the
strain and stress fields next to the input file.

    python demos/linear_elasticity.py
"""

from pathlib import Path
from time import time

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxmat.tensors import IsotropicTensor4, SymmetricTensor2

import eshelbyx as ex  # enables float64 globally on import

HERE = Path(__file__).parent
INPUT = HERE / "3_phase_microstructure_32.vti"

# --------------------------------------------------------------------------- #
# Microstructure
# --------------------------------------------------------------------------- #
grid = ex.Grid.from_vti(INPUT, marker_field="phase")

contrast = jnp.inf  # rigid inclusions and voids
E_ref = 1.0
grid = grid.apply_material_map(
    {
        0: {"E": E_ref, "nu": 0.1},  # matrix
        1: {"E": E_ref / contrast, "nu": 0.35},  # pores
        2: {"E": E_ref * contrast, "nu": 0.2},  # inclusions
    }
)
material_params = (grid["E"].ravel(), grid["nu"].ravel())

# --------------------------------------------------------------------------- #
# Reference medium and Green operator
# --------------------------------------------------------------------------- #
E0, nu0 = 1e-5, 0.3
lambda_0 = E0 * nu0 / (1 + nu0) / (1 - 2 * nu0)
mu0 = E0 / 2 / (1 + nu0)
C0 = IsotropicTensor4(kappa=lambda_0 + 2 / 3 * mu0, mu=mu0)

Gamma_0 = ex.GreenOperator(grid, C0, ex.WillotDiscretization(grid))


# --------------------------------------------------------------------------- #
# Local Eyre-Milton resolvent  alpha = 2 (C + s C0)^-1 (s C0)
#
# This block is boilerplate that every caller currently has to write. Deriving
# it from a jaxmat behaviour is the next API change; see the roadmap.
# --------------------------------------------------------------------------- #
def _apply_alpha(eps, params, C0):
    E, nu = params
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    C = IsotropicTensor4(kappa=lmbda + 2 / 3 * mu, mu=mu)
    return (C + C0).inv @ (C0 * 2.0) @ eps


@eqx.filter_jit
def apply_alpha(eps, params, C0):
    eps_flat = SymmetricTensor2(tensor=jnp.asarray(eps).reshape(-1, 3, 3))
    out = jax.vmap(_apply_alpha, in_axes=(0, 0, None))(eps_flat, params, C0)
    return out.tensor.reshape(eps.shape)


def alpha(eps, scale=1.0):
    return apply_alpha(eps, material_params, C0 * scale)


# --------------------------------------------------------------------------- #
# Solve
# --------------------------------------------------------------------------- #
E_macro = grid.broadcast_to_grid(
    SymmetricTensor2(array=jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
)

eps = sig = None
for solver in (ex.A2EMSolver, ex.AEMSolver, ex.DRSolver):
    tic = time()
    eps, sig, info = solver(E_macro, alpha, Gamma_0, tol=1e-6, max_iter=1000)
    print(f"{solver.__name__:12s} {time() - tic:6.2f} s  info={info}")
    print(f"{'':12s} mean stress = {jnp.mean(sig, axis=(0, 1, 2))[0, 0]:.6f}\n")

grid.save(
    HERE / "3_phase_microstructure_32_results.vti",
    fields={"strain": eps, "stress": sig},
)
