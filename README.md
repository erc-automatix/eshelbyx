# `eshelbyx`: FFT-based computational homogenization in JAX

`eshelbyx` is an open-source library for computational homogenization of
heterogeneous materials, solving the periodic cell problem on voxel grids with
FFT-based iterative solvers.

## Aim and scope

The library is implemented in [JAX](https://github.com/google/jax) and builds on
[`jaxmat`](https://erc-automatix.github.io/jaxmat/) for constitutive modelling
and tensor algebra, following the same design principles. Specifically:

1. **Robustness to contrast**: `eshelbyx` is built around the adaptive and
   accelerated Eyre–Milton schemes. Reformulating the Lippmann–Schwinger
   equation in terms of the local resolvent gives an operator whose spectrum is
   bounded away from zero and infinity independently of the phase contrast, so
   the solvers converge on microstructures containing rigid inclusions and
   voids, where the classical Moulinec–Suquet fixed point diverges. The
   accelerated variant additionally adapts the reference-medium stiffness as it
   iterates.

2. **Constitutive generality**: any `jaxmat` behaviour can be used as the local
   law, from linear elasticity to elastoviscoplasticity with internal variables.
   Consistent tangents come from automatic differentiation, so no
   behaviour-specific solver code is needed.

3. **Differentiability**: the whole pipeline is written in JAX, so converged
   fields and homogenised properties are differentiable with respect to material
   parameters and microstructure. This is what makes inverse identification of
   constitutive parameters from RVE response tractable.

4. **Efficiency**: solves are JIT-compiled and run on CPU or GPU through XLA.
   The Green operator is matrix-free, costing one forward and one inverse FFT
   per application, and integrates with
   [`lineax`](https://docs.kidger.site/lineax/) and
   [`optimistix`](https://docs.kidger.site/optimistix/) so that Krylov and
   root-finding machinery can be reused rather than reimplemented.

Several discrete Green operators are available (Moulinec–Suquet, Willot,
Schneider) behind a common interface, since the choice of discretization
materially affects accuracy at phase interfaces.

## Installation

Simply run `pip install`. Use `eshelbyx[viz]` for VTK input/output and plotting,
and `eshelbyx[gpu]` for GPU support.

```bash
pip install eshelbyx
```

## Double precision

Importing `eshelbyx` enables JAX double precision globally, as `jaxmat` does.
The solvers are formulated in float64 and do not converge in single precision,
so this is enforced rather than offered as an option. Import `eshelbyx` before
allocating any JAX array; see the *Sharp bits* section of the documentation for
details.

## Example

```python
import eshelbyx as ex  # enables float64 globally on import

import jax.numpy as jnp
import jaxmat.materials as jm
from jaxmat.tensors import SymmetricTensor2

# Spherical inclusion in a matrix
grid = ex.Grid.from_function(
    shape=(32, 32, 32),
    marker_fn=lambda x, y, z: jnp.where(
        (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 < 0.04, 1, 0
    ),
)
grid = grid.apply_material_map({0: {"E": 1.0, "nu": 0.3}, 1: {"E": 100.0, "nu": 0.2}})

# Reference medium and its Green operator
C0 = jm.LinearElasticIsotropic(E=1.0, nu=0.3).C
Gamma_0 = ex.GreenOperator(grid, C0, ex.WillotDiscretization(grid))

# Macroscopic uniaxial strain
E_macro = grid.broadcast_to_grid(SymmetricTensor2(array=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])))

eps, sig, info = ex.AEMSolver(E_macro, alpha, Gamma_0, tol=1e-8)
print(jnp.mean(sig, axis=(0, 1, 2)))
```

The local Eyre–Milton resolvent `alpha` must currently be supplied by the
caller; see `demos/linear_elasticity.py` for the full version. Deriving it
automatically from a `jaxmat` behaviour is the next planned API change.

## Citing

The project can be cited using the metadata in `CITATION.cff`. A Zenodo DOI will
accompany the first tagged release.
