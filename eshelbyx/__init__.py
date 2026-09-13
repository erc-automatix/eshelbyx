"""eshelbyx: FFT-based computational homogenization of heterogeneous materials in JAX.

Double precision
----------------
Importing ``eshelbyx`` enables double precision globally, exactly as ``jaxmat``
does::

    jax.config.update("jax_enable_x64", True)

This is deliberate and not optional. The spectral solvers are formulated in
float64 throughout: the Green operator declares a ``float64`` input structure,
the Fourier kernels are built in ``complex128``, and the Eyre-Milton iterations
are terminated on residuals down to ``1e-10``, which float32 cannot resolve.
Running in single precision does not merely lose accuracy, it stops the solvers
converging at all.

Two consequences to be aware of:

* The setting is process-global and must be applied before any JAX array is
  created, so ``import eshelbyx`` has to come before the first array allocation
  in a script. Importing ``jax`` or ``jax.numpy`` first is fine.
* Any other JAX code in the same process also runs in float64 afterwards.

To opt out, which is unsupported, set ``JAX_DEFAULT_DTYPE_BITS`` or revert the
flag after import at your own risk.
"""

import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
# jax.config.update("jax_debug_nans", True)  # raise when encountering nan

from importlib.metadata import version  # noqa: E402

from eshelbyx.discretization import (  # noqa: E402
    DiscretizationScheme,
    MoulinecSuquetDiscretization,
    SchneiderDiscretization,
    WillotDiscretization,
)
from eshelbyx.green_operator import GreenOperator  # noqa: E402
from eshelbyx.grid import Grid  # noqa: E402
from eshelbyx.material import Material  # noqa: E402
from eshelbyx.io import read_vti, write_vti  # noqa: E402
from eshelbyx.krylov_solver import KrylovAEMSolver  # noqa: E402
from eshelbyx.solver import A2EMSolver, AEMSolver, DRSolver, MSSolver  # noqa: E402

__version__ = version("eshelbyx")

__all__ = [
    "A2EMSolver",
    "AEMSolver",
    "DRSolver",
    # Discretization schemes
    "DiscretizationScheme",
    # Green operator
    "GreenOperator",
    # Geometry and data
    "Grid",
    "KrylovAEMSolver",
    # Material
    "Material",
    # Solvers
    "MSSolver",
    "MoulinecSuquetDiscretization",
    "SchneiderDiscretization",
    "WillotDiscretization",
    "__version__",
    "read_vti",
    "write_vti",
]
