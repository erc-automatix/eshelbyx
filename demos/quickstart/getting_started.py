import jax.numpy as jnp
from jaxmat.tensors import IsotropicTensor4, SymmetricTensor2, HillTensor
import jaxmat.materials as jm
import eshelbyx
from eshelbyx.microstructure import (
    VoxelMicrostructure,
    PeriodicGrid,
    MatrixPhase,
    EllipsoidInclusion,
    MatrixInclusionComposite,
)
from eshelbyx.fft import (
    FFTSolver,
    BasicGreenOperator,
    StaggeredGreenOperator,
    RotatedGreenOperator,
    FinelGreenOperator,
)
from eshelbyx.models import Dilute, MoriTanaka, SelfConsistent, PhaseCluster

print(eshelbyx.__version__)

phase_field = jnp.load("microstructure.py")
E1, nu1 = 30e3, 0.3
E2, nu2 = 200e3, 0.2
materials = {
    1: jm.LinearElasticIsotropic(E1, nu1),
    2: jm.LinearElasticIsotropic(E2, nu2),
}

# 1. Microstructure = geometrical concept only (phases) + reference medium for computing Hill or interaction tensors

## FFT-based microstructure (full-field) -> no reference medium yet ?
N = 128  # can down-sample or up-sample phase_field if grid size is different
micro1 = VoxelMicrostructure(phase_field, grid=PeriodicGrid(N, N, N))
phi1 = micro1.volume_fractions(1)
phi2 = micro1.volume_fractions(2)

## Micromechanics microstructure (mean-field)
matrix = MatrixPhase(phase_id=1)
ell = EllipsoidInclusion(phase_id=2, aspect_ratio=(2, 1, 1), angles=None)
micro2 = MatrixInclusionComposite(matrix, incl=[ell], volume_fractions=[phi1, phi2])
# Ponte Castaneda WIllis: micro = EllipsoidalDistribution(matrix, ellipsoids)

# 2. microstructure + reference medium to compute concentration/localization operators
C0 = IsotropicTensor4(E=1.0, nu=0.3)
## FFT
Gamma = BasicGreenOperator(C0, micro1)
Gamma = StaggeredGreenOperator(C0, micro1)  # staggered
Gamma = RotatedGreenOperator(C0, micro1)  # Willot's rotated grid
Gamma = FinelGreenOperator(C0, micro1)  # Finel discretization

## Micromechanics
P0 = HillTensor(C0, micro2)

## Pre-computed microstructural analysis e.g. using Self-Consistent Clustering Analysis
# (intermediate between full and mean field)
phases = PhaseCluster(file="interaction_tensors.npy", C0=C0)
print(phases.size)
print(phases.volume_fractions())
i, j = 0, 1
print(phases.interaction_tensor(i, j))


# 3. Model
## FFT Lipmann Schwinger resolution
solver = FFTSolver(
    max_iters=100, tol=1e-6, preconditioner="lipmann", fourier_backend="jax"
)

## Micromechanical scheme
model = Dilute(micro2, materials)
C1 = materials[1].C
P0 = P0.update(C0=C1)
model = MoriTanaka(materials, P0)
model = SelfConsistent(P0, micro2)  # fixed point scheme

# 4. Problem
E_macro = SymmetricTensor2()
E_macro = E_macro.at[0, 0].set(0.01)
problem = eshelbyx.SmallStrainHomogenization(macro_strain=E_macro)
## FFT resolution
state = solver.solve(problem, micro1)
Sigma = state.homogenized_stress()
sigma = state.local_stress()
eps = state.local_strain()

## Mean-field homogenization
state = model.solve(problem, micro2)  # micro needed ?
Sigma = state.homogenized_stress()
sigma = state.phase_average_stress()
sigma_fluct = state.second_moment_stress()  # computes average(sig*sig) over each phase
eps = state.phase_average_strain()
eps_fluct = state.second_moment_strain()  # computes average(eps*eps) over each phase
