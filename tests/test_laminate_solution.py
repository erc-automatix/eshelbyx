import pytest
import jax


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import equinox as eqx
from jaxmat.tensors import SymmetricTensor2, IsotropicTensor4, SymmetricTensor4
import jaxmat.materials as jm

from eshelbyx.utils import flatten_batch_dims, reshape_batch_dims
from eshelbyx.green_operator import (
    GreenOperator,
)
from eshelbyx.discretization import (
    MoulinecSuquetDiscretization,
    WillotDiscretization,
    SchneiderDiscretization,
)

from eshelbyx.grid import Grid
from eshelbyx.krylov_solver import KrylovAEMSolver
from eshelbyx.solver import A2EMSolver, AEMSolver, MSSolver

chi = [1.01, 10.0, 0.1]
solvers = [AEMSolver, A2EMSolver, KrylovAEMSolver]
discretizations = [
    MoulinecSuquetDiscretization,
    WillotDiscretization,
    SchneiderDiscretization,
]


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("grid_size", [2])
@pytest.mark.parametrize("contrast", chi)
@pytest.mark.parametrize("discretization", discretizations)
def test_laminate(contrast, solver, grid_size, discretization):

    def layer(x, y, z):
        return jnp.where(z >= 0.5, 2, 1)

    N = grid_size  # Number of voxels in each direction
    grid = Grid.from_function(shape=(N, N, N), marker_fn=layer)

    # Material parameters
    E1 = 1.0
    nu1 = 0.3

    E2 = E1 * contrast
    nu2 = 0.0

    # Reference medium
    E0 = 1.0
    nu0 = 0.15

    # Reference medium stiffness tensor
    C0 = jm.LinearElasticIsotropic(E=E0, nu=nu0).C

    # Material parameters mapping
    materials = {1: {"E": E1, "nu": nu1}, 2: {"E": E2, "nu": nu2}}
    grid = grid.apply_material_map(materials)
    E = grid["E"]
    nu = grid["nu"]

    def elastic_law(eps, material_params):
        # Elastic law at the material point level. Takes a `SymmetricTensor2` as strain input and outputs stress as a `SymmetricTensor2`.
        E, nu = material_params
        lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
        mu = E / 2 / (1 + nu)
        sig = lmbda * jnp.linalg.trace(eps) * jnp.eye(3) + 2 * mu * eps
        return SymmetricTensor2(tensor=sig)

    @eqx.filter_jit
    def constitutive_law(eps, params):
        eps_flatten = flatten_batch_dims(eps)
        sig_flatten = jax.vmap(elastic_law)(eps_flatten, params)
        return reshape_batch_dims(sig_flatten, grid.shape)

    # Material parameters mapping of reference medium for Moulinec and Suquet's solver
    E0 = jnp.full_like(E, E0)
    nu0 = jnp.full_like(nu, nu0)

    material_params_0 = (E0.ravel(), nu0.ravel())

    C_eps_0 = lambda eps: constitutive_law(eps, material_params_0)

    # Computation and application of the alpha 4th-order tensor
    def _apply_alpha(eps, material_params, C0):

        E, nu = material_params

        lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
        mu = E / 2 / (1 + nu)
        C = IsotropicTensor4(kappa=lmbda + 2 / 3 * mu, mu=mu)

        iC = (C + C0).inv
        alp = iC @ (C0 * 2.0)
        return alp @ eps  # here alp is a IsotropicTensor4

    @eqx.filter_jit
    def apply_alpha(eps, params, C0):
        eps_flatten = SymmetricTensor2(tensor=jnp.asarray(eps).reshape(-1, 3, 3))
        sig_tensor = jax.vmap(_apply_alpha, in_axes=(0, 0, None))(
            eps_flatten, params, C0
        )
        return sig_tensor.tensor.reshape(eps.shape)

    material_params = (E.ravel(), nu.ravel())

    alpha = lambda eps, scale=1.0: apply_alpha(eps, material_params, C0 * scale)
    C_eps = lambda eps: constitutive_law(eps, material_params)

    # Green operator
    Gamma_0 = GreenOperator(grid=grid, C0=C0, discretization=discretization(grid))

    def comput_num(E_array):
        E_macro = SymmetricTensor2(array=jnp.asarray(E_array))
        E_macro = grid.broadcast_to_grid(E_macro)  # create a field on the grid

        # Numerical computation depending on the solver being used
        if solver == MSSolver:
            # Moulinec and Suquet's solver
            eps, sig, info = solver(
                E_macro, C_eps, C_eps_0, Gamma_0, tol=1e-10, max_iter=200
            )
        else:
            eps, sig, info = solver(E_macro, alpha, Gamma_0, tol=1e-10, max_iter=200)
        sig_mean_num = jnp.mean(sig, axis=[0, 1, 2])

        return sig_mean_num

    # -----------------------------------------------------------------
    # Analytical solution
    # -----------------------------------------------------------------
    def P_func(n):
        n = jnp.asarray(n)
        delta = jnp.eye(3)

        # Construction du tenseur P_ijkl (3x3x3x3)
        term1 = jnp.einsum("i,jl,m->ijlm", n, delta, n)
        term2 = jnp.einsum("i,jm,l->ijlm", n, delta, n)
        term3 = jnp.einsum("j,il,m->ijlm", n, delta, n)
        term4 = jnp.einsum("j,im,l->ijlm", n, delta, n)

        P4 = SymmetricTensor4(
            tensor=0.5 * (term1 + term2 + term3 + term4)
            - jnp.einsum("i,j,l,m->ijlm", n, n, n, n)
        )
        return P4.array

    mat1 = jm.LinearElasticIsotropic(E=E1, nu=nu1)
    C1 = mat1.C.to_symmetric().array

    mat2 = jm.LinearElasticIsotropic(E=E2, nu=nu2)
    C2 = mat2.C.to_symmetric().array

    K1 = mat1.kappa
    K2 = mat2.kappa
    mu1 = mat1.mu
    mu2 = mat2.mu

    secu = 2  # Safety parameter
    eigenvalues = jnp.array([3 * K1, 3 * K2, 2 * mu1, 2 * mu2])
    lambda_param = jnp.max(eigenvalues) * secu

    n = [0, 0, 1]
    P = P_func(n)
    Id = SymmetricTensor4.identity().array

    A1 = jnp.linalg.inv(P + (lambda_param * (jnp.linalg.inv(C1 - lambda_param * Id))))
    A2 = jnp.linalg.inv(P + (lambda_param * (jnp.linalg.inv(C2 - lambda_param * Id))))
    A = A1 * 0.5 + A2 * 0.5

    C_lam = (Id + (jnp.linalg.inv(jnp.linalg.inv(A) - P))) * lambda_param

    def comput_an(E_array):
        sig_array = jnp.einsum("ij,j->i", C_lam, jnp.array(E_array))
        sig = SymmetricTensor2(array=sig_array)
        return sig

    # ---------------------------------------------------------------------
    # Verification
    # ---------------------------------------------------------------------

    # Macroscopic strain E applied
    E_array_xx = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    E_array_yy = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    E_array_zz = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    E_array_xy = [0.0, 0.0, 0.0, jnp.sqrt(2.0), 0.0, 0.0]

    E_array_xz = [0.0, 0.0, 0.0, 0.0, jnp.sqrt(2.0), 0.0]

    E_array_yz = [0.0, 0.0, 0.0, 0.0, 0.0, jnp.sqrt(2.0)]

    # Numerical stress solutions
    sig_num_xx = comput_num(E_array_xx)
    sig_num_yy = comput_num(E_array_yy)
    sig_num_zz = comput_num(E_array_zz)
    sig_num_xy = comput_num(E_array_xy)
    sig_num_xz = comput_num(E_array_xz)
    sig_num_yz = comput_num(E_array_yz)

    # Analytical stress solutions
    sig_an_xx = comput_an(E_array_xx)
    sig_an_yy = comput_an(E_array_yy)
    sig_an_zz = comput_an(E_array_zz)
    sig_an_xy = comput_an(E_array_xy)
    sig_an_xz = comput_an(E_array_xz)
    sig_an_yz = comput_an(E_array_yz)

    assert jnp.allclose(sig_num_xx, sig_an_xx)
    assert jnp.allclose(sig_num_yy, sig_an_yy)
    assert jnp.allclose(sig_num_zz, sig_an_zz)
    assert jnp.allclose(sig_num_xy, sig_an_xy)
    assert jnp.allclose(sig_num_xz, sig_an_xz)
    assert jnp.allclose(sig_num_yz, sig_an_yz)
    
    
