"""Tests for eshelbyx.material.

Run with: pytest tests/test_material.py -v
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxmat.materials as jm
import pytest
from jaxmat.tensors import IsotropicTensor4, SymmetricTensor2

from eshelbyx.grid import Grid
from eshelbyx.material import Material

N = 8


def two_phase_grid():
    grid = Grid.from_function(shape=(N, N, N), marker_fn=lambda x, y, z: jnp.where(x < 0.5, 1, 0))
    return grid.apply_material_map({0: {"E": 1.0, "nu": 0.3}, 1: {"E": 10.0, "nu": 0.2}})


def make_material(grid, C0):
    behavior = jm.LinearElasticIsotropic(E=grid["E"].ravel(), nu=grid["nu"].ravel())
    return Material(behavior, C0)


def random_strain(seed=0):
    eps = jax.random.normal(jax.random.PRNGKey(seed), (N, N, N, 3, 3))
    return 0.5 * (eps + jnp.swapaxes(eps, -1, -2))


# --------------------------------------------------------------------------- #
# The resolvent must reproduce the block it replaces
# --------------------------------------------------------------------------- #
def reference_alpha(eps, material_params, C0):
    """The hand-written resolvent that used to be copied into every caller."""

    def _apply(e, mp, C0):
        E, nu = mp
        lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
        mu = E / 2 / (1 + nu)
        C = IsotropicTensor4(kappa=lmbda + 2 / 3 * mu, mu=mu)
        return ((C + C0).inv @ (C0 * 2.0)) @ e

    flat = SymmetricTensor2(tensor=jnp.asarray(eps).reshape(-1, 3, 3))
    out = jax.vmap(_apply, in_axes=(0, 0, None))(flat, material_params, C0)
    return out.tensor.reshape(eps.shape)


@pytest.mark.parametrize("scale", [1.0, 0.3, 7.5])
def test_resolvent_matches_reference(scale):
    grid = two_phase_grid()
    C0 = jm.LinearElasticIsotropic(E=1.0, nu=0.3).C
    material = make_material(grid, C0)
    eps = random_strain()

    expected = reference_alpha(eps, (grid["E"].ravel(), grid["nu"].ravel()), C0 * scale)
    assert jnp.allclose(material.resolvent(eps, scale), expected, atol=1e-14)


def test_stress_matches_closed_form():
    grid = two_phase_grid()
    material = make_material(grid, jm.LinearElasticIsotropic(E=1.0, nu=0.3).C)
    eps = random_strain()

    E, nu = grid["E"].ravel(), grid["nu"].ravel()
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    flat = eps.reshape(-1, 3, 3)
    tr = jnp.trace(flat, axis1=-2, axis2=-1)[:, None, None]
    expected = (lmbda[:, None, None] * tr * jnp.eye(3) + 2 * mu[:, None, None] * flat).reshape(
        eps.shape
    )
    assert jnp.allclose(material.stress(eps), expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# Defining properties of the resolvent
# --------------------------------------------------------------------------- #
def test_resolvent_is_identity_when_C_equals_C0():
    """alpha = 2 (C + C0)^-1 C0 reduces to the identity for C == C0."""
    grid = Grid(shape=(N, N, N))
    C0 = jm.LinearElasticIsotropic(E=1.0, nu=0.3).C
    n = grid.num_cells
    behavior = jm.LinearElasticIsotropic(E=jnp.ones(n), nu=jnp.full((n,), 0.3))
    eps = random_strain()
    assert jnp.allclose(Material(behavior, C0).resolvent(eps), eps, atol=1e-12)


def test_resolvent_is_linear():
    """AEM applies alpha to search directions, so it must be linear."""
    grid = two_phase_grid()
    material = make_material(grid, jm.LinearElasticIsotropic(E=1.0, nu=0.3).C)
    a, b = random_strain(0), random_strain(1)
    lhs = material.resolvent(2.0 * a - 3.0 * b)
    rhs = 2.0 * material.resolvent(a) - 3.0 * material.resolvent(b)
    assert jnp.allclose(lhs, rhs, atol=1e-12)


def test_polarization_is_stress_minus_reference():
    grid = two_phase_grid()
    C0 = jm.LinearElasticIsotropic(E=1.0, nu=0.3).C
    material = make_material(grid, C0)
    eps = random_strain()
    C0_eps = eqx.filter_vmap(lambda e: C0 @ e)(
        SymmetricTensor2(tensor=eps.reshape(-1, 3, 3))
    ).tensor.reshape(eps.shape)
    assert jnp.allclose(material.polarization(eps), material.stress(eps) - C0_eps, atol=1e-12)


# --------------------------------------------------------------------------- #
# Heterogeneity and misuse
# --------------------------------------------------------------------------- #
def test_parameters_are_per_voxel():
    """Batching the behaviour's parameters must give every voxel its own
    material, unlike init_state(Nbatch=...) which shares one parameter set."""
    grid = two_phase_grid()
    material = make_material(grid, jm.LinearElasticIsotropic(E=1.0, nu=0.3).C)
    eps = jnp.broadcast_to(jnp.eye(3), (N, N, N, 3, 3))
    sig = material.stress(eps)
    stiff = grid["E"] > 5.0
    assert float(jnp.abs(sig[stiff][..., 0, 0]).mean()) > float(
        jnp.abs(sig[~stiff][..., 0, 0]).mean()
    )


def test_rejects_a_behaviour_without_a_stiffness():
    C0 = jm.LinearElasticIsotropic(E=1.0, nu=0.3).C
    behavior = jm.vonMisesIsotropicHardening(
        elasticity=jm.LinearElasticIsotropic(E=1.0, nu=0.3),
        yield_stress=jm.VoceHardening(sig0=1.0, sigu=2.0, b=1.0),
    )
    with pytest.raises(TypeError, match="stiffness"):
        Material(behavior, C0)


# --------------------------------------------------------------------------- #
# Several behaviour classes on different regions
# --------------------------------------------------------------------------- #
def fibre_behavior():
    return jm.LinearElasticTransverseIsotropic(
        axis=jnp.array([0.0, 0.0, 1.0]), EL=10.0, ET=2.0, nuL=0.2, nuT=0.3, muL=1.0
    )


def mixed_material(C0):
    grid = Grid.from_function(shape=(N, N, N), marker_fn=lambda x, y, z: jnp.where(x < 0.5, 0, 1))
    matrix = jm.LinearElasticIsotropic(E=1.0, nu=0.3)
    material = Material.from_map(grid, {0: matrix, 1: fibre_behavior()}, C0)
    return grid, material, matrix


@pytest.mark.parametrize("scale", [1.0, 4.0])
def test_each_region_uses_its_own_behaviour(scale):
    """An isotropic and a transversely isotropic phase are different pytree
    structures with different constitutive formulas, so a single vmap cannot
    span them; each region must be evaluated with its own behaviour."""
    C0 = jm.LinearElasticIsotropic(E=1.0, nu=0.3).C
    grid, material, matrix = mixed_material(C0)

    e = jnp.diag(jnp.array([0.0, 0.0, 1.0]))
    eps = jnp.broadcast_to(e, (N, N, N, 3, 3))
    sig, alpha = material.stress(eps), material.resolvent(eps, scale)
    markers = grid.markers

    for marker, behavior in ((0, matrix), (1, fibre_behavior())):
        C = behavior.C
        # an anisotropic phase needs the reference promoted to the general form
        C0r = C0 if marker == 0 else C0.to_symmetric()
        C0s = C0r * scale
        unit = SymmetricTensor2(tensor=e)
        assert jnp.allclose(sig[markers == marker][0], (C @ unit).tensor, atol=1e-12)
        assert jnp.allclose(
            alpha[markers == marker][0],
            ((C + C0s).inv @ (C0s * 2.0) @ unit).tensor,
            atol=1e-12,
        )


def test_mixed_material_drives_a_solve():
    """The mixed material must work end to end, and the solvers must agree."""
    from eshelbyx.discretization import WillotDiscretization
    from eshelbyx.green_operator import GreenOperator
    from eshelbyx.krylov_solver import KrylovAEMSolver
    from eshelbyx.solver import AEMSolver

    C0 = jm.LinearElasticIsotropic(E=2.0, nu=0.3).C
    grid, material, _ = mixed_material(C0)
    Gamma_0 = GreenOperator(grid, C0, WillotDiscretization(grid))
    E_macro = grid.broadcast_to_grid(
        SymmetricTensor2(array=jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    )
    alpha = material.as_alpha()

    eps_a, sig_a, _ = AEMSolver(E_macro, alpha, Gamma_0, tol=1e-10, max_iter=2000)
    _, sig_k, _ = KrylovAEMSolver(E_macro, alpha, Gamma_0, tol=1e-10, max_iter=2000)

    assert jnp.allclose(jnp.mean(sig_a, axis=(0, 1, 2)), jnp.mean(sig_k, axis=(0, 1, 2)), atol=1e-6)
    assert jnp.allclose(
        jnp.mean(eps_a, axis=(0, 1, 2)), jnp.diag(jnp.array([0.0, 0.0, 1.0])), atol=1e-8
    )


def test_from_map_rejects_uncovered_voxels():
    """A marker present in the grid but missing from the map would leave voxels
    with no behaviour; that must be an error, not a silent hole."""
    grid = Grid.from_function(shape=(N, N, N), marker_fn=lambda x, y, z: jnp.where(x < 0.5, 0, 1))
    C0 = jm.LinearElasticIsotropic(E=1.0, nu=0.3).C
    with pytest.raises(ValueError, match="absent"):
        Material.from_map(grid, {0: jm.LinearElasticIsotropic(E=1.0, nu=0.3)}, C0)


def test_from_map_rejects_an_unused_marker():
    grid = Grid.from_function(shape=(N, N, N), marker_fn=lambda x, y, z: jnp.zeros_like(x))
    C0 = jm.LinearElasticIsotropic(E=1.0, nu=0.3).C
    with pytest.raises(ValueError, match="no voxel"):
        Material.from_map(
            grid,
            {
                0: jm.LinearElasticIsotropic(E=1.0, nu=0.3),
                7: jm.LinearElasticIsotropic(E=2.0, nu=0.3),
            },
            C0,
        )


def test_regions_partition_agrees_with_single_behaviour():
    """Splitting a uniform material into two regions must change nothing."""
    C0 = jm.LinearElasticIsotropic(E=1.0, nu=0.3).C
    grid = Grid.from_function(shape=(N, N, N), marker_fn=lambda x, y, z: jnp.where(x < 0.5, 0, 1))
    behavior = jm.LinearElasticIsotropic(E=3.0, nu=0.25)
    single = Material(behavior, C0)
    split = Material.from_map(grid, {0: behavior, 1: behavior}, C0)
    eps = random_strain()
    assert jnp.allclose(single.stress(eps), split.stress(eps), atol=1e-14)
    assert jnp.allclose(single.resolvent(eps, 2.0), split.resolvent(eps, 2.0), atol=1e-14)
