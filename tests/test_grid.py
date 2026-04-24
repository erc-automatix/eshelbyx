"""
Simple Unit Tests for Grid System

Run with: pytest test_simple_grid.py -v
Or: python test_simple_grid.py
"""

try:
    import pytest
except ImportError:
    pytest = None

import jax.numpy as jnp
import numpy as np
from eshelbyx.grid import Grid


# ============================================================================
# TEST: Basic Grid Creation
# ============================================================================


def test_uniform_grid():
    """Test creating uniform grid"""
    grid = Grid(shape=(10, 10, 10))

    assert grid.shape == (10, 10, 10)
    assert grid.size == (1.0, 1.0, 1.0)
    assert jnp.all(grid.markers == 0)


def test_grid_from_function():
    """Test creating grid from function"""

    def sphere(x, y, z):
        r = jnp.sqrt(x**2 + y**2 + z**2)
        return jnp.where(r < 0.3, 1, 0)

    grid = Grid.from_function(shape=(32, 32, 32), marker_fn=sphere)

    assert grid.shape == (32, 32, 32)
    assert "markers" in grid.fields


def create_layered_grid(phi, N):

    def layer(x, y, z):
        return jnp.where(x < phi, 1, 0)

    return Grid.from_function(shape=(N, N, N), marker_fn=layer)


def test_layered_grid():
    """Test creating layered grid"""

    grid = create_layered_grid(0.25, 16)
    # Check two phases exist
    unique = jnp.unique(grid.markers)
    assert len(unique) == 2
    assert jnp.isclose(grid.volume_fractions[1], 0.25)
    assert jnp.isclose(grid.volume_fractions[0], 0.75)


def test_add_field():
    """Test adding a field to grid"""
    grid = Grid(shape=(8, 8, 8))

    temperature = jnp.ones(grid.shape) * 25.0
    grid = grid.add_field("temperature", temperature)

    assert "temperature" in grid.fields
    assert jnp.all(grid["temperature"] == 25.0)


def test_field_access():
    """Test different ways to access fields"""
    grid = Grid(shape=(4, 4, 4))
    grid = grid.add_field("test", jnp.ones(grid.shape))

    # All these should work
    assert grid.has_field("test")
    assert grid["test"] is not None
    assert grid.get_field("test") is not None


# ============================================================================
# TEST: Material Properties
# ============================================================================


def test_material_map():
    """Test applying material map"""
    grid = create_layered_grid(0.5, 16)
    # Apply materials
    materials = {
        0: {"E": 1.0, "nu": 0.3},
        1: {"E": 100.0, "nu": 0.2},
    }
    grid = grid.apply_material_map(materials)

    assert jnp.min(grid["E"]) == 1.0
    assert jnp.max(grid["E"]) == 100.0
    assert jnp.min(grid["nu"]) == 0.2
    assert jnp.max(grid["nu"]) == 0.3
