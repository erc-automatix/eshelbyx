"""Tests for eshelbyx.io: VTK round-tripping of grids and result fields.

Run with: pytest tests/test_io.py -v
"""

import numpy as np
import pytest

import eshelbyx as ex
from eshelbyx.io import _flatten_cells

pv = pytest.importorskip("pyvista", reason="eshelbyx.io needs the [viz] extra")


SHAPE = (4, 5, 6)
SIZE = (1.0, 2.0, 3.0)


def _grid():
    markers = np.arange(int(np.prod(SHAPE))).reshape(SHAPE) % 3
    return ex.Grid(shape=SHAPE, size=SIZE, fields={"markers": markers})


def _cell_index(i, j, k, shape=SHAPE):
    """VTK cell numbering: first index varies fastest."""
    return i + j * shape[0] + k * shape[0] * shape[1]


# --------------------------------------------------------------------------- #
# Flattening: spatial order and component order
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("components", [(), (3,), (3, 3), (6,)])
def test_flatten_cells_preserves_every_value(components):
    """_flatten_cells must place field[i, j, k] at VTK cell index, components
    in C order. Checked at every cell, not a sample."""
    field = np.arange(int(np.prod(SHAPE + components)), dtype=float).reshape(SHAPE + components)
    flat = _flatten_cells(field, len(SHAPE))

    expected_shape = (
        (int(np.prod(SHAPE)),)
        if not components
        else (
            int(np.prod(SHAPE)),
            int(np.prod(components)),
        )
    )
    assert flat.shape == expected_shape

    for i in range(SHAPE[0]):
        for j in range(SHAPE[1]):
            for k in range(SHAPE[2]):
                assert np.array_equal(
                    np.ravel(flat[_cell_index(i, j, k)]), np.ravel(field[i, j, k])
                )


def test_tensor_components_are_not_transposed():
    """Regression: handing VTK an (n, 3, 3) array makes it ravel the component
    block in Fortran order, silently storing the transpose of every tensor.
    Symmetric fields hide this, so the check uses a non-symmetric one."""
    field = np.arange(int(np.prod(SHAPE)) * 9, dtype=float).reshape((*SHAPE, 3, 3))
    assert not np.allclose(field, np.swapaxes(field, -1, -2)), "field must be non-symmetric"

    flat = _flatten_cells(field, 3)
    n = _cell_index(1, 2, 3)
    assert np.array_equal(flat[n].reshape(3, 3), field[1, 2, 3])
    assert not np.array_equal(flat[n].reshape(3, 3), field[1, 2, 3].T)


# --------------------------------------------------------------------------- #
# Round-trip through a file
# --------------------------------------------------------------------------- #
def test_roundtrip_scalar_vector_and_tensor(tmp_path):
    grid = _grid()
    rng = np.random.default_rng(0)
    tensor = rng.normal(size=(*SHAPE, 3, 3))
    vector = rng.normal(size=(*SHAPE, 3))

    path = tmp_path / "fields.vti"
    grid.save(path, fields={"stress": tensor, "flux": vector})
    back = ex.read_vti(path, marker_field="markers")

    assert back.shape == SHAPE
    assert np.allclose(back.size, SIZE)
    assert np.array_equal(np.asarray(back["markers"]), np.asarray(grid["markers"]))
    assert np.allclose(np.asarray(back["flux"]), vector)
    # VTK stores a component count, not a tensor rank, so 3x3 comes back as 9.
    assert np.allclose(np.asarray(back["stress"]).reshape((*SHAPE, 3, 3)), tensor)


def test_read_vti_rejects_unknown_marker_field(tmp_path):
    path = tmp_path / "g.vti"
    _grid().save(path)
    with pytest.raises(KeyError, match="phase"):
        ex.read_vti(path, marker_field="phase")


def test_read_vti_warns_when_marker_field_is_guessed(tmp_path):
    """Falling back to the first cell array depends on file ordering, so it must
    not happen silently. A Grid always carries a markers field, so the file is
    built directly to reach this path."""
    path = tmp_path / "g.vti"
    image = pv.ImageData()
    image.dimensions = tuple(n + 1 for n in SHAPE)
    image.spacing = tuple(L / n for L, n in zip(SIZE, SHAPE))
    image.cell_data["phase"] = np.zeros(int(np.prod(SHAPE)))
    image.save(path)

    with pytest.warns(UserWarning, match="marker_field"):
        grid = ex.read_vti(path)
    assert np.array_equal(np.asarray(grid["markers"]), np.asarray(grid["phase"]))


def test_material_map_applied_on_read(tmp_path):
    path = tmp_path / "g.vti"
    _grid().save(path)
    back = ex.read_vti(
        path,
        marker_field="markers",
        material_map={0: {"E": 1.0}, 1: {"E": 10.0}, 2: {"E": 100.0}},
    )
    assert back.has_field("E")
    assert float(np.max(np.asarray(back["E"]))) == 100.0
