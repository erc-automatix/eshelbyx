# Changelog

All notable changes to this project are documented here.

## [Unreleased] — 0.1.0

### Added
- `eshelbyx.io`: all VTK reading and writing, with PyVista imported lazily.
- `eshelbyx.microstructure`: mesh voxelisation helpers salvaged from `microgen`.
- Explicit public API via `eshelbyx.__all__`.
- Project setup aligned with `jaxmat`: ruff configuration, pre-commit hooks,
  `CITATION.cff`, issue and pull-request templates, a `tests` CI workflow with a
  pinned/latest dependency matrix, a `publish` workflow, and a Jupyter Book
  documentation skeleton under `docs/`.
- `docs/sharp_bits.md` documenting the double-precision enforcement and the
  current limitations (memory scaling, Moulinec–Suquet on even grids, Willot on
  non-cubic voxels, no 2D, isotropic reference media only).
- `tests/test_io.py` covering VTK round-tripping and component ordering.

### Changed
- PyVista is now an optional dependency (`pip install eshelbyx[viz]`); the
  library imports and solves without it.
- Tensor cell fields are written to VTK with components flattened in C order.
  Handing VTK an `(n, 3, 3)` array, as before, makes it ravel the component
  block in Fortran order and silently store the transpose of each tensor.
  Symmetric stress and strain were unaffected; any non-symmetric tensor field
  was written transposed.
- `Grid.save()` accepts result fields directly, replacing manual PyVista calls
  in the demos.
- Tests are named after the module they cover: `test_green_plane_wave.py` became
  `test_green_operator.py`, `test_solvers.py` became `test_solver.py`, and
  `test_laminate_solution.py` became `test_solver_laminate.py`.
- Licence updated to LGPL-3.0, matching `jaxmat`.
- Minimum Python raised to 3.11, matching `jaxmat`.

### Removed
- `solver_old.py`, a superseded duplicate of three solvers.
- `nonlinear_krylov_solver.py`, which was non-functional.
- `fft_homog.py` and `Hashin.py`, dead top-level scripts.
- `microgen.py`; its reusable part became `eshelbyx.microstructure` and its
  `main_*` scripts moved to `demos/`.
- `Grid.OldGrid`, `load_vtk_grid` and `Grid.__setitem__`.
- `spectral_kernels.compute_isotropic_G0_hat_inv`, which was unused.
- `diffrax` dependency, which was never imported.
