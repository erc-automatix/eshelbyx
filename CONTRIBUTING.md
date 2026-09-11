# Contributing to eshelbyx

## Setup

```bash
git clone https://github.com/erc-automatix/eshelbyx && cd eshelbyx
pip install -e ".[test]"
pre-commit install
```

## Before opening a pull request

```bash
pre-commit run --all-files
pytest tests/ -q
```

## Conventions

- **Formatting and linting** are handled by `ruff` through pre-commit, with the
  same configuration as `jaxmat`.
- **Docstrings** follow the numpydoc convention, rendered by Sphinx napoleon.
  Public functions need at least a summary, `Parameters` and `Returns`.
- **Double precision** is enabled by importing `eshelbyx`. Never call
  `jax.config.update` in a test or a demo.
- **Tests** live in `tests/` and are named after the module they cover:
  `tests/test_<module>.py`. When a module needs more than one file, prefix with
  the module name, as in `tests/test_solver_laminate.py`.
- **Known limitations** are encoded as `pytest.mark.xfail(strict=True)` with a
  reason, so that fixing them turns the suite red and forces the marker to be
  retired. Do not use `strict=False` to silence a flaky test.

## Adding a discretization scheme

A scheme supplies `D⁺(ξ)` and `D⁻(ξ)`; the base class derives the divergence and
symmetric-gradient operators so that one is the negative adjoint of the other,
which is what makes the Green operator self-adjoint.

If you override `apply_divergence` or `apply_symmetric_gradient`, check that
`spectral_kernels` still builds the correct kernel for your convention: it
assumes the acoustic tensor has the form `μ D² I + (λ+μ) D⁻⊗D⁺`. The consistency
test in `tests/test_green_operator.py` checks `G0_hat · A = I` per scheme and
will catch a mismatch.
