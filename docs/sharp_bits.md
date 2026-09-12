# Sharp bits

Behaviours that are deliberate, surprising, or currently limited.

## Importing `eshelbyx` enables double precision globally

Importing the package runs

```python
import jax

jax.config.update("jax_enable_x64", True)
```

as a side effect, exactly as `jaxmat` does. This is intentional and is not
configurable.

**Why it is enforced rather than offered.** The spectral solvers are formulated
in float64 from end to end. The Green operator declares a `float64` input
structure, the Fourier kernels are assembled in `complex128`, and the
Eyre–Milton iterations terminate on relative residuals down to $10^{-10}$,
roughly three orders of magnitude below what float32 can represent. In single
precision the iterations do not merely lose digits, they stop converging: the
residual stalls near the float32 epsilon and the `while_loop` runs to
`max_iter`. Making precision a user choice would mean shipping a configuration
in which every solver silently fails.

**What this means in practice.**

- The flag is process-global and must be set before the first JAX array is
  created. Put `import eshelbyx` at the top of your script, before any array
  allocation. Importing `jax` or `jax.numpy` beforehand is fine; allocating is
  not.

  ```python
  import eshelbyx as ex  # ok: enables x64 before anything is allocated
  import jax.numpy as jnp

  x = jnp.zeros(3)  # float64
  ```

  ```python
  import jax.numpy as jnp

  x = jnp.zeros(3)  # float32, created before the flag is set
  import eshelbyx as ex  # too late for x
  ```

- Any other JAX code in the same process runs in float64 afterwards. If you are
  combining `eshelbyx` with a neural-network training loop that expects float32,
  run them in separate processes.

- `jaxmat` sets the same flag on import, so in a stack that already uses
  `jaxmat` this changes nothing.

## PyVista is optional

All VTK reading, writing and plotting lives in {py:mod}`eshelbyx.io` and imports
PyVista lazily, so `import eshelbyx` works on a machine without a visualisation
stack. Install the extra when you need file input and output:

```bash
pip install "eshelbyx[viz]"
```

## VTK loses tensor rank

A cell field written as `(N1, N2, N3, 3, 3)` comes back from a `.vti` file with
a trailing axis of 9, because VTK records a component count rather than a tensor
rank. Components are stored in C order, so `field.reshape(shape + (3, 3))`
recovers the original.

## Memory grows as $O(N^3)$ in the Green operator

`GreenOperator` caches the Fourier kernel `G0_hat` as a dense
`(N1, N2, N3, 3, 3)` complex array, and each discretization scheme caches `xi`,
`D_plus` and `D_minus` at full grid resolution. Measured in float64:

| grid | cached operator arrays |
|---|---|
| $32^3$ | 8.7 MB |
| $64^3$ | 69.2 MB |
| $96^3$ | 234 MB |

This extrapolates to several GB at $256^3$, before any field storage, and is
currently the binding constraint on resolution rather than the FFT itself.
Reducing it is scheduled work.

## Moulinec–Suquet is not an exact projector on even grids

`MoulinecSuquetDiscretization` masks only the $k=0$ mode and never the Nyquist
mode. As a result $\Gamma^0 C^0$ is a projector on odd grids but not on even
ones. The patch test still passes; the idempotency test does not, and is marked
`xfail` for that case. Prefer `WillotDiscretization` unless you specifically
need the exact spectral derivative.

## The Willot scheme assumes cubic voxels

`WillotDiscretization` currently divides all three components of the
differential vector by the voxel size in $x$. On a grid whose spacing differs
between directions the derivative is wrong by the ratio of spacings, with no
warning. Use cubic voxels (`size` proportional to `shape`) until this is fixed.

## 2D is not yet supported

`Grid.from_function` accepts a two-dimensional `shape`, but the discretization
schemes unpack three frequency axes and will raise. Plane strain and plane
stress are not yet available; model them as a 3D grid with one voxel through the
thickness.

## The reference medium must be isotropic

`GreenOperator` supports `IsotropicTensor4` reference media. Passing a general
`SymmetricTensor4` currently raises, because the anisotropic kernel path has a
mismatched `vmap` signature. This only constrains the *reference* medium $C^0$;
the phases themselves may be arbitrarily anisotropic.

## The basic Moulinec–Suquet solver has a convergence condition

`MSSolver` is a fixed-point iteration on the Lippmann–Schwinger equation and
converges only when the reference medium is stiff enough, roughly
$C < 2 C^0$. With a softer reference it diverges rather than converging slowly.
The Eyre–Milton solvers (`AEMSolver`, `A2EMSolver`, `KrylovAEMSolver`) have no
such restriction and converge at arbitrary contrast, including rigid inclusions
and voids.
