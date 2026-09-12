# Discrete Green operators

A discretization scheme is defined by a pair of differential vectors
$\bD^+(\bxi)$ and $\bD^-(\bxi)$ in Fourier space. Everything else is derived
from them by the base class {py:class}`eshelbyx.discretization.DiscretizationScheme`:

- the divergence contracts the stress with $\bD^-$,
- the symmetric gradient is the symmetrised outer product with $\bD^+$.

Because the two are built as negative adjoints of one another, the resulting
Green operator is self-adjoint by construction. The Fourier kernel is then the
inverse of the discrete acoustic tensor
$\bA(\bxi) = \div_h \, \bC^0 \, \sym\tgrad_h$, which for an isotropic reference
medium has the closed form

$$
\bG^0 = \dfrac{\bI}{\mu \, D^2}
 - \dfrac{\lambda + \mu}{\mu (\lambda + 2\mu)}
   \dfrac{\bD^- \otimes \bD^+}{D^4}
$$

evaluated on the non-singular modes and zeroed elsewhere.

## Available schemes

```{list-table}
:header-rows: 1

* - Scheme
  - $\bD$
  - Notes
* - {py:class}`~eshelbyx.discretization.MoulinecSuquetDiscretization`
  - $i\bxi$
  - Exact spectral derivative. Spectrally accurate on smooth fields, but rings
    at phase interfaces and is not an exact projector on even grids.
* - {py:class}`~eshelbyx.discretization.WillotDiscretization`
  - rotated finite differences
  - The usual choice for high contrast. Assumes cubic voxels.
* - {py:class}`~eshelbyx.discretization.SchneiderDiscretization`
  - staggered $(e^{i\bxi h} - 1)/h$
  - Forward/backward differences on a staggered grid.
```

## Adding a scheme

Subclass `DiscretizationScheme` and provide `_D`, or override `_D_plus` and
`_D_minus` directly for a staggered scheme.

```{warning}
If you also override `apply_divergence` or `apply_symmetric_gradient`, check
that the kernel in {py:mod}`eshelbyx.spectral_kernels` is still correct for your
convention: it assumes the acoustic tensor has the form
$\mu D^2 \bI + (\lambda + \mu) \bD^- \otimes \bD^+$. The consistency test in
`tests/test_green_operator.py` verifies $\bG^0 \bA = \bI$ per scheme and will
catch a mismatch.
```
