# Solvers

All solvers share the signature `solver(E_macro, ..., Gamma_0, tol=, max_iter=)`
and return `(eps, sigma, info)`.

```{list-table}
:header-rows: 1

* - Solver
  - Iterates on
  - Notes
* - {py:func}`~eshelbyx.solver.MSSolver`
  - Lippmann–Schwinger
  - Basic fixed point. Converges only for $\bC \lesssim 2\bC^0$.
* - {py:func}`~eshelbyx.solver.AEMSolver`
  - Eyre–Milton
  - Adaptive scheme: one-dimensional minimal-residual step with optimal
    $\lambda = \langle X, Z\rangle_{\bC^0} / \langle Z, Z\rangle_{\bC^0}$.
    Contrast-independent.
* - {py:func}`~eshelbyx.solver.A2EMSolver`
  - Eyre–Milton
  - Accelerated variant: rescales the reference medium each iteration and
    carries a Douglas–Rachford memory term.
* - {py:func}`~eshelbyx.solver.DRSolver`
  - Douglas–Rachford splitting
  - Projection onto compatible strains against the local constitutive graph,
    with an optional adaptive scale.
* - {py:func}`~eshelbyx.krylov_solver.KrylovAEMSolver`
  - Eyre–Milton
  - GMRES on the auxiliary operator $\Zz$. Each iteration costs the same as an
    AEM iteration but typically far fewer are needed.
```

## Choosing a solver

For linear elasticity, `KrylovAEMSolver` usually needs the fewest iterations,
at the cost of storing the Krylov basis. `AEMSolver` has a constant memory
footprint and is the safe default. `MSSolver` is included mainly for reference
and comparison.

## Nonlinear problems

A nonlinear local law cannot be passed to the Eyre–Milton solvers directly,
because they apply the resolvent $\balpha$ to *search directions*, which is only
valid when $\balpha$ is linear. Instead the global residual

$$
\Rr(\beps) = \beps + \bGamma^0 : \left(\bsig(\beps) - \bC^0 : \beps\right) - \bE
$$

is linearised by Newton's method and each correction is solved with the linear
AEM using the consistent tangent. See {py:mod}`eshelbyx.nonlinear` and
{py:mod}`eshelbyx.newton_optx`.
