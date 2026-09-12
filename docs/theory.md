# The periodic cell problem

`eshelbyx` solves the periodic homogenization problem on a regular voxel grid.
Given a unit cell $\Omega$ occupied by a heterogeneous material with local
stiffness $\bC(\bx)$ and an imposed macroscopic strain $\bE$, we look for a
periodic strain field $\beps$ such that

$$
\begin{cases}
\div \bsig = 0 & \text{in } \Omega \\
\bsig = \bC(\bx) : \beps & \\
\beps = \bE + \sym \tgrad \bu, \quad \bu \text{ periodic} &
\end{cases}
$$

Introducing a homogeneous *reference medium* $\bC^0$ and its periodic Green
operator $\bGamma^0$, this is equivalent to the Lippmann–Schwinger equation

$$
\beps + \bGamma^0 : \left( \bsig(\beps) - \bC^0 : \beps \right) = \bE
$$

which is what every solver in the library iterates on, in one form or another.
The operator $\bGamma^0$ is diagonal in Fourier space, so each application costs
one forward and one inverse FFT.

## The Eyre–Milton reformulation

Iterating the Lippmann–Schwinger equation directly converges at a rate that
degrades with the phase contrast, and diverges outright once the material is
much stiffer than the reference. The Eyre–Milton reformulation removes that
dependence. Introducing the local *resolvent*

$$
\balpha = 2 \left( \bC + \bC^0 \right)^{-1} : \bC^0
$$

and the auxiliary strain $\beps^*$ with $\beps = \balpha : \beps^*$, the problem
becomes

$$
\Zz(\beps^*) := \balpha : \beps^* + 2\, \bGamma^0 \bC^0 :
\left( \beps^* - \balpha : \beps^* \right) = \bE
$$

The spectrum of $\Zz$ is bounded away from both zero and infinity independently
of the contrast between phases, including the limiting cases of rigid inclusions
and voids. This is why `eshelbyx` converges on microstructures with infinite
contrast.

See [](solvers.md) for the iterative schemes built on this reformulation, and
[](discretization.md) for the discrete Green operator.
