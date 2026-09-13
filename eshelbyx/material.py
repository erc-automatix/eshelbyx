"""Local constitutive law over the grid, and the Eyre-Milton resolvent.

The Eyre-Milton solvers do not act on the stiffness directly. They need the
*resolvent*

.. math::
    \\boldsymbol{\\alpha} = 2 (\\mathbb{C} + s\\mathbb{C}^0)^{-1} : (s\\mathbb{C}^0)

which maps the auxiliary strain to the true strain and is what makes the
iteration converge at arbitrary phase contrast. Deriving it is a short piece of
tensor algebra, but it has to be right, it depends on the reference medium and
its scale, and every caller needs it. This module derives it once from a
``jaxmat`` behaviour so that no caller has to write it again.

A behaviour is batched by giving its *parameters* a leading voxel axis::

    behavior = jm.LinearElasticIsotropic(E=grid["E"].ravel(), nu=grid["nu"].ravel())

Its pytree leaves are then per-voxel, so ``eqx.filter_vmap`` maps the behaviour
itself and every voxel gets its own material. Note this is not the same as
``behavior.init_state(Nbatch=...)``, which batches internal variables only and
leaves one shared parameter set.

Mixing different behaviour *classes*
------------------------------------
Parameter batching only covers one behaviour class. An isotropic matrix around
transversely isotropic fibres, or an elastic phase next to an elastoplastic one,
are different pytree structures with different constitutive formulas, and a
single ``vmap`` cannot span them. Use :meth:`Material.from_map`::

    material = Material.from_map(
        grid,
        {0: jm.LinearElasticIsotropic(E=1.0, nu=0.3),
         1: jm.LinearElasticTransverseIsotropic(axis=..., EL=..., ...)},
        C0,
    )

Each behaviour is then evaluated only on the voxels carrying its marker, by
gathering that region, mapping the behaviour over it and scattering the result
back. The voxel indices are fixed by the microstructure, so they are computed
once at construction. The alternative -- evaluating every behaviour everywhere
and selecting with ``where`` -- costs the sum of all phases at every voxel, and
is what ``lax.switch`` degrades to under ``vmap``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxmat.tensors import IsotropicTensor4, SymmetricTensor2, Tensor

__all__ = ["Material"]


class Material(eqx.Module):
    """A ``jaxmat`` behaviour evaluated over a grid, against a reference medium.

    Parameters
    ----------
    behavior : jaxmat behaviour
        Parameters may be scalars (uniform material) or arrays with a leading
        axis of ``n_voxels`` (heterogeneous).
    C0 : Tensor
        Stiffness of the reference medium.

    Examples
    --------
    >>> material = Material(                                   # doctest: +SKIP
    ...     jm.LinearElasticIsotropic(E=grid["E"].ravel(), nu=grid["nu"].ravel()),
    ...     C0,
    ... )
    >>> sigma = material.stress(eps)                           # doctest: +SKIP
    >>> eps = material.resolvent(eps_aux, scale)               # doctest: +SKIP
    """

    behaviors: tuple
    regions: tuple
    C0: Tensor

    def __init__(self, behavior, C0, regions=None):
        """
        Parameters
        ----------
        behavior : jaxmat behaviour, or tuple of them
            With ``regions``, one behaviour per region.
        C0 : Tensor
            Stiffness of the reference medium.
        regions : tuple of arrays, optional
            Flat voxel indices for each behaviour. ``None`` means a single
            behaviour covering the whole grid, which needs no gather.
        """
        self.behaviors = tuple(behavior) if isinstance(behavior, tuple) else (behavior,)
        self.regions = tuple(regions) if regions is not None else None
        self.C0 = C0

    def __check_init__(self):
        for behavior in self.behaviors:
            if not hasattr(behavior, "C"):
                raise TypeError(
                    f"{type(behavior).__name__} does not expose a stiffness `C`. "
                    "Material currently covers linear elastic behaviours; a "
                    "nonlinear law has to be linearised, which eshelbyx.nonlinear "
                    "handles."
                )
        if self.regions is not None and len(self.regions) != len(self.behaviors):
            raise ValueError(f"{len(self.behaviors)} behaviours but {len(self.regions)} regions.")

    @classmethod
    def from_map(cls, grid, behavior_map, C0, marker_field="markers"):
        """Build a material from ``{marker: behaviour}``.

        Every voxel must be covered by exactly one marker in the map; an
        uncovered voxel is an error rather than a silently zero-stiffness hole.

        A behaviour's parameters may be scalars, meaning uniform over its
        region, or arrays whose length equals that region's voxel count.
        """
        markers = np.asarray(grid[marker_field]).ravel()
        behaviors, regions = [], []
        covered = np.zeros(markers.size, dtype=bool)
        for marker, behavior in behavior_map.items():
            idx = np.flatnonzero(markers == marker)
            if idx.size == 0:
                raise ValueError(f"marker {marker!r} matches no voxel in the grid")
            covered[idx] = True
            behaviors.append(behavior)
            regions.append(jnp.asarray(idx))
        if not covered.all():
            missing = np.unique(markers[~covered])
            raise ValueError(
                f"markers {missing.tolist()} are present in the grid but absent "
                "from the behaviour map; every voxel needs a behaviour."
            )
        return cls(tuple(behaviors), C0, regions=tuple(regions))

    @property
    def behavior(self):
        """The behaviour, when there is only one."""
        if len(self.behaviors) != 1:
            raise AttributeError(
                f"this Material holds {len(self.behaviors)} behaviours; use "
                "`.behaviors` or index `.regions`."
            )
        return self.behaviors[0]

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _reference_for(self, behavior):
        """The reference medium in a representation this phase can combine with.

        ``jaxmat`` cannot evaluate ``SymmetricTensor4 + IsotropicTensor4``, so
        an anisotropic phase needs the reference promoted to the general form.
        The isotropic case keeps the cheaper structured representation.
        """
        C0 = self.C0
        if isinstance(behavior.C, IsotropicTensor4) or not hasattr(C0, "to_symmetric"):
            return C0
        return C0.to_symmetric()

    @staticmethod
    def _vmap_over(fn, behavior, values, C0):
        """vmap fn over a region, mapping the behaviour only if it is batched."""
        leaves = [leaf for leaf in jax.tree_util.tree_leaves(behavior) if eqx.is_array(leaf)]
        batched = bool(leaves) and all(
            leaf.ndim > 0 and leaf.shape[0] == values.array.shape[0] for leaf in leaves
        )
        in_axes = (0 if batched else None, 0, None)
        return eqx.filter_vmap(fn, in_axes=in_axes)(behavior, values, C0)

    def _map(self, fn, eps):
        """Apply a per-voxel tensor operation to a field of shape (..., 3, 3).

        ``fn`` takes ``(behavior, strain, C0)``; the reference is passed in
        rather than closed over because its representation depends on the phase.
        """
        shape = jnp.shape(eps)
        flat = jnp.asarray(eps).reshape(-1, 3, 3)

        if self.regions is None:
            behavior = self.behaviors[0]
            out = self._vmap_over(
                fn,
                behavior,
                SymmetricTensor2(tensor=flat),
                self._reference_for(behavior),
            )
            return out.tensor.reshape(shape)

        out = jnp.zeros_like(flat)
        for behavior, idx in zip(self.behaviors, self.regions, strict=True):
            values = SymmetricTensor2(tensor=flat[idx])
            result = self._vmap_over(fn, behavior, values, self._reference_for(behavior))
            out = out.at[idx].set(result.tensor)
        return out.reshape(shape)

    # ------------------------------------------------------------------ #
    # constitutive law
    # ------------------------------------------------------------------ #
    @eqx.filter_jit
    def stress(self, eps):
        """Local stress ``sigma = C : eps``."""
        return self._map(lambda behavior, e, C0: behavior.C @ e, eps)

    @eqx.filter_jit
    def polarization(self, eps):
        """Polarization stress ``tau = (C - C0) : eps``."""
        return self._map(lambda behavior, e, C0: (behavior.C - C0) @ e, eps)

    # ------------------------------------------------------------------ #
    # Eyre-Milton resolvent
    # ------------------------------------------------------------------ #
    @eqx.filter_jit
    def resolvent(self, eps, scale=1.0):
        """Apply ``alpha = 2 (C + s C0)^-1 (s C0)`` to a strain field.

        Parameters
        ----------
        eps : array of shape ``grid.shape + (3, 3)``
            Auxiliary strain field.
        scale : float
            Reference-medium scale ``s``; the accelerated solvers vary it.
        """

        def alpha(behavior, e, C0):
            C0s = C0 * scale
            return ((behavior.C + C0s).inv @ (C0s * 2.0)) @ e

        return self._map(alpha, eps)

    def as_alpha(self):
        """Return ``alpha(eps, scale=1.0)`` in the form the solvers expect.

        The solvers take the resolvent as a plain callable and ``eqx.filter_jit``
        keys callables by identity, so calling this repeatedly forces a
        recompilation each time. Build it once, outside any loop.
        """
        material = self

        def alpha(eps, scale=1.0):
            return material.resolvent(eps, scale)

        return alpha

    def as_stress_fn(self):
        """Return ``C_eps(eps)`` in the form ``MSSolver`` expects."""
        material = self

        def C_eps(eps):
            return material.stress(eps)

        return C_eps
