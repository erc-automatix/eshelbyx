from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import equinox as eqx


class Tensor(eqx.Module):
    dim: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    base_shape: Tuple[int, ...] = eqx.field(static=True)
    base_array_shape: Tuple[int, ...] = eqx.field(static=True)
    _array: jax.Array

    def __init__(
        self,
        tensor: Optional[jax.Array] = None,
        array: Optional[jax.Array] = None,
    ):
        if tensor is not None:
            if tensor.shape[-self.rank :] != self.base_shape:
                raise ValueError(
                    f"Wrong shape {tensor.shape[-self.rank :]} <> {self.base_shape}"
                )
            self._array = self._as_array(tensor)
        elif array is not None:
            if array.shape[-self.array_rank :] != self.base_array_shape:
                raise ValueError(
                    f"Wrong shape {array.shape[-self.array_rank :]} <> {self.base_array_shape}"
                )
            self._array = jnp.asarray(array)
        else:
            self._array = jnp.zeros(self.base_array_shape)

    # def flatten_batch_dims(self):
    #     return self.__class__(tensor=self.tensor.reshape((-1,) + self.base_shape))

    def flatten_batch_dims(self):
        return self.__class__(
            array=self._array.reshape((-1,) + self.base_array_shape),
        )

    @property
    def shape(self):
        return self.batch_shape + self.base_shape

    @property
    def batch_shape(self):
        return self._array.shape[: -self.array_rank]

    @property
    def tensor(self):
        return self._as_tensor(self._array)

    @property
    def T(self):
        return self.__class__(tensor=jnp.transpose(self.tensor))

    @property
    def array(self):
        return self._array

    @property
    def array_shape(self):
        return self._array.shape

    # @property
    # def batch_shape(self):
    #     return self.array_shape[: -self.array_rank]

    @property
    def array_rank(self):
        return len(self.base_array_shape)

    def __getitem__(self, idx):
        return self.tensor[idx]

    def __jax_array__(self):
        return self.tensor

    def __array__(self, dtype=None):
        return jnp.asarray(self.tensor, dtype=dtype)

    def __add__(self, other):
        cls = self._weaken_with(other)
        other_array = jnp.asarray(other).reshape(self.tensor.shape)
        return cls(tensor=self.tensor + other_array)

    def __sub__(self, other):
        cls = self._weaken_with(other)
        other_array = jnp.asarray(other).reshape(self.tensor.shape)
        return cls(tensor=self.tensor - other_array)

    def __mul__(self, other):
        return self.__class__(tensor=jnp.asarray(other) * self.tensor)

    def __truediv__(self, other):
        return self.__class__(tensor=self.tensor / jnp.asarray(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        return self.__class__(tensor=jnp.asarray(self) @ jnp.asarray(other))

    def __rmatmul__(self, other):
        return self.__class__(tensor=jnp.asarray(other) @ self.tensor)

    def __neg__(self):
        return self.__class__(tensor=-self.tensor)

    def __repr__(self):
        return f"{self.tensor}"

    def _as_array(self, tensor):
        return tensor.ravel()

    def _as_tensor(self, array):
        return array.reshape(*array.shape[:-1], self.shape)

    def _weaken_with(self, other):
        return self.__class__


class Tensor2(Tensor):
    dim = 3
    rank = 2
    base_shape = (dim, dim)
    base_array_shape = (dim**rank,)

    @classmethod
    def identity(cls):
        return cls(tensor=jnp.eye(cls.dim))

    def _as_array(self, tensor):
        tensor = jnp.asarray(tensor)
        d = self.dim
        vec = jnp.zeros(tensor.shape[: -self.rank] + self.base_array_shape)
        buff = 0
        for i in range(d):
            vec = vec.at[..., i].set(tensor[..., i, i])
            for j in range(i + 1, d):
                vec = vec.at[..., d + buff].set(tensor[..., i, j])
                vec = vec.at[..., d + buff + 1].set(tensor[..., j, i])
                buff += 2
        return vec

    def _as_tensor(self, array):
        d = self.dim
        tensor = jnp.zeros(array.shape[: -self.array_rank] + (d, d))
        # Diagonal terms
        for i in range(d):
            tensor = tensor.at[..., i, i].set(array[..., i])

        # Off-diagonal terms
        offset = d
        for i in range(d):
            for j in range(i + 1, d):
                tensor = tensor.at[..., i, j].set(array[..., offset])
                tensor = tensor.at[..., j, i].set(array[..., offset + 1])
                offset += 2

        return tensor

    # ----------------- custom pytree -----------------
    def __eqx_pytree__(self):
        # expose the tensor view as the leaf
        leaves = (self.tensor,)
        aux = (self._batch_shape, self.dim, self.rank)
        return leaves, aux

    @classmethod
    def __eqx_pytree_unflatten__(cls, aux, leaves):
        batch_shape, dim, rank = aux
        (tensor,) = leaves
        obj = cls.__new__(cls)
        # reconstruct minimal _array from the tensor
        obj._array = obj._as_array(tensor)
        obj._batch_shape = batch_shape
        obj.dim = dim
        obj.rank = rank
        obj.base_shape = (dim, dim)
        obj.base_array_shape = (dim**rank,)
        return obj

    @property
    def sym(self):
        return SymmetricTensor2(tensor=0.5 * (self.tensor + self.T.tensor))

    @property
    def inv(self):
        return self.__class__(tensor=jnp.linalg.inv(self.tensor))

    @property
    def eigenvalues(self):
        eivenvalues, eigendyads = jnp.linalg.eig(self.tensor)
        return eivenvalues, jnp.asarray(
            [SymmetricTensor2(tensor=N) for N in eigendyads]
        )

    @property
    def T(self):
        # we transpose only the last two indices in case of a batched tensor
        return self.__class__(tensor=jnp.swapaxes(self.tensor, -1, -2))


class SymmetricTensor2(Tensor2):
    dim = 3
    base_array_shape = (dim * (dim + 1) // 2,)

    def is_symmetric(self):
        return jnp.allclose(self, self.T)

    def _as_array(self, tensor):
        d = self.dim
        vec = jnp.zeros(tensor.shape[: -self.rank] + self.base_array_shape)
        buff = 0
        for i in range(d):
            vec = vec.at[..., i].set(tensor[..., i, i])
            for j in range(i + 1, d):
                vec = vec.at[..., d + buff].set(jnp.sqrt(2) * tensor[..., i, j])
                buff += 1
        return vec

    def _as_tensor(self, array):
        d = self.dim
        tensor = jnp.zeros(array.shape[: -self.array_rank] + (d, d))

        # Diagonal entries
        for i in range(d):
            tensor = tensor.at[..., i, i].set(array[..., i])

        # Off-diagonal entries (upper triangle) scaled by 1/sqrt(2)
        offset = d
        for i in range(d):
            for j in range(i + 1, d):
                val = array[..., offset] / jnp.sqrt(2)
                tensor = tensor.at[..., i, j].set(val)
                tensor = tensor.at[..., j, i].set(val)  # symmetry
                offset += 1

        return tensor

    def __matmul__(self, other):
        # Multiplication of symmetric tensors cannot be ensured to remain symmetric
        return Tensor2(tensor=jnp.asarray(self.tensor) @ jnp.asarray(other))

    def _weaken_with(self, other):
        if isinstance(other, self.__class__):
            return self.__class__
        return Tensor2


# @lru_cache(maxsize=None)
def symmetric_kelvin_mandel_index_map(d):
    """
    Returns:
        - km_to_ij: list mapping KM index → (i,j)
        - ij_to_km: dict mapping (i,j) → KM index
    """
    km_to_ij = []
    ij_to_km = {}
    idx = 0
    sqrt2 = 2**0.5
    for i in range(d):
        ij_to_km[(i, i)] = (idx, 1.0)
        km_to_ij.append(((i, i), 1.0))
        idx += 1
    for i in range(d):
        for j in range(i + 1, d):
            km_to_ij.append(((i, j), sqrt2))
            ij_to_km[(i, j)] = (idx, sqrt2)
            ij_to_km[(j, i)] = (idx, sqrt2)  # symmetry
            idx += 1
    return km_to_ij, ij_to_km


def symmetric_kelvin_mandel_index_map(d: int):
    """Vectorized map (same ordering & scaling)."""
    sqrt2 = jnp.sqrt(2.0)
    i_diag = jnp.arange(d, dtype=jnp.int16)
    j_diag = i_diag
    i_off, j_off = jnp.asarray([0, 0, 1], dtype=jnp.int16), jnp.asarray(
        [1, 2, 2], dtype=jnp.int16
    )
    km_i = jnp.concatenate([i_diag, i_off], dtype=int)
    km_j = jnp.concatenate([j_diag, j_off], dtype=int)
    KM_SCALE = jnp.concatenate(
        [jnp.ones((d,)), jnp.full(i_off.shape, fill_value=sqrt2)]
    )
    KM_MAP = jnp.stack([km_i, km_j], axis=1)

    # Generate all pair combinations (a,b),)
    Nkm = KM_MAP.shape[0]

    a_idx, b_idx = jnp.meshgrid(jnp.arange(Nkm), jnp.arange(Nkm), indexing="ij")
    ij = KM_MAP[a_idx]  # shape (Nkm, Nkm, 2)
    kl = KM_MAP[b_idx]  # shape (Nkm, Nkm, 2)

    # Extract indices
    i_, j_ = ij[..., 0], ij[..., 1]
    k_, l_ = kl[..., 0], kl[..., 1]
    return (i_, j_, k_, l_), KM_SCALE[a_idx] * KM_SCALE[b_idx], (a_idx, b_idx)


class SymmetricTensor4(Tensor):
    dim = 3
    rank = 4
    array_rank = 2
    base_shape = (dim, dim, dim, dim)
    base_array_shape = (dim * (dim + 1) // 2, dim * (dim + 1) // 2)

    @classmethod
    def identity(cls):
        d = cls.dim
        n = d * (d + 1) // 2
        return cls(array=jnp.eye(n))

    @classmethod
    def J(cls):
        I2 = SymmetricTensor2.identity()
        J = jnp.einsum("ij,kl->ijkl", I2, I2) / cls.dim
        return cls(tensor=J)

    @classmethod
    def K(cls):
        return cls.identity() - cls.J()

    @property
    def array_shape(self):
        vdim = self.dim * (self.dim + 1) // 2
        return (vdim, vdim)

    def is_symmetric(self):
        return jnp.allclose(self, self.T)

    @staticmethod
    @jax.jit
    def _as_array(tensor):
        (i_, j_, k_, l_), scale, _ = symmetric_kelvin_mandel_index_map(3)
        return scale * tensor[i_, j_, k_, l_]

    @staticmethod
    @jax.jit
    def _as_tensor(array: jax.Array) -> jax.Array:
        """
        Converts a KM matrix (n,n) back to full symmetric 4th-order tensor (d,d,d,d)
        """
        d = 3
        (i_, j_, k_, l_), scale, (a_idx, b_idx) = symmetric_kelvin_mandel_index_map(d)
        vals = array[a_idx, b_idx] / scale  # (Nkm, Nkm)

        # Initialize tensor
        tensor = jnp.zeros((d, d, d, d))

        # All symmetric permutations: (i,j,k,l), (i,j,l,k), (j,i,k,l), (j,i,l,k)
        tensor = tensor.at[i_, j_, k_, l_].set(vals)
        tensor = tensor.at[i_, j_, l_, k_].set(vals)
        tensor = tensor.at[j_, i_, k_, l_].set(vals)
        tensor = tensor.at[j_, i_, l_, k_].set(vals)
        return tensor

    def __matmul__(self, other):
        return other.__class__(
            tensor=jnp.tensordot(jnp.asarray(self), jnp.asarray(other).T)
        )

    @property
    def inv(self):
        return self.__class__(array=jnp.linalg.inv(self.array))


def _eval_basis(coeffs, basis):
    return sum([c * b for (c, b) in zip(coeffs, basis)])


class IsotropicTensor4(SymmetricTensor4):
    kappa: float
    mu: float

    def __init__(self, kappa, mu):
        self.kappa = kappa
        self.mu = mu
        super().__init__(self.eval())

    @property
    def basis(self):
        J = SymmetricTensor4.J()
        K = SymmetricTensor4.K()
        return [J, K]

    def basis_jvp(self, eps):
        return [
            jnp.linalg.trace(eps) / self.dim,
            eps - jnp.linalg.trace(eps) / self.dim * SymmetricTensor2.identity(),
        ]

    @property
    def coeffs(self):
        return jnp.asarray([3 * self.kappa, 2 * self.mu])

    def eval(self):
        return _eval_basis(self.coeffs, self.basis)

    def eval_jvp(self, eps):
        return _eval_basis(self.coeffs, self.basis_jvp(eps))

    def __add__(self, other):
        return IsotropicTensor4(self.kappa + other.kappa, self.mu + other.mu)

    def __sub__(self, other):
        return IsotropicTensor4(self.kappa - other.kappa, self.mu - other.mu)

    def __mul__(self, other):
        return IsotropicTensor4(self.kappa * other, self.mu * other)

    def __matmul__(self, other):
        if isinstance(other, IsotropicTensor4):
            return IsotropicTensor4(self.kappa * other.kappa, self.mu * other.mu)
        elif isinstance(other, Tensor2):
            return other.__class__(array=self.array @ other.array)
        else:
            raise NotImplementedError

    @property
    def inv(self):
        return IsotropicTensor4(1 / 9 / self.kappa, 1 / 4 / self.mu)
