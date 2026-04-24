import jax.numpy as jnp
from jaxmat.tensors import SymmetricTensor2, SymmetricTensor4, IsotropicTensor4


eps = SymmetricTensor2(tensor=jnp.diag(jnp.asarray([1.0, 0.0, 0.0])))


E0 = 1.0
nu0 = 0.3
lambda_0, mu0 = E0 * nu0 / (1 + nu0) / (1 - 2 * nu0), E0 / 2 / (1 + nu0)
C0 = IsotropicTensor4(lambda_0 + 2 / 3 * mu0, mu0)

E = 1.0
nu = 0.3
lambda_, mu = E * nu / (1 + nu) / (1 - 2 * nu), E / 2 / (1 + nu)
C = IsotropicTensor4(lambda_ + 2 / 3 * mu, mu)
alpha = (C0 @ (C + C0).inv) * 2.0

print(eps)
sig = C0 @ eps
sig_ = lambda_0 * jnp.linalg.trace(eps) * jnp.eye(3) + 2 * mu0 * eps
assert jnp.allclose(sig, sig_)
print(C0 @ eps)
print(C0.eval_jvp(eps))

iC = (C + C0).inv
C2 = C @ C
print(C.coeffs)
print(C2.coeffs)
# print((C + C0).coeffs)

# print(iC.coeffs)
# print(iC.inv.coeffs)

# print(alpha.coeffs)
