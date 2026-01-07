import jax.numpy as jnp


def compute_G0_hat_inv(D_plus, D_minus, C0):
    """
    Computes the inverse of G0_hat for a given D and C0

    Input:
    D [numpy.ndarray]: global array of differential vectors [Fourier domain]
    C0: stiffness tensor of the reference medium

    Output:
    G0_hat_inv [numpy.ndarray]: global array, defined in Einstein's notation by: G0_hat_inv_ij = conjugate(D_m) * C0_mijn * D_n
    """
    return jnp.einsum("m, mijn, n -> ij", D_minus, C0.tensor, D_plus)


def compute_G0_hat(D_plus, D_minus, C0, tol=1e-14):
    """
    Compute G0_hat with explicit zero-frequency handling.
    """
    G0_hat_inv = compute_G0_hat_inv(D_plus, D_minus, C0)

    norm_D = jnp.sqrt(jnp.dot(D_minus, D_plus))
    return jnp.where(
        norm_D > tol,
        jnp.linalg.inv(G0_hat_inv),
        jnp.zeros_like(G0_hat_inv),
    )


def compute_isotropic_G0_hat_inv(D_plus, D_minus, singular_mask, C0):
    kappa, mu = C0.kappa, C0.mu
    lam = kappa - 2 / 3 * mu

    D2 = jnp.sum(jnp.real(D_minus * D_plus), axis=-1)

    I = jnp.eye(3)
    DD = jnp.outer(D_minus, D_plus)

    G0_inv = mu * I * D2 + (lam + mu) * DD

    return G0_inv


def compute_isotropic_G0_hat(D_plus, D_minus, singular_mask, C0):
    kappa, mu = C0.kappa, C0.mu
    lam = kappa - 2 / 3 * mu

    D2 = jnp.sum(jnp.real(D_minus * D_plus), axis=-1)

    D2s = jnp.where(singular_mask, 1.0, D2)

    I = jnp.eye(3)
    DD = jnp.outer(D_minus, D_plus)

    G0_1 = jnp.where(singular_mask, 0.0, I / D2s)
    G0_2 = jnp.where(singular_mask, 0.0, DD / D2s**2)

    G0 = G0_1 / mu - ((lam + mu) / (mu * (lam + 2 * mu))) * G0_2

    return G0, (G0_1, G0_2)
