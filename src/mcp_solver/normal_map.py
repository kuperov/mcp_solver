import jax
import jax.numpy as jnp
import numpy as np


@jax.custom_jvp
def project(x, lb, ub):
    """pi_B: componentwise clip onto [lb, ub] with a pinned subgradient.

    Stock jnp.clip has gradient 0.5 at exact-bound points (JAX splits
    ties); projected iterates sit exactly on bounds by construction, so
    we pin the derivative to 1 there (boundary treated as interior),
    selecting a definite B-subdifferential element. See spec.
    """
    return jnp.clip(x, lb, ub)


@project.defjvp
def _project_jvp(primals, tangents):
    x, lb, ub = primals
    dx, _, _ = tangents  # bounds are constants
    inside = (x >= lb) & (x <= ub)
    return jnp.clip(x, lb, ub), jnp.where(inside, dx, 0.0)


def make_normal_map(f, lb, ub):
    """Robinson's normal map f_B(x) = f(pi_B(x)) + x - pi_B(x)."""
    lb = jnp.asarray(lb)
    ub = jnp.asarray(ub)

    def fB(x):
        z = project(x, lb, ub)
        return f(z) + x - z

    return fB


def decompose(x, lb, ub):
    """x -> (z, w, v): z = pi_B(x), w = (z-x)_+, v = (x-z)_+ (paper eq. 5)."""
    z = np.clip(x, lb, ub)
    w = np.maximum(z - x, 0.0)
    v = np.maximum(x - z, 0.0)
    return z, w, v


def natural_residual(z, f_of_z, lb, ub):
    """||z - pi_B(z - f(z))||_inf — solver-independent optimality measure."""
    return float(np.abs(z - np.clip(z - f_of_z, lb, ub)).max())


def fB_np(f_np, x, lb, ub):
    """Normal map f_B(x) = f(pi_B(x)) + x - pi_B(x), plain numpy."""
    z = np.clip(x, lb, ub)
    return f_np(z) + x - z


def merit(f_np, x, lb, ub):
    """Theta(x) = ||f_B(x)||_2; +inf when f_B is undefined (spec merit)."""
    val = fB_np(f_np, x, lb, ub)
    if not np.all(np.isfinite(val)):
        return np.inf
    return float(np.linalg.norm(val))
