import jax
import jax.numpy as jnp
import numpy as np

from mcp_solver.normal_map import (decompose, make_normal_map,
                                   natural_residual, project)


def test_project_derivative_is_one_at_exact_bounds():
    lb, ub = jnp.array([0.0]), jnp.array([2.0])
    g = jax.grad(lambda x: project(jnp.array([x]), lb, ub)[0])
    assert float(g(0.0)) == 1.0     # exactly on lower bound -> 1, not 0.5
    assert float(g(2.0)) == 1.0     # exactly on upper bound -> 1, not 0.5
    assert float(g(1.0)) == 1.0     # interior
    assert float(g(-1.0)) == 0.0    # strictly outside
    assert float(g(3.0)) == 0.0


def test_project_values_match_clip():
    lb = jnp.array([0.0, -jnp.inf, 1.0])
    ub = jnp.array([2.0, jnp.inf, 1.0])
    x = jnp.array([-5.0, 3.0, 7.0])
    np.testing.assert_allclose(project(x, lb, ub), jnp.clip(x, lb, ub))


def test_normal_map_zero_iff_mcp_solution():
    # f(z) = z - 1 on [0, 2]: solution z*=1 interior, x*=z*=1
    f = lambda z: z - 1.0
    lb, ub = jnp.array([0.0]), jnp.array([2.0])
    fB = make_normal_map(f, lb, ub)
    np.testing.assert_allclose(fB(jnp.array([1.0])), [0.0], atol=1e-15)
    # f(z) = z + 1 on [0, 2]: solution z*=0 (f>0 at lower bound),
    # x* = z* - w* = -1 and fB(x*) = f(0) + x* - 0 = 1 - 1 = 0
    f2 = lambda z: z + 1.0
    fB2 = make_normal_map(f2, lb, ub)
    np.testing.assert_allclose(fB2(jnp.array([-1.0])), [0.0], atol=1e-15)


def test_decompose_roundtrip():
    lb, ub = np.array([0.0, 0.0]), np.array([2.0, 2.0])
    x = np.array([-0.5, 2.7])
    z, w, v = decompose(x, lb, ub)
    np.testing.assert_allclose(z, [0.0, 2.0])
    np.testing.assert_allclose(w, [0.5, 0.0])   # w = (z - x)_+
    np.testing.assert_allclose(v, [0.0, 0.7])   # v = (x - z)_+
    np.testing.assert_allclose(z - w + v, x)


def test_natural_residual_zero_at_solution():
    lb, ub = np.array([0.0]), np.array([np.inf])
    assert natural_residual(np.array([0.0]), np.array([3.0]), lb, ub) == 0.0
    assert natural_residual(np.array([1.0]), np.array([3.0]), lb, ub) >= 1.0
