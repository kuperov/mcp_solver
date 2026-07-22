import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mcp_solver.problem import MCPProblem


def _dense_f(z):
    # every output depends on every input -> coloring cannot win
    return jnp.tanh(z) + 0.01 * jnp.sum(z**2) * jnp.ones_like(z)


def _banded_f(z):
    # tridiagonal structure -> 3-colorable
    lower = jnp.concatenate([jnp.zeros(1), z[:-1]])
    upper = jnp.concatenate([z[1:], jnp.zeros(1)])
    return z**2 + 0.5 * lower - 0.25 * upper


@pytest.mark.parametrize("f", [_dense_f, _banded_f])
def test_jacobian_matches_jacfwd(f):
    n = 30
    lb, ub = np.full(n, -10.0), np.full(n, 10.0)
    x0 = np.linspace(0.1, 1.0, n)
    p = MCPProblem(f, lb, ub, x0, jac_chunk=7)
    z = x0 + 0.03
    expected = np.asarray(jax.jacfwd(f)(jnp.asarray(z)))
    np.testing.assert_allclose(p.jac(z), expected, rtol=1e-10, atol=1e-12)


def test_coloring_wins_on_banded_and_falls_back_on_dense():
    n = 30
    lb, ub = np.full(n, -10.0), np.full(n, 10.0)
    x0 = np.full(n, 0.5)
    banded = MCPProblem(_banded_f, lb, ub, x0)
    dense = MCPProblem(_dense_f, lb, ub, x0)
    assert banded.n_jac_tangents <= 3
    assert dense.n_jac_tangents == n


def test_jac_boxed_zeroes_out_of_bounds_columns_and_pins_boundary():
    n = 3
    lb, ub = np.zeros(n), np.full(n, 2.0)
    f = lambda z: z**2 + jnp.roll(z, 1)
    p = MCPProblem(f, lb, ub, np.full(n, 1.0))
    z = np.array([-1.0, 0.0, 1.0])   # below bound / exactly on bound / interior
    Jb = p.jac_boxed(z)
    assert np.all(Jb[:, 0] == 0.0)                  # clipped column zeroed
    zc = np.clip(z, lb, ub)
    Jraw = np.asarray(jax.jacfwd(f)(jnp.asarray(zc)))
    np.testing.assert_allclose(Jb[:, 1], Jraw[:, 1])  # boundary column intact
    np.testing.assert_allclose(Jb[:, 2], Jraw[:, 2])


def test_f_boxed_evaluates_inside_bounds_only():
    # raw f would be log(-5) = nan; boxed evaluation clips to 0.1 first
    lb, ub = np.array([0.1]), np.array([np.inf])
    p = MCPProblem(lambda z: jnp.log(z), lb, ub, np.array([1.0]))
    assert np.all(np.isfinite(p.f_boxed(np.array([-5.0]))))
    assert not np.all(np.isfinite(p.f_np(np.array([-5.0]))))


def test_validation_errors():
    with pytest.raises(ValueError):
        MCPProblem(lambda z: z, np.zeros(2), -np.ones(2), np.zeros(2))  # lb > ub
    with pytest.raises(ValueError):
        MCPProblem(lambda z: z, np.zeros(2), np.ones(2), np.zeros(3))   # bad x0
    with pytest.raises(ValueError):
        MCPProblem(lambda z: z, np.array([np.inf]), np.array([np.inf]),
                   np.array([0.0]))   # fixed at infinity


# --- parameterized residual f(z, params) + set_params (compile/coloring reuse) --
def _param_f(z, params):
    # z^2 - c  (c a per-component parameter); root at z = sqrt(c) for z >= 0
    return z**2 - params["c"]


def test_parameterized_residual_and_set_params():
    from mcp_solver import SolverOptions, solve_path

    n = 5
    lb, ub = np.zeros(n), np.full(n, np.inf)
    x0 = np.full(n, 0.5)
    c0 = jnp.asarray(np.full(n, 4.0))
    p = MCPProblem(_param_f, lb, ub, x0, params={"c": c0})

    # f and jac read the current params
    z = np.full(n, 2.0)
    np.testing.assert_allclose(p.f_np(z), np.zeros(n), atol=1e-12)  # 2^2 - 4 = 0
    np.testing.assert_allclose(np.diag(p.jac(z)), np.full(n, 4.0), rtol=1e-10)  # 2z

    r1 = solve_path(p, SolverOptions())
    assert r1.converged
    np.testing.assert_allclose(r1.z, np.full(n, 2.0), atol=1e-8)

    # swap params on the SAME problem object (no rebuild) -> new root sqrt(9)=3
    p.set_params({"c": jnp.asarray(np.full(n, 9.0))})
    np.testing.assert_allclose(p.f_np(z), np.full(n, -5.0), atol=1e-12)  # 4 - 9
    r2 = solve_path(p, SolverOptions())
    assert r2.converged
    np.testing.assert_allclose(r2.z, np.full(n, 3.0), atol=1e-8)


def test_set_params_requires_parameterized_problem():
    p = MCPProblem(lambda z: z - 1.0, np.zeros(1), np.full(1, np.inf), np.array([0.5]))
    with pytest.raises(ValueError):
        p.set_params({"c": jnp.asarray([1.0])})
