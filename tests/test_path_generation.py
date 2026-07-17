import numpy as np
import pytest

from mcp_solver import SolverOptions
from mcp_solver.path.linearize import LinearMCP
from mcp_solver.path.pivot import PathStatus, generate_path

INF = np.inf


def _lin(M, q, lb, ub, z0, w0, v0):
    n = q.size
    r = M @ z0 + q - w0 + v0
    ones = np.ones(n)
    return LinearMCP(n=n, M=M, q=q, r=r, lb=lb, ub=ub, z0=z0, w0=w0, v0=v0,
                     Ms=M.copy(), qs=q.copy(), rs=r.copy(), lbs=lb.copy(),
                     ubs=ub.copy(), z0s=z0.copy(), w0s=w0.copy(),
                     v0s=v0.copy(), R=ones, C=ones,
                     free=~np.isfinite(lb) & ~np.isfinite(ub), fixed=lb == ub)


def _check_lcp_solution(lin, triple, tol=1e-7):
    z, w, v = triple
    res = lin.M @ z + lin.q - w + v
    assert np.abs(res).max() < tol
    assert np.all(z >= lin.lb - tol) and np.all(z <= lin.ub + tol)
    assert np.all(w >= -tol) and np.all(v >= -tol)
    fixed = lin.lb == lin.ub
    assert np.all(w[(z > lin.lb + 1e-6) & ~fixed] < tol)
    assert np.all(v[(z < lin.ub - 1e-6) & ~fixed] < tol)


def _random_monotone(n, seed, frac_active=0.4):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) / np.sqrt(n)
    M = A @ A.T + np.eye(n)
    q = rng.standard_normal(n) * 2.0
    lb, ub = np.zeros(n), np.full(n, INF)
    z0 = np.abs(rng.standard_normal(n))
    z0[rng.random(n) < frac_active] = 0.0
    fz = M @ z0 + q
    w0 = np.where(z0 == 0.0, np.maximum(fz, 0.0), 0.0)
    v0 = np.zeros(n)
    return _lin(M, q, lb, ub, z0, w0, v0)


@pytest.mark.parametrize("seed", range(6))
def test_random_monotone_lcps_reach_newton_point(seed):
    lin = _random_monotone(12, seed)
    res = generate_path(lin, SolverOptions(), debug=True)
    assert res.status is PathStatus.NEWTON_POINT, res.status
    _check_lcp_solution(lin, res.final_triple)
    # breakpoints: first is the start, last is the Newton point (t = 1)
    t0, x0 = res.breakpoints[0]
    tN, xN = res.breakpoints[-1]
    assert t0 == 0.0 and abs(tN - 1.0) < 1e-9
    z, w, v = res.final_triple
    np.testing.assert_allclose(xN, z - w + v, atol=1e-8)


def test_box_bounds_and_fixed_and_free():
    # mixed structure: solution known by construction
    M = np.eye(4)
    lb = np.array([0.0, 0.0, -INF, 3.0])
    ub = np.array([2.0, 2.0, INF, 3.0])
    # f(z) = z - target: z* = clip(target), free solves exactly, fixed pinned
    target = np.array([5.0, -1.0, 0.7, 9.0])
    q = -target
    z0 = np.array([1.0, 1.0, 0.0, 3.0])
    w0 = np.zeros(4)
    v0 = np.zeros(4)
    lin = _lin(M, q, lb, ub, z0, w0, v0)
    res = generate_path(lin, SolverOptions(), debug=True)
    assert res.status is PathStatus.NEWTON_POINT
    z, w, v = res.final_triple
    np.testing.assert_allclose(z, [2.0, 0.0, 0.7, 3.0], atol=1e-8)
    assert v[0] > 0.5 and w[1] > 0.5          # active-bound multipliers


def test_murty_family_terminates():
    for n in (4, 8):
        M = np.eye(n) + 2.0 * np.tril(np.ones((n, n)), -1)
        q = -np.ones(n)
        lb, ub = np.zeros(n), np.full(n, INF)
        z0 = np.zeros(n)
        w0 = np.maximum(q, 0.0)               # = 0: degenerate start
        lin = _lin(M, q, lb, ub, z0, w0, np.zeros(n))
        res = generate_path(lin, SolverOptions(max_pivots=5000), debug=True)
        assert res.status is PathStatus.NEWTON_POINT
        z, w, v = res.final_triple
        expected = np.zeros(n)
        expected[0] = 1.0                     # derived in plan: z* = e_1
        np.testing.assert_allclose(z, expected, atol=1e-8)


def test_infeasible_lcp_rays():
    # f(z) = -1 forever, z >= 0: no solution; Lemke must hit a ray
    n = 2
    lin = _lin(np.zeros((n, n)), -np.ones(n), np.zeros(n), np.full(n, INF),
               np.zeros(n), np.zeros(n), np.zeros(n))
    res = generate_path(lin, SolverOptions(), debug=True)
    assert res.status is PathStatus.RAY_TERMINATION


def test_lemke_start_option_also_solves():
    lin = _random_monotone(10, 42)
    res = generate_path(lin, SolverOptions(lemke_start=True), debug=True)
    assert res.status is PathStatus.NEWTON_POINT
    assert res.used_slack_start
    _check_lcp_solution(lin, res.final_triple)


def test_scaled_problem_solves_and_reports_original_units():
    # badly scaled diagonal problem through the real linearize()
    import jax.numpy as jnp
    from mcp_solver.path.linearize import linearize
    from mcp_solver.problem import MCPProblem
    D = np.array([1e6, 1.0, 1e-4])
    p = MCPProblem(lambda z: jnp.asarray(D) * (z - 2.0), np.zeros(3),
                   np.full(3, INF), np.ones(3))
    lin = linearize(p, np.ones(3))
    res = generate_path(lin, SolverOptions(), debug=True)
    assert res.status is PathStatus.NEWTON_POINT
    z, w, v = res.final_triple
    np.testing.assert_allclose(z, np.full(3, 2.0), rtol=1e-8)
