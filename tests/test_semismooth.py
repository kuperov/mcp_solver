import jax.numpy as jnp
import numpy as np

from mcp_solver import SolverOptions, Status
from mcp_solver.problem import MCPProblem
from mcp_solver.semismooth import solve_semismooth

INF = np.inf


def _check_mcp(p, res, tol=1e-6):
    assert res.status is Status.CONVERGED
    z, f = res.z, p.f_np(res.z)
    assert np.all(z >= p.lb - tol) and np.all(z <= p.ub + tol)
    at_l = z <= p.lb + tol
    at_u = z >= p.ub - tol
    fixed = p.lb == p.ub
    interior = ~at_l & ~at_u
    assert np.all(f[at_l & ~fixed] >= -tol)
    assert np.all(f[at_u & ~fixed] <= tol)
    assert np.all(np.abs(f[interior]) <= tol)


def test_linear_system_free_vars_converges_in_one_iteration():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    p = MCPProblem(lambda z: jnp.asarray(A) @ z - jnp.asarray(b),
                   np.full(2, -INF), np.full(2, INF), np.zeros(2))
    res = solve_semismooth(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, np.linalg.solve(A, b), atol=1e-8)
    assert len(res.iterations) <= 2


def test_simple_ncp_with_active_bound():
    # f(z) = z + 1 >= 0 for all z >= 0 -> solution z = 0, w = f = 1
    p = MCPProblem(lambda z: z + 1.0, np.zeros(1), np.array([INF]),
                   np.array([5.0]))
    res = solve_semismooth(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [0.0], atol=1e-8)
    np.testing.assert_allclose(res.w, [1.0], atol=1e-6)


def test_box_bounds_upper_active():
    # f(z) = z - 5 on [0, 2]: f < 0 at solution -> z at upper bound 2
    p = MCPProblem(lambda z: z - 5.0, np.zeros(1), np.array([2.0]),
                   np.array([1.0]))
    res = solve_semismooth(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [2.0], atol=1e-8)


def test_domain_hazard_log_never_nans_out():
    # f(z) = log(z) - 1 on [1e-8, inf): solution z = e. A full Newton step
    # from z=5 overshoots negative without boxing; must still converge.
    p = MCPProblem(lambda z: jnp.log(z) - 1.0, np.full(1, 1e-8),
                   np.array([INF]), np.array([5.0]))
    res = solve_semismooth(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [np.e], rtol=1e-7)


def test_nonmonotone_helps_on_nasty_scalar():
    # f(z) = atan(z - 10): monotone but flat; plain Newton oscillates badly
    p = MCPProblem(lambda z: jnp.arctan(z - 10.0), np.full(1, -INF),
                   np.array([INF]), np.array([0.0]))
    res = solve_semismooth(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [10.0], atol=1e-6)


def test_stalls_gracefully_on_infeasible_problem():
    # f(z) = 1 with z <= 0 <= ... no: use f(z) = -1 forever on z >= 0:
    # requires f >= 0 at bound -> no solution exists
    p = MCPProblem(lambda z: -jnp.ones_like(z), np.zeros(1),
                   np.array([INF]), np.array([1.0]))
    res = solve_semismooth(p, SolverOptions(max_iter=50))
    assert res.status in (Status.STALLED, Status.MAX_ITERATIONS)


def test_badly_scaled_problem_converges():
    # quantities ~1e6 against prices ~1 (CGE-style scaling)
    D = np.array([1e6, 1.0, 1e-4])
    A = np.diag(D)
    p = MCPProblem(lambda z: jnp.asarray(A) @ z - jnp.asarray(D * 2.0),
                   np.zeros(3), np.full(3, INF), np.ones(3))
    res = solve_semismooth(p)
    _check_mcp(p, res, tol=1e-4)
    np.testing.assert_allclose(res.z, np.full(3, 2.0), rtol=1e-6)
