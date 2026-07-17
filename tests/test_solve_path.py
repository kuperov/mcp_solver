import jax.numpy as jnp
import numpy as np

from mcp_solver import SolverOptions, Status
from mcp_solver.path.solver import solve_path
from mcp_solver.problem import MCPProblem

INF = np.inf


def _check_mcp(p, res, tol=1e-6):
    assert res.status is Status.CONVERGED, res.status
    z, f = res.z, p.f_np(res.z)
    assert np.all(z >= p.lb - tol) and np.all(z <= p.ub + tol)
    fixed = p.lb == p.ub
    at_l = (z <= p.lb + tol) & ~fixed
    at_u = (z >= p.ub - tol) & ~fixed
    interior = ~at_l & ~at_u & ~fixed
    assert np.all(f[at_l] >= -tol)
    assert np.all(f[at_u] <= tol)
    assert np.all(np.abs(f[interior]) <= tol)


def test_linear_free_system_one_path():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    p = MCPProblem(lambda z: jnp.asarray(A) @ z - jnp.asarray(b),
                   np.full(2, -INF), np.full(2, INF), np.zeros(2))
    res = solve_path(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, np.linalg.solve(A, b), atol=1e-8)
    assert len(res.iterations) <= 2


def test_active_bounds_and_multipliers():
    # f(z) = z - t on [0,2]^3, t = (-1, 0.5, 3): z* = (0, 0.5, 2)
    t = jnp.array([-1.0, 0.5, 3.0])
    p = MCPProblem(lambda z: z - t, np.zeros(3), np.full(3, 2.0),
                   np.full(3, 1.0))
    res = solve_path(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [0.0, 0.5, 2.0], atol=1e-8)
    assert res.w[0] > 0.5 and res.v[2] > 0.5


def test_nonlinear_ncp():
    # nonlinear complementarity: f(z) = z^2 - 4 on z >= 0 -> z* = 2
    p = MCPProblem(lambda z: z**2 - 4.0, np.zeros(1), np.array([INF]),
                   np.array([0.5]))
    res = solve_path(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [2.0], rtol=1e-7)


def test_nasty_atan_converges():
    p = MCPProblem(lambda z: jnp.arctan(z - 10.0), np.full(1, -INF),
                   np.array([INF]), np.array([0.0]))
    res = solve_path(p, SolverOptions(max_iter=200))
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [10.0], atol=1e-6)
    # damping machinery must have engaged on this problem: the plain
    # full-Newton m-step cannot be the only step type used
    kinds = {r.step_type for r in res.iterations}
    assert kinds != {"m"}, kinds


def test_dsteps_taken_near_solution():
    # start near the solution of a smooth problem: steps are tiny -> d-steps
    p = MCPProblem(lambda z: z - 1.0, np.zeros(3), np.full(3, INF),
                   np.full(3, 0.999))
    res = solve_path(p)
    _check_mcp(p, res)
    assert any(r.step_type == "d" for r in res.iterations)


def test_infeasible_reports_ray_or_stall():
    p = MCPProblem(lambda z: -jnp.ones_like(z), np.zeros(1),
                   np.array([INF]), np.array([1.0]))
    res = solve_path(p, SolverOptions(max_iter=30))
    assert res.status in (Status.RAY_TERMINATION, Status.STALLED,
                          Status.MAX_ITERATIONS)


def test_domain_error_at_start():
    p = MCPProblem(lambda z: jnp.log(z), np.zeros(1), np.array([INF]),
                   np.array([0.0]))   # log(0) at the only feasible start
    res = solve_path(p)
    assert res.status is Status.DOMAIN_ERROR


def test_log_domain_problem_converges():
    # f(z) = log(z) - 1 on [1e-8, inf): solution z = e; path points that
    # leave the domain must be handled by the merit(inf) machinery
    p = MCPProblem(lambda z: jnp.log(z) - 1.0, np.full(1, 1e-8),
                   np.array([INF]), np.array([5.0]))
    res = solve_path(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [np.e], rtol=1e-7)
