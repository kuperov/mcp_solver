"""Test-problem library. Solutions are verified by MCP residuals, never
by literature constants (except closed forms derived here)."""
import jax.numpy as jnp
import numpy as np

from mcp_solver.problem import MCPProblem

INF = np.inf


def assert_mcp_solution(problem, z, tol=1e-6):
    f = problem.f_np(z)
    lb, ub = problem.lb, problem.ub
    assert np.all(z >= lb - tol) and np.all(z <= ub + tol), "bounds violated"
    fixed = lb == ub
    at_l = (z <= lb + tol) & ~fixed
    at_u = (z >= ub - tol) & ~fixed
    interior = ~at_l & ~at_u & ~fixed
    assert np.all(f[at_l] >= -tol), "f must be >= 0 at active lower bounds"
    assert np.all(f[at_u] <= tol), "f must be <= 0 at active upper bounds"
    assert np.all(np.abs(f[interior]) <= tol), "f must vanish at interior"


def kojima_shindo():
    """Classic 4-variable NCP; degenerate solution set (two solutions)."""
    def f(z):
        z1, z2, z3, z4 = z[0], z[1], z[2], z[3]
        return jnp.stack([
            3 * z1**2 + 2 * z1 * z2 + 2 * z2**2 + z3 + 3 * z4 - 6,
            2 * z1**2 + z1 + z2**2 + 10 * z3 + 2 * z4 - 2,
            3 * z1**2 + z1 * z2 + 2 * z2**2 + 2 * z3 + 9 * z4 - 9,
            z1**2 + 3 * z2**2 + 2 * z3 + 3 * z4 - 3,
        ])
    return MCPProblem(f, np.zeros(4), np.full(4, INF), np.full(4, 1.0))


def cournot_duopoly():
    """2-firm Cournot, linear demand p=10-Q, cost c_i q_i + d_i q_i^2 with
    c=(1,1), d=(1,1). Interior equilibrium solves 4q1+q2=9, q1+4q2=9,
    i.e. q* = (1.8, 1.8)."""
    a, b = 10.0, 1.0
    c = jnp.array([1.0, 1.0])
    d = jnp.array([1.0, 1.0])

    def f(q):
        Q = jnp.sum(q)
        return c + 2 * d * q - (a - b * Q) + b * q
    return MCPProblem(f, np.zeros(2), np.full(2, INF), np.full(2, 1.0))


COURNOT_SOLUTION = np.array([1.8, 1.8])


def synthetic_lcp(n=20, seed=0, frac_active=0.4):
    """Monotone LCP with a constructed known solution.

    Build M = A A^T + I (positive definite), pick z* with a fraction of
    components at the bound, w* complementary, then q = w* - M z*.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) / np.sqrt(n)
    M = A @ A.T + np.eye(n)
    z_star = rng.uniform(0.5, 2.0, n)
    active = rng.random(n) < frac_active
    z_star[active] = 0.0
    w_star = np.zeros(n)
    w_star[active] = rng.uniform(0.5, 2.0, active.sum())
    q = w_star - M @ z_star
    Mj, qj = jnp.asarray(M), jnp.asarray(q)
    p = MCPProblem(lambda z: Mj @ z + qj, np.zeros(n), np.full(n, INF),
                   np.ones(n))
    p.known_solution = z_star
    return p


def upper_bounded_lcp():
    """All three bound regimes active in one problem, solution constructed:
    f(z) = z - t with t = (-1, 0.5, 3) on [0,2]^3 ->
    z* = (0, 0.5, 2), f(z*) = (1, 0, -1)."""
    t = jnp.array([-1.0, 0.5, 3.0])
    p = MCPProblem(lambda z: z - t, np.zeros(3), np.full(3, 2.0),
                   np.full(3, 1.0))
    p.known_solution = np.array([0.0, 0.5, 2.0])
    return p


LIBRARY = {
    "kojima_shindo": kojima_shindo,
    "cournot": cournot_duopoly,
    "lcp_n20": lambda: synthetic_lcp(20, seed=0),
    "lcp_n80_degenerate": lambda: synthetic_lcp(80, seed=3, frac_active=0.7),
    "upper_bounded": upper_bounded_lcp,
}
