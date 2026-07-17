import numpy as np
import pytest

from mcp_solver import SolverOptions
from mcp_solver.path.linearize import LinearMCP
from mcp_solver.path.pivot import (AT_LOWER, AT_UPPER, BASIC, _Tableau)

INF = np.inf


def _lin(M, q, lb, ub, z0, w0, v0):
    """Build an UNSCALED LinearMCP directly (R = C = 1) for unit tests."""
    n = q.size
    r = M @ z0 + q - w0 + v0
    ones = np.ones(n)
    free = ~np.isfinite(lb) & ~np.isfinite(ub)
    fixed = lb == ub
    return LinearMCP(n=n, M=M, q=q, r=r, lb=lb, ub=ub, z0=z0, w0=w0, v0=v0,
                     Ms=M.copy(), qs=q.copy(), rs=r.copy(), lbs=lb.copy(),
                     ubs=ub.copy(), z0s=z0.copy(), w0s=w0.copy(),
                     v0s=v0.copy(), R=ones, C=ones, free=free, fixed=fixed)


def _mixed_lin(seed=0):
    rng = np.random.default_rng(seed)
    n = 6
    A = rng.standard_normal((n, n))
    M = A @ A.T + np.eye(n)
    lb = np.array([-INF, 0.0, 0.0, 0.0, 1.0, -INF])
    ub = np.array([INF, INF, 2.0, INF, 1.0, INF])
    z0 = np.array([0.3, 0.0, 2.0, 0.7, 1.0, -0.2])   # free, at-l, at-u, int, fixed, free
    w0 = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0])
    q = rng.standard_normal(n)
    return _lin(M, q, lb, ub, z0, w0, v0)


def _init_from_triple(tab):
    """Mirror of the initial-basis rule (Task 4 wires this into the loop)."""
    lin, n = tab.lin, tab.n
    for j in range(n):
        if lin.free[j]:
            tab.basis[j], tab.z_stat[j] = j, BASIC
        elif lin.fixed[j]:
            tab.basis[j], tab.z_stat[j] = n + j, AT_LOWER
        elif lin.w0s[j] > 0 or (np.isfinite(lin.lbs[j])
                                and abs(lin.z0s[j] - lin.lbs[j]) <= 1e-12):
            tab.basis[j], tab.z_stat[j] = n + j, AT_LOWER
        elif lin.v0s[j] > 0 or (np.isfinite(lin.ubs[j])
                                and abs(lin.z0s[j] - lin.ubs[j]) <= 1e-12):
            tab.basis[j], tab.z_stat[j] = 2 * n + j, AT_UPPER
        else:
            tab.basis[j], tab.z_stat[j] = j, BASIC
    return tab.factorize()


def test_initial_basis_reproduces_start_triple():
    lin = _mixed_lin()
    tab = _Tableau(lin, SolverOptions())
    assert _init_from_triple(tab)
    t, x, (z, w, v) = tab.current_point()
    assert t == 0.0
    np.testing.assert_allclose(z, lin.z0, atol=1e-9)
    np.testing.assert_allclose(w, lin.w0, atol=1e-9)
    np.testing.assert_allclose(v, lin.v0, atol=1e-9)
    np.testing.assert_allclose(x, lin.z0 - lin.w0 + lin.v0, atol=1e-9)
    assert tab.check_residual()


def test_replace_matches_fresh_factorization():
    lin = _mixed_lin(3)
    tab = _Tableau(lin, SolverOptions())
    assert _init_from_triple(tab)
    n = tab.n
    # pivot W_1 (basic, pos 1) out for Z_1
    assert tab.replace(1, 1)
    tab.z_stat[1] = BASIC
    tab.recompute_xB()
    Binv_updated = tab.Binv.copy()
    xB_updated = tab.xB.copy()
    assert tab.factorize()                    # fresh inverse from scratch
    np.testing.assert_allclose(Binv_updated, tab.Binv, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(xB_updated, tab.xB, rtol=1e-8, atol=1e-10)


def test_residual_monitor_survives_many_replacements():
    lin = _mixed_lin(5)
    tab = _Tableau(lin, SolverOptions())
    assert _init_from_triple(tab)
    rng = np.random.default_rng(0)
    # random walk of legal basis swaps within families 1..3 (bounded indices)
    for _ in range(60):
        j = int(rng.integers(1, 4))
        cur = tab.basis[j]
        alts = [j, tab.n + j] + ([2 * tab.n + j] if np.isfinite(lin.ubs[j]) else [])
        alts = [a for a in alts if a != cur]
        new = int(rng.choice(alts))
        if not tab.replace(j, new):
            assert tab.factorize()
            continue
        tab.z_stat[j] = (BASIC if new < tab.n
                         else (AT_LOWER if new < 2 * tab.n else AT_UPPER))
        # nonbasic z must sit at a finite bound; keep status consistent
        if new >= tab.n and not np.isfinite(lin.lbs[j]):
            tab.z_stat[j] = AT_UPPER
        if not tab.check_residual():
            assert tab.factorize()
        tab.recompute_xB()
        assert tab.check_residual()


def test_bounds_of_and_columns():
    lin = _mixed_lin()
    tab = _Tableau(lin, SolverOptions())
    n = tab.n
    np.testing.assert_allclose(tab.column(0), lin.Ms[:, 0])       # Z col
    np.testing.assert_allclose(tab.column(n + 2), -np.eye(n)[2])  # W col
    np.testing.assert_allclose(tab.column(2 * n + 2), np.eye(n)[2])
    np.testing.assert_allclose(tab.column(tab.T), lin.rs)
    assert tab.bounds_of(n + 4) == (-INF, INF)     # fixed j=4: free slack
    assert tab.bounds_of(n + 1) == (0.0, INF)      # ordinary w
    assert tab.bounds_of(tab.T) == (0.0, 1.0)
    tab.t_relaxed = True
    assert tab.bounds_of(tab.T) == (-INF, 1.0)
