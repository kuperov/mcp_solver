"""Spec testing layer 3: both solvers on every problem, agreement checked."""
import numpy as np
import pytest

from examples.shoven_whalley import build_model
from examples.synthetic_cge import make_exchange_economy
from mcp_solver import SolverOptions
from mcp_solver.path.solver import solve_path
from mcp_solver.semismooth import solve_semismooth
from tests.problems import LIBRARY, assert_mcp_solution


@pytest.mark.parametrize("name", sorted(LIBRARY))
def test_both_solvers_agree(name):
    p1, p2 = LIBRARY[name](), LIBRARY[name]()
    r1 = solve_semismooth(p1)
    r2 = solve_path(p2)
    assert r1.converged and r2.converged, (r1.status, r2.status)
    assert_mcp_solution(p1, r1.z)
    assert_mcp_solution(p2, r2.z)
    if hasattr(p1, "known_solution"):
        np.testing.assert_allclose(r1.z, p1.known_solution, atol=1e-5)
        np.testing.assert_allclose(r2.z, p1.known_solution, atol=1e-5)


def test_shoven_whalley_through_path():
    m1, m2 = build_model(), build_model()
    r1 = solve_semismooth(m1.build())
    r2 = solve_path(m2.build())
    assert r1.converged and r2.converged
    s1, s2 = m1.unpack(r1.z), m2.unpack(r2.z)
    for key in ("y", "p", "pf"):
        np.testing.assert_allclose(s1[key], s2[key], rtol=1e-5)


@pytest.mark.parametrize("n", [20, 200])
def test_exchange_economy_through_path(n):
    m = make_exchange_economy(n, seed=42)
    prob = m.build()
    res = solve_path(prob)
    assert res.converged, f"n={n}: {res.status}"
    assert np.abs(prob.f_np(res.z)).max() < 1e-6


@pytest.mark.slow
def test_exchange_economy_1000_through_path():
    m = make_exchange_economy(1000, seed=7)
    prob = m.build(jac_coloring=False)
    res = solve_path(prob, SolverOptions(max_iter=200, max_pivots=20000))
    assert res.converged, res.status
    assert np.abs(prob.f_np(res.z)).max() < 1e-5
