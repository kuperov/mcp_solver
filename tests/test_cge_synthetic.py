import numpy as np
import pytest

from examples.synthetic_cge import make_exchange_economy
from mcp_solver import SolverOptions
from mcp_solver.semismooth import solve_semismooth


@pytest.mark.parametrize("n", [5, 20, 200])
def test_exchange_economy_solves_and_clears(n):
    m = make_exchange_economy(n, seed=42)
    prob = m.build()
    res = solve_semismooth(prob)
    assert res.converged, f"n={n}: {res.status}, residual={res.residual:.2e}"
    sol = m.unpack(res.z)
    assert np.all(sol["p"] > 0.0), "equilibrium prices must be positive"
    excess_supply = prob.f_np(res.z)
    assert np.abs(excess_supply).max() < 1e-6          # all markets clear
    assert abs(sol["p"] @ excess_supply) < 1e-8        # Walras


@pytest.mark.slow
def test_large_economy_2000():
    m = make_exchange_economy(2000, seed=7)
    prob = m.build(jac_coloring=False)   # income terms make coloring lose
    res = solve_semismooth(prob, SolverOptions(max_iter=200))
    assert res.converged
    assert np.abs(prob.f_np(res.z)).max() < 1e-5
