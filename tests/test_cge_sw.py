import numpy as np

from examples.shoven_whalley import build_model
from mcp_solver.semismooth import solve_semismooth


def _solve(numeraire=1.0):
    m = build_model(numeraire_value=numeraire)
    prob = m.build()
    res = solve_semismooth(prob)
    assert res.converged, res.status
    return m, prob, res


def test_equilibrium_conditions_hold():
    m, prob, res = _solve()
    f = prob.f_np(res.z)
    tol = 1e-6
    # all prices/activities strictly positive here -> every f component ~ 0
    assert np.all(res.z[2:] > 0.01), "prices should be strictly positive"
    assert np.abs(f).max() < tol


def test_walras_law():
    m, prob, res = _solve()
    # value of excess supplies at solution prices ~ 0 (goods + factors)
    sol = m.unpack(res.z)
    f = prob.f_np(res.z)
    prices = np.concatenate([sol["p"], sol["pf"]])
    excess = f[2:]                      # goods + factor market equations
    assert abs(prices @ excess) < 1e-6


def test_homogeneity_doubling_numeraire_doubles_prices():
    m1, _, r1 = _solve(1.0)
    m2, _, r2 = _solve(2.0)
    s1, s2 = m1.unpack(r1.z), m2.unpack(r2.z)
    np.testing.assert_allclose(s2["p"], 2 * s1["p"], rtol=1e-6)
    np.testing.assert_allclose(s2["pf"], 2 * s1["pf"], rtol=1e-6)
    np.testing.assert_allclose(s2["y"], s1["y"], rtol=1e-6)  # real side unchanged


def test_diagnose_warns_without_numeraire():
    from examples.shoven_whalley import build_model_no_numeraire
    m = build_model_no_numeraire()
    assert any("numeraire" in w for w in m.diagnose())
