"""Tests for the AgCGE sample (examples/agcge_national.py).

Correctness bar mirrors the agcge package's own: exact benchmark
replication, the two dropped identities verified ex post, and scenario
solutions cross-checked between the PATH and semismooth solvers.
"""
import numpy as np
import pytest

from examples.agcge_national import (NATIONAL_ELAS, NATIONAL_SAM, TOY_ELAS,
                                     TOY_SAM, apply_shock, build_model,
                                     calibrate, dropped_identity_residuals,
                                     real_gdp, solve)
from mcp_solver import SolverOptions
from mcp_solver.path.solver import solve_path
from mcp_solver.semismooth import solve_semismooth


@pytest.fixture(scope="module")
def toy_cal():
    return calibrate(TOY_SAM, TOY_ELAS)


@pytest.fixture(scope="module")
def national_cal():
    return calibrate(NATIONAL_SAM, NATIONAL_ELAS)


def _assert_benchmark(cal, v):
    np.testing.assert_allclose(v["Z"], cal.Z0, rtol=1e-7)
    np.testing.assert_allclose(v["Q"], cal.Q0, rtol=1e-7)
    np.testing.assert_allclose(v["pq"], 1.0, atol=1e-7)
    np.testing.assert_allclose(v["pd"], 1.0, atol=1e-7)
    np.testing.assert_allclose(v["pf"], 1.0, atol=1e-7)
    np.testing.assert_allclose(float(v["eps"][0]), 1.0, atol=1e-7)
    np.testing.assert_allclose(v["F"].reshape(cal.F0.shape), cal.F0, atol=1e-7)
    va_r, ca_r = dropped_identity_residuals(cal, v)
    assert va_r < 1e-8 and ca_r < 1e-8


def test_toy_benchmark_replicates(toy_cal, solver):
    v, res = solve(toy_cal, solver=solver)
    _assert_benchmark(toy_cal, v)


def test_national_benchmark_replicates(national_cal, solver):
    v, res = solve(national_cal, solver=solver)
    _assert_benchmark(national_cal, v)
    assert res.residual < 1e-8


def test_model_is_square_and_diagnosable(national_cal):
    m = build_model(national_cal)
    # numeraire fixed -> no missing-numeraire warning, no zero rows/cols
    assert m.diagnose() == []


def test_fmd_scenario_and_cross_solver_agreement(national_cal):
    fmd = apply_shock(national_cal,
                      tfp={"beef": 0.95, "pork": 0.95},
                      export_demand={"beef": 0.5, "pork": 0.5})
    v_path, _ = solve(fmd, solver=solve_path)
    v_semi, _ = solve(fmd, solver=solve_semismooth,
                      options=SolverOptions(max_iter=300))
    k_beef = national_cal.sectors.index("beef")
    # beef collapses: output and exports well down, its price up
    assert v_path["Z"][k_beef] < 0.85 * national_cal.Z0[k_beef]
    assert v_path["E"][k_beef] < 0.80 * national_cal.E0[k_beef]
    assert v_path["pq"][k_beef] > 1.0
    # dropped identities still hold at the shocked solution
    va_r, ca_r = dropped_identity_residuals(fmd, v_path)
    assert max(va_r, ca_r) < 1e-6
    # the two solvers agree on the whole variable dictionary
    for key in ("Z", "E", "M", "pq", "pd", "pf", "Xp", "eps"):
        np.testing.assert_allclose(v_path[key], v_semi[key], rtol=2e-5,
                                   atol=1e-7, err_msg=key)


def test_tariff_scenario_directions(national_cal):
    tar = apply_shock(national_cal, tariff_add={"mfg": 0.10})
    v, _ = solve(tar, solver=solve_path)
    k = national_cal.sectors.index("mfg")
    assert v["M"][k] < national_cal.M0[k]          # imports fall
    assert v["Z"][k] > national_cal.Z0[k]          # protected output rises
    assert float(v["eps"][0]) < 1.0                # real appreciation
    assert real_gdp(tar, v) == pytest.approx(
        real_gdp(national_cal, v), rel=1e-12)      # same base-price aggregate


def test_shock_helper_leaves_base_untouched(national_cal):
    before = national_cal.E0.copy()
    apply_shock(national_cal, export_demand={"beef": 0.5})
    np.testing.assert_array_equal(national_cal.E0, before)
