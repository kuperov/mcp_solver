import jax.numpy as jnp
import numpy as np
import pytest

from mcp_solver import SolverOptions, Status
from mcp_solver.normal_map import fB_np, merit
from mcp_solver.problem import MCPProblem
from mcp_solver.semismooth import fb_masks, fb_residual, fb_system, solve_semismooth

INF = np.inf


def test_merit_and_fB_np():
    lb, ub = np.array([0.0]), np.array([2.0])
    f_np = lambda z: np.asarray(z) - 1.0
    # x = 1 is the solution: fB = 0
    np.testing.assert_allclose(fB_np(f_np, np.array([1.0]), lb, ub), [0.0])
    assert merit(f_np, np.array([1.0]), lb, ub) == 0.0
    # x = -1: z = 0, fB = f(0) + x - 0 = -1 - 1 = -2
    np.testing.assert_allclose(fB_np(f_np, np.array([-1.0]), lb, ub), [-2.0])
    assert merit(f_np, np.array([-1.0]), lb, ub) == 2.0
    # undefined f -> merit inf, no raise
    f_nan = lambda z: np.full_like(z, np.nan)
    assert merit(f_nan, np.array([1.0]), lb, ub) == np.inf


def test_options_construction_knobs_removed():
    with pytest.raises(TypeError):
        SolverOptions(jac_chunk=64)
    with pytest.raises(TypeError):
        SolverOptions(jac_coloring=False)


def test_fb_residual_matches_fb_system():
    rng = np.random.default_rng(2)
    lb = np.array([-INF, 0.0, -INF, 0.0, 3.0])
    ub = np.array([INF, INF, 2.0, 2.0, 3.0])
    masks = fb_masks(lb, ub)
    z = rng.uniform(0.1, 1.9, 5)
    fval = rng.standard_normal(5)
    J = rng.standard_normal((5, 5))
    Phi_only = fb_residual(z, fval, lb, ub, masks)
    Phi_sys, _ = fb_system(z, fval, J, lb, ub, masks)
    np.testing.assert_allclose(Phi_only, Phi_sys, rtol=1e-14)


def test_linesearch_does_not_recompute_jacobian_on_rejected_trials():
    # atan problem forces many backtracks (stage-1 suite's nasty scalar)
    p = MCPProblem(lambda z: jnp.arctan(z - 10.0), np.full(1, -INF),
                   np.array([INF]), np.array([0.0]))
    calls = {"jac": 0}
    orig = p.jac_boxed
    p.jac_boxed = lambda z: (calls.__setitem__("jac", calls["jac"] + 1), orig(z))[1]
    res = solve_semismooth(p)
    assert res.status is Status.CONVERGED
    # one Jacobian per accepted iterate (+1 initial, +small slack for the
    # H-finiteness re-check path); rejected trials must not cost Jacobians
    assert calls["jac"] <= len(res.iterations) + 3
