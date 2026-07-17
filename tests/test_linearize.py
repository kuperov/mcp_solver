import jax.numpy as jnp
import numpy as np

from mcp_solver.normal_map import fB_np
from mcp_solver.path.linearize import linearize
from mcp_solver.problem import MCPProblem

INF = np.inf


def _problem():
    # nonlinear, badly scaled on purpose
    D = np.array([1e5, 1.0, 1e-3])
    # Use z-affine form to avoid zero Jacobian entries at intermediate points
    f = lambda z: jnp.asarray(D) * (z - 1.0)
    return MCPProblem(f, np.zeros(3), np.full(3, INF), np.full(3, 1.5))


def test_linearization_identities():
    p = _problem()
    x = np.array([1.2, -0.3, 0.7])      # -0.3 below bound: z=0, w=0.3
    lin = linearize(p, x)
    z = np.clip(x, p.lb, p.ub)
    # M, q reproduce f at the linearization point: M z + q = f(z)
    np.testing.assert_allclose(lin.M @ z + lin.q, p.f_np(z), rtol=1e-10)
    # r is the normal-map residual at x
    np.testing.assert_allclose(lin.r, fB_np(p.f_np, x, p.lb, p.ub), rtol=1e-10)
    # r equals the triple identity M z0 + q - w0 + v0
    np.testing.assert_allclose(lin.M @ lin.z0 + lin.q - lin.w0 + lin.v0,
                               lin.r, rtol=1e-8, atol=1e-10)
    # scaled t=0 path equation holds
    np.testing.assert_allclose(lin.Ms @ lin.z0s + lin.qs - lin.w0s + lin.v0s,
                               lin.rs, rtol=1e-8, atol=1e-10)
    # scaled matrix is equilibrated (max-norms near 1) despite 1e5..1e-3 rows
    assert np.abs(lin.Ms).max(axis=1).max() < 1.5
    assert np.abs(lin.Ms).max(axis=1).min() > 0.4
    # unscale inverts scale
    z2, w2, v2 = lin.unscale_triple(lin.z0s, lin.w0s, lin.v0s)
    np.testing.assert_allclose(z2, lin.z0, rtol=1e-12)
    np.testing.assert_allclose(w2, lin.w0, rtol=1e-12)
    np.testing.assert_allclose(v2, lin.v0, rtol=1e-12)
    # scaled bounds ordered
    assert np.all(lin.lbs <= lin.ubs)


def test_linearize_returns_none_on_undefined():
    p = MCPProblem(lambda z: jnp.log(z), np.zeros(1), np.array([INF]),
                   np.array([1.0]))
    # f(pi_B(x)) = log(0) = -inf at x <= 0
    assert linearize(p, np.array([-1.0])) is None


def test_masks_present():
    lb = np.array([-INF, 0.0, 2.0])
    ub = np.array([INF, INF, 2.0])
    p = MCPProblem(lambda z: z, lb, ub, np.array([0.0, 1.0, 2.0]))
    lin = linearize(p, p.x0.copy())
    assert lin.free.tolist() == [True, False, False]
    assert lin.fixed.tolist() == [False, False, True]
