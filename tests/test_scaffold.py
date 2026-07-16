import jax.numpy as jnp
import numpy as np

from mcp_solver import IterationRecord, SolveResult, SolverOptions, Status


def test_x64_enabled():
    assert jnp.array(1.0).dtype == jnp.float64


def test_options_defaults():
    o = SolverOptions()
    assert o.tol == 1e-8
    assert 0 < o.sigma < 1 and 0 < o.beta < 1
    assert o.m_bar >= 1 and o.n_bar >= 1


def test_result_converged_and_table():
    z = np.zeros(2)
    recs = [IterationRecord(k=0, merit=1.0, step_type="m", step_len=1.0)]
    r = SolveResult(status=Status.CONVERGED, z=z, w=z, v=z,
                    residual=1e-12, iterations=recs)
    assert r.converged
    assert "merit" in r.table() and "1" in r.table()
    r2 = SolveResult(status=Status.STALLED, z=z, w=z, v=z,
                     residual=1.0, iterations=[])
    assert not r2.converged
