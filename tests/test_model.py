import jax.numpy as jnp
import numpy as np
import pytest

from mcp_solver.model import Model


def _two_block_model():
    m = Model()
    m.add_variables("x", 2, lb=0.0, start=1.0)
    m.add_variables("y", 1, start=2.0)                    # free
    m.add_equations("ex", lambda v: v["x"] - v["y"], complements="x")
    m.add_equations("ey", lambda v: jnp.sum(v["x"], keepdims=True) - 3.0,
                    complements="y")
    return m


def test_build_pack_unpack_and_eval():
    m = _two_block_model()
    p = m.build()
    assert p.n == 3
    np.testing.assert_allclose(p.x0, [1.0, 1.0, 2.0])
    np.testing.assert_allclose(p.lb, [0.0, 0.0, -np.inf])
    # f ordering follows variable declaration order: [ex(0), ex(1), ey]
    np.testing.assert_allclose(p.f_np(np.array([1.0, 1.0, 2.0])),
                               [-1.0, -1.0, -1.0])
    d = m.unpack(np.array([5.0, 6.0, 7.0]))
    np.testing.assert_allclose(d["x"], [5.0, 6.0])
    np.testing.assert_allclose(d["y"], [7.0])


def test_fix_sets_bounds_and_start():
    m = _two_block_model()
    m.fix("x", 1, 4.0)
    p = m.build()
    assert p.lb[1] == p.ub[1] == 4.0 and p.x0[1] == 4.0


def test_unpaired_and_double_paired_blocks_raise():
    m = Model()
    m.add_variables("x", 2)
    with pytest.raises(ValueError, match="unpaired"):
        m.build()
    m.add_equations("e1", lambda v: v["x"], complements="x")
    with pytest.raises(ValueError, match="already paired"):
        m.add_equations("e2", lambda v: v["x"], complements="x")


def test_wrong_equation_size_raises():
    m = Model()
    m.add_variables("x", 2)
    m.add_equations("e", lambda v: jnp.sum(v["x"], keepdims=True),
                    complements="x")   # size 1 != 2
    with pytest.raises(ValueError, match="size"):
        m.build()


def test_diagnose_flags_zero_column_and_rank():
    m = Model()
    m.add_variables("x", 2, start=1.0)
    # f ignores x[1] entirely -> zero column, rank deficiency
    m.add_equations("e", lambda v: jnp.stack([v["x"][0] - 1.0,
                                              v["x"][0] + 1.0]),
                    complements="x")
    warnings = m.diagnose()
    assert any("column" in w for w in warnings)
    assert any("rank" in w for w in warnings)


def test_diagnose_clean_model_is_quiet():
    m = _two_block_model()
    assert m.diagnose() == []
