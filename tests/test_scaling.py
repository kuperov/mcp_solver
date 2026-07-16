import numpy as np

from mcp_solver.scaling import ruiz


def test_ruiz_equilibrates_badly_scaled_matrix():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 40))
    A[:5, :] *= 1e6      # CGE-style: some rows in "quantity" units
    A[:, -5:] *= 1e-7
    As, R, C = ruiz(A)
    np.testing.assert_allclose(As, np.diag(R) @ A @ np.diag(C), rtol=1e-13)
    row_max = np.abs(As).max(axis=1)
    col_max = np.abs(As).max(axis=0)
    assert np.all(row_max < 1.3) and np.all(row_max > 0.5)
    assert np.all(col_max < 1.3) and np.all(col_max > 0.5)
    assert np.all(R > 0) and np.all(C > 0)


def test_ruiz_handles_zero_rows_without_dividing_by_zero():
    A = np.array([[1.0, 2.0], [0.0, 0.0]])
    As, R, C = ruiz(A)
    assert np.all(np.isfinite(As)) and np.all(np.isfinite(R))
