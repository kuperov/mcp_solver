import numpy as np
import pytest

from tests.problems import COURNOT_SOLUTION, LIBRARY, assert_mcp_solution


@pytest.mark.parametrize("name", sorted(LIBRARY))
def test_library_problem_solves(name, solver):
    p = LIBRARY[name]()
    res = solver(p)
    assert res.converged, f"{name}: {res.status} residual={res.residual:.2e}"
    assert_mcp_solution(p, res.z)
    if hasattr(p, "known_solution"):
        np.testing.assert_allclose(res.z, p.known_solution, atol=1e-5)


def test_cournot_closed_form(solver):
    p = LIBRARY["cournot"]()
    res = solver(p)
    np.testing.assert_allclose(res.z, COURNOT_SOLUTION, atol=1e-7)
