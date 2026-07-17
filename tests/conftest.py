import pytest

from mcp_solver.path.solver import solve_path
from mcp_solver.semismooth import solve_semismooth

SOLVERS = {"semismooth": solve_semismooth, "path": solve_path}


@pytest.fixture(params=sorted(SOLVERS), ids=sorted(SOLVERS))
def solver(request):
    return SOLVERS[request.param]
