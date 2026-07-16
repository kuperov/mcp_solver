import pytest

from mcp_solver.semismooth import solve_semismooth

# Stage 2 adds "path": solve_path here; every parametrized test then
# runs through both solvers automatically (the spec's cross-check layer).
SOLVERS = {"semismooth": solve_semismooth}


@pytest.fixture(params=sorted(SOLVERS), ids=sorted(SOLVERS))
def solver(request):
    return SOLVERS[request.param]
