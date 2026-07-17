import jax

# Solver correctness requires float64; must run before any submodule import.
jax.config.update("jax_enable_x64", True)

from mcp_solver.model import Model
from mcp_solver.options import SolverOptions
from mcp_solver.path.solver import solve_path
from mcp_solver.problem import MCPProblem
from mcp_solver.result import IterationRecord, SolveResult, Status
from mcp_solver.semismooth import solve_semismooth

__all__ = ["SolverOptions", "Status", "SolveResult", "IterationRecord",
           "MCPProblem", "Model", "solve_semismooth", "solve_path"]
