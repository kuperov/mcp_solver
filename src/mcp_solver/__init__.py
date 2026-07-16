import jax

# Solver correctness requires float64; must run before any submodule import.
jax.config.update("jax_enable_x64", True)

from mcp_solver.options import SolverOptions
from mcp_solver.result import IterationRecord, SolveResult, Status

__all__ = ["SolverOptions", "Status", "SolveResult", "IterationRecord"]
