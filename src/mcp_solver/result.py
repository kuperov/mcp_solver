from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class Status(Enum):
    CONVERGED = auto()
    RAY_TERMINATION = auto()
    MAX_ITERATIONS = auto()
    SINGULAR_BASIS = auto()
    DOMAIN_ERROR = auto()
    STALLED = auto()


@dataclass
class IterationRecord:
    k: int
    merit: float
    step_type: str        # "m", "d", "w" (watchdog), "ls" (linesearch)
    step_len: float
    pivots: int = 0
    T: float = 0.0


@dataclass
class SolveResult:
    status: Status
    z: np.ndarray
    w: np.ndarray
    v: np.ndarray
    residual: float       # natural residual ||z - pi_B(z - f(z))||_inf
    iterations: list[IterationRecord] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return self.status is Status.CONVERGED

    def table(self) -> str:
        lines = [f"{'k':>4} {'merit':>14} {'type':>4} {'step':>10} "
                 f"{'pivots':>6} {'T':>6}"]
        for r in self.iterations:
            lines.append(f"{r.k:>4} {r.merit:>14.6e} {r.step_type:>4} "
                         f"{r.step_len:>10.3e} {r.pivots:>6} {r.T:>6.3f}")
        lines.append(f"status: {self.status.name}  residual: {self.residual:.3e}")
        return "\n".join(lines)
