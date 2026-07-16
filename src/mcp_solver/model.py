from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from mcp_solver.problem import MCPProblem


@dataclass
class _Block:
    name: str
    size: int
    lb: np.ndarray
    ub: np.ndarray
    start: np.ndarray
    offset: int = 0
    equation: object = None      # (eq_name, func) once paired


class Model:
    """Light GAMS-MCP-style helpers: named blocks, equation pairing, fix()."""

    def __init__(self):
        self._blocks: dict[str, _Block] = {}

    def add_variables(self, name, size, lb=-np.inf, ub=np.inf, start=None):
        if name in self._blocks:
            raise ValueError(f"variable block {name!r} already exists")
        lb_a = np.broadcast_to(np.asarray(lb, float), (size,)).copy()
        ub_a = np.broadcast_to(np.asarray(ub, float), (size,)).copy()
        if start is None:
            start_a = np.clip(np.zeros(size), lb_a, ub_a)
        else:
            start_a = np.broadcast_to(np.asarray(start, float), (size,)).copy()
        self._blocks[name] = _Block(name, size, lb_a, ub_a, start_a)

    def add_equations(self, name, func, complements):
        block = self._require(complements)
        if block.equation is not None:
            raise ValueError(f"block {complements!r} is already paired "
                             f"with {block.equation[0]!r}")
        block.equation = (name, func)

    def fix(self, name, index, value):
        b = self._require(name)
        b.lb[index] = b.ub[index] = b.start[index] = float(value)

    def build(self, **mcp_kwargs) -> MCPProblem:
        blocks = list(self._blocks.values())
        unpaired = [b.name for b in blocks if b.equation is None]
        if unpaired:
            raise ValueError(f"unpaired variable blocks: {unpaired}")
        off = 0
        for b in blocks:
            b.offset = off
            off += b.size
        n = off

        def f(z):
            vars_ = {b.name: z[b.offset:b.offset + b.size] for b in blocks}
            outs = []
            for b in blocks:
                out = jnp.atleast_1d(b.equation[1](vars_))
                outs.append(out)
            return jnp.concatenate(outs)

        lb = np.concatenate([b.lb for b in blocks])
        ub = np.concatenate([b.ub for b in blocks])
        x0 = np.concatenate([b.start for b in blocks])
        # eager size check with a plain-numpy trial evaluation
        vars_np = {b.name: jnp.asarray(x0[b.offset:b.offset + b.size])
                   for b in blocks}
        for b in blocks:
            got = np.atleast_1d(np.asarray(b.equation[1](vars_np)))
            if got.shape != (b.size,):
                raise ValueError(
                    f"equation {b.equation[0]!r} returns size {got.shape}, "
                    f"complement block {b.name!r} has size {b.size}")
        return MCPProblem(f, lb, ub, x0, **mcp_kwargs)

    def unpack(self, z):
        out, off = {}, 0
        for b in self._blocks.values():
            out[b.name] = np.asarray(z[off:off + b.size])
            off += b.size
        return out

    def diagnose(self, problem: MCPProblem | None = None) -> list[str]:
        p = problem if problem is not None else self.build()
        J = p.jac(np.clip(p.x0, p.lb, p.ub))
        warnings = []
        zero_rows = np.flatnonzero(np.abs(J).max(axis=1) == 0.0)
        zero_cols = np.flatnonzero(np.abs(J).max(axis=0) == 0.0)
        if zero_rows.size:
            warnings.append(f"zero Jacobian row(s) at {zero_rows.tolist()} "
                            "— equation ignores all variables")
        if zero_cols.size:
            warnings.append(f"zero Jacobian column(s) at {zero_cols.tolist()} "
                            "— variable appears in no equation")
        rank = np.linalg.matrix_rank(J)
        if rank < p.n:
            msg = f"Jacobian rank {rank} < n = {p.n} at start point"
            if rank == p.n - 1 and not np.any(p.lb == p.ub):
                msg += (" — deficiency of exactly 1 with no fixed variable: "
                        "likely a missing numeraire (use Model.fix)")
            warnings.append(msg)
        return warnings

    def _require(self, name) -> _Block:
        if name not in self._blocks:
            raise ValueError(f"unknown variable block {name!r}")
        return self._blocks[name]
