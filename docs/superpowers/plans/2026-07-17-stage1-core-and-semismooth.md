# Stage 1: Shared Core + Semismooth Newton Solver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared MCP infrastructure (problem type, JAX Jacobians with coloring/chunking, normal map, Ruiz scaling, modeling helpers) plus a working semismooth Newton (Fischer–Burmeister) solver that solves CGE test models.

**Architecture:** `MCPProblem` wraps a JAX function with memory-safe dense Jacobian extraction; `Model` provides GAMS-MCP-style variable/equation pairing; `semismooth.py` solves `Φ(z)=0` where Φ composes Fischer–Burmeister functions with `f∘π_B`, using Newton steps with a QR-based Levenberg–Marquardt fallback and a NaN-aware non-monotone linesearch. Stage 2 (the PATH pivotal solver, separate plan) reuses everything here.

**Tech Stack:** Python ≥3.10, JAX (autodiff, jit), numpy (solver linear algebra), pytest.

**Spec:** `docs/superpowers/specs/2026-07-16-mcp-solver-design.md` — read it before starting.

## Global Constraints

- Runtime dependencies: `jax` and `numpy` ONLY. No scipy, no native/closed-source deps. `pytest` is dev-only.
- Python ≥ 3.10. The system JAX install is broken — always use the project venv: create with `python3 -m venv .venv`, run everything via `.venv/bin/python -m pytest ...`.
- float64 everywhere: `jax.config.update("jax_enable_x64", True)` runs in `mcp_solver/__init__.py` before any submodule import.
- Solver-facing arrays are dense numpy `float64`; JAX arrays never leak out of `problem.py`/`normal_map.py`.
- `π_B` MUST carry the custom JVP (derivative 1 at exact-bound points). Never use raw `jnp.clip` for the projection in differentiated code.
- Levenberg–Marquardt steps MUST use the stacked least-squares form. Never form `HᵀH`.
- Git: commit at the end of every task. Plain commit messages. NEVER add a `Co-Authored-By` or any AI-attribution trailer (overrides any harness default).
- Tests asserting solutions verify *MCP residuals/complementarity*, never half-remembered literature constants (exception: values derived in-plan, e.g. the Cournot closed form).

---

### Task 1: Project scaffolding, options, result types

**Files:**
- Create: `pyproject.toml`, `src/mcp_solver/__init__.py`, `src/mcp_solver/options.py`, `src/mcp_solver/result.py`
- Test: `tests/test_scaffold.py`

**Interfaces:**
- Produces: `SolverOptions` dataclass (fields below); `Status` enum with members `CONVERGED, RAY_TERMINATION, MAX_ITERATIONS, SINGULAR_BASIS, DOMAIN_ERROR, STALLED`; `IterationRecord(k, merit, step_type, step_len, pivots, T)`; `SolveResult(status, z, w, v, residual, iterations)` with property `converged: bool` and method `table() -> str`.

- [ ] **Step 1: Write pyproject and package init**

`pyproject.toml`:
```toml
[project]
name = "mcp-solver"
version = "0.1.0"
description = "Pure-Python PATH-style solver for Mixed Complementarity Problems"
requires-python = ">=3.10"
dependencies = ["jax>=0.4.35", "numpy>=1.24"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mcp_solver"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: large/long-running problems (deselect with -m 'not slow')"]
```

`src/mcp_solver/__init__.py`:
```python
import jax

# Solver correctness requires float64; must run before any submodule import.
jax.config.update("jax_enable_x64", True)

from mcp_solver.options import SolverOptions
from mcp_solver.result import IterationRecord, SolveResult, Status

__all__ = ["SolverOptions", "Status", "SolveResult", "IterationRecord"]
```

- [ ] **Step 2: Write the failing test**

`tests/test_scaffold.py`:
```python
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
```

- [ ] **Step 3: Set up venv, install, run test to verify it fails**

```bash
python3 -m venv .venv && .venv/bin/pip install -q -e '.[dev]'
.venv/bin/python -m pytest tests/test_scaffold.py -v
```
Expected: FAIL (ImportError: `options`/`result` modules missing).

- [ ] **Step 4: Implement options.py and result.py**

`src/mcp_solver/options.py`:
```python
from dataclasses import dataclass


@dataclass
class SolverOptions:
    # convergence
    tol: float = 1e-8            # ||Phi||_inf (stage 1) / ||f_B||_inf (stage 2)
    max_iter: int = 500
    # non-monotone reference values (paper section 2.4)
    m_bar: int = 10              # merit-memory length (paper m-bar)
    n_bar: int = 5               # d-steps between forced m-steps (stage 2)
    sigma: float = 0.01          # descent relaxation, sigma in (0,1)
    beta: float = 0.5            # d-step radius shrink factor (stage 2)
    delta0: float = 1.0          # initial d-step radius (stage 2)
    # linesearch (stage 1)
    armijo_c: float = 1e-4
    alpha_min: float = 1e-12
    lm_mu: float = 1e-6          # LM regularization scale
    # pivoting (stage 2; declared now so both stages share one options type)
    max_pivots: int = 3000
    refactor_every: int = 50
    pivot_tol: float = 1e-9
    basis_residual_tol: float = 1e-7
    lemke_start: bool = False
    # jacobian extraction
    jac_chunk: int = 256
    jac_coloring: bool = True
    verbose: bool = False
```

`src/mcp_solver/result.py`:
```python
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
```

- [ ] **Step 5: Run tests, commit**

```bash
.venv/bin/python -m pytest tests/test_scaffold.py -v
```
Expected: 3 passed.
```bash
printf '.venv/\n__pycache__/\n*.egg-info/\n' > .gitignore
git add -A && git commit -m "feat: project scaffolding, SolverOptions, SolveResult/Status"
```

---

### Task 2: Ruiz equilibration (`scaling.py`)

**Files:**
- Create: `src/mcp_solver/scaling.py`
- Test: `tests/test_scaling.py`

**Interfaces:**
- Produces: `ruiz(A: np.ndarray, max_iter: int = 20, tol: float = 1e-2) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(A_scaled, R, C)` with `A_scaled = diag(R) @ A @ diag(C)`, R/C strictly positive 1-D arrays.

- [ ] **Step 1: Write the failing test**

`tests/test_scaling.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_scaling.py -v
```
Expected: FAIL (ModuleNotFoundError: `mcp_solver.scaling`).

- [ ] **Step 3: Implement**

`src/mcp_solver/scaling.py`:
```python
import numpy as np


def ruiz(A: np.ndarray, max_iter: int = 20, tol: float = 1e-2):
    """Ruiz equilibration: iterated sqrt max-norm row/column scaling.

    Returns (A_scaled, R, C) with A_scaled = diag(R) @ A @ diag(C).
    Zero rows/columns are left unscaled (scale factor 1).
    """
    n, m = A.shape
    R = np.ones(n)
    C = np.ones(m)
    As = A.astype(np.float64, copy=True)
    for _ in range(max_iter):
        r = np.sqrt(np.abs(As).max(axis=1))
        c = np.sqrt(np.abs(As).max(axis=0))
        r[r == 0] = 1.0
        c[c == 0] = 1.0
        As /= r[:, None]
        As /= c[None, :]
        R /= r
        C /= c
        if max(np.abs(1 - r * r).max(), np.abs(1 - c * c).max()) < tol:
            break
    return As, R, C
```

- [ ] **Step 4: Run tests, commit**

```bash
.venv/bin/python -m pytest tests/test_scaling.py -v
```
Expected: 2 passed.
```bash
git add -A && git commit -m "feat: Ruiz equilibration"
```

---

### Task 3: Normal map with pinned projection subgradient (`normal_map.py`)

**Files:**
- Create: `src/mcp_solver/normal_map.py`
- Test: `tests/test_normal_map.py`

**Interfaces:**
- Produces (all JAX-traceable): `project(x, lb, ub)` — clip with custom JVP (derivative 1 at exact bounds); `make_normal_map(f, lb, ub) -> fB` where `fB(x) = f(project(x)) + x - project(x)`; numpy helpers `decompose(x, lb, ub) -> (z, w, v)` (paper eq. 5) and `natural_residual(z, f_of_z, lb, ub) -> float` = `||z - clip(z - f(z), lb, ub)||_inf`.

- [ ] **Step 1: Write the failing test**

`tests/test_normal_map.py`:
```python
import jax
import jax.numpy as jnp
import numpy as np

from mcp_solver.normal_map import (decompose, make_normal_map,
                                   natural_residual, project)


def test_project_derivative_is_one_at_exact_bounds():
    lb, ub = jnp.array([0.0]), jnp.array([2.0])
    g = jax.grad(lambda x: project(jnp.array([x]), lb, ub)[0])
    assert float(g(0.0)) == 1.0     # exactly on lower bound -> 1, not 0.5
    assert float(g(2.0)) == 1.0     # exactly on upper bound -> 1, not 0.5
    assert float(g(1.0)) == 1.0     # interior
    assert float(g(-1.0)) == 0.0    # strictly outside
    assert float(g(3.0)) == 0.0


def test_project_values_match_clip():
    lb = jnp.array([0.0, -jnp.inf, 1.0])
    ub = jnp.array([2.0, jnp.inf, 1.0])
    x = jnp.array([-5.0, 3.0, 7.0])
    np.testing.assert_allclose(project(x, lb, ub), jnp.clip(x, lb, ub))


def test_normal_map_zero_iff_mcp_solution():
    # f(z) = z - 1 on [0, 2]: solution z*=1 interior, x*=z*=1
    f = lambda z: z - 1.0
    lb, ub = jnp.array([0.0]), jnp.array([2.0])
    fB = make_normal_map(f, lb, ub)
    np.testing.assert_allclose(fB(jnp.array([1.0])), [0.0], atol=1e-15)
    # f(z) = z + 1 on [0, 2]: solution z*=0 (f>0 at lower bound),
    # x* = z* - w* = -1 and fB(x*) = f(0) + x* - 0 = 1 - 1 = 0
    f2 = lambda z: z + 1.0
    fB2 = make_normal_map(f2, lb, ub)
    np.testing.assert_allclose(fB2(jnp.array([-1.0])), [0.0], atol=1e-15)


def test_decompose_roundtrip():
    lb, ub = np.array([0.0, 0.0]), np.array([2.0, 2.0])
    x = np.array([-0.5, 2.7])
    z, w, v = decompose(x, lb, ub)
    np.testing.assert_allclose(z, [0.0, 2.0])
    np.testing.assert_allclose(w, [0.5, 0.0])   # w = (z - x)_+
    np.testing.assert_allclose(v, [0.0, 0.7])   # v = (x - z)_+
    np.testing.assert_allclose(z - w + v, x)


def test_natural_residual_zero_at_solution():
    lb, ub = np.array([0.0]), np.array([np.inf])
    assert natural_residual(np.array([0.0]), np.array([3.0]), lb, ub) == 0.0
    assert natural_residual(np.array([1.0]), np.array([3.0]), lb, ub) > 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_normal_map.py -v
```
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

`src/mcp_solver/normal_map.py`:
```python
import jax
import jax.numpy as jnp
import numpy as np


@jax.custom_jvp
def project(x, lb, ub):
    """pi_B: componentwise clip onto [lb, ub] with a pinned subgradient.

    Stock jnp.clip has gradient 0.5 at exact-bound points (JAX splits
    ties); projected iterates sit exactly on bounds by construction, so
    we pin the derivative to 1 there (boundary treated as interior),
    selecting a definite B-subdifferential element. See spec.
    """
    return jnp.clip(x, lb, ub)


@project.defjvp
def _project_jvp(primals, tangents):
    x, lb, ub = primals
    dx, _, _ = tangents  # bounds are constants
    inside = (x >= lb) & (x <= ub)
    return jnp.clip(x, lb, ub), jnp.where(inside, dx, 0.0)


def make_normal_map(f, lb, ub):
    """Robinson's normal map f_B(x) = f(pi_B(x)) + x - pi_B(x)."""
    lb = jnp.asarray(lb)
    ub = jnp.asarray(ub)

    def fB(x):
        z = project(x, lb, ub)
        return f(z) + x - z

    return fB


def decompose(x, lb, ub):
    """x -> (z, w, v): z = pi_B(x), w = (z-x)_+, v = (x-z)_+ (paper eq. 5)."""
    z = np.clip(x, lb, ub)
    w = np.maximum(z - x, 0.0)
    v = np.maximum(x - z, 0.0)
    return z, w, v


def natural_residual(z, f_of_z, lb, ub):
    """||z - pi_B(z - f(z))||_inf — solver-independent optimality measure."""
    return float(np.abs(z - np.clip(z - f_of_z, lb, ub)).max())
```

- [ ] **Step 4: Run tests, commit**

```bash
.venv/bin/python -m pytest tests/test_normal_map.py -v
```
Expected: 5 passed. The first test is the review-driven regression test: it fails with 0.5 if anyone swaps in raw `jnp.clip`.
```bash
git add -A && git commit -m "feat: normal map with pinned projection subgradient"
```

---

### Task 4: MCPProblem with colored/chunked Jacobians (`problem.py`)

**Files:**
- Create: `src/mcp_solver/problem.py`
- Test: `tests/test_problem.py`

**Interfaces:**
- Consumes: `project` from `mcp_solver.normal_map`.
- Produces: class `MCPProblem(f, lb, ub, x0, *, jac_chunk=256, jac_coloring=True)` where `f` is a JAX callable `R^n -> R^n`. Members: `n: int`; `lb, ub, x0: np.ndarray`; `f_np(z: np.ndarray) -> np.ndarray` (raw f); `f_boxed(z) -> np.ndarray` (= f∘π_B); `jac(z) -> np.ndarray` dense (n,n) of raw f; `jac_boxed(z) -> np.ndarray` of f∘π_B (differentiated through the pinned projection); `n_jac_tangents: int` (number of JVP passes per Jacobian: n_colors if coloring won, else n).

- [ ] **Step 1: Write the failing test**

`tests/test_problem.py`:
```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mcp_solver.problem import MCPProblem


def _dense_f(z):
    # every output depends on every input -> coloring cannot win
    return jnp.tanh(z) + 0.01 * jnp.sum(z**2) * jnp.ones_like(z)


def _banded_f(z):
    # tridiagonal structure -> 3-colorable
    lower = jnp.concatenate([jnp.zeros(1), z[:-1]])
    upper = jnp.concatenate([z[1:], jnp.zeros(1)])
    return z**2 + 0.5 * lower - 0.25 * upper


@pytest.mark.parametrize("f", [_dense_f, _banded_f])
def test_jacobian_matches_jacfwd(f):
    n = 30
    lb, ub = np.full(n, -10.0), np.full(n, 10.0)
    x0 = np.linspace(0.1, 1.0, n)
    p = MCPProblem(f, lb, ub, x0, jac_chunk=7)
    z = x0 + 0.03
    expected = np.asarray(jax.jacfwd(f)(jnp.asarray(z)))
    np.testing.assert_allclose(p.jac(z), expected, rtol=1e-10, atol=1e-12)


def test_coloring_wins_on_banded_and_falls_back_on_dense():
    n = 30
    lb, ub = np.full(n, -10.0), np.full(n, 10.0)
    x0 = np.full(n, 0.5)
    banded = MCPProblem(_banded_f, lb, ub, x0)
    dense = MCPProblem(_dense_f, lb, ub, x0)
    assert banded.n_jac_tangents <= 3
    assert dense.n_jac_tangents == n


def test_jac_boxed_zeroes_out_of_bounds_columns_and_pins_boundary():
    n = 3
    lb, ub = np.zeros(n), np.full(n, 2.0)
    f = lambda z: z**2 + jnp.roll(z, 1)
    p = MCPProblem(f, lb, ub, np.full(n, 1.0))
    z = np.array([-1.0, 0.0, 1.0])   # below bound / exactly on bound / interior
    Jb = p.jac_boxed(z)
    assert np.all(Jb[:, 0] == 0.0)                  # clipped column zeroed
    zc = np.clip(z, lb, ub)
    Jraw = np.asarray(jax.jacfwd(f)(jnp.asarray(zc)))
    np.testing.assert_allclose(Jb[:, 1], Jraw[:, 1])  # boundary column intact
    np.testing.assert_allclose(Jb[:, 2], Jraw[:, 2])


def test_f_boxed_evaluates_inside_bounds_only():
    # raw f would be log(-5) = nan; boxed evaluation clips to 0.1 first
    lb, ub = np.array([0.1]), np.array([np.inf])
    p = MCPProblem(lambda z: jnp.log(z), lb, ub, np.array([1.0]))
    assert np.all(np.isfinite(p.f_boxed(np.array([-5.0]))))
    assert not np.all(np.isfinite(p.f_np(np.array([-5.0]))))


def test_validation_errors():
    with pytest.raises(ValueError):
        MCPProblem(lambda z: z, np.zeros(2), -np.ones(2), np.zeros(2))  # lb > ub
    with pytest.raises(ValueError):
        MCPProblem(lambda z: z, np.zeros(2), np.ones(2), np.zeros(3))   # bad x0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_problem.py -v
```
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

`src/mcp_solver/problem.py`:
```python
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from mcp_solver.normal_map import project


class MCPProblem:
    """An MCP: find l <= z <= u with f(z) = w - v, complementarity holding.

    Jacobians are extracted by batched JVPs. At construction the sparsity
    pattern is probed at two perturbed points and greedily column-colored;
    if coloring beats one-tangent-per-column (dense rows in CGE income
    equations defeat it), compressed seed vectors are used, otherwise
    identity tangents. Either way tangents are batched in chunks of
    `jac_chunk` so peak AD memory is bounded regardless of structure.
    A build-time verification compares the colored Jacobian against the
    direct one and silently falls back on mismatch.
    """

    def __init__(self, f, lb, ub, x0, *, jac_chunk=256, jac_coloring=True):
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.n = self.lb.size
        if self.ub.shape != self.lb.shape or self.x0.shape != self.lb.shape:
            raise ValueError("lb, ub, x0 must have identical shapes")
        if np.any(self.lb > self.ub):
            raise ValueError("lb > ub for some component")
        self._chunk = int(jac_chunk)
        self._lb_j = jnp.asarray(self.lb)
        self._ub_j = jnp.asarray(self.ub)
        self._raw = f
        self._boxed = lambda z: f(project(z, self._lb_j, self._ub_j))
        self._f_jit = jax.jit(f)
        self._fboxed_jit = jax.jit(self._boxed)
        # batched JVP: S is (k, n) tangents -> (k, n) directional derivatives
        self._jvp_raw = jax.jit(
            lambda z, S: jax.vmap(lambda s: jax.jvp(f, (z,), (s,))[1])(S))
        self._jvp_boxed = jax.jit(
            lambda z, S: jax.vmap(
                lambda s: jax.jvp(self._boxed, (z,), (s,))[1])(S))
        self._groups = None          # list of np.ndarray column-index groups
        self._pattern = None         # (rows_nz, cols_nz) of probed sparsity
        if jac_coloring:
            self._build_coloring()
        self.n_jac_tangents = (
            len(self._groups) if self._groups is not None else self.n)

    # ---- public API -------------------------------------------------
    def f_np(self, z):
        return np.asarray(self._f_jit(jnp.asarray(z)))

    def f_boxed(self, z):
        return np.asarray(self._fboxed_jit(jnp.asarray(z)))

    def jac(self, z):
        return self._jac(z, self._jvp_raw)

    def jac_boxed(self, z):
        # boxed pattern is a subset of the raw pattern (clipped columns are
        # zeroed), so the raw coloring remains valid for scattering.
        return self._jac(z, self._jvp_boxed)

    # ---- internals ---------------------------------------------------
    def _batched_jvp(self, z, S, jvp):
        zj = jnp.asarray(z)
        out = [np.asarray(jvp(zj, jnp.asarray(S[s:s + self._chunk])))
               for s in range(0, S.shape[0], self._chunk)]
        return np.concatenate(out, axis=0)

    def _direct_jac(self, z, jvp):
        D = self._batched_jvp(z, np.eye(self.n), jvp)
        return np.ascontiguousarray(D.T)      # rows of D are J columns

    def _jac(self, z, jvp):
        if self._groups is None:
            return self._direct_jac(z, jvp)
        S = np.zeros((len(self._groups), self.n))
        colmap = np.empty(self.n, dtype=np.intp)
        for g, cols in enumerate(self._groups):
            S[g, cols] = 1.0
            colmap[cols] = g
        C = self._batched_jvp(z, S, jvp)      # (n_groups, n)
        J = np.zeros((self.n, self.n))
        rows_nz, cols_nz = self._pattern
        J[rows_nz, cols_nz] = C[colmap[cols_nz], rows_nz]
        return J

    def _build_coloring(self):
        rng = np.random.default_rng(20260717)
        pattern = np.zeros((self.n, self.n), dtype=bool)
        for _ in range(2):   # two probe points; OR of nonzeros
            zp = self.x0 + rng.uniform(0.01, 0.1, self.n) * (
                1.0 + np.abs(self.x0))
            zp = np.clip(zp, self.lb, self.ub)
            pattern |= self._direct_jac(zp, self._jvp_raw) != 0.0
        # greedy column coloring: columns sharing a nonzero row conflict
        order = np.argsort(-pattern.sum(axis=0))
        color_rows: list[np.ndarray] = []     # per color: bool row coverage
        color_of = np.empty(self.n, dtype=np.intp)
        for j in order:
            rows_j = pattern[:, j]
            for c, covered in enumerate(color_rows):
                if not np.any(covered & rows_j):
                    color_of[j] = c
                    color_rows[c] = covered | rows_j
                    break
            else:
                color_of[j] = len(color_rows)
                color_rows.append(rows_j.copy())
        if len(color_rows) >= self.n:
            return                             # no win; keep identity mode
        groups = [np.flatnonzero(color_of == c)
                  for c in range(len(color_rows))]
        self._groups = groups
        self._pattern = np.nonzero(pattern)
        # build-time verification at a third point
        zv = np.clip(self.x0 + rng.uniform(0.01, 0.1, self.n), self.lb, self.ub)
        direct = self._direct_jac(zv, self._jvp_raw)
        colored = self._jac(zv, self._jvp_raw)
        scale = max(1.0, np.abs(direct).max())
        if np.abs(direct - colored).max() > 1e-9 * scale:
            warnings.warn("Jacobian coloring failed verification; "
                          "falling back to direct extraction")
            self._groups = None
            self._pattern = None
```

- [ ] **Step 4: Run tests, verify pass**

```bash
.venv/bin/python -m pytest tests/test_problem.py -v
```
Expected: 6 passed. If `test_coloring_wins_on_banded...` fails on the banded bound, print `banded.n_jac_tangents` — greedy coloring of a tridiagonal pattern must need ≤3 colors.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: MCPProblem with colored/chunked JVP Jacobians"
```

---

### Task 5: Modeling helpers (`model.py`)

**Files:**
- Create: `src/mcp_solver/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `MCPProblem`.
- Produces: class `Model` with methods `add_variables(name, size: int, lb=-inf, ub=+inf, start=None)`; `add_equations(name, func, complements: str)` where `func(vars: dict[str, jnp.ndarray]) -> jnp.ndarray` of the complement block's size; `fix(name, index: int, value: float)`; `build(**mcp_kwargs) -> MCPProblem`; `unpack(z: np.ndarray) -> dict[str, np.ndarray]`; `diagnose(problem=None) -> list[str]` (warning strings, empty = healthy).

- [ ] **Step 1: Write the failing test**

`tests/test_model.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_model.py -v
```
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

`src/mcp_solver/model.py`:
```python
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
```

- [ ] **Step 4: Run tests, commit**

```bash
.venv/bin/python -m pytest tests/test_model.py -v
```
Expected: 6 passed.
```bash
git add -A && git commit -m "feat: Model helpers — blocks, pairing, fix, diagnose"
```

---

### Task 6: Fischer–Burmeister residual and generalized Jacobian (`semismooth.py`, part 1)

**Files:**
- Create: `src/mcp_solver/semismooth.py`
- Test: `tests/test_fb.py`

**Interfaces:**
- Consumes: `MCPProblem` (uses `f_boxed`, `jac_boxed`).
- Produces: `fb_masks(lb, ub) -> dict[str, np.ndarray]` (bool masks `free, lower, upper, both, fixed`); `fb_system(z, fval, J, lb, ub, masks) -> tuple[np.ndarray, np.ndarray]` returning `(Phi, H)` where `H = diag(alpha) + diag(beta) @ J` is an element of the generalized Jacobian. `fval`/`J` must be the *boxed* function/Jacobian values at `z`.

The FB function is `phi(a,b) = a + b - sqrt(a^2+b^2)`, partials `d_a = 1 - a/rho`, `d_b = 1 - b/rho` with `rho = sqrt(a^2+b^2)`; at the kink `rho = 0` use the perturbed element `d_a = d_b = 1 - 1/sqrt(2)`. Case table (spec):

| case | Phi_i | alpha_i | beta_i |
|---|---|---|---|
| free | `f_i` | 0 | 1 |
| fixed (`lb==ub`) | `z_i - lb_i` | 1 | 0 |
| lower | `phi(z_i-l_i, f_i)` | `d_a` | `d_b` |
| upper | `-phi(u_i-z_i, -f_i)` | `d_a` | `d_b` |
| both | outer `phi(a, psi)`, `a=z_i-l_i`, `psi=-phi(c,d)`, `c=u_i-z_i`, `d=-f_i` | `d_a + d_psi*d_c` | `d_psi*d_d` |

(derivations: for upper, `dPhi/dz = -[d_a*(-e_i) + d_b*(-J_i)] = d_a*e_i + d_b*J_i`; for both, chain rule through `psi`.)

- [ ] **Step 1: Write the failing test**

`tests/test_fb.py`:
```python
import numpy as np

from mcp_solver.semismooth import fb_masks, fb_system

INF = np.inf


def _fd_jacobian(phi_of_z, z, h=1e-7):
    n = z.size
    J = np.zeros((n, n))
    for j in range(n):
        zp, zm = z.copy(), z.copy()
        zp[j] += h
        zm[j] -= h
        J[:, j] = (phi_of_z(zp) - phi_of_z(zm)) / (2 * h)
    return J


def test_phi_zero_iff_solution_each_case():
    lb = np.array([-INF, 0.0, -INF, 0.0, 3.0])
    ub = np.array([INF, INF, 2.0, 2.0, 3.0])
    masks = fb_masks(lb, ub)
    assert masks["free"][0] and masks["lower"][1] and masks["upper"][2]
    assert masks["both"][3] and masks["fixed"][4]
    # a solution point: f free = 0; z at lower with f>0; z at upper with
    # f<0; both-bounded interior with f=0; fixed anywhere
    z = np.array([1.0, 0.0, 2.0, 1.0, 3.0])
    f = np.array([0.0, 5.0, -5.0, 0.0, 9.9])
    Phi, _ = fb_system(z, f, np.eye(5), lb, ub, masks)
    np.testing.assert_allclose(Phi, np.zeros(5), atol=1e-14)
    # perturb each: nonzero Phi
    f_bad = np.array([0.1, -0.1, 0.1, 0.3, 0.0])
    z_bad = np.array([1.0, 0.5, 1.5, 1.0, 2.0])
    Phi_bad, _ = fb_system(z_bad, f_bad, np.eye(5), lb, ub, masks)
    assert np.all(np.abs(Phi_bad) > 1e-3)


def test_H_matches_finite_differences_at_generic_point():
    rng = np.random.default_rng(1)
    n = 6
    lb = np.array([-INF, 0.0, -INF, 0.0, -1.0, 2.0])
    ub = np.array([INF, INF, 2.0, 2.0, 1.0, 2.0])
    masks = fb_masks(lb, ub)
    A = rng.standard_normal((n, n))
    b = rng.standard_normal(n)
    f_of = lambda z: A @ z + b          # linear f -> J constant = A
    z0 = np.array([0.3, 0.7, 1.1, 0.4, 0.2, 2.0])  # generic (no kinks)

    def phi_of(z):
        return fb_system(z, f_of(z), A, lb, ub, masks)[0]

    _, H = fb_system(z0, f_of(z0), A, lb, ub, masks)
    np.testing.assert_allclose(H, _fd_jacobian(phi_of, z0),
                               rtol=1e-5, atol=1e-6)


def test_kink_uses_perturbed_element_and_stays_finite():
    lb, ub = np.array([0.0]), np.array([np.inf])
    masks = fb_masks(lb, ub)
    # z at bound and f = 0: exact FB kink (a = b = 0)
    Phi, H = fb_system(np.array([0.0]), np.array([0.0]),
                       np.array([[2.0]]), lb, ub, masks)
    assert np.all(np.isfinite(Phi)) and np.all(np.isfinite(H))
    expected = (1 - 1 / np.sqrt(2)) + (1 - 1 / np.sqrt(2)) * 2.0
    np.testing.assert_allclose(H[0, 0], expected)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_fb.py -v
```
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

Create `src/mcp_solver/semismooth.py` (solver loop is Task 7; this task only these functions):

```python
import numpy as np


def fb_masks(lb, ub):
    finite_l = np.isfinite(lb)
    finite_u = np.isfinite(ub)
    fixed = lb == ub
    return {
        "free": ~finite_l & ~finite_u,
        "lower": finite_l & ~finite_u,
        "upper": ~finite_l & finite_u,
        "both": finite_l & finite_u & ~fixed,
        "fixed": fixed,
    }


def _fb(a, b):
    """phi(a,b)=a+b-sqrt(a^2+b^2) and partials; perturbed element at kink."""
    rho = np.sqrt(a * a + b * b)
    phi = a + b - rho
    safe = np.where(rho == 0.0, 1.0, rho)
    da = 1.0 - a / safe
    db = 1.0 - b / safe
    kink = rho == 0.0
    da = np.where(kink, 1.0 - 1.0 / np.sqrt(2.0), da)
    db = np.where(kink, 1.0 - 1.0 / np.sqrt(2.0), db)
    return phi, da, db


def fb_system(z, fval, J, lb, ub, masks):
    """(Phi, H): FB residual and a generalized-Jacobian element.

    H = diag(alpha) + diag(beta) @ J, with fval/J the boxed f and Jacobian.
    """
    n = z.size
    Phi = np.empty(n)
    alpha = np.empty(n)
    beta = np.empty(n)

    m = masks["free"]
    Phi[m], alpha[m], beta[m] = fval[m], 0.0, 1.0

    m = masks["fixed"]
    Phi[m], alpha[m], beta[m] = z[m] - lb[m], 1.0, 0.0

    m = masks["lower"]
    phi, da, db = _fb(z[m] - lb[m], fval[m])
    Phi[m], alpha[m], beta[m] = phi, da, db

    m = masks["upper"]
    phi, da, db = _fb(ub[m] - z[m], -fval[m])
    Phi[m], alpha[m], beta[m] = -phi, da, db

    m = masks["both"]
    phi2, dc, dd = _fb(ub[m] - z[m], -fval[m])
    psi = -phi2
    phi1, da, dpsi = _fb(z[m] - lb[m], psi)
    Phi[m] = phi1
    alpha[m] = da + dpsi * dc
    beta[m] = dpsi * dd

    H = np.diag(alpha) + beta[:, None] * J
    return Phi, H
```

- [ ] **Step 4: Run tests, commit**

```bash
.venv/bin/python -m pytest tests/test_fb.py -v
```
Expected: 3 passed. The finite-difference test is the load-bearing one — if it fails, check the `upper`/`both` sign derivations against the case table above before touching code.
```bash
git add -A && git commit -m "feat: Fischer-Burmeister residual and generalized Jacobian"
```

---

### Task 7: Semismooth Newton solver loop (`semismooth.py`, part 2)

**Files:**
- Modify: `src/mcp_solver/semismooth.py` (append)
- Modify: `src/mcp_solver/__init__.py` (export `solve_semismooth`)
- Test: `tests/test_semismooth.py`

**Interfaces:**
- Consumes: `fb_masks`, `fb_system`, `ruiz`, `natural_residual`, `decompose`, `SolverOptions`, `SolveResult`, `Status`, `IterationRecord`, `MCPProblem`.
- Produces: `solve_semismooth(problem: MCPProblem, options: SolverOptions | None = None) -> SolveResult`.

Algorithm (spec §Stage 1): Newton on `Φ` with equilibrated solves, stacked-QR LM fallback when the Newton system is singular/non-descent, NaN-aware non-monotone Armijo linesearch (reference = max of last `m_bar` accepted `Ψ` values), start point projected into bounds.

- [ ] **Step 1: Write the failing test**

`tests/test_semismooth.py`:
```python
import jax.numpy as jnp
import numpy as np

from mcp_solver import SolverOptions, Status
from mcp_solver.problem import MCPProblem
from mcp_solver.semismooth import solve_semismooth

INF = np.inf


def _check_mcp(p, res, tol=1e-6):
    assert res.status is Status.CONVERGED
    z, f = res.z, p.f_np(res.z)
    assert np.all(z >= p.lb - tol) and np.all(z <= p.ub + tol)
    at_l = z <= p.lb + tol
    at_u = z >= p.ub - tol
    fixed = p.lb == p.ub
    interior = ~at_l & ~at_u
    assert np.all(f[at_l & ~fixed] >= -tol)
    assert np.all(f[at_u & ~fixed] <= tol)
    assert np.all(np.abs(f[interior]) <= tol)


def test_linear_system_free_vars_converges_in_one_iteration():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    p = MCPProblem(lambda z: jnp.asarray(A) @ z - jnp.asarray(b),
                   np.full(2, -INF), np.full(2, INF), np.zeros(2))
    res = solve_semismooth(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, np.linalg.solve(A, b), atol=1e-8)
    assert len(res.iterations) <= 2


def test_simple_ncp_with_active_bound():
    # f(z) = z + 1 >= 0 for all z >= 0 -> solution z = 0, w = f = 1
    p = MCPProblem(lambda z: z + 1.0, np.zeros(1), np.array([INF]),
                   np.array([5.0]))
    res = solve_semismooth(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [0.0], atol=1e-8)
    np.testing.assert_allclose(res.w, [1.0], atol=1e-6)


def test_box_bounds_upper_active():
    # f(z) = z - 5 on [0, 2]: f < 0 at solution -> z at upper bound 2
    p = MCPProblem(lambda z: z - 5.0, np.zeros(1), np.array([2.0]),
                   np.array([1.0]))
    res = solve_semismooth(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [2.0], atol=1e-8)


def test_domain_hazard_log_never_nans_out():
    # f(z) = log(z) - 1 on [1e-8, inf): solution z = e. A full Newton step
    # from z=5 overshoots negative without boxing; must still converge.
    p = MCPProblem(lambda z: jnp.log(z) - 1.0, np.full(1, 1e-8),
                   np.array([INF]), np.array([5.0]))
    res = solve_semismooth(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [np.e], rtol=1e-7)


def test_nonmonotone_helps_on_nasty_scalar():
    # f(z) = atan(z - 10): monotone but flat; plain Newton oscillates badly
    p = MCPProblem(lambda z: jnp.arctan(z - 10.0), np.full(1, -INF),
                   np.array([INF]), np.array([0.0]))
    res = solve_semismooth(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [10.0], atol=1e-6)


def test_stalls_gracefully_on_infeasible_problem():
    # f(z) = 1 with z <= 0 <= ... no: use f(z) = -1 forever on z >= 0:
    # requires f >= 0 at bound -> no solution exists
    p = MCPProblem(lambda z: -jnp.ones_like(z), np.zeros(1),
                   np.array([INF]), np.array([1.0]))
    res = solve_semismooth(p, SolverOptions(max_iter=50))
    assert res.status in (Status.STALLED, Status.MAX_ITERATIONS)


def test_badly_scaled_problem_converges():
    # quantities ~1e6 against prices ~1 (CGE-style scaling)
    D = np.array([1e6, 1.0, 1e-4])
    A = np.diag(D)
    p = MCPProblem(lambda z: jnp.asarray(A) @ z - jnp.asarray(D * 2.0),
                   np.zeros(3), np.full(3, INF), np.ones(3))
    res = solve_semismooth(p)
    _check_mcp(p, res, tol=1e-4)
    np.testing.assert_allclose(res.z, np.full(3, 2.0), rtol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_semismooth.py -v
```
Expected: FAIL (ImportError: `solve_semismooth`).

- [ ] **Step 3: Implement the solver loop**

Append to `src/mcp_solver/semismooth.py`:

```python
from mcp_solver.normal_map import decompose, natural_residual
from mcp_solver.options import SolverOptions
from mcp_solver.result import IterationRecord, SolveResult, Status
from mcp_solver.scaling import ruiz


def _newton_step(H, Phi, opts):
    """Equilibrated Newton solve; stacked-QR LM fallback. Returns (d, g)."""
    Hs, R, C = ruiz(H)
    rhs = -(R * Phi)
    g = H.T @ Phi                       # gradient of Psi = 0.5||Phi||^2
    try:
        d = C * np.linalg.solve(Hs, rhs)
        if np.all(np.isfinite(d)) and g @ d < 0.0:
            return d, g
    except np.linalg.LinAlgError:
        pass
    # LM: min ||[Hs; sqrt(mu) I] y + [R*Phi; 0]|| in scaled space, d = C*y
    n = H.shape[0]
    mu = opts.lm_mu * max(1.0, float(np.linalg.norm(Phi)))
    A = np.vstack([Hs, np.sqrt(mu) * np.eye(n)])
    b = np.concatenate([rhs, np.zeros(n)])
    y, *_ = np.linalg.lstsq(A, b, rcond=None)
    return C * y, g


def solve_semismooth(problem, options=None):
    opts = options or SolverOptions()
    lb, ub = problem.lb, problem.ub
    masks = fb_masks(lb, ub)
    z = np.clip(problem.x0, lb, ub)

    def system(z):
        fval = problem.f_boxed(z)
        if not np.all(np.isfinite(fval)):
            return fval, None, None
        Phi, H = fb_system(z, fval, problem.jac_boxed(z), lb, ub, masks)
        return fval, Phi, H

    fval, Phi, H = system(z)
    if Phi is None:
        return SolveResult(Status.DOMAIN_ERROR, z, np.zeros_like(z),
                           np.zeros_like(z), np.inf, [])

    psi_hist = [0.5 * float(Phi @ Phi)]
    records = []
    status = Status.MAX_ITERATIONS

    for k in range(opts.max_iter):
        if np.abs(Phi).max() <= opts.tol:
            status = Status.CONVERGED
            break
        d, g = _newton_step(H, Phi, opts)
        ref = max(psi_hist[-opts.m_bar:])
        gTd = float(g @ d)
        alpha, accepted = 1.0, False
        while alpha >= opts.alpha_min:
            zt = np.asarray(z + alpha * d)
            ft, Phit, Ht = system(zt)
            if Phit is not None:
                psit = 0.5 * float(Phit @ Phit)
                if np.isfinite(psit) and \
                        psit <= ref + opts.armijo_c * alpha * gTd:
                    z, fval, Phi, H = zt, ft, Phit, Ht
                    psi_hist.append(psit)
                    accepted = True
                    break
            alpha *= 0.5
        if not accepted:
            status = Status.STALLED
            break
        records.append(IterationRecord(k=k, merit=np.sqrt(2 * psi_hist[-1]),
                                       step_type="ls", step_len=alpha))
        if opts.verbose:
            print(records[-1])

    f_final = problem.f_boxed(z)
    w = np.maximum(f_final, 0.0)
    v = np.maximum(-f_final, 0.0)
    res = natural_residual(z, f_final, lb, ub)
    if status is Status.CONVERGED and not np.isfinite(res):
        status = Status.DOMAIN_ERROR
    return SolveResult(status, z, w, v, res, records)
```

Update `src/mcp_solver/__init__.py` exports:
```python
import jax

# Solver correctness requires float64; must run before any submodule import.
jax.config.update("jax_enable_x64", True)

from mcp_solver.model import Model
from mcp_solver.options import SolverOptions
from mcp_solver.problem import MCPProblem
from mcp_solver.result import IterationRecord, SolveResult, Status
from mcp_solver.semismooth import solve_semismooth

__all__ = ["SolverOptions", "Status", "SolveResult", "IterationRecord",
           "MCPProblem", "Model", "solve_semismooth"]
```

- [ ] **Step 4: Run the full suite**

```bash
.venv/bin/python -m pytest -v
```
Expected: all tests pass. If `test_domain_hazard_log_never_nans_out` fails, the boxing (`f_boxed`/`jac_boxed`) is not being used — the solver must never call raw `f_np`/`jac` on trial points.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: semismooth Newton solver with LM fallback and NaN-aware linesearch"
```

---

### Task 8: Literature problems and the cross-solver harness

**Files:**
- Create: `tests/problems.py` (problem library), `tests/conftest.py` (solver registry), `tests/test_literature.py`
- Test: `tests/test_literature.py`

**Interfaces:**
- Produces: `tests/problems.py` exposing `LIBRARY: dict[str, Callable[[], MCPProblem]]` and `assert_mcp_solution(problem, z, tol=1e-6)`; `tests/conftest.py` exposing fixture `solver` parametrized over `SOLVERS: dict[str, Callable]` (stage 2 adds `"path"` here — this dict is the cross-check hook).

- [ ] **Step 1: Write the problem library**

`tests/problems.py`:
```python
"""Test-problem library. Solutions are verified by MCP residuals, never
by literature constants (except closed forms derived here)."""
import jax.numpy as jnp
import numpy as np

from mcp_solver.problem import MCPProblem

INF = np.inf


def assert_mcp_solution(problem, z, tol=1e-6):
    f = problem.f_np(z)
    lb, ub = problem.lb, problem.ub
    assert np.all(z >= lb - tol) and np.all(z <= ub + tol), "bounds violated"
    fixed = lb == ub
    at_l = (z <= lb + tol) & ~fixed
    at_u = (z >= ub - tol) & ~fixed
    interior = ~at_l & ~at_u & ~fixed
    assert np.all(f[at_l] >= -tol), "f must be >= 0 at active lower bounds"
    assert np.all(f[at_u] <= tol), "f must be <= 0 at active upper bounds"
    assert np.all(np.abs(f[interior]) <= tol), "f must vanish at interior"


def kojima_shindo():
    """Classic 4-variable NCP; degenerate solution set (two solutions)."""
    def f(z):
        z1, z2, z3, z4 = z[0], z[1], z[2], z[3]
        return jnp.stack([
            3 * z1**2 + 2 * z1 * z2 + 2 * z2**2 + z3 + 3 * z4 - 6,
            2 * z1**2 + z1 + z2**2 + 10 * z3 + 2 * z4 - 2,
            3 * z1**2 + z1 * z2 + 2 * z2**2 + 2 * z3 + 9 * z4 - 9,
            z1**2 + 3 * z2**2 + 2 * z3 + 3 * z4 - 3,
        ])
    return MCPProblem(f, np.zeros(4), np.full(4, INF), np.full(4, 1.0))


def cournot_duopoly():
    """2-firm Cournot, linear demand p=10-Q, cost c_i q_i + d_i q_i^2 with
    c=(1,1), d=(1,1). Interior equilibrium solves 4q1+q2=9, q1+4q2=9,
    i.e. q* = (1.8, 1.8)."""
    a, b = 10.0, 1.0
    c = jnp.array([1.0, 1.0])
    d = jnp.array([1.0, 1.0])

    def f(q):
        Q = jnp.sum(q)
        return c + 2 * d * q - (a - b * Q) + b * q
    return MCPProblem(f, np.zeros(2), np.full(2, INF), np.full(2, 1.0))


COURNOT_SOLUTION = np.array([1.8, 1.8])


def synthetic_lcp(n=20, seed=0, frac_active=0.4):
    """Monotone LCP with a constructed known solution.

    Build M = A A^T + I (positive definite), pick z* with a fraction of
    components at the bound, w* complementary, then q = w* - M z*.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) / np.sqrt(n)
    M = A @ A.T + np.eye(n)
    z_star = rng.uniform(0.5, 2.0, n)
    active = rng.random(n) < frac_active
    z_star[active] = 0.0
    w_star = np.zeros(n)
    w_star[active] = rng.uniform(0.5, 2.0, active.sum())
    q = w_star - M @ z_star
    Mj, qj = jnp.asarray(M), jnp.asarray(q)
    p = MCPProblem(lambda z: Mj @ z + qj, np.zeros(n), np.full(n, INF),
                   np.ones(n))
    p.known_solution = z_star
    return p


def upper_bounded_lcp():
    """All three bound regimes active in one problem, solution constructed:
    f(z) = z - t with t = (-1, 0.5, 3) on [0,2]^3 ->
    z* = (0, 0.5, 2), f(z*) = (1, 0, -1)."""
    t = jnp.array([-1.0, 0.5, 3.0])
    p = MCPProblem(lambda z: z - t, np.zeros(3), np.full(3, 2.0),
                   np.full(3, 1.0))
    p.known_solution = np.array([0.0, 0.5, 2.0])
    return p


LIBRARY = {
    "kojima_shindo": kojima_shindo,
    "cournot": cournot_duopoly,
    "lcp_n20": lambda: synthetic_lcp(20, seed=0),
    "lcp_n80_degenerate": lambda: synthetic_lcp(80, seed=3, frac_active=0.7),
    "upper_bounded": upper_bounded_lcp,
}
```

`tests/conftest.py`:
```python
import pytest

from mcp_solver.semismooth import solve_semismooth

# Stage 2 adds "path": solve_path here; every parametrized test then
# runs through both solvers automatically (the spec's cross-check layer).
SOLVERS = {"semismooth": solve_semismooth}


@pytest.fixture(params=sorted(SOLVERS), ids=sorted(SOLVERS))
def solver(request):
    return SOLVERS[request.param]
```

- [ ] **Step 2: Write the failing test**

`tests/test_literature.py`:
```python
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
```

- [ ] **Step 3: Run tests**

```bash
.venv/bin/python -m pytest tests/test_literature.py -v
```
Expected: all pass. Kojima–Shindo is the hard one (degenerate); if the semismooth solver stalls on it from `x0 = 1`, that is a genuine solver-robustness bug — debug via `SolverOptions(verbose=True)`, do not weaken the test.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "test: literature problem library and cross-solver harness"
```

---

### Task 9: Shoven–Whalley-style CGE example

**Files:**
- Create: `examples/shoven_whalley.py`
- Test: `tests/test_cge_sw.py`

**Interfaces:**
- Produces: `examples/shoven_whalley.py` exposing `build_model(numeraire_value: float = 1.0) -> Model` and a `__main__` block that solves and prints the solution.

A 2-good, 2-factor, 2-consumer economy with Cobb–Douglas production and CES demand, formulated as a square MCP exactly per the spec's CGE conventions:

- Variables: `y` (2 activity levels, lb=0), `p` (2 goods prices, lb=0), `pf` (2 factor prices, lb=0), numeraire `pf[0]` fixed.
- zero profit ⟂ `y`: `unit_cost_j(pf) - p_j` where Cobb–Douglas unit cost `c_j = prod_f (pf_f / alpha_jf)^alpha_jf`.
- goods market ⟂ `p`: `y_i - sum_h x_hi(p, m_h)` with CES demand `x_hi = a_hi^sigma p_i^(-sigma) m_h / sum_k a_hk^sigma p_k^(1-sigma)`, income `m_h = sum_f pf_f e_hf`.
- factor market ⟂ `pf`: `E_f - sum_j (alpha_jf c_j / pf_f) y_j` (Shephard's lemma factor demand).

- [ ] **Step 1: Write the failing test**

`tests/test_cge_sw.py`:
```python
import numpy as np

from examples.shoven_whalley import build_model
from mcp_solver.semismooth import solve_semismooth


def _solve(numeraire=1.0):
    m = build_model(numeraire_value=numeraire)
    prob = m.build()
    res = solve_semismooth(prob)
    assert res.converged, res.status
    return m, prob, res


def test_equilibrium_conditions_hold():
    m, prob, res = _solve()
    f = prob.f_np(res.z)
    tol = 1e-6
    # all prices/activities strictly positive here -> every f component ~ 0
    assert np.all(res.z[2:] > 0.01), "prices should be strictly positive"
    assert np.abs(f).max() < tol


def test_walras_law():
    m, prob, res = _solve()
    # value of excess supplies at solution prices ~ 0 (goods + factors)
    sol = m.unpack(res.z)
    f = prob.f_np(res.z)
    prices = np.concatenate([sol["p"], sol["pf"]])
    excess = f[2:]                      # goods + factor market equations
    assert abs(prices @ excess) < 1e-6


def test_homogeneity_doubling_numeraire_doubles_prices():
    m1, _, r1 = _solve(1.0)
    m2, _, r2 = _solve(2.0)
    s1, s2 = m1.unpack(r1.z), m2.unpack(r2.z)
    np.testing.assert_allclose(s2["p"], 2 * s1["p"], rtol=1e-6)
    np.testing.assert_allclose(s2["pf"], 2 * s1["pf"], rtol=1e-6)
    np.testing.assert_allclose(s2["y"], s1["y"], rtol=1e-6)  # real side unchanged


def test_diagnose_warns_without_numeraire():
    from examples.shoven_whalley import build_model_no_numeraire
    m = build_model_no_numeraire()
    assert any("numeraire" in w for w in m.diagnose())
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_cge_sw.py -v
```
Expected: FAIL (ModuleNotFoundError: examples).

- [ ] **Step 3: Implement**

`examples/shoven_whalley.py`:
```python
"""Shoven-Whalley-style 2x2x2 general equilibrium model as an MCP.

Cobb-Douglas production, CES consumers, factor endowments. Correctness
is asserted through equilibrium conditions (market clearing, zero
profit, Walras, homogeneity), not published constants.
"""
import jax.numpy as jnp
import numpy as np

from mcp_solver.model import Model

# calibration
ALPHA = np.array([[0.6, 0.4],      # factor shares by activity (rows sum 1)
                  [0.3, 0.7]])
A_SHARE = np.array([[0.5, 0.5],    # consumer CES share params
                    [0.2, 0.8]])
SIGMA = np.array([1.5, 0.8])       # consumer elasticities
ENDOW = np.array([[3.0, 1.0],      # e[h, f]: endowment of factor f
                  [1.0, 4.0]])


def _unit_cost(pf):
    alpha = jnp.asarray(ALPHA)
    return jnp.prod((pf[None, :] / alpha) ** alpha, axis=1)


def _demand(p, m_h, a_h, sigma_h):
    num = a_h**sigma_h * p**(-sigma_h) * m_h
    den = jnp.sum(a_h**sigma_h * p**(1.0 - sigma_h))
    return num / den


def _build(fix_numeraire, numeraire_value=1.0):
    model = Model()
    model.add_variables("y", 2, lb=0.0, start=1.0)
    model.add_variables("p", 2, lb=0.0, start=1.0)
    model.add_variables("pf", 2, lb=0.0, start=1.0)

    def zero_profit(v):
        return _unit_cost(v["pf"]) - v["p"]

    def goods_market(v):
        m = jnp.asarray(ENDOW) @ v["pf"]
        demand = sum(
            _demand(v["p"], m[h], jnp.asarray(A_SHARE[h]), SIGMA[h])
            for h in range(2))
        return v["y"] - demand

    def factor_market(v):
        c = _unit_cost(v["pf"])
        a_fj = jnp.asarray(ALPHA).T * c[None, :] / v["pf"][:, None]
        return jnp.sum(jnp.asarray(ENDOW), axis=0) - a_fj @ v["y"]

    model.add_equations("zero_profit", zero_profit, complements="y")
    model.add_equations("goods_market", goods_market, complements="p")
    model.add_equations("factor_market", factor_market, complements="pf")
    if fix_numeraire:
        model.fix("pf", 0, numeraire_value)
    return model


def build_model(numeraire_value=1.0):
    return _build(True, numeraire_value)


def build_model_no_numeraire():
    return _build(False)


if __name__ == "__main__":
    from mcp_solver import SolverOptions
    from mcp_solver.semismooth import solve_semismooth

    m = build_model()
    res = solve_semismooth(m.build(), SolverOptions(verbose=True))
    print(res.table())
    for name, val in m.unpack(res.z).items():
        print(f"{name} = {val}")
```

Note on `factor_market`: `a_fj` has shape (factors, activities); check index orientation against ALPHA's (activities, factors) layout — the transpose is deliberate.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_cge_sw.py -v
```
Expected: 4 passed. If the solve diverges: run `examples/shoven_whalley.py` directly with verbose output; the usual culprit is a sign error making excess *demand* complementary to prices instead of excess *supply*.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: Shoven-Whalley 2x2x2 CGE example with equilibrium tests"
```

---

### Task 10: Scalable synthetic CGE generator

**Files:**
- Create: `examples/synthetic_cge.py`
- Test: `tests/test_cge_synthetic.py`

**Interfaces:**
- Produces: `make_exchange_economy(n_goods: int, n_households: int | None = None, sigma: float = 1.2, seed: int = 0) -> Model` — a pure-exchange CES economy (gross substitutes for `sigma >= 1`, hence unique equilibrium) with numeraire `p[0]` fixed at 1. Market equations are excess supply ⟂ prices.

- [ ] **Step 1: Write the failing test**

`tests/test_cge_synthetic.py`:
```python
import numpy as np
import pytest

from examples.synthetic_cge import make_exchange_economy
from mcp_solver import SolverOptions
from mcp_solver.semismooth import solve_semismooth


@pytest.mark.parametrize("n", [5, 20, 200])
def test_exchange_economy_solves_and_clears(n):
    m = make_exchange_economy(n, seed=42)
    prob = m.build()
    res = solve_semismooth(prob)
    assert res.converged, f"n={n}: {res.status}, residual={res.residual:.2e}"
    sol = m.unpack(res.z)
    assert np.all(sol["p"] > 0.0), "equilibrium prices must be positive"
    excess_supply = prob.f_np(res.z)
    assert np.abs(excess_supply).max() < 1e-6          # all markets clear
    assert abs(sol["p"] @ excess_supply) < 1e-8        # Walras


@pytest.mark.slow
def test_large_economy_2000():
    m = make_exchange_economy(2000, seed=7)
    prob = m.build(jac_coloring=False)   # income terms make coloring lose
    res = solve_semismooth(prob, SolverOptions(max_iter=200))
    assert res.converged
    assert np.abs(prob.f_np(res.z)).max() < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_cge_synthetic.py -m 'not slow' -v
```
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

`examples/synthetic_cge.py`:
```python
"""Scalable synthetic pure-exchange CES economies.

H households with CES preferences (common elasticity sigma >= 1 ->
gross substitutes -> unique equilibrium) and random endowments,
calibrated so p = 1 is *near* (not at) the equilibrium. MCP form:
excess supply(p) >= 0  ⟂  p >= 0, numeraire p[0] = 1.
"""
import jax.numpy as jnp
import numpy as np

from mcp_solver.model import Model


def make_exchange_economy(n_goods, n_households=None, sigma=1.2, seed=0):
    H = n_households or max(4, n_goods // 5)
    rng = np.random.default_rng(seed)
    endow = rng.uniform(0.5, 1.5, (H, n_goods))
    shares = rng.uniform(0.5, 1.5, (H, n_goods))
    shares /= shares.sum(axis=1, keepdims=True)
    e_j = jnp.asarray(endow)
    a_j = jnp.asarray(shares)
    total_supply = jnp.sum(e_j, axis=0)

    def excess_supply(v):
        p = v["p"]
        m = e_j @ p                                   # incomes (H,)
        weights = a_j**sigma * p[None, :] ** (1.0 - sigma)  # (H, n)
        denom = jnp.sum(weights, axis=1)              # price indices (H,)
        demand = jnp.sum((weights / p[None, :]) * (m / denom)[:, None],
                         axis=0)
        return total_supply - demand

    model = Model()
    model.add_variables("p", n_goods, lb=0.0, start=1.0)
    model.add_equations("markets", excess_supply, complements="p")
    model.fix("p", 0, 1.0)
    return model


if __name__ == "__main__":
    from mcp_solver import SolverOptions
    from mcp_solver.semismooth import solve_semismooth

    m = make_exchange_economy(200, seed=1)
    res = solve_semismooth(m.build(), SolverOptions(verbose=True))
    print(res.table())
```

(Derivation of `demand`: CES demand `x_hi = a_hi^sigma p_i^{-sigma} m_h / sum_k a_hk^sigma p_k^{1-sigma}`; the code's `weights/p * m/denom` is exactly this, vectorized over households.)

- [ ] **Step 4: Run tests including slow**

```bash
.venv/bin/python -m pytest tests/test_cge_synthetic.py -v         # fast sizes
.venv/bin/python -m pytest tests/test_cge_synthetic.py -m slow -v # n=2000
```
Expected: all pass. The n=2000 case is the stage-1 scale gate: it exercises chunked Jacobians (dense income rows) and the equilibrated Newton solve at 2000×2000. Runtime should be minutes, not hours; if it OOMs, `jac_chunk` is not being respected in `_batched_jvp`.

- [ ] **Step 5: Commit and close out stage 1**

```bash
.venv/bin/python -m pytest -m 'not slow' -q   # full suite green
git add -A && git commit -m "feat: scalable synthetic exchange-economy generator; stage 1 complete"
```

---

## Self-Review Notes (already applied)

- Spec coverage: problem/normal-map/scaling/model/options/result/semismooth all tasked; `diagnose()` covers the spec's diagnostics bullet; the spec's iteration-table requirement is `SolveResult.table()` (Task 1) + `verbose` prints (Task 7). Stage-2 spec sections (path/, pivot, NMS, pathsearch, cross-check with PATH) are deliberately deferred to the stage-2 plan; the cross-check *hook* (`SOLVERS` registry) ships now in Task 8.
- Type consistency: `solve_semismooth(problem, options)` signature consistent across Tasks 7–10; `fb_system` returns `(Phi, H)` consistently in Tasks 6–7; `ruiz` returns `(As, R, C)` with `As = diag(R) A diag(C)` in Tasks 2 and 7.
- Known judgment calls an implementer must NOT "fix": pinned projection JVP (test enforces 1.0 at bounds); LM via stacked lstsq (never `HᵀH`); tests verify residuals, not literature constants.
