# Stage 2: PATH Pivotal Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the PATH algorithm (Dirkse & Ferris 1993) — linearization, complementary-pivot path generation, backward pathsearch over stored breakpoints, and the non-monotone watchdog — as `solve_path`, cross-checked against the stage-1 semismooth solver on the full test suite.

**Architecture:** A `path/` package on top of the as-built stage-1 core. `linearize.py` builds the Ruiz-equilibrated linear MCP at the current iterate; `pivot.py` traces the piecewise-linear path from the current triple to the Newton point with an explicit-inverse basis (Sherman–Morrison updates, residual-monitored refactorization) and records `(t, x)` breakpoints; `pathsearch.py` walks stored breakpoints backward with segment Armijo; `nms.py` + `solver.py` run Algorithm PATH (p. 14): d-steps, m-steps, watchdog, rule-(15) reference values. Task 1 first clears the stage-1 carryover items the new code depends on.

**Tech Stack:** Python ≥3.10, JAX (model functions/Jacobians via stage-1 core), numpy (all pivot linear algebra), pytest. Venv at `.venv` (exists; `.venv/bin/python -m pytest`).

**References:** spec `docs/superpowers/specs/2026-07-16-mcp-solver-design.md` (as amended 2026-07-17: snapshot pathsearch), carryover notes `docs/superpowers/specs/2026-07-17-stage2-carryover-notes.md`, paper `cstr1179.pdf`.

## Global Constraints

- Runtime deps `jax` + `numpy` ONLY; pytest dev-only. Python ≥ 3.10.
- All pivot linear algebra in numpy; JAX arrays never leak from public APIs.
- Basis maintained as explicit inverse + Sherman–Morrison column replacement; per-pivot residual check `‖B·x_B − rhs‖∞ ≤ basis_residual_tol·scale`, refactorize on drift and unconditionally every `refactor_every` pivots.
- The linearized system is Ruiz-equilibrated before pivoting; merit values and all acceptance tests are computed in ORIGINAL units.
- Degeneracy: Harris-style two-pass ratio test with `pivot_tol`; no lexicographic rules.
- Pathsearch: stored `(t, x)` breakpoint snapshots (spec as amended); no reverse pivoting.
- Solver returns `SolveResult`; numerical failure NEVER raises (statuses: CONVERGED, RAY_TERMINATION, MAX_ITERATIONS, SINGULAR_BASIS, DOMAIN_ERROR, STALLED).
- Run tests via `.venv/bin/python -m pytest` from repo root; full non-slow suite green at the end of every task.
- Git: commit at the end of every task, plain message, NEVER any Co-Authored-By/AI-attribution trailer (overrides any harness default).
- Tests verify MCP residuals/complementarity, never unverified literature constants (in-plan derived constants allowed).

## Variable/naming conventions used throughout (memorize)

For problem size `n`, pivot variable ids are integers: `Z_j = j`, `W_j = n + j`, `V_j = 2n + j`, `T = 3n`. Index masks come from `fb_masks(lb, ub)`: a variable is FREE (`free` mask: only `Z_j`, permanently basic), FIXED (`fixed` mask: `z_j` pinned at the bound forever; `W_j` doubles as the *free slack* `s_j = w_j − v_j`, column `−e_j`, bounds `(−inf, inf)`, permanently basic), or BOUNDED (normal pivoting among `Z_j`/`W_j`/`V_j`). The scaled system is `Ms·z − w + v + rs·t = bs` with `bs = −qs + rs`, `lbs ≤ z ≤ ubs`, `w, v ≥ 0`, `t ∈ [0, 1]` (t's lower bound relaxed after it first enters). z-status per index: `BASIC=0, AT_LOWER=1, AT_UPPER=2`.

---

### Task 1: Carryover cleanups (merit helper, options pruning, linesearch Jacobian fix)

**Files:**
- Modify: `src/mcp_solver/normal_map.py` (append two functions)
- Modify: `src/mcp_solver/options.py` (delete two fields)
- Modify: `src/mcp_solver/semismooth.py` (split residual from Jacobian; restructure linesearch)
- Test: `tests/test_carryover.py` (new), existing suite must stay green

**Interfaces:**
- Consumes: existing `fb_masks`, `fb_system`, `MCPProblem.f_boxed/jac_boxed`.
- Produces: `normal_map.fB_np(f_np, x, lb, ub) -> np.ndarray` (normal map in numpy; `f_np` is a callable like `problem.f_np`); `normal_map.merit(f_np, x, lb, ub) -> float` (2-norm of `f_B`, `inf` when undefined); `semismooth.fb_residual(z, fval, lb, ub, masks) -> np.ndarray` (Phi only, no Jacobian needed). `SolverOptions` no longer has `jac_chunk`/`jac_coloring` (construction-time knobs live on `MCPProblem`).

- [ ] **Step 1: Write the failing tests**

`tests/test_carryover.py`:
```python
import jax.numpy as jnp
import numpy as np
import pytest

from mcp_solver import SolverOptions, Status
from mcp_solver.normal_map import fB_np, merit
from mcp_solver.problem import MCPProblem
from mcp_solver.semismooth import fb_masks, fb_residual, fb_system, solve_semismooth

INF = np.inf


def test_merit_and_fB_np():
    lb, ub = np.array([0.0]), np.array([2.0])
    f_np = lambda z: np.asarray(z) - 1.0
    # x = 1 is the solution: fB = 0
    np.testing.assert_allclose(fB_np(f_np, np.array([1.0]), lb, ub), [0.0])
    assert merit(f_np, np.array([1.0]), lb, ub) == 0.0
    # x = -1: z = 0, fB = f(0) + x - 0 = -1 - 1 = -2
    np.testing.assert_allclose(fB_np(f_np, np.array([-1.0]), lb, ub), [-2.0])
    assert merit(f_np, np.array([-1.0]), lb, ub) == 2.0
    # undefined f -> merit inf, no raise
    f_nan = lambda z: np.full_like(z, np.nan)
    assert merit(f_nan, np.array([1.0]), lb, ub) == np.inf


def test_options_construction_knobs_removed():
    with pytest.raises(TypeError):
        SolverOptions(jac_chunk=64)
    with pytest.raises(TypeError):
        SolverOptions(jac_coloring=False)


def test_fb_residual_matches_fb_system():
    rng = np.random.default_rng(2)
    lb = np.array([-INF, 0.0, -INF, 0.0, 3.0])
    ub = np.array([INF, INF, 2.0, 2.0, 3.0])
    masks = fb_masks(lb, ub)
    z = rng.uniform(0.1, 1.9, 5)
    fval = rng.standard_normal(5)
    J = rng.standard_normal((5, 5))
    Phi_only = fb_residual(z, fval, lb, ub, masks)
    Phi_sys, _ = fb_system(z, fval, J, lb, ub, masks)
    np.testing.assert_allclose(Phi_only, Phi_sys, rtol=1e-14)


def test_linesearch_does_not_recompute_jacobian_on_rejected_trials():
    # atan problem forces many backtracks (stage-1 suite's nasty scalar)
    p = MCPProblem(lambda z: jnp.arctan(z - 10.0), np.full(1, -INF),
                   np.array([INF]), np.array([0.0]))
    calls = {"jac": 0}
    orig = p.jac_boxed
    p.jac_boxed = lambda z: (calls.__setitem__("jac", calls["jac"] + 1), orig(z))[1]
    res = solve_semismooth(p)
    assert res.status is Status.CONVERGED
    # one Jacobian per accepted iterate (+1 initial, +small slack for the
    # H-finiteness re-check path); rejected trials must not cost Jacobians
    assert calls["jac"] <= len(res.iterations) + 3
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_carryover.py -v`
Expected: FAIL (ImportError: `fB_np`/`fb_residual`; `SolverOptions` accepts the kwargs).

- [ ] **Step 3: Implement**

Append to `src/mcp_solver/normal_map.py`:
```python
def fB_np(f_np, x, lb, ub):
    """Normal map f_B(x) = f(pi_B(x)) + x - pi_B(x), plain numpy."""
    z = np.clip(x, lb, ub)
    return f_np(z) + x - z


def merit(f_np, x, lb, ub):
    """Theta(x) = ||f_B(x)||_2; +inf when f_B is undefined (spec merit)."""
    val = fB_np(f_np, x, lb, ub)
    if not np.all(np.isfinite(val)):
        return np.inf
    return float(np.linalg.norm(val))
```

In `src/mcp_solver/options.py`, DELETE the two lines:
```python
    # jacobian extraction
    jac_chunk: int = 256
    jac_coloring: bool = True
```
(the comment line too; these are `MCPProblem` construction knobs, not solve-time options — carryover item 1 resolved by removal).

In `src/mcp_solver/semismooth.py`: refactor so the per-case Phi/alpha/beta assembly is shared, expose `fb_residual`, and restructure `solve_semismooth`'s linesearch to evaluate ONLY Phi at trial points, computing the Jacobian solely at accepted points (carryover item 2). Replace `fb_system` with:

```python
def _phi_terms(z, fval, lb, ub, masks):
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
    return Phi, alpha, beta


def fb_residual(z, fval, lb, ub, masks):
    """FB residual Phi only — no Jacobian required (cheap trial-point eval)."""
    return _phi_terms(z, fval, lb, ub, masks)[0]


def fb_system(z, fval, J, lb, ub, masks):
    """(Phi, H): FB residual and a generalized-Jacobian element.

    H = diag(alpha) + diag(beta) @ J, with fval/J the boxed f and Jacobian.
    """
    Phi, alpha, beta = _phi_terms(z, fval, lb, ub, masks)
    H = np.diag(alpha) + beta[:, None] * J
    return Phi, H
```

Replace the body of `solve_semismooth` with (same signature, same statuses; the
old inner `system()` disappears):

```python
def solve_semismooth(problem, options=None):
    opts = options or SolverOptions()
    lb, ub = problem.lb, problem.ub
    masks = fb_masks(lb, ub)
    z = np.clip(problem.x0, lb, ub)

    def eval_phi(zz):
        fval = problem.f_boxed(zz)
        if not np.all(np.isfinite(fval)):
            return None, None
        Phi = fb_residual(zz, fval, lb, ub, masks)
        if not np.all(np.isfinite(Phi)):
            return None, None
        return fval, Phi

    def eval_H(zz, fval):
        J = problem.jac_boxed(zz)
        if not np.all(np.isfinite(J)):
            return None
        return fb_system(zz, fval, J, lb, ub, masks)[1]

    fval, Phi = eval_phi(z)
    if Phi is None:
        return SolveResult(Status.DOMAIN_ERROR, z, np.zeros_like(z),
                           np.zeros_like(z), np.inf, [])
    H = eval_H(z, fval)
    if H is None:
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
            ft, Phit = eval_phi(zt)
            if Phit is not None:
                psit = 0.5 * float(Phit @ Phit)
                if psit <= ref + opts.armijo_c * alpha * gTd:
                    Ht = eval_H(zt, ft)      # Jacobian ONLY at accepted points
                    if Ht is not None:
                        z, fval, Phi, H = zt, ft, Phit, Ht
                        psi_hist.append(psit)
                        accepted = True
                        break
                    # H undefined here: treat like a rejected trial, keep
                    # backtracking toward the current (H-finite) iterate
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

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -m 'not slow' -q`
Expected: all pass (47 prior + 4 new = 51; the two existing sqrt-Jacobian regression tests in `tests/test_semismooth.py` must still pass — the initial-point path still returns DOMAIN_ERROR when H is non-finite at x0).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: merit helper, prune dead options, Jacobian-free trial evaluations"
```

---

### Task 2: Linearization with equilibration (`path/linearize.py`)

**Files:**
- Create: `src/mcp_solver/path/__init__.py` (empty), `src/mcp_solver/path/linearize.py`
- Test: `tests/test_linearize.py`

**Interfaces:**
- Consumes: `MCPProblem.f_np/jac`, `normal_map.decompose`, `scaling.ruiz`, `fb_masks`.
- Produces:
```python
@dataclass
class LinearMCP:
    n: int
    # original units
    M, q, r, lb, ub, z0, w0, v0: np.ndarray
    # Ruiz-scaled (Ms = diag(R) M diag(C); zs = z / C; ws = R*w; vs = R*v)
    Ms, qs, rs, lbs, ubs, z0s, w0s, v0s, R, C: np.ndarray
    free, fixed: np.ndarray            # bool masks
    def unscale_triple(self, zs, ws, vs) -> tuple  # (z, w, v) original units

def linearize(problem, x) -> LinearMCP | None      # None if f or J non-finite at pi_B(x)
```
Scaling identities (used by every later task): `Ms @ z0s + qs - w0s + v0s == rs` (the t=0 path equation) and `unscale_triple` inverts exactly: `z = C*zs, w = ws/R, v = vs/R`.

- [ ] **Step 1: Write the failing test**

`tests/test_linearize.py`:
```python
import jax.numpy as jnp
import numpy as np

from mcp_solver.normal_map import fB_np
from mcp_solver.path.linearize import linearize
from mcp_solver.problem import MCPProblem

INF = np.inf


def _problem():
    # nonlinear, badly scaled on purpose
    D = np.array([1e5, 1.0, 1e-3])
    f = lambda z: jnp.asarray(D) * (z**2 - 2.0)
    return MCPProblem(f, np.zeros(3), np.full(3, INF), np.full(3, 1.5))


def test_linearization_identities():
    p = _problem()
    x = np.array([1.2, -0.3, 0.7])      # -0.3 below bound: z=0, w=0.3
    lin = linearize(p, x)
    z = np.clip(x, p.lb, p.ub)
    # M, q reproduce f at the linearization point: M z + q = f(z)
    np.testing.assert_allclose(lin.M @ z + lin.q, p.f_np(z), rtol=1e-10)
    # r is the normal-map residual at x
    np.testing.assert_allclose(lin.r, fB_np(p.f_np, x, p.lb, p.ub), rtol=1e-10)
    # r equals the triple identity M z0 + q - w0 + v0
    np.testing.assert_allclose(lin.M @ lin.z0 + lin.q - lin.w0 + lin.v0,
                               lin.r, rtol=1e-8, atol=1e-10)
    # scaled t=0 path equation holds
    np.testing.assert_allclose(lin.Ms @ lin.z0s + lin.qs - lin.w0s + lin.v0s,
                               lin.rs, rtol=1e-8, atol=1e-10)
    # scaled matrix is equilibrated (max-norms near 1) despite 1e5..1e-3 rows
    assert np.abs(lin.Ms).max(axis=1).max() < 1.5
    assert np.abs(lin.Ms).max(axis=1).min() > 0.4
    # unscale inverts scale
    z2, w2, v2 = lin.unscale_triple(lin.z0s, lin.w0s, lin.v0s)
    np.testing.assert_allclose(z2, lin.z0, rtol=1e-12)
    np.testing.assert_allclose(w2, lin.w0, rtol=1e-12)
    np.testing.assert_allclose(v2, lin.v0, rtol=1e-12)
    # scaled bounds ordered
    assert np.all(lin.lbs <= lin.ubs)


def test_linearize_returns_none_on_undefined():
    p = MCPProblem(lambda z: jnp.log(z), np.zeros(1), np.array([INF]),
                   np.array([1.0]))
    # f(pi_B(x)) = log(0) = -inf at x <= 0
    assert linearize(p, np.array([-1.0])) is None


def test_masks_present():
    lb = np.array([-INF, 0.0, 2.0])
    ub = np.array([INF, INF, 2.0])
    p = MCPProblem(lambda z: z, lb, ub, np.array([0.0, 1.0, 2.0]))
    lin = linearize(p, p.x0.copy())
    assert lin.free.tolist() == [True, False, False]
    assert lin.fixed.tolist() == [False, False, True]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_linearize.py -v`
Expected: FAIL (ModuleNotFoundError: `mcp_solver.path`).

- [ ] **Step 3: Implement**

`src/mcp_solver/path/__init__.py`: empty file.

`src/mcp_solver/path/linearize.py`:
```python
"""Linearize the MCP at the current iterate (paper eq. 4) and equilibrate.

M = grad f(z_k), q = f(z_k) - M z_k, covering vector r = f_B(x_k).
Ruiz scaling: Ms = diag(R) M diag(C); variables transform as
zs = z / C, ws = R*w, vs = R*v; bounds as lbs = lb / C, ubs = ub / C
(C > 0 preserves order). Merit/acceptance always in ORIGINAL units.
"""
from dataclasses import dataclass

import numpy as np

from mcp_solver.normal_map import decompose
from mcp_solver.scaling import ruiz
from mcp_solver.semismooth import fb_masks


@dataclass
class LinearMCP:
    n: int
    M: np.ndarray
    q: np.ndarray
    r: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    z0: np.ndarray
    w0: np.ndarray
    v0: np.ndarray
    Ms: np.ndarray
    qs: np.ndarray
    rs: np.ndarray
    lbs: np.ndarray
    ubs: np.ndarray
    z0s: np.ndarray
    w0s: np.ndarray
    v0s: np.ndarray
    R: np.ndarray
    C: np.ndarray
    free: np.ndarray
    fixed: np.ndarray

    def unscale_triple(self, zs, ws, vs):
        return self.C * zs, ws / self.R, vs / self.R


def linearize(problem, x):
    lb, ub = problem.lb, problem.ub
    z, w, v = decompose(x, lb, ub)
    fz = problem.f_np(z)
    if not np.all(np.isfinite(fz)):
        return None
    M = problem.jac(z)
    if not np.all(np.isfinite(M)):
        return None
    q = fz - M @ z
    r = fz + x - z                       # f_B(x) = f(z) - w + v
    Ms, R, C = ruiz(M)
    masks = fb_masks(lb, ub)
    with np.errstate(invalid="ignore"):  # inf/C stays inf, sign preserved
        lbs = lb / C
        ubs = ub / C
    return LinearMCP(
        n=lb.size, M=M, q=q, r=r, lb=lb, ub=ub, z0=z, w0=w, v0=v,
        Ms=Ms, qs=R * q, rs=R * r, lbs=lbs, ubs=ubs,
        z0s=z / C, w0s=R * w, v0s=R * v, R=R, C=C,
        free=masks["free"], fixed=masks["fixed"])
```

- [ ] **Step 4: Run tests, commit**

Run: `.venv/bin/python -m pytest tests/test_linearize.py -v` — expected 3 passed;
then `.venv/bin/python -m pytest -m 'not slow' -q` — all green.
```bash
git add -A && git commit -m "feat: MCP linearization with Ruiz equilibration"
```

<!-- CHUNK-BOUNDARY-1 -->

---

### Task 3: Basis machinery (`path/pivot.py` part 1: `_Tableau`)

**Files:**
- Create: `src/mcp_solver/path/pivot.py` (this task: constants, `PathStatus`, `PathResult`, `_Tableau`; the path loop comes in Task 4)
- Test: `tests/test_tableau.py`

**Interfaces:**
- Consumes: `LinearMCP` (Task 2), `SolverOptions.basis_residual_tol/refactor_every`.
- Produces (Task 4 builds directly on ALL of this):
```python
BASIC, AT_LOWER, AT_UPPER = 0, 1, 2
class PathStatus(Enum): NEWTON_POINT, RAY_TERMINATION, PIVOT_LIMIT, SINGULAR_BASIS
@dataclass
class PathResult:
    status: PathStatus
    breakpoints: list      # [(t, x)] ORIGINAL units; first entry is the path start
    n_pivots: int
    used_slack_start: bool
    final_triple: tuple | None   # (z, w, v) original units at t = 1

class _Tableau:
    # attributes: lin, opts, n, T (= 3n), bs, z_stat, t_val, t_basic,
    #             t_relaxed, basis (int64[n]), B, Binv, xB, n_refactor
    def column(self, vid) -> np.ndarray
    def bounds_of(self, vid) -> tuple[float, float]
    def rhs_effective(self) -> np.ndarray
    def factorize(self) -> bool               # False = singular
    def recompute_xB(self) -> None
    def check_residual(self) -> bool
    def replace(self, pos, new_vid) -> bool   # Sherman-Morrison; False = singular
    def current_point(self) -> tuple          # (t, x, (z, w, v)) original units
    def current_scaled(self) -> tuple         # (t, zs, ws, vs) scaled units (debug)
```
Basis position invariant: position `p` holds the basic variable of index-family `p` (exactly one of `Z_p`/`W_p`/`V_p`), except one position may hold `T` once t enters.

- [ ] **Step 1: Write the failing test**

`tests/test_tableau.py`:
```python
import numpy as np
import pytest

from mcp_solver import SolverOptions
from mcp_solver.path.linearize import LinearMCP
from mcp_solver.path.pivot import (AT_LOWER, AT_UPPER, BASIC, _Tableau)

INF = np.inf


def _lin(M, q, lb, ub, z0, w0, v0):
    """Build an UNSCALED LinearMCP directly (R = C = 1) for unit tests."""
    n = q.size
    r = M @ z0 + q - w0 + v0
    ones = np.ones(n)
    free = ~np.isfinite(lb) & ~np.isfinite(ub)
    fixed = lb == ub
    return LinearMCP(n=n, M=M, q=q, r=r, lb=lb, ub=ub, z0=z0, w0=w0, v0=v0,
                     Ms=M.copy(), qs=q.copy(), rs=r.copy(), lbs=lb.copy(),
                     ubs=ub.copy(), z0s=z0.copy(), w0s=w0.copy(),
                     v0s=v0.copy(), R=ones, C=ones, free=free, fixed=fixed)


def _mixed_lin(seed=0):
    rng = np.random.default_rng(seed)
    n = 6
    A = rng.standard_normal((n, n))
    M = A @ A.T + np.eye(n)
    lb = np.array([-INF, 0.0, 0.0, 0.0, 1.0, -INF])
    ub = np.array([INF, INF, 2.0, INF, 1.0, INF])
    z0 = np.array([0.3, 0.0, 2.0, 0.7, 1.0, -0.2])   # free, at-l, at-u, int, fixed, free
    w0 = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0])
    q = rng.standard_normal(n)
    return _lin(M, q, lb, ub, z0, w0, v0)


def _init_from_triple(tab):
    """Mirror of the initial-basis rule (Task 4 wires this into the loop)."""
    lin, n = tab.lin, tab.n
    for j in range(n):
        if lin.free[j]:
            tab.basis[j], tab.z_stat[j] = j, BASIC
        elif lin.fixed[j]:
            tab.basis[j], tab.z_stat[j] = n + j, AT_LOWER
        elif lin.w0s[j] > 0 or (np.isfinite(lin.lbs[j])
                                and abs(lin.z0s[j] - lin.lbs[j]) <= 1e-12):
            tab.basis[j], tab.z_stat[j] = n + j, AT_LOWER
        elif lin.v0s[j] > 0 or (np.isfinite(lin.ubs[j])
                                and abs(lin.z0s[j] - lin.ubs[j]) <= 1e-12):
            tab.basis[j], tab.z_stat[j] = 2 * n + j, AT_UPPER
        else:
            tab.basis[j], tab.z_stat[j] = j, BASIC
    return tab.factorize()


def test_initial_basis_reproduces_start_triple():
    lin = _mixed_lin()
    tab = _Tableau(lin, SolverOptions())
    assert _init_from_triple(tab)
    t, x, (z, w, v) = tab.current_point()
    assert t == 0.0
    np.testing.assert_allclose(z, lin.z0, atol=1e-9)
    np.testing.assert_allclose(w, lin.w0, atol=1e-9)
    np.testing.assert_allclose(v, lin.v0, atol=1e-9)
    np.testing.assert_allclose(x, lin.z0 - lin.w0 + lin.v0, atol=1e-9)
    assert tab.check_residual()


def test_replace_matches_fresh_factorization():
    lin = _mixed_lin(3)
    tab = _Tableau(lin, SolverOptions())
    assert _init_from_triple(tab)
    n = tab.n
    # pivot W_1 (basic, pos 1) out for Z_1
    assert tab.replace(1, 1)
    tab.z_stat[1] = BASIC
    tab.recompute_xB()
    Binv_updated = tab.Binv.copy()
    xB_updated = tab.xB.copy()
    assert tab.factorize()                    # fresh inverse from scratch
    np.testing.assert_allclose(Binv_updated, tab.Binv, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(xB_updated, tab.xB, rtol=1e-8, atol=1e-10)


def test_residual_monitor_survives_many_replacements():
    lin = _mixed_lin(5)
    tab = _Tableau(lin, SolverOptions())
    assert _init_from_triple(tab)
    rng = np.random.default_rng(0)
    # random walk of legal basis swaps within families 1..3 (bounded indices)
    for _ in range(60):
        j = int(rng.integers(1, 4))
        cur = tab.basis[j]
        alts = [j, tab.n + j] + ([2 * tab.n + j] if np.isfinite(lin.ubs[j]) else [])
        alts = [a for a in alts if a != cur]
        new = int(rng.choice(alts))
        if not tab.replace(j, new):
            assert tab.factorize()
            continue
        tab.z_stat[j] = (BASIC if new < tab.n
                         else (AT_LOWER if new < 2 * tab.n else AT_UPPER))
        # nonbasic z must sit at a finite bound; keep status consistent
        if new >= tab.n and not np.isfinite(lin.lbs[j]):
            tab.z_stat[j] = AT_UPPER
        if not tab.check_residual():
            assert tab.factorize()
        tab.recompute_xB()
        assert tab.check_residual()


def test_bounds_of_and_columns():
    lin = _mixed_lin()
    tab = _Tableau(lin, SolverOptions())
    n = tab.n
    np.testing.assert_allclose(tab.column(0), lin.Ms[:, 0])       # Z col
    np.testing.assert_allclose(tab.column(n + 2), -np.eye(n)[2])  # W col
    np.testing.assert_allclose(tab.column(2 * n + 2), np.eye(n)[2])
    np.testing.assert_allclose(tab.column(tab.T), lin.rs)
    assert tab.bounds_of(n + 4) == (-INF, INF)     # fixed j=4: free slack
    assert tab.bounds_of(n + 1) == (0.0, INF)      # ordinary w
    assert tab.bounds_of(tab.T) == (0.0, 1.0)
    tab.t_relaxed = True
    assert tab.bounds_of(tab.T) == (-INF, 1.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_tableau.py -v`
Expected: FAIL (ModuleNotFoundError / ImportError).

- [ ] **Step 3: Implement**

`src/mcp_solver/path/pivot.py`:
```python
"""Pivotal path generation for the linearized MCP (paper section 2.2, eq. 11).

Scaled system:  Ms z - w + v + rs t = bs,   bs = -qs + rs,
with lbs <= z <= ubs, w >= 0, v >= 0, t in [0, 1] (t's lower bound is
relaxed once t has entered the basis, per the paper).

Variable ids: Z_j = j, W_j = n + j, V_j = 2n + j, T = 3n. For FIXED j,
W_j doubles as the free slack s_j = w_j - v_j (column -e_j, bounds
(-inf, inf), permanently basic). FREE j has only Z_j, permanently basic.
"""
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

BASIC, AT_LOWER, AT_UPPER = 0, 1, 2


class PathStatus(Enum):
    NEWTON_POINT = auto()
    RAY_TERMINATION = auto()
    PIVOT_LIMIT = auto()
    SINGULAR_BASIS = auto()


@dataclass
class PathResult:
    status: PathStatus
    breakpoints: list
    n_pivots: int
    used_slack_start: bool
    final_triple: tuple | None


class _Tableau:
    """Basis + explicit inverse for the rectangular pivoting scheme."""

    def __init__(self, lin, opts):
        self.lin = lin
        self.opts = opts
        n = lin.n
        self.n = n
        self.T = 3 * n
        self.bs = -lin.qs + lin.rs
        self.z_stat = np.full(n, BASIC, dtype=np.int8)
        self.t_val = 0.0
        self.t_basic = False
        self.t_relaxed = False
        self.basis = np.empty(n, dtype=np.int64)
        self.B = np.empty((n, n))
        self.Binv = None
        self.xB = np.zeros(n)
        self.n_refactor = 0

    # -- columns and bounds -------------------------------------------
    def column(self, vid):
        n, lin = self.n, self.lin
        if vid == self.T:
            return self.lin.rs.copy()
        j, k = vid % n, vid // n
        if k == 0:
            return lin.Ms[:, j].copy()
        col = np.zeros(n)
        col[j] = -1.0 if k == 1 else 1.0
        return col

    def bounds_of(self, vid):
        n, lin = self.n, self.lin
        if vid == self.T:
            return (-np.inf if self.t_relaxed else 0.0), 1.0
        j, k = vid % n, vid // n
        if k == 0:
            return lin.lbs[j], lin.ubs[j]
        if k == 1 and lin.fixed[j]:
            return -np.inf, np.inf          # free slack for fixed variable
        return 0.0, np.inf

    # -- assembly -------------------------------------------------------
    def rhs_effective(self):
        lin = self.lin
        rhs = self.bs.copy()
        at_l = self.z_stat == AT_LOWER
        at_u = self.z_stat == AT_UPPER
        if at_l.any():
            rhs -= lin.Ms[:, at_l] @ lin.lbs[at_l]
        if at_u.any():
            rhs -= lin.Ms[:, at_u] @ lin.ubs[at_u]
        if not self.t_basic and self.t_val != 0.0:
            rhs -= self.t_val * lin.rs
        return rhs

    def factorize(self):
        for p, vid in enumerate(self.basis):
            self.B[:, p] = self.column(vid)
        try:
            self.Binv = np.linalg.inv(self.B)
        except np.linalg.LinAlgError:
            return False
        if not np.all(np.isfinite(self.Binv)):
            return False
        self.n_refactor += 1
        self.recompute_xB()
        return True

    def recompute_xB(self):
        self.xB = self.Binv @ self.rhs_effective()

    def check_residual(self):
        rhs = self.rhs_effective()
        res = self.B @ self.xB - rhs
        scale = 1.0 + np.abs(rhs).max()
        return np.abs(res).max() <= self.opts.basis_residual_tol * scale

    def replace(self, pos, new_vid):
        """Sherman-Morrison column replacement; refactor when ill-pivoted."""
        a = self.column(new_vid)
        u = self.Binv @ a
        if not np.all(np.isfinite(u)) or abs(u[pos]) < 1e-11:
            self.basis[pos] = new_vid
            return self.factorize()
        e = np.zeros(self.n)
        e[pos] = 1.0
        self.Binv -= np.outer(u - e, self.Binv[pos]) / u[pos]
        self.B[:, pos] = a
        self.basis[pos] = new_vid
        return True

    # -- point extraction ------------------------------------------------
    def current_scaled(self):
        n, lin = self.n, self.lin
        z = np.where(self.z_stat == AT_LOWER, lin.lbs,
                     np.where(self.z_stat == AT_UPPER, lin.ubs, 0.0))
        w = np.zeros(n)
        v = np.zeros(n)
        t = self.t_val
        for p in range(n):
            vid = self.basis[p]
            val = self.xB[p]
            if vid == self.T:
                t = val
                continue
            j, k = vid % n, vid // n
            if k == 0:
                z[j] = val
            elif k == 1:
                if lin.fixed[j]:               # free slack s = w - v
                    w[j] = max(val, 0.0)
                    v[j] = max(-val, 0.0)
                else:
                    w[j] = val
            else:
                v[j] = val
        return t, z, w, v

    def current_point(self):
        t, zs, ws, vs = self.current_scaled()
        z, w, v = self.lin.unscale_triple(zs, ws, vs)
        return t, z - w + v, (z, w, v)
```

- [ ] **Step 4: Run tests, commit**

Run: `.venv/bin/python -m pytest tests/test_tableau.py -v` — 4 passed;
`.venv/bin/python -m pytest -m 'not slow' -q` — all green.
```bash
git add -A && git commit -m "feat: pivot tableau with explicit inverse and residual monitor"
```

---

### Task 4: Path generation (`path/pivot.py` part 2: ratio test, pivot rules, loop)

**Files:**
- Modify: `src/mcp_solver/path/pivot.py` (append)
- Test: `tests/test_path_generation.py`

**Interfaces:**
- Consumes: `_Tableau` and everything from Task 3, `LinearMCP`.
- Produces: `generate_path(lin: LinearMCP, opts: SolverOptions, debug: bool = False) -> PathResult`. `debug=True` asserts the path invariants (eq. 10 + complementarity) at every breakpoint — tests use it; the solver calls with default False.

Algorithm recap (implementer: this is the whole of paper §2.2 — read the docstrings in Task 3's file first):
- Initial basis from the start triple; on singularity (or `opts.lemke_start`), the all-slack Lemke start: bounded z to its nearest bound, slack basic on the matching side, covering vector REDEFINED as the residual of that start point.
- `t` enters first (from 0, increasing). Ratio test over basic variables (Harris two-pass with `pivot_tol`), plus the entering variable's own opposite bound (bound flip for z; t reaching 1 = Newton point).
- Pivot rules on the leaving variable: `W_j`→enter `Z_j` at lower (+1); `V_j`→enter `Z_j` at upper (−1); `Z_j` at lower→enter `W_j`; `Z_j` at upper→enter `V_j`; `T` at 1→Newton point. After t first enters, `t_relaxed = True` (lower bound dropped).
- Every pivot appends a `(t, x)` breakpoint (original units). No blocking variable and infinite self-range → RAY_TERMINATION. `opts.max_pivots` → PIVOT_LIMIT.

- [ ] **Step 1: Write the failing test**

`tests/test_path_generation.py`:
```python
import numpy as np
import pytest

from mcp_solver import SolverOptions
from mcp_solver.path.linearize import LinearMCP
from mcp_solver.path.pivot import PathStatus, generate_path

INF = np.inf


def _lin(M, q, lb, ub, z0, w0, v0):
    n = q.size
    r = M @ z0 + q - w0 + v0
    ones = np.ones(n)
    return LinearMCP(n=n, M=M, q=q, r=r, lb=lb, ub=ub, z0=z0, w0=w0, v0=v0,
                     Ms=M.copy(), qs=q.copy(), rs=r.copy(), lbs=lb.copy(),
                     ubs=ub.copy(), z0s=z0.copy(), w0s=w0.copy(),
                     v0s=v0.copy(), R=ones, C=ones,
                     free=~np.isfinite(lb) & ~np.isfinite(ub), fixed=lb == ub)


def _check_lcp_solution(lin, triple, tol=1e-7):
    z, w, v = triple
    res = lin.M @ z + lin.q - w + v
    assert np.abs(res).max() < tol
    assert np.all(z >= lin.lb - tol) and np.all(z <= lin.ub + tol)
    assert np.all(w >= -tol) and np.all(v >= -tol)
    fixed = lin.lb == lin.ub
    assert np.all(w[(z > lin.lb + 1e-6) & ~fixed] < tol)
    assert np.all(v[(z < lin.ub - 1e-6) & ~fixed] < tol)


def _random_monotone(n, seed, frac_active=0.4):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) / np.sqrt(n)
    M = A @ A.T + np.eye(n)
    q = rng.standard_normal(n) * 2.0
    lb, ub = np.zeros(n), np.full(n, INF)
    z0 = np.abs(rng.standard_normal(n))
    z0[rng.random(n) < frac_active] = 0.0
    fz = M @ z0 + q
    w0 = np.where(z0 == 0.0, np.maximum(fz, 0.0), 0.0)
    v0 = np.zeros(n)
    return _lin(M, q, lb, ub, z0, w0, v0)


@pytest.mark.parametrize("seed", range(6))
def test_random_monotone_lcps_reach_newton_point(seed):
    lin = _random_monotone(12, seed)
    res = generate_path(lin, SolverOptions(), debug=True)
    assert res.status is PathStatus.NEWTON_POINT, res.status
    _check_lcp_solution(lin, res.final_triple)
    # breakpoints: first is the start, last is the Newton point (t = 1)
    t0, x0 = res.breakpoints[0]
    tN, xN = res.breakpoints[-1]
    assert t0 == 0.0 and abs(tN - 1.0) < 1e-9
    z, w, v = res.final_triple
    np.testing.assert_allclose(xN, z - w + v, atol=1e-8)


def test_box_bounds_and_fixed_and_free():
    # mixed structure: solution known by construction
    M = np.eye(4)
    lb = np.array([0.0, 0.0, -INF, 3.0])
    ub = np.array([2.0, 2.0, INF, 3.0])
    # f(z) = z - target: z* = clip(target), free solves exactly, fixed pinned
    target = np.array([5.0, -1.0, 0.7, 9.0])
    q = -target
    z0 = np.array([1.0, 1.0, 0.0, 3.0])
    w0 = np.zeros(4)
    v0 = np.zeros(4)
    lin = _lin(M, q, lb, ub, z0, w0, v0)
    res = generate_path(lin, SolverOptions(), debug=True)
    assert res.status is PathStatus.NEWTON_POINT
    z, w, v = res.final_triple
    np.testing.assert_allclose(z, [2.0, 0.0, 0.7, 3.0], atol=1e-8)
    assert v[0] > 0.5 and w[1] > 0.5          # active-bound multipliers


def test_murty_family_terminates():
    for n in (4, 8):
        M = np.eye(n) + 2.0 * np.tril(np.ones((n, n)), -1)
        q = -np.ones(n)
        lb, ub = np.zeros(n), np.full(n, INF)
        z0 = np.zeros(n)
        w0 = np.maximum(q, 0.0)               # = 0: degenerate start
        lin = _lin(M, q, lb, ub, z0, w0, np.zeros(n))
        res = generate_path(lin, SolverOptions(max_pivots=5000), debug=True)
        assert res.status is PathStatus.NEWTON_POINT
        z, w, v = res.final_triple
        expected = np.zeros(n)
        expected[0] = 1.0                     # derived in plan: z* = e_1
        np.testing.assert_allclose(z, expected, atol=1e-8)


def test_infeasible_lcp_rays():
    # f(z) = -1 forever, z >= 0: no solution; Lemke must hit a ray
    n = 2
    lin = _lin(np.zeros((n, n)), -np.ones(n), np.zeros(n), np.full(n, INF),
               np.zeros(n), np.zeros(n), np.zeros(n))
    res = generate_path(lin, SolverOptions(), debug=True)
    assert res.status is PathStatus.RAY_TERMINATION


def test_lemke_start_option_also_solves():
    lin = _random_monotone(10, 42)
    res = generate_path(lin, SolverOptions(lemke_start=True), debug=True)
    assert res.status is PathStatus.NEWTON_POINT
    assert res.used_slack_start
    _check_lcp_solution(lin, res.final_triple)


def test_scaled_problem_solves_and_reports_original_units():
    # badly scaled diagonal problem through the real linearize()
    import jax.numpy as jnp
    from mcp_solver.path.linearize import linearize
    from mcp_solver.problem import MCPProblem
    D = np.array([1e6, 1.0, 1e-4])
    p = MCPProblem(lambda z: jnp.asarray(D) * (z - 2.0), np.zeros(3),
                   np.full(3, INF), np.ones(3))
    lin = linearize(p, np.ones(3))
    res = generate_path(lin, SolverOptions(), debug=True)
    assert res.status is PathStatus.NEWTON_POINT
    z, w, v = res.final_triple
    np.testing.assert_allclose(z, np.full(3, 2.0), rtol=1e-8)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_path_generation.py -v`
Expected: FAIL (ImportError: `generate_path`).

- [ ] **Step 3: Implement**

Append to `src/mcp_solver/path/pivot.py`:
```python
def _initial_basis_from_triple(tab):
    lin, n = tab.lin, tab.n
    for j in range(n):
        if lin.free[j]:
            tab.basis[j], tab.z_stat[j] = j, BASIC
        elif lin.fixed[j]:
            tab.basis[j], tab.z_stat[j] = n + j, AT_LOWER
        elif lin.w0s[j] > 0 or (np.isfinite(lin.lbs[j])
                                and abs(lin.z0s[j] - lin.lbs[j])
                                <= 1e-12 * (1.0 + abs(lin.lbs[j]))):
            tab.basis[j], tab.z_stat[j] = n + j, AT_LOWER
        elif lin.v0s[j] > 0 or (np.isfinite(lin.ubs[j])
                                and abs(lin.z0s[j] - lin.ubs[j])
                                <= 1e-12 * (1.0 + abs(lin.ubs[j]))):
            tab.basis[j], tab.z_stat[j] = 2 * n + j, AT_UPPER
        else:
            tab.basis[j], tab.z_stat[j] = j, BASIC
    return tab.factorize()


def _slack_start(tab):
    """All-slack (Lemke) start: bounded z to nearest bound, slack basic on
    the matching side, covering vector redefined as this point's residual."""
    lin, n = tab.lin, tab.n
    z_bar = np.where(np.isfinite(lin.z0s), lin.z0s, 0.0).copy()
    for j in range(n):
        if lin.free[j]:
            tab.basis[j], tab.z_stat[j] = j, BASIC
        elif lin.fixed[j]:
            tab.basis[j], tab.z_stat[j] = n + j, AT_LOWER
            z_bar[j] = lin.lbs[j]
        else:
            l, u, z = lin.lbs[j], lin.ubs[j], lin.z0s[j]
            go_upper = np.isfinite(u) and (not np.isfinite(l)
                                           or (u - z) < (z - l))
            if go_upper:
                tab.basis[j], tab.z_stat[j] = 2 * n + j, AT_UPPER
                z_bar[j] = u
            else:
                tab.basis[j], tab.z_stat[j] = n + j, AT_LOWER
                z_bar[j] = l
    rho = lin.Ms @ z_bar + lin.qs
    w_bar = np.zeros(n)
    v_bar = np.zeros(n)
    rs_new = rho.copy()
    for j in range(n):
        if lin.fixed[j]:
            rs_new[j] = 0.0                       # slack absorbs the row
        elif tab.z_stat[j] == AT_LOWER and not lin.free[j]:
            w_bar[j] = max(rho[j], 0.0)
            rs_new[j] = min(rho[j], 0.0)
        elif tab.z_stat[j] == AT_UPPER:
            v_bar[j] = max(-rho[j], 0.0)
            rs_new[j] = max(rho[j], 0.0)
    # free rows keep rs_new = rho (their z is basic; t column drives them)
    lin.rs = rs_new
    lin.r = rs_new / lin.R
    tab.bs = -lin.qs + lin.rs
    return tab.factorize()


def _ratio_test(tab, delta, theta_self, opts):
    """Harris two-pass. Returns (blocking, theta): blocking is a basis
    position, the string "self", or None (ray)."""
    tol = opts.pivot_tol
    n = tab.n
    lo = np.empty(n)
    hi = np.empty(n)
    for p in range(n):
        lo[p], hi[p] = tab.bounds_of(tab.basis[p])
    xB = tab.xB
    theta_i = np.full(n, np.inf)
    dec = delta > tol
    theta_i[dec] = (xB[dec] - lo[dec] + tol) / delta[dec]
    inc = delta < -tol
    theta_i[inc] = (xB[inc] - hi[inc] - tol) / delta[inc]
    theta_min = theta_i.min()
    if min(theta_min, theta_self) == np.inf:
        return None, np.inf
    if theta_self <= theta_min:
        return "self", max(theta_self, 0.0)
    cand = theta_i <= theta_min + tol
    p = int(np.argmax(np.where(cand, np.abs(delta), -1.0)))
    if delta[p] > 0:
        theta = (xB[p] - lo[p]) / delta[p]
    else:
        theta = (xB[p] - hi[p]) / delta[p]
    return p, max(theta, 0.0)


def _debug_invariants(tab, t):
    lin = tab.lin
    _, zs, ws, vs = tab.current_scaled()
    lhs = lin.Ms @ zs + lin.qs - ws + vs - (1.0 - t) * lin.rs
    scale = 1.0 + np.abs(lin.rs).max() + np.abs(lin.qs).max()
    assert np.abs(lhs).max() < 1e-6 * scale, "path equation (10) violated"
    bounded = ~lin.free & ~lin.fixed
    tol = 1e-7 * scale
    at_l = np.abs(zs - lin.lbs) <= 1e-9 * (1.0 + np.abs(lin.lbs))
    at_u = np.abs(zs - lin.ubs) <= 1e-9 * (1.0 + np.abs(lin.ubs))
    assert np.all(ws[bounded & ~at_l] <= tol), "w > 0 off the lower bound"
    assert np.all(vs[bounded & ~at_u] <= tol), "v > 0 off the upper bound"


def generate_path(lin, opts, debug=False):
    tab = _Tableau(lin, opts)
    used_slack = False
    if opts.lemke_start or not _initial_basis_from_triple(tab):
        if not _slack_start(tab):
            return PathResult(PathStatus.SINGULAR_BASIS, [], 0, True, None)
        used_slack = True
    bps = []
    t, x, triple = tab.current_point()
    bps.append((t, x))
    if debug:
        _debug_invariants(tab, t)
    entering, direction, enter_from = tab.T, 1.0, 0.0
    n = tab.n
    for pivot in range(opts.max_pivots):
        a = tab.column(entering)
        delta = direction * (tab.Binv @ a)
        lo_e, hi_e = tab.bounds_of(entering)
        theta_self = (hi_e - enter_from) if direction > 0 else (enter_from - lo_e)
        blocking, theta = _ratio_test(tab, delta, theta_self, opts)
        if blocking is None:
            return PathResult(PathStatus.RAY_TERMINATION, bps, pivot,
                              used_slack, None)
        if blocking == "self":
            if entering == tab.T:              # t ran straight to 1
                tab.t_val, tab.t_basic = 1.0, False
                tab.recompute_xB()
                t, x, triple = tab.current_point()
                bps.append((1.0, x))
                if debug:
                    _debug_invariants(tab, 1.0)
                return PathResult(PathStatus.NEWTON_POINT, bps, pivot + 1,
                                  used_slack, triple)
            j = entering                       # bound flip of entering Z_j
            reached_upper = direction > 0
            tab.z_stat[j] = AT_UPPER if reached_upper else AT_LOWER
            tab.recompute_xB()
            t, x, triple = tab.current_point()
            bps.append((t, x))
            if debug:
                _debug_invariants(tab, t)
            if reached_upper:
                entering, direction, enter_from = 2 * n + j, 1.0, 0.0
            else:
                entering, direction, enter_from = n + j, 1.0, 0.0
            continue
        p = int(blocking)
        leaving = int(tab.basis[p])
        went_upper = delta[p] < 0
        # book-keep the leaving variable's new nonbasic state
        if leaving == tab.T:
            tab.t_basic = False
            tab.t_val = 1.0                    # only exit is at the top
        elif leaving < n:
            tab.z_stat[leaving] = AT_UPPER if went_upper else AT_LOWER
        # entering variable's new state
        if entering == tab.T:
            tab.t_basic, tab.t_relaxed = True, True
        elif entering < n:
            tab.z_stat[entering] = BASIC
        if not tab.replace(p, entering):
            return PathResult(PathStatus.SINGULAR_BASIS, bps, pivot + 1,
                              used_slack, None)
        if (pivot + 1) % opts.refactor_every == 0 or not tab.check_residual():
            if not tab.factorize():
                return PathResult(PathStatus.SINGULAR_BASIS, bps, pivot + 1,
                                  used_slack, None)
        else:
            tab.recompute_xB()
        t, x, triple = tab.current_point()
        bps.append((t, x))
        if debug:
            _debug_invariants(tab, t)
        if leaving == tab.T:
            return PathResult(PathStatus.NEWTON_POINT, bps, pivot + 1,
                              used_slack, triple)
        # complementarity pivot rules choose the next entering variable
        j, kind = leaving % n, leaving // n
        if kind == 1:                          # w_j left -> z_j from lower
            entering, direction, enter_from = j, 1.0, tab.lin.lbs[j]
        elif kind == 2:                        # v_j left -> z_j from upper
            entering, direction, enter_from = j, -1.0, tab.lin.ubs[j]
        elif went_upper:                       # z_j left at upper -> v_j
            entering, direction, enter_from = 2 * n + j, 1.0, 0.0
        else:                                  # z_j left at lower -> w_j
            entering, direction, enter_from = n + j, 1.0, 0.0
    return PathResult(PathStatus.PIVOT_LIMIT, bps, opts.max_pivots,
                      used_slack, None)
```

Implementation notes for the engineer (read before debugging a failure):
- `recompute_xB` after every pivot makes ALL basic values (including the just-entered variable's) come out of one solve — never track the entering value by hand.
- After a bound flip there is NO leaving variable and NO basis change; the flip only moves the nonbasic contribution to the other bound.
- `theta_self` is infinite for entering W/V (their far bound is +inf); only Z can bound-flip and only T can self-terminate at 1.
- If `test_murty_family_terminates` cycles at the degenerate start (`w0 = 0`), check that `_ratio_test` clamps `theta` at 0 and that ties break toward the largest `|delta[p]|` (the Harris pass-2 rule) — that combination is what breaks Murty-style degeneracy in practice.

- [ ] **Step 4: Run tests, commit**

Run: `.venv/bin/python -m pytest tests/test_path_generation.py -v` — 11 passed
(6 seeds + 5 others); `.venv/bin/python -m pytest -m 'not slow' -q` — all green.
```bash
git add -A && git commit -m "feat: complementary-pivot path generation with Lemke fallback"
```

<!-- CHUNK-BOUNDARY-2 -->

---

### Task 5: Backward pathsearch (`path/pathsearch.py`)

**Files:**
- Create: `src/mcp_solver/path/pathsearch.py`
- Test: `tests/test_pathsearch.py`

**Interfaces:**
- Consumes: nothing project-specific — pure function over breakpoints.
- Produces: `pathsearch(merit_fn, breakpoints, reference, sigma, shrink=0.5, max_halvings=25) -> tuple[np.ndarray, float, float] | None` returning `(x, t, merit)` of the accepted point, or `None` if no point on the path satisfies the non-monotone descent condition (NmD): `merit(x) <= (1 - sigma * clamp(t, 0, 1)) * reference`. `merit_fn(x) -> float` must return `inf` for undefined points (Task 1's `normal_map.merit` partial-applied does this).

- [ ] **Step 1: Write the failing test**

`tests/test_pathsearch.py`:
```python
import numpy as np

from mcp_solver.path.pathsearch import pathsearch


def _bp(ts, xs):
    return [(t, np.array([x])) for t, x in zip(ts, xs)]


def test_accepts_endpoint_when_good():
    # merit = |x|; endpoint x=0 is perfect
    bps = _bp([0.0, 0.5, 1.0], [4.0, 2.0, 0.0])
    got = pathsearch(lambda x: abs(float(x[0])), bps, reference=4.0, sigma=0.01)
    assert got is not None
    x, t, m = got
    assert t == 1.0 and m == 0.0


def test_walks_back_when_endpoint_bad():
    # merit spikes at the endpoint; the middle breakpoint is acceptable
    def merit(x):
        return abs(float(x[0]))
    bps = _bp([0.0, 0.6, 1.0], [4.0, 1.0, 50.0])
    got = pathsearch(merit, bps, reference=4.0, sigma=0.01)
    assert got is not None
    x, t, m = got
    assert m <= (1 - 0.01 * t) * 4.0
    assert m <= 1.0 + 1e-12          # found the good middle region


def test_segment_armijo_finds_interior_point():
    # endpoint region undefined (inf merit); an interior point of the last
    # segment must be located by halving from the far end
    def merit(x):
        v = float(x[0])
        if v > 3.5:
            return np.inf            # undefined region near the endpoint
        return abs(v - 2.0) + 1.0    # best merit 1.0 at v = 2
    bps = _bp([0.0, 1.0], [0.0, 4.0])
    got = pathsearch(merit, bps, reference=2.0, sigma=0.01)
    assert got is not None
    x, t, m = got
    assert np.isfinite(m) and m <= (1 - 0.01 * min(max(t, 0), 1)) * 2.0
    assert float(x[0]) <= 3.5


def test_returns_none_when_nothing_acceptable():
    bps = _bp([0.0, 0.5, 1.0], [5.0, 6.0, 7.0])
    got = pathsearch(lambda x: abs(float(x[0])), bps, reference=1.0, sigma=0.01)
    assert got is None


def test_negative_t_clamped_in_acceptance_factor():
    # t < 0 must not INFLATE the acceptance threshold above reference
    bps = _bp([0.0, -0.5], [4.0, 3.999])
    got = pathsearch(lambda x: abs(float(x[0])), bps, reference=4.0, sigma=0.5)
    assert got is not None            # 3.999 <= (1 - 0.5*0) * 4.0
    x, t, m = got
    assert m <= 4.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_pathsearch.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

`src/mcp_solver/path/pathsearch.py`:
```python
"""Backward pathsearch over stored breakpoints (spec as amended 2026-07-17).

Walk the piecewise-linear path from the Newton end backward; accept the
first point satisfying the non-monotone descent condition (NmD)
    merit(p(t)) <= (1 - sigma * clamp(t, 0, 1)) * reference,
searching each segment by geometric halving from its far end. Checking
the endpoint first means a good Newton point costs one merit evaluation.
"""
import numpy as np


def pathsearch(merit_fn, breakpoints, reference, sigma,
               shrink=0.5, max_halvings=25):
    def acceptable(t, m):
        return np.isfinite(m) and (
            m <= (1.0 - sigma * min(max(t, 0.0), 1.0)) * reference)

    for i in range(len(breakpoints) - 1, 0, -1):
        t_hi, x_hi = breakpoints[i]
        t_lo, x_lo = breakpoints[i - 1]
        m = merit_fn(x_hi)
        if acceptable(t_hi, m):
            return x_hi, t_hi, m
        alpha = 1.0
        for _ in range(max_halvings):
            alpha *= shrink
            xt = x_lo + alpha * (x_hi - x_lo)
            tt = t_lo + alpha * (t_hi - t_lo)
            m = merit_fn(xt)
            if acceptable(tt, m):
                return xt, tt, m
    return None
```

- [ ] **Step 4: Run tests, commit**

Run: `.venv/bin/python -m pytest tests/test_pathsearch.py -v` — 5 passed;
full non-slow suite green.
```bash
git add -A && git commit -m "feat: backward pathsearch over stored breakpoints"
```

---

### Task 6: Non-monotone stabilization and Algorithm PATH (`path/nms.py`, `path/solver.py`)

**Files:**
- Create: `src/mcp_solver/path/nms.py`, `src/mcp_solver/path/solver.py`
- Modify: `src/mcp_solver/__init__.py` (export `solve_path`)
- Test: `tests/test_solve_path.py`

**Interfaces:**
- Consumes: `linearize`, `generate_path`, `PathStatus`, `pathsearch`, `normal_map.merit/fB_np/decompose/natural_residual`, `SolverOptions`, `SolveResult`/`Status`/`IterationRecord`.
- Produces: `solve_path(problem: MCPProblem, options: SolverOptions | None = None) -> SolveResult` (the stage-2 public API; Task 7 registers it in the cross-check registry); `nms.NMSState` with `d_allowed(k)`, `after_d()`, `new_checkpoint(merit, k)`, attributes `reference`, `delta`, `checkpoint_k`.

Algorithm PATH (paper p. 14), as the spec's outer-loop section states it:
- Convergence test `‖f_B(x)‖∞ ≤ tol` at the top of each iteration.
- d-step: allowed while `k < checkpoint_k + n_bar`, requires a full Newton path (`NEWTON_POINT`), step shorter than `delta`, and finite merit at the endpoint; accepts WITHOUT a merit test, shrinks `delta *= beta`. Per p. 14, if the (already computed) merit beats the reference value, promote the point to a check point.
- m-step: accept the Newton endpoint iff `merit <= (1 - sigma*T_k) * reference`; new check point (rule 15: reference = max of last `m_bar` check-point merits).
- watchdog: on m-step failure (or ray/pivot-limit/undefined endpoint): return to the last check point, regenerate its path, `pathsearch` it against the reference; acceptance creates a new check point; `None` → STALLED (or RAY_TERMINATION when the failing path rayed).

- [ ] **Step 1: Write the failing test**

`tests/test_solve_path.py`:
```python
import jax.numpy as jnp
import numpy as np

from mcp_solver import SolverOptions, Status
from mcp_solver.path.solver import solve_path
from mcp_solver.problem import MCPProblem

INF = np.inf


def _check_mcp(p, res, tol=1e-6):
    assert res.status is Status.CONVERGED, res.status
    z, f = res.z, p.f_np(res.z)
    assert np.all(z >= p.lb - tol) and np.all(z <= p.ub + tol)
    fixed = p.lb == p.ub
    at_l = (z <= p.lb + tol) & ~fixed
    at_u = (z >= p.ub - tol) & ~fixed
    interior = ~at_l & ~at_u & ~fixed
    assert np.all(f[at_l] >= -tol)
    assert np.all(f[at_u] <= tol)
    assert np.all(np.abs(f[interior]) <= tol)


def test_linear_free_system_one_path():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    p = MCPProblem(lambda z: jnp.asarray(A) @ z - jnp.asarray(b),
                   np.full(2, -INF), np.full(2, INF), np.zeros(2))
    res = solve_path(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, np.linalg.solve(A, b), atol=1e-8)
    assert len(res.iterations) <= 2


def test_active_bounds_and_multipliers():
    # f(z) = z - t on [0,2]^3, t = (-1, 0.5, 3): z* = (0, 0.5, 2)
    t = jnp.array([-1.0, 0.5, 3.0])
    p = MCPProblem(lambda z: z - t, np.zeros(3), np.full(3, 2.0),
                   np.full(3, 1.0))
    res = solve_path(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [0.0, 0.5, 2.0], atol=1e-8)
    assert res.w[0] > 0.5 and res.v[2] > 0.5


def test_nonlinear_ncp():
    # nonlinear complementarity: f(z) = z^2 - 4 on z >= 0 -> z* = 2
    p = MCPProblem(lambda z: z**2 - 4.0, np.zeros(1), np.array([INF]),
                   np.array([0.5]))
    res = solve_path(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [2.0], rtol=1e-7)


def test_nasty_atan_converges():
    p = MCPProblem(lambda z: jnp.arctan(z - 10.0), np.full(1, -INF),
                   np.array([INF]), np.array([0.0]))
    res = solve_path(p, SolverOptions(max_iter=200))
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [10.0], atol=1e-6)
    # damping machinery must have engaged on this problem: the plain
    # full-Newton m-step cannot be the only step type used
    kinds = {r.step_type for r in res.iterations}
    assert kinds != {"m"}, kinds


def test_dsteps_taken_near_solution():
    # start near the solution of a smooth problem: steps are tiny -> d-steps
    p = MCPProblem(lambda z: z - 1.0, np.zeros(3), np.full(3, INF),
                   np.full(3, 0.999))
    res = solve_path(p)
    _check_mcp(p, res)
    assert any(r.step_type == "d" for r in res.iterations)


def test_infeasible_reports_ray_or_stall():
    p = MCPProblem(lambda z: -jnp.ones_like(z), np.zeros(1),
                   np.array([INF]), np.array([1.0]))
    res = solve_path(p, SolverOptions(max_iter=30))
    assert res.status in (Status.RAY_TERMINATION, Status.STALLED,
                          Status.MAX_ITERATIONS)


def test_domain_error_at_start():
    p = MCPProblem(lambda z: jnp.log(z), np.zeros(1), np.array([INF]),
                   np.array([0.0]))   # log(0) at the only feasible start
    res = solve_path(p)
    assert res.status is Status.DOMAIN_ERROR


def test_log_domain_problem_converges():
    # f(z) = log(z) - 1 on [1e-8, inf): solution z = e; path points that
    # leave the domain must be handled by the merit(inf) machinery
    p = MCPProblem(lambda z: jnp.log(z) - 1.0, np.full(1, 1e-8),
                   np.array([INF]), np.array([5.0]))
    res = solve_path(p)
    _check_mcp(p, res)
    np.testing.assert_allclose(res.z, [np.e], rtol=1e-7)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_solve_path.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

`src/mcp_solver/path/nms.py`:
```python
"""Non-monotone stabilization state (paper section 2.4, rules NmD/15)."""


class NMSState:
    def __init__(self, merit0, opts):
        self.opts = opts
        self.checkpoint_merits = [merit0]
        self.reference = merit0
        self.delta = opts.delta0
        self.checkpoint_k = 0

    def d_allowed(self, k):
        return k < self.checkpoint_k + self.opts.n_bar

    def after_d(self):
        self.delta *= self.opts.beta

    def new_checkpoint(self, merit, k):
        self.checkpoint_merits.append(merit)
        recent = self.checkpoint_merits[-max(self.opts.m_bar, 1):]
        self.reference = max(recent)          # rule (15)
        self.delta = self.opts.delta0
        self.checkpoint_k = k + 1
```

`src/mcp_solver/path/solver.py`:
```python
"""Algorithm PATH (Dirkse & Ferris 1993, p. 14): the outer loop."""
import numpy as np

from mcp_solver.normal_map import decompose, fB_np, merit, natural_residual
from mcp_solver.options import SolverOptions
from mcp_solver.path.linearize import linearize
from mcp_solver.path.nms import NMSState
from mcp_solver.path.pathsearch import pathsearch
from mcp_solver.path.pivot import PathStatus, generate_path
from mcp_solver.result import IterationRecord, SolveResult, Status


def solve_path(problem, options=None):
    opts = options or SolverOptions()
    lb, ub = problem.lb, problem.ub

    def merit_fn(x):
        return merit(problem.f_np, x, lb, ub)

    x = np.clip(problem.x0, lb, ub).astype(float)
    m0 = merit_fn(x)
    zeros = np.zeros_like(x)
    if not np.isfinite(m0):
        return SolveResult(Status.DOMAIN_ERROR, x, zeros, zeros, np.inf, [])
    if linearize(problem, x) is None:
        return SolveResult(Status.DOMAIN_ERROR, x, zeros, zeros, np.inf, [])

    state = NMSState(m0, opts)
    checkpoint_x = x.copy()
    records = []
    status = Status.MAX_ITERATIONS
    k = 0

    for _ in range(opts.max_iter):
        fB = fB_np(problem.f_np, x, lb, ub)
        if np.all(np.isfinite(fB)) and np.abs(fB).max() <= opts.tol:
            status = Status.CONVERGED
            break

        lin = linearize(problem, x)
        path = generate_path(lin, opts) if lin is not None else None
        if path is not None and path.status is PathStatus.SINGULAR_BASIS:
            status = Status.SINGULAR_BASIS
            break

        newton_ok = (path is not None
                     and path.status is PathStatus.NEWTON_POINT)
        if path is not None and path.breakpoints:
            T_k, xT = path.breakpoints[-1]
        else:
            T_k, xT = 0.0, x
        step = float(np.linalg.norm(xT - x))
        mT = merit_fn(xT) if path is not None else np.inf

        if (newton_ok and state.d_allowed(k) and step < state.delta
                and np.isfinite(mT)):
            x = xT
            state.after_d()
            if mT < state.reference:          # p. 14: promote good d-steps
                state.new_checkpoint(mT, k)
                checkpoint_x = x.copy()
            records.append(IterationRecord(k=k, merit=mT, step_type="d",
                                           step_len=step,
                                           pivots=path.n_pivots, T=T_k))
        elif (newton_ok and np.isfinite(mT)
              and mT <= (1.0 - opts.sigma * min(max(T_k, 0.0), 1.0))
              * state.reference):
            x = xT
            state.new_checkpoint(mT, k)
            checkpoint_x = x.copy()
            records.append(IterationRecord(k=k, merit=mT, step_type="m",
                                           step_len=step,
                                           pivots=path.n_pivots, T=T_k))
        else:
            # watchdog: back to the check point, search its path
            if np.array_equal(x, checkpoint_x) and path is not None:
                cp_path = path                # already at the check point
            else:
                lin_cp = linearize(problem, checkpoint_x)
                cp_path = (generate_path(lin_cp, opts)
                           if lin_cp is not None else None)
            found = None
            if cp_path is not None and len(cp_path.breakpoints) > 1:
                found = pathsearch(merit_fn, cp_path.breakpoints,
                                   state.reference, opts.sigma)
            if found is None:
                rayed = (path is not None
                         and path.status is PathStatus.RAY_TERMINATION)
                status = (Status.RAY_TERMINATION if rayed else Status.STALLED)
                break
            prev = checkpoint_x
            x, t_acc, m_acc = found
            state.new_checkpoint(m_acc, k)
            checkpoint_x = x.copy()
            records.append(IterationRecord(
                k=k, merit=m_acc, step_type="w",
                step_len=float(np.linalg.norm(x - prev)),
                pivots=cp_path.n_pivots, T=t_acc))
        k += 1
        if opts.verbose:
            print(records[-1])

    z, w, v = decompose(x, lb, ub)
    fz = problem.f_np(z)
    if np.all(np.isfinite(fz)):
        resid = natural_residual(z, fz, lb, ub)
    else:
        resid = np.inf
        if status is Status.CONVERGED:
            status = Status.DOMAIN_ERROR
    return SolveResult(status, z, w, v, resid, records)
```

Update `src/mcp_solver/__init__.py` — add the import and export:
```python
from mcp_solver.path.solver import solve_path
```
and append `"solve_path"` to `__all__` (keep the x64 config line FIRST).

Implementation notes:
- `w`/`v` come from `decompose(x, ...)` — at a normal-map solution these ARE the multipliers (paper eq. 5), unlike the semismooth solver's `(f)+/(−f)+` split. Both satisfy the MCP conditions at a solution; do not "unify" them.
- The watchdog reuses the current path when the failing step was launched from the check point itself — regenerating would produce the identical path.
- A ray/pivot-limit path falls through to the watchdog (its breakpoints may still contain an acceptable point); RAY_TERMINATION is reported only when the watchdog also finds nothing.

- [ ] **Step 4: Run tests, commit**

Run: `.venv/bin/python -m pytest tests/test_solve_path.py -v` — 8 passed;
full non-slow suite green.
```bash
git add -A && git commit -m "feat: Algorithm PATH outer loop with NMS watchdog"
```

---

### Task 7: Cross-check registry, new literature problems, CGE through PATH

**Files:**
- Modify: `tests/conftest.py` (register `solve_path`), `tests/problems.py` (add `murty`, `josephy_variant`), `examples/shoven_whalley.py` (`__main__` runs both solvers)
- Test: `tests/test_cross_check.py` (new), plus the whole existing parametrized suite now runs ×2 solvers

**Interfaces:**
- Consumes: everything.
- Produces: `SOLVERS = {"semismooth": solve_semismooth, "path": solve_path}` — the spec's cross-check layer goes live.

- [ ] **Step 1: Register the solver and add problems**

`tests/conftest.py` — replace the SOLVERS dict:
```python
import pytest

from mcp_solver.path.solver import solve_path
from mcp_solver.semismooth import solve_semismooth

SOLVERS = {"semismooth": solve_semismooth, "path": solve_path}


@pytest.fixture(params=sorted(SOLVERS), ids=sorted(SOLVERS))
def solver(request):
    return SOLVERS[request.param]
```

Append to `tests/problems.py`:
```python
def murty(n=8):
    """Murty-style exponential LCP family: M = I + 2*tril(ones, -1),
    q = -ones. Solution derived in-plan: z* = e_1 (row 1: z1 - 1 = 0;
    row i>1: 2 z1 + z_i - 1 = 1 > 0 with z_i = 0). Classic worst case
    for Lemke pivot counts."""
    M = np.eye(n) + 2.0 * np.tril(np.ones((n, n)), -1)
    q = -np.ones(n)
    Mj, qj = jnp.asarray(M), jnp.asarray(q)
    p = MCPProblem(lambda z: Mj @ z + qj, np.zeros(n), np.full(n, INF),
                   np.zeros(n))
    sol = np.zeros(n)
    sol[0] = 1.0
    p.known_solution = sol
    return p


def josephy_variant():
    """4-variable NCP in the Josephy/Kojima-Shindo family. Coefficients
    are plan-defined (NOT a literature citation); correctness is verified
    by MCP residuals only."""
    def f(z):
        z1, z2, z3, z4 = z[0], z[1], z[2], z[3]
        return jnp.stack([
            3 * z1**2 + 2 * z1 * z2 + 2 * z2**2 + z3 + 3 * z4 - 6,
            2 * z1**2 + z1 + z2**2 + 3 * z3 + 2 * z4 - 2,
            3 * z1**2 + z1 * z2 + 2 * z2**2 + 2 * z3 + 3 * z4 - 1,
            z1**2 + 3 * z2**2 + 2 * z3 + 3 * z4 - 3,
        ])
    return MCPProblem(f, np.zeros(4), np.full(4, INF), np.full(4, 1.0))
```
And extend `LIBRARY`:
```python
LIBRARY = {
    "kojima_shindo": kojima_shindo,
    "cournot": cournot_duopoly,
    "lcp_n20": lambda: synthetic_lcp(20, seed=0),
    "lcp_n80_degenerate": lambda: synthetic_lcp(80, seed=3, frac_active=0.7),
    "upper_bounded": upper_bounded_lcp,
    "murty_n8": lambda: murty(8),
    "josephy_variant": josephy_variant,
}
```

- [ ] **Step 2: Write the cross-check test**

`tests/test_cross_check.py`:
```python
"""Spec testing layer 3: both solvers on every problem, agreement checked."""
import numpy as np
import pytest

from examples.shoven_whalley import build_model
from examples.synthetic_cge import make_exchange_economy
from mcp_solver import SolverOptions
from mcp_solver.path.solver import solve_path
from mcp_solver.semismooth import solve_semismooth
from tests.problems import LIBRARY, assert_mcp_solution


@pytest.mark.parametrize("name", sorted(LIBRARY))
def test_both_solvers_agree(name):
    p1, p2 = LIBRARY[name](), LIBRARY[name]()
    r1 = solve_semismooth(p1)
    r2 = solve_path(p2)
    assert r1.converged and r2.converged, (r1.status, r2.status)
    assert_mcp_solution(p1, r1.z)
    assert_mcp_solution(p2, r2.z)
    if hasattr(p1, "known_solution"):
        np.testing.assert_allclose(r1.z, p1.known_solution, atol=1e-5)
        np.testing.assert_allclose(r2.z, p1.known_solution, atol=1e-5)


def test_shoven_whalley_through_path():
    m1, m2 = build_model(), build_model()
    r1 = solve_semismooth(m1.build())
    r2 = solve_path(m2.build())
    assert r1.converged and r2.converged
    s1, s2 = m1.unpack(r1.z), m2.unpack(r2.z)
    for key in ("y", "p", "pf"):
        np.testing.assert_allclose(s1[key], s2[key], rtol=1e-5)


@pytest.mark.parametrize("n", [20, 200])
def test_exchange_economy_through_path(n):
    m = make_exchange_economy(n, seed=42)
    prob = m.build()
    res = solve_path(prob)
    assert res.converged, f"n={n}: {res.status}"
    assert np.abs(prob.f_np(res.z)).max() < 1e-6


@pytest.mark.slow
def test_exchange_economy_1000_through_path():
    m = make_exchange_economy(1000, seed=7)
    prob = m.build(jac_coloring=False)
    res = solve_path(prob, SolverOptions(max_iter=200, max_pivots=20000))
    assert res.converged, res.status
    assert np.abs(prob.f_np(res.z)).max() < 1e-5
```

Also update `examples/shoven_whalley.py`'s `__main__` block to run both solvers:
```python
if __name__ == "__main__":
    from mcp_solver import SolverOptions
    from mcp_solver.path.solver import solve_path
    from mcp_solver.semismooth import solve_semismooth

    m = build_model()
    for name, solver in (("semismooth", solve_semismooth),
                         ("path", solve_path)):
        res = solver(m.build(), SolverOptions(verbose=True))
        print(f"--- {name} ---")
        print(res.table())
        for vname, val in m.unpack(res.z).items():
            print(f"{vname} = {val}")
```

- [ ] **Step 3: Run everything**

Run: `.venv/bin/python -m pytest -m 'not slow' -q`
Expected: all green — note the parametrized literature tests now run ×2
(semismooth + path). Kojima–Shindo through PATH is the hardest gate in the
suite; if it fails, debug with `SolverOptions(verbose=True)` and the
`debug=True` flag of `generate_path` — do NOT weaken the test.

Then the slow gates:
`.venv/bin/python -m pytest -m slow -v` (n=2000 semismooth + n=1000 path;
expect minutes — report runtimes in the task report).

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "test: cross-solver checks, Murty/Josephy problems, CGE through PATH"
```

---

## Self-Review Notes (already applied)

- Spec coverage: every stage-2 spec section has a task — outer loop/d-m-watchdog/rule 15 (Task 6); path generation incl. pivot rules, t-relaxation, Lemke start, `lemke_start` option, Harris ratio test, explicit-inverse + Sherman–Morrison + residual monitor + periodic refactorization (Tasks 3–4); equilibration with consistent transforms and original-unit merit (Task 2, exercised in 4/6); snapshot pathsearch per the amended spec section (Task 5); domain-error watchdog (Task 6 via `merit = inf` + `linearize -> None`); cross-check layer + Murty/Josephy (Task 7); carryover items 1–4 (Task 1 and Task 7). Carryover item 6 (LM descent fallback in the semismooth solver) is deliberately NOT included — stage-1 behavior is spec-accepted as a known limitation; changing it is out of stage-2 scope.
- Type consistency: `LinearMCP` field names identical across Tasks 2/3/4 (test `_lin` helpers construct every field by keyword); `PathResult.breakpoints` is `[(t, x)]` in original units everywhere; `pathsearch` consumes exactly that; every `SolverOptions` field referenced (`pivot_tol`, `basis_residual_tol`, `refactor_every`, `max_pivots`, `lemke_start`, `n_bar`, `m_bar`, `sigma`, `beta`, `delta0`, `tol`, `max_iter`, `verbose`) exists in stage-1 `options.py`.
- Placeholder scan: none — every step carries complete code.
- Known judgment calls the implementer must NOT "fix": `recompute_xB` full-solve after every pivot (correctness over micro-optimization; still O(n²), same order as the update itself); the slack-start covering-vector redefinition (`rs_new`) is deliberate — that path legitimately starts at a different point than `x_k`, exactly as the paper describes; `w/v` conventions differ between the two solvers (Task 6 notes).
