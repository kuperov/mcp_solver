# mcp_solver

A pure-Python solver for **Mixed Complementarity Problems (MCPs)**, built for
solving **computable general equilibrium (CGE) models** without native or
closed-source dependencies. It implements the PATH algorithm of Dirkse &
Ferris (1993) — pivotal path generation, backward pathsearch, and non-monotone
watchdog stabilization — plus an independent semismooth Newton solver used as
a cross-check, on a shared JAX/numpy core.

**Dependencies: `jax` and `numpy`. That's it.** Exact Jacobians come from
JAX automatic differentiation; all solver-side linear algebra is plain numpy.
No PATH binary, no GAMS, no licence keys, no compiled extensions.

---

## The problem class

Given `f : Rⁿ → Rⁿ` and bounds `l ≤ z ≤ u` (entries may be ±∞), find `z` and
nonnegative multipliers `w, v` such that

```
f(z) = w − v,    l ≤ z ≤ u,    (z − l)ᵀw = 0,    (u − z)ᵀv = 0
```

Equivalently, for each component: `z` is at its lower bound and `f ≥ 0`, at
its upper bound and `f ≤ 0`, or strictly interior with `f = 0`. This is the
standard format for Walrasian equilibrium models (Mathiesen/Rutherford
complementarity format), KKT systems, Nash equilibria, and (L)CPs. Square
nonlinear systems are the special case with all bounds infinite.

Target scale: **500–5000 variables** (single-country CGE models with real SAM
data). Dense linear algebra is used throughout; memory-safe Jacobian
extraction makes this workable well past n = 2000.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -m 'not slow' -q   # full test suite
.venv/bin/python examples/shoven_whalley.py   # 2x2x2 CGE through both solvers
```

### Raw API

`f` is written in JAX over a flat vector; bounds and the start point are
numpy arrays:

```python
import jax.numpy as jnp
import numpy as np
from mcp_solver import MCPProblem, SolverOptions, solve_path

# 0 <= z  ⊥  z^2 - 4 >= 0   ->   z* = 2
problem = MCPProblem(
    f=lambda z: z**2 - 4.0,
    lb=np.zeros(1), ub=np.full(1, np.inf), x0=np.array([0.5]),
)
res = solve_path(problem)                 # or solve_semismooth(problem)
print(res.status, res.z, res.w, res.v)    # Status.CONVERGED [2.] ...
print(res.table())                        # per-iteration solver log
```

### Modeling helpers (GAMS-MCP style)

For anything bigger than a toy, `Model` handles variable blocks, explicit
equation↔variable pairing, and numeraire fixing:

```python
from mcp_solver import Model, solve_path

m = Model()
m.add_variables("p", n_goods, lb=0.0, start=1.0)     # prices
m.add_equations("markets", excess_supply, complements="p")
m.fix("p", 0, 1.0)                                   # numeraire
result = solve_path(m.build())
prices = m.unpack(result.z)["p"]
```

- `add_equations(name, func, complements=block)` — `func` receives a dict of
  named JAX arrays and returns the equation block paired with `block`
  (elementwise, GAMS-MCP semantics).
- `fix(name, index, value)` — sets `lb = ub`, the standard CGE numeraire
  device (a fixed variable's paired equation is allowed to be slack).
- `diagnose()` — checks Jacobian rank at the start point, flags all-zero
  rows/columns (forgotten pairings), and warns when a homogeneous price
  system has no fixed numeraire.

## The two solvers

| | `solve_path` | `solve_semismooth` |
|---|---|---|
| method | Dirkse–Ferris PATH: Newton on Robinson's normal map, each step solved by complementary pivoting along a piecewise-linear path | semismooth Newton on a Fischer–Burmeister reformulation `Φ(z) = 0` |
| globalization | backward pathsearch over the pivot path + non-monotone watchdog (d-steps / m-steps / reference values) | non-monotone Armijo linesearch, Levenberg–Marquardt fallback |
| strengths | robust on hard/stiff CGE counterfactuals; handles active-set changes through the pivot path | simple, fast on well-behaved problems; fully independent implementation |
| role | production solver | reference oracle / fallback |

Both take `(MCPProblem, SolverOptions)` and return a `SolveResult`. The test
suite runs **every** problem through both and asserts agreement — a
disagreement is treated as a bug in whichever solver is wrong.

### How `solve_path` works (one paragraph)

At each iterate the MCP is linearized and rewritten via Robinson's normal map
`f_B(x) = f(π_B(x)) + x − π_B(x)`. The linearized subproblem is solved by a
Lemke-style **complementary pivot method** with the current residual as
covering vector: pivoting traces a piecewise-linear path from the current
point toward the Newton point, recording a breakpoint at every basis change.
If the endpoint fails the (non-monotone) descent test, a **backward
pathsearch** walks the stored breakpoints toward the current point; a
**watchdog** allows several cheap "d-steps" between full merit checks and
falls back to the last check point when things go wrong. Rays on nonmonotone
linearizations are retried once from the all-slack (Lemke) basis before
failure is reported. Details: 
[Dirkse & Ferris (1993)](https://pages.cs.wisc.edu/~ferris/techreports/cstr1179.pdf),
and the annotated spec in `docs/superpowers/specs/`.

### Numerical machinery worth knowing about

- **Jacobians** are extracted by batched JVPs with automatic **graph
  coloring** (structure probed at build time) and a fixed-chunk fallback, so
  peak AD memory is bounded regardless of sparsity — CGE income equations
  have dense rows that defeat coloring, and the chunked path handles them.
- **The projection `π_B` carries a custom JVP** pinning its derivative to 1
  at exact-bound points. Stock `jnp.clip` gives 0.5 at ties (JAX splits
  them), which silently halves Jacobian columns for any iterate sitting on a
  bound — which projected iterates do by construction.
- **Ruiz equilibration** is applied to every linearized system before
  pivoting (CGE systems mix quantities ~1e6 with prices ~1); merit values and
  acceptance tests always use original units.
- The pivot basis is kept as an **explicit inverse with Sherman–Morrison
  rank-1 updates**, a per-pivot residual monitor, and periodic
  refactorization. Degeneracy is handled by a Harris two-pass ratio test.
- **Numerical failure never raises.** Every solve returns a `SolveResult`
  with a status: `CONVERGED`, `RAY_TERMINATION` (often a modeling error or a
  genuinely infeasible problem), `MAX_ITERATIONS`, `SINGULAR_BASIS`,
  `DOMAIN_ERROR` (f or its Jacobian undefined at the start), or `STALLED`.
  Structural errors (shape mismatches, `lb > ub`) raise immediately at build.

## Options

`SolverOptions` fields follow the paper's Table 1 where applicable:

| field | default | meaning |
|---|---|---|
| `tol` | 1e-8 | convergence: `‖Φ‖∞` (semismooth) / `‖f_B‖∞` (path) |
| `max_iter` | 500 | outer iteration cap |
| `m_bar` | 10 | non-monotone merit memory (reference = max of last m̄) |
| `n_bar` | 5 | d-steps allowed between forced merit checks (path) |
| `sigma` | 0.01 | descent relaxation in `(1 − σt)·R` |
| `beta`, `delta0` | 0.5, 1.0 | d-step radius shrink factor and initial radius |
| `max_pivots` | 3000 | per-path pivot cap |
| `refactor_every` | 50 | unconditional basis refactorization cadence |
| `pivot_tol`, `basis_residual_tol` | 1e-9, 1e-7 | ratio-test and basis-accuracy tolerances |
| `lemke_start` | False | force the all-slack (Lemke) start for every subproblem |
| `verbose` | False | print the per-iteration record as it happens |

Jacobian extraction knobs (`jac_chunk`, `jac_coloring`) are constructor
arguments of `MCPProblem` / `Model.build(...)`, not solve-time options.

## Examples

**`examples/shoven_whalley.py`** — a 2-good, 2-factor, 2-consumer general
equilibrium model (Cobb-Douglas production, CES demand) built with `Model`.
Tests assert market clearing, Walras' law, and homogeneity (doubling the
numeraire doubles all prices and changes nothing real).

**`examples/synthetic_cge.py`** — a scalable pure-exchange CES economy
generator (gross substitutes, hence unique equilibrium) used as the scale
gate: n = 2000 through the semismooth solver, n = 1000 through PATH.

## Testing

```bash
.venv/bin/python -m pytest -m 'not slow' -q    # ~110 tests, <1 min
.venv/bin/python -m pytest -m slow -v          # large-n scale gates (minutes)
```

Four layers: unit tests with invariant checks (the pivot engine asserts the
path equation and complementarity at every breakpoint in debug mode);
literature problems (Kojima–Shindo, Murty's exponential LCP family, a
Josephy-style NCP, Cournot with a closed-form solution); **cross-solver
agreement on every problem** (`tests/conftest.py` parametrizes the whole
suite over both solvers); and CGE economics tests (equilibrium conditions,
Walras, homogeneity, benchmark replication). Solution tests verify MCP
residuals and complementarity directly rather than trusting literature
constants.

## Project layout

```
src/mcp_solver/
  problem.py      MCPProblem: JAX f, colored/chunked Jacobian extraction
  normal_map.py   π_B (pinned custom JVP), normal map, merit function
  scaling.py      Ruiz equilibration
  model.py        Model helpers: blocks, pairing, fix(), diagnose()
  semismooth.py   Fischer–Burmeister semismooth Newton solver
  path/
    linearize.py  linearization + equilibration at the current iterate
    pivot.py      complementary-pivot path generation (the core)
    pathsearch.py backward pathsearch over stored breakpoints
    nms.py        non-monotone stabilization state (watchdog bookkeeping)
    solver.py     Algorithm PATH outer loop
docs/superpowers/ design spec (with amendment history), implementation
                  plans, and stage carryover notes
```

## Known limitations

- **Dense only.** Jacobians and basis algebra are dense; fine to ~5000
  variables. The sparse route (coloring + sparse LU) is a documented future
  stage, not present.
- The 1993 paper is implemented faithfully, but PATH-the-product's later
  additions (crash phase, proximal perturbation, logical presolve) are out
  of scope.
- Deliberate deviations from the paper (snapshot pathsearch instead of
  entering-stack backtracing; watchdog Lemke restarts; Harris-only
  degeneracy handling) are annotated inline in
  `docs/superpowers/specs/2026-07-16-mcp-solver-design.md`.

## References

- S. P. Dirkse & M. C. Ferris (1993), *The PATH Solver: A Non-Monotone
  Stabilization Scheme for Mixed Complementarity Problems*
  [link](https://pages.cs.wisc.edu/~ferris/techreports/cstr1179.pdf).
- S. M. Robinson (1992), normal maps and the normal-map equation `f_B(x)=0`.
- L. Mathiesen (1985), the complementarity format for Walrasian equilibrium.
- A. Fischer (1992), the Fischer–Burmeister NCP function.
- D. Ralph (1994), pathsearch damping for Newton methods on nonsmooth
  equations.
