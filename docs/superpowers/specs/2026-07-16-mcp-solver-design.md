# mcp_solver — a pure-Python PATH-style solver for Mixed Complementarity Problems

**Date:** 2026-07-16
**Status:** Approved design, pre-implementation
**Reference:** Dirkse & Ferris (1993), *The PATH Solver: A Non-Monotone Stabilization
Scheme for Mixed Complementarity Problems* (`cstr1179.pdf` in repo root). Page/equation
numbers below refer to that paper.

## Goal

A pure-Python solver for Mixed Complementarity Problems (MCPs), targeted at solving
CGE models, implementing the PATH algorithm from the paper. No native or closed-source
dependencies; JAX is used for function evaluation and automatic differentiation, numpy
for solver-side linear algebra.

**Problem class:** given `f : Rⁿ → Rⁿ` and bounds `l ≤ z ≤ u` (entries may be ±∞),
find `z, w ≥ 0, v ≥ 0` with `f(z) = w − v`, `l ≤ z ≤ u`, `(z−l)ᵀw = 0`, `(u−z)ᵀv = 0`
(paper Definition 1).

**Target scale:** 500–5000 variables (single-country CGE models with real SAM data).
Dense linear algebra throughout is acceptable at this scale.

## Strategy: two solvers, one core (staged)

- **Stage 1:** shared core infrastructure plus a **semismooth Newton** solver
  (Fischer–Burmeister reformulation). Fast to build (~500 lines), gets CGE models
  solving early, and serves as the reference oracle for stage 2.
- **Stage 2:** the **PATH algorithm** proper — pivotal path generation, backtracing
  pathsearch, non-monotone stabilization (watchdog) — built on the same core and
  cross-checked against the stage-1 solver on the full test suite.

Rationale: the pivotal engine is ~80% of the total difficulty; building it against a
working reference solver makes its bugs distinguishable from modeling bugs. The
stage-1 solver remains useful afterward as a fallback.

## Dependencies

`jax` (+ `jaxlib`) and `numpy` only. Model functions are written in JAX; Jacobians come
from JAX AD, jit-compiled, and materialized as **dense** numpy arrays.

JAX's sparse math support (`jax.experimental.sparse`) is explicitly rejected: no sparse
factorizations and patchy op coverage. However, to avoid `vmap` memory exhaustion from 
naive `jax.jacfwd` at $N=5000$, the architecture will use **Jacobian coloring** to 
evaluate the derivatives efficiently. This colored JVP will scatter into a standard 
dense array, bypassing sparse math entirely while mitigating memory scaling issues. If the
project ever outgrows dense (>5000 vars), the route is `scipy.sparse.linalg.splu` — a 
hypothetical stage 3, out of scope here.

## Package layout

```
src/mcp_solver/
  problem.py      # MCPProblem: f (JAX callable), lb, ub, x0; jit f and Jacobian
  normal_map.py   # projection π_B (custom_jvp), normal map f_B, merit ‖f_B‖
  scaling.py      # Ruiz equilibration of a linear system (shared by both solvers)
  model.py        # light modeling helpers (variable blocks, equation pairing, fix())
  result.py       # SolveResult: status enum, z, w, v, residual, iteration log
  options.py      # solver options (paper Table 1: σ, β, Δ, n̄, m̄, tolerances, limits)
  semismooth.py   # stage 1 solver
  path/
    linearize.py  # M = ∇f(z_k), q = f(z_k) − M z_k, residual r = f_B(x_k)
    pivot.py      # rectangular-Lemke path generation (§2.2, eq. 11)
    pathsearch.py # backtracing pathsearch (§2.3)
    nms.py        # non-monotone stabilization: d-steps/m-steps/watchdog (§2.4)
    solver.py     # Algorithm PATH outer loop (p. 14)
tests/            # see Testing
examples/         # shoven_whalley.py, synthetic scalable CGE
```

## Shared core

### MCPProblem (`problem.py`)

`f` (JAX callable over a flat vector), `lb`, `ub`, `x0`. Provides jit-compiled `f(z)`
and dense Jacobian `J(z)` (as numpy arrays to the solvers). Validates shapes and
`lb ≤ ub` at construction; raises immediately on structural errors.

**Jacobian Coloring Extension:** To safely evaluate models at scale, `Model.build()` 
will sequentially compute the Jacobian structure once at initialization to determine 
a color map, and package a colored JVP into the `J(z)` callable. The solver remains 
oblivious to this, receiving a fast, memory-safe dense matrix.

Caveat: CGE income/aggregation equations produce **dense Jacobian rows**, and a single
dense row makes column coloring degenerate toward n colors. The implementation
therefore uses one mechanism — batched JVPs over column groups — where a group is a
color class when the structure permits, and otherwise a fixed-size chunk (~256
columns), which bounds peak memory regardless of sparsity structure.

### Normal map (`normal_map.py`)

`π_B(x) = clip(x, l, u)`; `f_B(x) = f(π_B(x)) + x − π_B(x)`; merit `Θ(x) = ‖f_B(x)‖`.
Conversion between `x` and the triple `(z, w, v)`: `z = π_B(x)`, `v = (x−z)₊`,
`w = (z−x)₊` (paper eq. 5).

`π_B` carries a `jax.custom_jvp` fixing its derivative to **1 at exact-bound points**
(boundary treated as interior). Measured behavior of stock `jnp.clip` (JAX 0.11) is a
gradient of **0.5** at ties — a Clarke-interval midpoint that silently halves Jacobian
columns whenever an iterate sits exactly on a bound, which projected iterates do by
construction. Pinning the derivative to 1 selects a specific, informative
B-subdifferential element and removes dependence on framework tie-breaking.

### Modeling helpers (`model.py`)

Deliberately small, GAMS-MCP in spirit — a convenience layer, not a DSL:

- `Model.add_variables(name, shape, lb=..., ub=...)` → named blocks; handles
  packing/unpacking the flat vector so equation code reads naturally.
- `Model.add_equations(name, func, complements=<block>)` — explicit
  equation↔variable pairing.
- `Model.fix(name, index)` — set `lb = ub` for a variable; the standard CGE numeraire
  device (every Walrasian system has a redundant equation/degenerate direction).
- `Model.build()` → `MCPProblem`; checks every variable is paired exactly once.

## Stage 1: semismooth Newton (`semismooth.py`)

Reformulate MCP as `Φ(z) = 0` with the Fischer–Burmeister function
`φ(a,b) = a + b − √(a² + b²)` (zero iff `a ≥ 0, b ≥ 0, ab = 0`), componentwise by
bound structure:

| bound structure | Φᵢ(z) |
|---|---|
| free | `fᵢ(z)` |
| lower only | `φ(zᵢ − lᵢ, fᵢ(z))` |
| upper only | `−φ(uᵢ − zᵢ, −fᵢ(z))` |
| both | `φ(zᵢ − lᵢ, −φ(uᵢ − zᵢ, −fᵢ(z)))` (reduces to the rows above as l → −∞ / u → +∞) |
| fixed | `zᵢ − lᵢ` |

Iteration:
1. Assemble `H ∈ ∂Φ(z)` (an element of the generalized Jacobian) analytically from
   `∇f` and the FB partials, with the standard perturbation at the kink `(0,0)`.
2. Solve `H d = −Φ`. If `H` is singular or `d` is not a descent direction for
   `Ψ = ½‖Φ‖²`, fall back to a Levenberg–Marquardt step computed as the least-squares
   solution of the stacked system `min‖[H; √μ·I]d + [Φ; 0]‖₂` via QR (never by forming
   `HᵀH`, which squares the condition number — fatal on ill-scaled CGE Jacobians).
3. Non-monotone Armijo linesearch on `Ψ` against `max` of the last `m̄` values —
   the same reference-value logic as stage 2, shared code where practical.

Termination: `‖Φ‖∞ ≤ tol` (converged), else max-iteration / stall statuses.

**Domain safety.** CGE functions (CES shares, logs) are undefined for out-of-bounds
arguments, and unlike the normal-map solver, the FB iteration is not confined to
`B`. Two measures keep it out of trouble:
- Φ evaluates `f` through the projection: `f̃ = f ∘ π_B`. The FB sign structure forces
  every root of the composed Φ into `l ≤ z ≤ u` (an out-of-bounds root would require
  `φ(uᵢ − zᵢ, ·) = 0` with `uᵢ − zᵢ < 0`, impossible), so the solution set is
  unchanged while `f` never sees out-of-bounds values of bounded variables. Free
  variables are not clipped; models whose functions blow up in a free variable should
  bound it.
- NaN-aware linesearch: a trial point where `Ψ` is NaN/Inf is treated as `+∞` and
  rejected; backtracking from an in-domain iterate always reaches defined territory
  since the domain is open around the current point.

## Stage 2: PATH (`path/`)

Faithful implementation of Algorithm PATH (paper p. 14).

### Outer loop (`solver.py`)

At iterate `x_k`: stop if `‖f_B(x_k)‖ ≤ tol`; linearize at `z_k = π_B(x_k)`; generate
the path `p_k : [0, T_k] → Rⁿ`; then:

- **d-step** (allowed while `k < check_point + n̄`): if `‖p(T_k) − p(0)‖ < Δ`, accept
  `x_{k+1} = p(T_k)` without a merit test and shrink `Δ ← βΔ`. (Per p. 14, `f_B` is
  still evaluated at the accepted point; if it beats the reference value, the check
  point and reference value are updated, and if it is undefined a watchdog step is
  taken.)
- **m-step** (otherwise): accept `p(T_k)` if `‖f_B(p(T_k))‖ ≤ (1 − σT_k)·R_j`.
- **watchdog** on m-step failure: return to the last check point, regenerate its path
  if needed, backtrace it for a point satisfying the non-monotone descent condition
  (NmPs); increment `j`, update `R_j`, reset `Δ`, set `check_point = k+1`.

Reference values use the paper's rule (15): `R_{j+1} = max` of the last `m(j+1) ≤ m̄`
check-point merit values.

### Path generation (`pivot.py`)

The linearized MCP in tableau form (eq. 11): columns `[M  −I  I  r]` over variables
`(z, w, v, t)` with `l ≤ z ≤ u`, `w, v ≥ 0`, `0 ≤ t ≤ 1`, right-hand side `−q + r`,
covering vector `r = f_B(x_k)`. Bounded-variable complementary pivoting:

- Initial basis from the triple `(z_k, w_k, v_k)` at `t = 0`; `t` always enters first.
- Pivot rules (p. 8): `w_j` leaves → `z_j` enters at `l_j`; `v_j` leaves → `z_j` enters
  at `u_j`; `z_j` leaves at lower → `w_j` enters at 0; `z_j` leaves at upper → `v_j`
  enters at 0; `t` leaves at 1 → Newton point found.
- After `t` first enters, its lower bound is relaxed (t may oscillate/go negative), per
  the paper's recommendation.
- Termination: `t` leaves at 1 (success), ray termination (no blocking variable), or
  pivot limit.
- Degeneracy: Harris-style tolerance ratio test plus deterministic tiny bound
  perturbations (EXPAND-like), removed at termination. Lexicographic rules are
  rejected: they need rows of `B⁻¹` per tie and add complexity for no practical
  benefit on empirical models.
- Rank-deficient initial basis: restart the path from a point corresponding to the
  all-slack basis (Lemke start), per pp. 8–9; a `lemke_start` option forces this
  always, reproducing Lemke's method for comparison (as PATH offers).

**Equilibration (required, not optional):** before pivoting, the linearized system is
Ruiz-equilibrated — iterated row/column max-norm scaling `M̃ = R·M·C` with `q`, `r`,
bounds, and slacks transformed consistently (`z̃ = C⁻¹z`, `w̃ = Rw`, `ṽ = Rv`;
positive diagonal scalings preserve the complementarity pairing). CGE systems are
natively terribly scaled (quantities ~10⁶ against prices ~1), and the explicit-inverse
scheme below is *not backward stable*: its residuals start near `cond(B)·ε`, so an
unscaled cond of 10⁸ would trip the refactorization monitor within a few pivots and
degrade the engine to O(n³) per pivot. Merit values and descent tests are computed in
original units; scaling is internal to the pivot engine. The same utility
preconditions the stage-1 Newton systems.

**Linear algebra:** numpy has no LU-update or factor-reuse routines, so the basis is
maintained as an **explicit inverse** `B⁻¹` with Sherman–Morrison rank-1
column-replacement updates — O(n²) per pivot, fully vectorized, no eta-file loop.
Stability is policed rather than assumed: every pivot checks the basic-solution
residual `‖B·x_B − b‖∞` and refactorizes (fresh `numpy.linalg.solve`/`inv`) when it
drifts past tolerance, with an unconditional refactorization at least every ~50
pivots. Refactorizing every pivot would be O(n³) per pivot — unacceptable at n = 5000.

### Backtracing pathsearch (`pathsearch.py`)

During path construction, store only the entering-variable stack (§2.3). If the
endpoint fails the descent test, unpivot back through the stack, checking the
non-monotone condition (NmD) at breakpoints; Armijo search within the first segment
whose near endpoint passes and far endpoint fails, yielding a step satisfying (NmPs).

### Domain errors during iteration

If `f` is undefined (NaN/Inf) at a prospective iterate, treat as a failed m-step and
take a watchdog step (paper p. 14).

## Error handling & diagnostics

- Numerical failure never raises; `SolveResult.status` is one of:
  `CONVERGED`, `RAY_TERMINATION`, `MAX_ITERATIONS`, `SINGULAR_BASIS`,
  `DOMAIN_ERROR` (f undefined at `x0`), `STALLED`.
- Structural errors (shape mismatch, `l > u`, unpaired variables) raise at
  `build()`/solve entry with clear messages.
- Per-iteration log on the result: merit, step type (d/m/watchdog), pivot count, `T`
  reached, steplength; printable iteration table with `verbose=True`.
- `Model.diagnose()`: Jacobian rank at `x0`, all-zero rows/columns (forgotten
  pairing), warning when a homogeneous price system has no fixed numeraire.

## Testing (pytest, four layers)

1. **Pivot-engine invariants** on random LCPs: complementarity of `(z, w, v)` at every
   breakpoint; eq. (10) `Mz(t) + q − w(t) + v(t) = (1−t)r` along the path; recovered
   Newton point satisfies the linearized MCP. Property tests on random monotone
   problems (`M = AAᵀ + εI`) which must always solve.
2. **Literature problems** with known solutions: Murty LCPs, Kojima–Shindo, Josephy,
   a Nash equilibrium problem — exercising degeneracy, nonmonotonicity, the watchdog.
3. **Cross-check:** every suite problem runs through both solvers; solutions must
   agree (up to known solution multiplicity) wherever both converge.
4. **CGE:** Shoven–Whalley-style 2×2×2 with an independently computed equilibrium;
   a parameterized synthetic CGE generator (nested-CES economies, random calibrated
   SAMs, ~20–2000 variables) verifying Walras' law and homogeneity at solutions.

## Out of scope

- Sparse basis factorization (stage-3 hypothetical; although Jacobian *extraction* uses coloring, the solver math remains dense).
- A full modeling DSL (MPSGE-style); only the light helpers above.
- PATH's later-era additions not in the 1993 paper (crash phase, proximal
  perturbation, *logical* presolve — fixed-variable elimination, redundancy
  detection). Numerical scaling (Ruiz equilibration) is explicitly **in** scope; it
  is load-bearing for the explicit-inverse basis scheme.
- GPU execution (JAX runs on CPU here; nothing precludes GPU later).
