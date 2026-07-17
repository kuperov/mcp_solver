# Stage-1 → Stage-2 carryover notes

Deferred findings from the stage-1 final whole-branch review (branch merged to
master at d542712, 47 tests green). Address during stage-2 planning/implementation.

## Deferred fixes / improvements

1. **Dead options fields:** `SolverOptions.jac_chunk` / `jac_coloring` are never
   read — the real knobs are `MCPProblem(..., jac_chunk=, jac_coloring=)` via
   `Model.build(**mcp_kwargs)`. Wire them through or remove the fields.
2. **Linesearch Jacobian waste:** `solve_semismooth`'s backtracking loop calls
   `system(zt)` (full `jac_boxed`) at every rejected trial point. Restructure so
   the Jacobian is computed only at accepted points. First lever if large models
   feel slow.
3. **Merit helper:** spec lists `Θ(x) = ‖f_B(x)‖` in `normal_map.py`; not yet
   implemented — stage 2 (PATH outer loop) is its only consumer; add it there.
4. **Literature problems:** spec names Murty and Josephy problems; add to
   `tests/problems.py` in stage 2 where the cross-check suite is the point
   (they mainly stress watchdog/pathsearch).
5. **Coloring probe limitation:** sparsity pattern probed at 3 random points; a
   structural nonzero vanishing at all three ⇒ silently wrong compressed
   Jacobian (cannot cause false CONVERGED; worst case slow/stall). Known
   limitation. Also: greedy coloring is ~O(n³) on dense patterns before it
   discovers it can't win — add an early bail-out on any full pattern row.
6. **LM descent in scaled space:** `_newton_step`'s LM solves in Ruiz-scaled
   y-space; with R≠I the step is not algebraically guaranteed descent for
   Ψ (Armijo then degrades to near-pure nonmonotone acceptance; graceful
   failure = STALLED). Cheap improvement: check the LM step's `gᵀd`, fall back
   to `−g` direction if positive.
7. **build()-time validation:** reject non-finite fixed bounds
   (`lb == ub == ±inf` currently maps to the `fixed` FB case, Φ = z − inf).
8. **DOMAIN_ERROR gloss:** status now also covers "f finite but ∂Φ non-finite
   at x0" (commit d542712), slightly wider than the spec's "f undefined at x0"
   wording — update the spec text when next edited.

## Plan-bug adjudications from stage 1 (for the record)

- `natural_residual` test: `> 1.0` → `>= 1.0` (value exactly 1.0).
- Greedy coloring needs `np.argsort(..., kind="stable")` for the tridiagonal
  ≤3-color bound.
- SW model: homogeneity rank deficiency only visible at a cost-covering start;
  `p` start = `unit_cost(pf=1)` (benchmark-price convention).

## Stage-2 disposition (2026-07-17)

- Items 1-4: done.
- Item 5 (coloring dense-row bail-out): explicitly carried to stage 3
  (perf only).
- Item 6 (LM descent in scaled space): accepted as a known limitation, out
  of scope, unchanged.
- Item 7 (reject non-finite fixed bounds): fixed in this commit
  (`MCPProblem.__init__` in `src/mcp_solver/problem.py`).
- Item 8 (DOMAIN_ERROR gloss): was already done.

Newly logged minors from the stage-2 final review (not actioned beyond what's
noted):

- Last-iterate convergence check (vs. best-iterate) — logged, not addressed.
- Dead `w_bar`/`v_bar` docs-code in `_slack_start`
  (`src/mcp_solver/path/pivot.py`) — logged, not addressed.
- Redundant `linearize` call in the watchdog Lemke fallback — logged, not
  addressed.
- `fb_system` recomputes Φ at accepted points — logged, not addressed.
- The `w`/`v` conventions intentionally differ between the two solvers
  (`path/solver.py` uses the box-preimage convention via
  `normal_map.decompose`; `semismooth.py` uses the FB-residual convention
  from `f` at `z`); cross-reference docstring/comment notes added at both
  call sites in this commit.
