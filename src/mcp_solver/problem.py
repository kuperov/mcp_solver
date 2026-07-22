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

    def __init__(self, f, lb, ub, x0, *, params=None, jac_chunk=256,
                 jac_coloring=True):
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.n = self.lb.size
        if self.ub.shape != self.lb.shape or self.x0.shape != self.lb.shape:
            raise ValueError("lb, ub, x0 must have identical shapes")
        if np.any(self.lb > self.ub):
            raise ValueError("lb > ub for some component")
        fixed = self.lb == self.ub
        if np.any(fixed & ~np.isfinite(self.lb)):
            raise ValueError("fixed variables (lb == ub) must be finite")
        self._chunk = int(jac_chunk)
        self._lb_j = jnp.asarray(self.lb)
        self._ub_j = jnp.asarray(self.ub)
        # Parameterized residual: f may be f(z) (params is None) or f(z, params),
        # where `params` is any JAX pytree of calibration values. Threading params
        # as a *traced argument* of the jitted residual + JVP means a compiled
        # kernel is REUSED across parameter values -- so re-solving the same
        # structure (homotopy steps, fresh scenarios) with `set_params(...)` skips
        # recompilation and the (structure-only) coloring probe. When params is
        # None the single-argument f(z) form is used, unchanged.
        self.params = params
        if params is None:
            self._f_of = lambda z, p: f(z)
        else:
            self._f_of = f
        self._raw = f
        self._boxed = lambda z, p: self._f_of(project(z, self._lb_j, self._ub_j), p)
        self._f_jit = jax.jit(self._f_of)
        self._fboxed_jit = jax.jit(self._boxed)
        # batched JVP: S is (k, n) tangents -> (k, n) directional derivatives;
        # differentiated w.r.t. z only, with params held fixed for that solve.
        self._jvp_raw = jax.jit(
            lambda z, S, p: jax.vmap(
                lambda s: jax.jvp(lambda zz: self._f_of(zz, p), (z,), (s,))[1])(S))
        self._jvp_boxed = jax.jit(
            lambda z, S, p: jax.vmap(
                lambda s: jax.jvp(lambda zz: self._boxed(zz, p), (z,), (s,))[1])(S))
        self._groups = None          # list of np.ndarray column-index groups
        self._pattern = None         # (rows_nz, cols_nz) of probed sparsity
        if jac_coloring:
            self._build_coloring()
        self.n_jac_tangents = (
            len(self._groups) if self._groups is not None else self.n)

    def set_params(self, params):
        """Swap the residual's parameter pytree in place. The shapes/dtypes must
        match the pytree passed at construction, so the compiled residual/JVP
        kernels and the (structure-only) coloring are reused -- no recompile."""
        if self.params is None:
            raise ValueError(
                "this MCPProblem was built without params (single-argument f); "
                "rebuild with params=... to use set_params")
        self.params = params

    # ---- public API -------------------------------------------------
    def f_np(self, z):
        return np.asarray(self._f_jit(jnp.asarray(z), self.params))

    def f_boxed(self, z):
        return np.asarray(self._fboxed_jit(jnp.asarray(z), self.params))

    def jac(self, z):
        return self._jac(z, self._jvp_raw)

    def jac_boxed(self, z):
        # boxed pattern is a subset of the raw pattern (clipped columns are
        # zeroed), so the raw coloring remains valid for scattering.
        return self._jac(z, self._jvp_boxed)

    # ---- internals ---------------------------------------------------
    def _batched_jvp(self, z, S, jvp):
        zj = jnp.asarray(z)
        out = [np.asarray(jvp(zj, jnp.asarray(S[s:s + self._chunk]), self.params))
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
        # stable tie-break: for banded patterns most columns tie on degree,
        # and an unstable sort's arbitrary tie order can force the greedy
        # pass into a non-optimal color count (e.g. 4 instead of 3 colors
        # on a tridiagonal pattern); a stable sort keeps ties in column
        # order, which is exactly the periodic assignment coloring wants.
        order = np.argsort(-pattern.sum(axis=0), kind="stable")
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
