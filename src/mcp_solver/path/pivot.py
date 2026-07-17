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
