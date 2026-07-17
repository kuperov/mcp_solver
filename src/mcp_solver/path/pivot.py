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
