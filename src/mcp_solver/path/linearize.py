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
