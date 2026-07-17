import numpy as np

from mcp_solver.normal_map import natural_residual
from mcp_solver.options import SolverOptions
from mcp_solver.result import IterationRecord, SolveResult, Status
from mcp_solver.scaling import ruiz


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
        if not (np.all(np.isfinite(Phi)) and np.all(np.isfinite(H))):
            return fval, None, None
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
