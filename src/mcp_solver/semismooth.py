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
