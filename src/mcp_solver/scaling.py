import numpy as np


def ruiz(A: np.ndarray, max_iter: int = 20, tol: float = 1e-2):
    """Ruiz equilibration: iterated sqrt max-norm row/column scaling.

    Returns (A_scaled, R, C) with A_scaled = diag(R) @ A @ diag(C).
    Zero rows/columns are left unscaled (scale factor 1).
    """
    n, m = A.shape
    R = np.ones(n)
    C = np.ones(m)
    As = A.astype(np.float64, copy=True)
    for _ in range(max_iter):
        r = np.sqrt(np.abs(As).max(axis=1))
        c = np.sqrt(np.abs(As).max(axis=0))
        r[r == 0] = 1.0
        c[c == 0] = 1.0
        As /= r[:, None]
        As /= c[None, :]
        R /= r
        C /= c
        if max(np.abs(1 - r * r).max(), np.abs(1 - c * c).max()) < tol:
            break
    return As, R, C
