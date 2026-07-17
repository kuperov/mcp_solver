"""Backward pathsearch over stored breakpoints (spec as amended 2026-07-17).

Walk the piecewise-linear path from the Newton end backward; accept the
first point satisfying the non-monotone descent condition (NmD)
    merit(p(t)) <= (1 - sigma * clamp(t, 0, 1)) * reference,
searching each segment by geometric halving from its far end. Checking
the endpoint first means a good Newton point costs one merit evaluation.
"""
import numpy as np


def pathsearch(merit_fn, breakpoints, reference, sigma,
               shrink=0.5, max_halvings=25):
    def acceptable(t, m):
        return np.isfinite(m) and (
            m <= (1.0 - sigma * min(max(t, 0.0), 1.0)) * reference)

    # Check all breakpoints first (backward from Newton end)
    for i in range(len(breakpoints) - 1, -1, -1):
        t, x = breakpoints[i]
        m = merit_fn(x)
        if acceptable(t, m):
            return x, t, m

    # If no breakpoint acceptable, search within segments by halving from far end
    for i in range(len(breakpoints) - 1, 0, -1):
        t_hi, x_hi = breakpoints[i]
        t_lo, x_lo = breakpoints[i - 1]
        alpha = 1.0
        for _ in range(max_halvings):
            alpha *= shrink
            xt = x_lo + alpha * (x_hi - x_lo)
            tt = t_lo + alpha * (t_hi - t_lo)
            m = merit_fn(xt)
            if acceptable(tt, m):
                return xt, tt, m
    return None
