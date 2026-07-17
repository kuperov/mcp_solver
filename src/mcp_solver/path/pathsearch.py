"""Two-phase backward pathsearch over stored breakpoints (spec as amended 2026-07-17).

Phase 1: check all stored breakpoints backward from the Newton end; accept the
first point satisfying the non-monotone descent condition (NmD)
    merit(p(t)) <= (1 - sigma * clamp(t, 0, 1)) * reference,
Phase 2: if no breakpoint acceptable, search within segments by geometric halving
from their far ends. Preferring stored breakpoints over interior points matches the
backtrace's intent and is what the implementation and test suite pin down.
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
