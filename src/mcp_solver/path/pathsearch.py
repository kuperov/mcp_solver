"""Two-phase backward pathsearch over stored breakpoints (spec as amended 2026-07-17).

Phase 1: check all stored breakpoints AFTER the starting point, backward from
the Newton end; accept the first point satisfying the non-monotone descent
condition (NmD)
    merit(p(t)) <= (1 - sigma * clamp(t, 0, 1)) * reference,
Phase 2: if no breakpoint acceptable, search within segments by geometric halving
from their far ends (this still uses breakpoints[0] as the low end of the first
segment). Preferring stored breakpoints over interior points matches the
backtrace's intent and is what the implementation and test suite pin down.

Index 0 is deliberately excluded from phase 1's direct candidate list:
accepting the path's own origin (breakpoints[0]) would be a null step, so the
search always continues past it to look for real progress along the path.
"""
import numpy as np


def pathsearch(merit_fn, breakpoints, reference, sigma,
               shrink=0.5, max_halvings=25):
    def acceptable(t, m):
        if not np.isfinite(m):
            return False
        tc = min(max(t, 0.0), 1.0)
        if tc <= 0.0:
            # The t-factor is inactive here, so the NmD bound degenerates to
            # m <= reference, which admits equality. A point numerically
            # coincident with the search's checkpoint (t~0, m == reference)
            # must not be accepted as "progress" -- require strict
            # improvement instead.
            return m < reference
        return m <= (1.0 - sigma * tc) * reference

    # Check breakpoints after the start first (backward from Newton end)
    for i in range(len(breakpoints) - 1, 0, -1):
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
