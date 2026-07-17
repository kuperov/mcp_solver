"""Algorithm PATH (Dirkse & Ferris 1993, p. 14): the outer loop."""
import dataclasses

import numpy as np

from mcp_solver.normal_map import decompose, fB_np, merit, natural_residual
from mcp_solver.options import SolverOptions
from mcp_solver.path.linearize import linearize
from mcp_solver.path.nms import NMSState
from mcp_solver.path.pathsearch import pathsearch
from mcp_solver.path.pivot import PathStatus, generate_path
from mcp_solver.result import IterationRecord, SolveResult, Status


def solve_path(problem, options=None):
    opts = options or SolverOptions()
    lb, ub = problem.lb, problem.ub

    def merit_fn(x):
        return merit(problem.f_np, x, lb, ub)

    x = np.clip(problem.x0, lb, ub).astype(float)
    m0 = merit_fn(x)
    zeros = np.zeros_like(x)
    if not np.isfinite(m0):
        return SolveResult(Status.DOMAIN_ERROR, x, zeros, zeros, np.inf, [])
    if linearize(problem, x) is None:
        return SolveResult(Status.DOMAIN_ERROR, x, zeros, zeros, np.inf, [])

    state = NMSState(m0, opts)
    checkpoint_x = x.copy()
    records = []
    status = Status.MAX_ITERATIONS
    k = 0

    for _ in range(opts.max_iter):
        fB = fB_np(problem.f_np, x, lb, ub)
        if np.all(np.isfinite(fB)) and np.abs(fB).max() <= opts.tol:
            status = Status.CONVERGED
            break

        lin = linearize(problem, x)
        path = generate_path(lin, opts) if lin is not None else None
        if path is not None and path.status is PathStatus.SINGULAR_BASIS:
            status = Status.SINGULAR_BASIS
            break

        newton_ok = (path is not None
                     and path.status is PathStatus.NEWTON_POINT)
        if path is not None and path.breakpoints:
            T_k, xT = path.breakpoints[-1]
        else:
            T_k, xT = 0.0, x
        step = float(np.linalg.norm(xT - x))
        mT = merit_fn(xT) if path is not None else np.inf

        if (newton_ok and state.d_allowed(k) and step < state.delta
                and np.isfinite(mT)):
            x = xT
            state.after_d()
            if mT < state.reference:          # p. 14: promote good d-steps
                state.new_checkpoint(mT, k)
                checkpoint_x = x.copy()
            records.append(IterationRecord(k=k, merit=mT, step_type="d",
                                           step_len=step,
                                           pivots=path.n_pivots, T=T_k))
        elif (newton_ok and np.isfinite(mT)
              and mT <= (1.0 - opts.sigma * min(max(T_k, 0.0), 1.0))
              * state.reference):
            x = xT
            state.new_checkpoint(mT, k)
            checkpoint_x = x.copy()
            records.append(IterationRecord(k=k, merit=mT, step_type="m",
                                           step_len=step,
                                           pivots=path.n_pivots, T=T_k))
        else:
            # watchdog: back to the check point, search its path
            if np.array_equal(x, checkpoint_x) and path is not None:
                cp_path = path                # already at the check point
            else:
                lin_cp = linearize(problem, checkpoint_x)
                cp_path = (generate_path(lin_cp, opts)
                           if lin_cp is not None else None)
            found = None
            if cp_path is not None and len(cp_path.breakpoints) > 1:
                found = pathsearch(merit_fn, cp_path.breakpoints,
                                   state.reference, opts.sigma)
            if found is None and not opts.lemke_start:
                # Last resort: the crash-start basis at the check point may
                # itself be the reason no descent direction exists (its
                # pivot sequence is one particular choice, not the only
                # one). Retry that single linearization with the all-slack
                # Lemke start before declaring a genuine stall/ray.
                lemke_opts = dataclasses.replace(opts, lemke_start=True)
                lin_cp2 = linearize(problem, checkpoint_x)
                cp_path2 = (generate_path(lin_cp2, lemke_opts)
                            if lin_cp2 is not None else None)
                if cp_path2 is not None and len(cp_path2.breakpoints) > 1:
                    found = pathsearch(merit_fn, cp_path2.breakpoints,
                                       state.reference, opts.sigma)
                    if found is not None:
                        cp_path = cp_path2
            if found is None:
                rayed = (path is not None
                         and path.status is PathStatus.RAY_TERMINATION)
                status = (Status.RAY_TERMINATION if rayed else Status.STALLED)
                break
            prev = checkpoint_x
            x, t_acc, m_acc = found
            state.new_checkpoint(m_acc, k)
            checkpoint_x = x.copy()
            records.append(IterationRecord(
                k=k, merit=m_acc, step_type="w",
                step_len=float(np.linalg.norm(x - prev)),
                pivots=cp_path.n_pivots, T=t_acc))
        k += 1
        if opts.verbose:
            print(records[-1])

    z, w, v = decompose(x, lb, ub)
    fz = problem.f_np(z)
    if np.all(np.isfinite(fz)):
        resid = natural_residual(z, fz, lb, ub)
    else:
        resid = np.inf
        if status is Status.CONVERGED:
            status = Status.DOMAIN_ERROR
    return SolveResult(status, z, w, v, resid, records)
