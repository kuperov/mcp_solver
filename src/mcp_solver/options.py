from dataclasses import dataclass


@dataclass
class SolverOptions:
    # convergence
    tol: float = 1e-8            # ||Phi||_inf (stage 1) / ||f_B||_inf (stage 2)
    max_iter: int = 500
    # non-monotone reference values (paper section 2.4)
    m_bar: int = 10              # merit-memory length (paper m-bar)
    n_bar: int = 5               # d-steps between forced m-steps (stage 2)
    sigma: float = 0.01          # descent relaxation, sigma in (0,1)
    beta: float = 0.5            # d-step radius shrink factor (stage 2)
    delta0: float = 1.0          # initial d-step radius (stage 2)
    # linesearch (stage 1)
    armijo_c: float = 1e-4
    alpha_min: float = 1e-12
    lm_mu: float = 1e-6          # LM regularization scale
    # pivoting (stage 2; declared now so both stages share one options type)
    max_pivots: int = 3000
    refactor_every: int = 50
    pivot_tol: float = 1e-9
    basis_residual_tol: float = 1e-7
    lemke_start: bool = False
    verbose: bool = False
