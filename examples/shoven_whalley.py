"""Shoven-Whalley-style 2x2x2 general equilibrium model as an MCP.

Cobb-Douglas production, CES consumers, factor endowments. Correctness
is asserted through equilibrium conditions (market clearing, zero
profit, Walras, homogeneity), not published constants.
"""
import jax.numpy as jnp
import numpy as np

from mcp_solver.model import Model

# calibration
ALPHA = np.array([[0.6, 0.4],      # factor shares by activity (rows sum 1)
                  [0.3, 0.7]])
A_SHARE = np.array([[0.5, 0.5],    # consumer CES share params
                    [0.2, 0.8]])
SIGMA = np.array([1.5, 0.8])       # consumer elasticities
ENDOW = np.array([[3.0, 1.0],      # e[h, f]: endowment of factor f
                  [1.0, 4.0]])


def _unit_cost(pf):
    alpha = jnp.asarray(ALPHA)
    return jnp.prod((pf[None, :] / alpha) ** alpha, axis=1)


def _demand(p, m_h, a_h, sigma_h):
    num = a_h**sigma_h * p**(-sigma_h) * m_h
    den = jnp.sum(a_h**sigma_h * p**(1.0 - sigma_h))
    return num / den


def _build(fix_numeraire, numeraire_value=1.0):
    model = Model()
    # Start at a cost-covering benchmark: p0 = unit_cost(pf0) so zero-profit
    # holds exactly at the start point. This is also what exposes the
    # textbook homogeneity/Walras-law rank deficiency (without a fixed
    # numeraire the Jacobian is singular right at this point) that
    # Model.diagnose is meant to catch.
    p_start = np.asarray(_unit_cost(jnp.ones(2)))
    model.add_variables("y", 2, lb=0.0, start=1.0)
    model.add_variables("p", 2, lb=0.0, start=p_start)
    model.add_variables("pf", 2, lb=0.0, start=1.0)

    def zero_profit(v):
        return _unit_cost(v["pf"]) - v["p"]

    def goods_market(v):
        m = jnp.asarray(ENDOW) @ v["pf"]
        demand = sum(
            _demand(v["p"], m[h], jnp.asarray(A_SHARE[h]), SIGMA[h])
            for h in range(2))
        return v["y"] - demand

    def factor_market(v):
        c = _unit_cost(v["pf"])
        a_fj = jnp.asarray(ALPHA).T * c[None, :] / v["pf"][:, None]
        return jnp.sum(jnp.asarray(ENDOW), axis=0) - a_fj @ v["y"]

    model.add_equations("zero_profit", zero_profit, complements="y")
    model.add_equations("goods_market", goods_market, complements="p")
    model.add_equations("factor_market", factor_market, complements="pf")
    if fix_numeraire:
        model.fix("pf", 0, numeraire_value)
    return model


def build_model(numeraire_value=1.0):
    return _build(True, numeraire_value)


def build_model_no_numeraire():
    return _build(False)


if __name__ == "__main__":
    from mcp_solver import SolverOptions
    from mcp_solver.semismooth import solve_semismooth

    m = build_model()
    res = solve_semismooth(m.build(), SolverOptions(verbose=True))
    print(res.table())
    for name, val in m.unpack(res.z).items():
        print(f"{name} = {val}")
