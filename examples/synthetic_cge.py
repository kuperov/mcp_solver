"""Scalable synthetic pure-exchange CES economies.

H households with CES preferences (common elasticity sigma >= 1 ->
gross substitutes -> unique equilibrium) and random endowments,
calibrated so p = 1 is *near* (not at) the equilibrium. MCP form:
excess supply(p) >= 0  ⟂  p >= 0, numeraire p[0] = 1.
"""
import jax.numpy as jnp
import numpy as np

from mcp_solver.model import Model


def make_exchange_economy(n_goods, n_households=None, sigma=1.2, seed=0):
    H = n_households or max(4, n_goods // 5)
    rng = np.random.default_rng(seed)
    endow = rng.uniform(0.5, 1.5, (H, n_goods))
    shares = rng.uniform(0.5, 1.5, (H, n_goods))
    shares /= shares.sum(axis=1, keepdims=True)
    e_j = jnp.asarray(endow)
    a_j = jnp.asarray(shares)
    total_supply = jnp.sum(e_j, axis=0)

    def excess_supply(v):
        p = v["p"]
        m = e_j @ p                                   # incomes (H,)
        weights = a_j**sigma * p[None, :] ** (1.0 - sigma)  # (H, n)
        denom = jnp.sum(weights, axis=1)              # price indices (H,)
        demand = jnp.sum((weights / p[None, :]) * (m / denom)[:, None],
                         axis=0)
        return total_supply - demand

    model = Model()
    model.add_variables("p", n_goods, lb=0.0, start=1.0)
    model.add_equations("markets", excess_supply, complements="p")
    model.fix("p", 0, 1.0)
    return model


if __name__ == "__main__":
    from mcp_solver import SolverOptions
    from mcp_solver.semismooth import solve_semismooth

    m = make_exchange_economy(200, seed=1)
    res = solve_semismooth(m.build(), SolverOptions(verbose=True))
    print(res.table())
