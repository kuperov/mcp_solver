"""AgCGE on mcp_solver: a single-country agricultural CGE as a square MCP.

Standalone port of the `agcge` package's model (wagland monorepo) to this
repository's PATH solver. The economics is identical to `agcge/model.py`
(v0 full-employment closure): Cobb-Douglas-free CES value added, Leontief
intermediates, CET export transformation with large-country foreign demand,
Armington import composites, LES (Stone-Geary) households, fixed-share
government and savings-driven investment.

The *formulation* follows `agcge/docs/path_squaring_audit.md` exactly: the
over-determined residual system is squared by dropping the VA zero-profit
identity (implied by CES factor demand, audit section 2.1) and the current
account (Walras-redundant, section 2.3), fixing the numeraire factor price,
and pairing equations to variables per the audit's Mathiesen table -- with
the numeraire factor's market clearing paired to the exchange rate `eps`.
The two dropped identities are verified ex post on every solution.

Data: `examples/data/agcge_national_sam.csv` (9 sectors x LAB/CAP/LND,
the agmodel-anchored scaffold SAM) and `agcge_toy_sam.csv` (3x3), copied
verbatim from the agcge package. Elasticities are that package's sourced
GTAP-based values, mapped onto the scaffold sector set.

Run:  .venv/bin/python examples/agcge_national.py
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from mcp_solver import Model, SolverOptions
from mcp_solver.path.solver import solve_path

DATA_DIR = Path(__file__).resolve().parent / "data"
NATIONAL_SAM = DATA_DIR / "agcge_national_sam.csv"
TOY_SAM = DATA_DIR / "agcge_toy_sam.csv"

_TAX = ("PTAX", "MTAX", "DTAX")
_INST = ("HOH", "GOV", "INV", "EXT")

# Per-sector elasticities: (sigma_va, sigma_arm, sigma_cet, eta_export,
# income_elasticity). Values from agcge/national.py's GTAP-based tables
# ("beef" carries the beef_grassfed values). sigma_va must differ from 1
# (the sample uses the pure CES unit-cost formula; 1.0 exactly would need
# the Cobb-Douglas branch).
NATIONAL_ELAS = {
    "grains":   (0.26, 2.6, 2.0, 5.5, 0.40),
    "beef":     (0.24, 4.0, 2.0, 3.0, 0.70),
    "cotton":   (0.26, 5.0, 2.0, 3.5, 0.50),
    "chickens": (0.26, 4.0, 2.0, 8.0, 0.70),
    "sugar":    (0.26, 5.4, 2.0, 3.0, 0.50),
    "pork":     (0.26, 4.0, 2.0, 8.0, 0.70),
    "other_ag": (0.30, 3.0, 2.0, 4.0, 0.70),
    "services": (1.30, 1.9, 2.0, 6.0, 1.15),
    "mfg":      (1.12, 3.8, 2.0, 6.0, 1.05),
}
TOY_ELAS = {
    "AGR": (0.30, 2.6, 2.0, 4.0, 0.60),
    "MFG": (1.12, 3.8, 2.0, 6.0, 1.05),
    "SRV": (1.30, 1.9, 2.0, 6.0, 1.15),
}
FRISCH = -2.5


# ---------------------------------------------------------------------------
# SAM loading (stdlib csv; convention: [row, col] = payment col -> row)
# ---------------------------------------------------------------------------
def load_sam(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    accounts = [c for c in rows[0][1:]]
    mat = np.zeros((len(accounts), len(accounts)))
    for r in rows[1:]:
        if not r or not r[0]:
            continue
        i = accounts.index(r[0])
        mat[i, :] = [float(x) if x else 0.0 for x in r[1:]]
    imbalance = mat.sum(axis=1) - mat.sum(axis=0)
    if np.abs(imbalance).max() > 1e-6:
        bad = {accounts[i]: imbalance[i]
               for i in np.flatnonzero(np.abs(imbalance) > 1e-6)}
        raise ValueError(f"SAM not balanced: {bad}")
    return accounts, mat


# ---------------------------------------------------------------------------
# Calibration (port of agcge/calibrate.py; benchmark prices all = 1)
# ---------------------------------------------------------------------------
@dataclass
class Calibrated:
    sectors: list
    factors: list
    numeraire: str
    # benchmark flows
    Z0: np.ndarray
    Y0: np.ndarray
    E0: np.ndarray
    M0: np.ndarray
    D0: np.ndarray
    Q0: np.ndarray
    Xp0: np.ndarray
    Xg0: np.ndarray
    Xv0: np.ndarray
    F0: np.ndarray          # (nf, n)
    FS0: np.ndarray
    # technology / behaviour
    ax: np.ndarray          # (n, n) intermediate use per unit output
    av: np.ndarray
    b: np.ndarray           # Hicks-neutral VA productivity (1 at benchmark)
    beta: np.ndarray        # (nf, n) VA cost shares
    AM: np.ndarray
    AD: np.ndarray
    cet_delta: np.ndarray
    cet_scale: np.ndarray
    cet_K: np.ndarray
    eta_export: np.ndarray
    lesbeta: np.ndarray
    gamma: np.ndarray
    mu: np.ndarray
    lam: np.ndarray
    tz: np.ndarray
    tm: np.ndarray
    td: float
    pwe: np.ndarray
    pwm: np.ndarray
    ssp: float
    ssg: float
    Sf0: float
    sigma_va: np.ndarray
    sigma_arm: np.ndarray
    sigma_cet: np.ndarray
    YH0: float
    Sp0: float
    Sg0: float
    Td0: float


def calibrate(sam_path, elas_table, numeraire="LAB"):
    accounts, mat = load_sam(sam_path)
    known = set(_TAX) | set(_INST)
    factors = [a for a in accounts if a in ("LAB", "CAP", "LND")]
    sectors = [a for a in accounts if a not in known and a not in factors]
    idx = {a: k for k, a in enumerate(accounts)}
    S = [idx[s] for s in sectors]
    Fc = [idx[f] for f in factors]
    n = len(sectors)

    def block(rows, cols):
        return mat[np.ix_(rows, cols)]

    IO0 = block(S, S)
    F0 = block(Fc, S)
    Xp0 = block(S, [idx["HOH"]]).ravel()
    Xg0 = block(S, [idx["GOV"]]).ravel()
    Xv0 = block(S, [idx["INV"]]).ravel()
    E0 = block(S, [idx["EXT"]]).ravel()
    M0_world = block([idx["EXT"]], S).ravel()
    Tz0 = block([idx["PTAX"]], S).ravel() if "PTAX" in idx else np.zeros(n)
    Tm0 = block([idx["MTAX"]], S).ravel() if "MTAX" in idx else np.zeros(n)
    Td0 = float(mat[idx["DTAX"], idx["HOH"]]) if "DTAX" in idx else 0.0

    M0 = M0_world + Tm0                        # imports at domestic prices
    Z0 = IO0.sum(axis=0) + F0.sum(axis=0) + Tz0
    Y0 = F0.sum(axis=0)
    D0 = Z0 - E0
    Q0 = D0 + M0
    FS0 = F0.sum(axis=1)

    ax = IO0 / Z0[None, :]
    av = Y0 / Z0
    beta = F0 / Y0[None, :]

    sigma_va = np.array([elas_table[s][0] for s in sectors])
    sigma_arm = np.array([elas_table[s][1] for s in sectors])
    sigma_cet = np.array([elas_table[s][2] for s in sectors])
    eta_export = np.array([elas_table[s][3] for s in sectors])
    eta_inc = np.array([elas_table[s][4] for s in sectors])
    if np.any(np.isclose(sigma_va, 1.0)):
        raise ValueError("sample uses the pure CES unit cost; sigma_va must "
                         "differ from 1 (use e.g. 1.001)")

    AM = np.where(Q0 > 0, M0 / Q0, 0.0)
    AD = np.where(Q0 > 0, D0 / Q0, 0.0)

    has_exp = (E0 > 0) & (D0 > 0)
    ed = np.where(has_exp, E0 / np.where(D0 > 0, D0, 1.0), 1.0)
    r = np.where(has_exp, np.power(ed, 1.0 / sigma_cet), 0.0)
    cet_delta = r / (1.0 + r)
    rho_t = (sigma_cet + 1.0) / sigma_cet
    denom = cet_delta * np.power(E0, rho_t) + (1.0 - cet_delta) * np.power(D0, rho_t)
    cet_scale = np.where(denom > 0, Z0 / np.power(np.where(denom > 0, denom, 1.0), 1.0 / rho_t), 1.0)
    cet_K = np.where(cet_delta < 1.0,
                     np.power(cet_delta / (1.0 - cet_delta), sigma_cet), 0.0)

    net_cost = Z0 - Tz0
    tz = np.where(net_cost > 0, Tz0 / np.where(net_cost > 0, net_cost, 1.0), 0.0)
    tm = np.where(M0_world > 0, Tm0 / np.where(M0_world > 0, M0_world, 1.0), 0.0)
    pwe = np.ones(n)
    pwm = 1.0 / (1.0 + tm)

    YH0 = float(F0.sum())
    td = Td0 / YH0 if YH0 > 0 else 0.0
    Sp0 = float(mat[idx["INV"], idx["HOH"]])
    Sg0 = float(mat[idx["INV"], idx["GOV"]])
    Sf0 = float(mat[idx["INV"], idx["EXT"]])
    ssp = Sp0 / YH0 if YH0 > 0 else 0.0
    GovRev0 = Td0 + Tz0.sum() + Tm0.sum()
    ssg = Sg0 / GovRev0 if GovRev0 > 0 else 0.0

    alpha = Xp0 / Xp0.sum()
    mu = Xg0 / Xg0.sum() if Xg0.sum() > 0 else np.zeros(n)
    lam = Xv0 / Xv0.sum() if Xv0.sum() > 0 else np.zeros(n)

    # LES calibration: marginal shares ~ Engel elasticity x budget share,
    # subsistence from the Frisch parameter (benchmark replicates for any
    # frisch because Xp0 = gamma + lesbeta * supernumerary by construction).
    C0 = Xp0.sum()
    raw = eta_inc * (Xp0 / C0)
    lesbeta = raw / raw.sum()
    supernum = -C0 / FRISCH
    gamma = Xp0 - lesbeta * supernum

    return Calibrated(
        sectors=sectors, factors=factors, numeraire=numeraire,
        Z0=Z0, Y0=Y0, E0=E0, M0=M0, D0=D0, Q0=Q0,
        Xp0=Xp0, Xg0=Xg0, Xv0=Xv0, F0=F0, FS0=FS0,
        ax=ax, av=av, b=np.ones(n), beta=beta, AM=AM, AD=AD,
        cet_delta=cet_delta, cet_scale=cet_scale, cet_K=cet_K,
        eta_export=eta_export, lesbeta=lesbeta, gamma=gamma,
        mu=mu, lam=lam, tz=tz, tm=tm, td=td, pwe=pwe, pwm=pwm,
        ssp=ssp, ssg=ssg, Sf0=Sf0,
        sigma_va=sigma_va, sigma_arm=sigma_arm, sigma_cet=sigma_cet,
        YH0=YH0, Sp0=Sp0, Sg0=Sg0, Td0=Td0,
    )


# ---------------------------------------------------------------------------
# Shocks: pure parameter perturbations on the calibrated model
# ---------------------------------------------------------------------------
def apply_shock(cal, tfp=None, export_demand=None, tariff_add=None):
    """Return a shocked copy of `cal`.

    tfp           : {sector: multiplier} on VA productivity b (drought/disease)
    export_demand : {sector: multiplier} on the foreign demand anchor E0
                    (market closure; the large-country curve shifts inward)
    tariff_add    : {sector: percentage points} added to the import tariff tm
    """
    b = cal.b.copy()
    E0 = cal.E0.copy()
    tm = cal.tm.copy()
    for name, mult in (tfp or {}).items():
        b[cal.sectors.index(name)] *= mult
    for name, mult in (export_demand or {}).items():
        E0[cal.sectors.index(name)] *= mult
    for name, add in (tariff_add or {}).items():
        tm[cal.sectors.index(name)] += add
    return replace(cal, b=b, E0=E0, tm=tm)


# ---------------------------------------------------------------------------
# The square MCP (audit sections 3-4)
# ---------------------------------------------------------------------------
def build_model(cal) -> Model:
    n, nf = len(cal.sectors), len(cal.factors)
    inum = cal.factors.index(cal.numeraire)
    j = jnp.asarray  # calibration constants as jax arrays

    ax, av, beta = j(cal.ax), j(cal.av), j(cal.beta)
    bprod, sig = j(cal.b), j(cal.sigma_va)
    Omega, rho_t = j(cal.sigma_cet), j((cal.sigma_cet + 1.0) / cal.sigma_cet)
    num_mask = j(np.arange(nf) == inum)

    def common(v):
        """Shared intermediates (XLA CSE dedupes across equation blocks)."""
        F = v["F"].reshape(nf, n)
        pf_sec = v["pf_sec"].reshape(nf, n)
        E, eps = v["E"], v["eps"][0]
        Epos = jnp.maximum(E, 1e-9)
        pwe = j(cal.pwe) * jnp.power(Epos / j(cal.E0), -1.0 / j(cal.eta_export))
        pe = pwe * eps
        pm = j(cal.pwm) * eps * (1.0 + j(cal.tm))
        c_va = jnp.power(jnp.sum(beta * jnp.power(pf_sec, 1.0 - sig), axis=0),
                         1.0 / (1.0 - sig))
        py = c_va / bprod
        YH = jnp.sum(pf_sec * F)
        netcost = av * py + jnp.sum(ax * v["pq"][:, None], axis=0)
        Tz = j(cal.tz) * netcost * v["Z"]
        Tm = j(cal.tm) * j(cal.pwm) * eps * v["M"]
        GovRev = v["Td"][0] + jnp.sum(Tz) + jnp.sum(Tm)
        return dict(F=F, pf_sec=pf_sec, pwe=pwe, pe=pe, pm=pm, py=py,
                    YH=YH, netcost=netcost, GovRev=GovRev)

    m = Model()
    # -- variables (declaration order = packing order; starts = benchmark) --
    m.add_variables("pf", nf, lb=1e-6, start=1.0)
    m.add_variables("pq", n, lb=1e-6, start=1.0)
    m.add_variables("pd", n, lb=1e-6, start=1.0)
    m.add_variables("pz", n, lb=1e-6, start=1.0)
    m.add_variables("Q", n, lb=0.0, start=cal.Q0)
    m.add_variables("Z", n, lb=0.0, start=cal.Z0)
    m.add_variables("Y", n, lb=0.0, start=cal.Y0)
    m.add_variables("E", n, lb=1e-9, start=np.maximum(cal.E0, 1e-9))
    m.add_variables("M", n, lb=0.0, start=cal.M0)
    m.add_variables("Ds", n, lb=0.0, start=cal.D0)
    m.add_variables("Dd", n, lb=0.0, start=cal.D0)
    m.add_variables("Xp", n, lb=0.0, start=cal.Xp0)
    m.add_variables("Xg", n, lb=0.0, start=cal.Xg0)
    m.add_variables("Xv", n, start=cal.Xv0)              # signed (inventories)
    m.add_variables("F", nf * n, lb=0.0, start=cal.F0.ravel())
    m.add_variables("pf_sec", nf * n, lb=1e-6, start=1.0)
    m.add_variables("eps", 1, lb=1e-6, start=1.0)
    m.add_variables("Td", 1, lb=0.0, start=cal.Td0)
    m.add_variables("Sp", 1, start=cal.Sp0)              # free (deficits)
    m.add_variables("Sg", 1, start=cal.Sg0)
    m.fix("pf", inum, 1.0)                               # numeraire
    # audit section 3 edge case: a sector with no benchmark exports has a
    # degenerate CET FOC; fix E = 0 (its FOC slot becomes the trivial E - 0).
    no_export = cal.E0 <= 0.0
    for k in np.flatnonzero(no_export):
        m.fix("E", int(k), 0.0)

    # -- equations, in the audit's Mathiesen pairing (eq ⟂ variable) --------
    def eq_factor_demand(v):                              # 1 ⟂ F
        c = common(v)
        fdem = beta * (v["Y"] / bprod)[None, :] * jnp.power(
            (c["py"] * bprod)[None, :] / c["pf_sec"], sig[None, :])
        return (c["F"] - fdem).ravel()

    def eq_leontief(v):                                   # 3 ⟂ Y
        return v["Y"] - av * v["Z"]

    def eq_zero_profit_cost(v):                           # 4 ⟂ pz
        return v["pz"] - (1.0 + j(cal.tz)) * common(v)["netcost"]

    def eq_cet_foc(v):                                    # 5 ⟂ E
        c = common(v)
        foc = v["E"] - v["Ds"] * j(cal.cet_K) * jnp.power(c["pe"] / v["pd"], Omega)
        return jnp.where(j(no_export), v["E"], foc)

    def eq_cet_frontier(v):                               # 6 ⟂ Ds
        agg = j(cal.cet_delta) * jnp.power(jnp.maximum(v["E"], 1e-12), rho_t) \
            + (1.0 - j(cal.cet_delta)) * jnp.power(v["Ds"], rho_t)
        return v["Z"] - j(cal.cet_scale) * jnp.power(agg, 1.0 / rho_t)

    def eq_cet_revenue(v):                                # 7 ⟂ Z
        c = common(v)
        return v["pz"] * v["Z"] - (c["pe"] * v["E"] + v["pd"] * v["Ds"])

    def eq_arm_import(v):                                 # 8 ⟂ M
        c = common(v)
        return v["M"] - j(cal.AM) * jnp.power(v["pq"] / c["pm"], j(cal.sigma_arm)) * v["Q"]

    def eq_arm_domestic(v):                               # 9 ⟂ Dd
        return v["Dd"] - j(cal.AD) * jnp.power(v["pq"] / v["pd"], j(cal.sigma_arm)) * v["Q"]

    def eq_arm_value(v):                                  # 10 ⟂ Q
        c = common(v)
        return v["pq"] * v["Q"] - (v["pd"] * v["Dd"] + c["pm"] * v["M"])

    def eq_domestic_clearing(v):                          # 11 ⟂ pd
        return v["Ds"] - v["Dd"]

    def eq_composite_clearing(v):                         # 12 ⟂ pq
        interm = jnp.sum(ax * v["Z"][None, :], axis=1)
        return v["Q"] - (interm + v["Xp"] + v["Xg"] + v["Xv"])

    def eq_household(v):                                  # 13 ⟂ Xp (LES)
        c = common(v)
        Yc = c["YH"] - v["Sp"][0] - v["Td"][0]
        supernum = Yc - jnp.sum(v["pq"] * j(cal.gamma))
        return v["Xp"] - (j(cal.gamma) + j(cal.lesbeta) * supernum / v["pq"])

    def eq_government(v):                                 # 14 ⟂ Xg
        c = common(v)
        return v["Xg"] - j(cal.mu) * (c["GovRev"] - v["Sg"][0]) / v["pq"]

    def eq_investment(v):                                 # 15 ⟂ Xv
        sav = v["Sp"][0] + v["Sg"][0] + v["eps"][0] * cal.Sf0
        return v["Xv"] - j(cal.lam) * sav / v["pq"]

    def eq_mobility(v):                                   # 16a ⟂ pf_sec
        c = common(v)                                     # mobile factors:
        return (c["pf_sec"] - v["pf"][:, None]).ravel()   # price equalisation

    def eq_factor_clearing(v):                            # 16b ⟂ pf (h != num)
        c = common(v)
        clearing = jnp.sum(c["F"], axis=1) - j(cal.FS0)
        # the numeraire's pf is FIXED, so its slot holds the trivially-true
        # pf[num] - 1; its clearing equation pairs with eps below (audit s4).
        return jnp.where(num_mask, v["pf"] - 1.0, clearing)

    def eq_numeraire_clearing(v):                         # 16b (h = num) ⟂ eps
        c = common(v)
        return jnp.sum(c["F"][inum, :])[None] - cal.FS0[inum]

    def eq_direct_tax(v):                                 # 18 ⟂ Td
        return v["Td"] - cal.td * common(v)["YH"]

    def eq_private_savings(v):                            # 19 ⟂ Sp
        return v["Sp"] - cal.ssp * common(v)["YH"]

    def eq_government_savings(v):                         # 20 ⟂ Sg
        return v["Sg"] - cal.ssg * common(v)["GovRev"]

    m.add_equations("factor_demand", eq_factor_demand, complements="F")
    m.add_equations("leontief", eq_leontief, complements="Y")
    m.add_equations("zero_profit_cost", eq_zero_profit_cost, complements="pz")
    m.add_equations("cet_foc", eq_cet_foc, complements="E")
    m.add_equations("cet_frontier", eq_cet_frontier, complements="Ds")
    m.add_equations("cet_revenue", eq_cet_revenue, complements="Z")
    m.add_equations("arm_import", eq_arm_import, complements="M")
    m.add_equations("arm_domestic", eq_arm_domestic, complements="Dd")
    m.add_equations("arm_value", eq_arm_value, complements="Q")
    m.add_equations("domestic_clearing", eq_domestic_clearing, complements="pd")
    m.add_equations("composite_clearing", eq_composite_clearing, complements="pq")
    m.add_equations("household", eq_household, complements="Xp")
    m.add_equations("government", eq_government, complements="Xg")
    m.add_equations("investment", eq_investment, complements="Xv")
    m.add_equations("mobility", eq_mobility, complements="pf_sec")
    m.add_equations("factor_clearing", eq_factor_clearing, complements="pf")
    m.add_equations("numeraire_clearing", eq_numeraire_clearing, complements="eps")
    m.add_equations("direct_tax", eq_direct_tax, complements="Td")
    m.add_equations("private_savings", eq_private_savings, complements="Sp")
    m.add_equations("government_savings", eq_government_savings, complements="Sg")
    return m


# ---------------------------------------------------------------------------
# Solving + verification
# ---------------------------------------------------------------------------
def solve(cal, solver=solve_path, options=None):
    """Solve the CGE; returns (vars_dict, SolveResult). Raises on failure."""
    model = build_model(cal)
    problem = model.build()
    res = solver(problem, options or SolverOptions(max_iter=200))
    if not res.converged:
        raise RuntimeError(f"CGE solve failed: {res.status}, "
                           f"residual={res.residual:.3e}")
    return model.unpack(res.z), res


def dropped_identity_residuals(cal, v):
    """Ex-post check of the two identities the square system drops (audit
    section 6): VA zero profit (eq 2) and the current account (eq 21)."""
    n, nf = len(cal.sectors), len(cal.factors)
    F = v["F"].reshape(nf, n)
    pf_sec = v["pf_sec"].reshape(nf, n)
    c_va = np.power((cal.beta * np.power(pf_sec, 1.0 - cal.sigma_va)).sum(axis=0),
                    1.0 / (1.0 - cal.sigma_va))
    py = c_va / cal.b
    va_zero_profit = py * v["Y"] - (pf_sec * F).sum(axis=0)
    Epos = np.maximum(v["E"], 1e-9)
    pwe = cal.pwe * np.power(Epos / cal.E0, -1.0 / cal.eta_export)
    current_account = (cal.pwm * v["M"]).sum() - ((pwe * v["E"]).sum() + cal.Sf0)
    return float(np.abs(va_zero_profit).max()), float(abs(current_account))


def real_gdp(cal, v):
    """Real GDP at benchmark prices = value added in base prices."""
    return float(cal.av @ v["Z"])


def report(cal, base, shocked, label):
    def pct(a, b):
        return 100.0 * (a - b) / np.where(np.abs(b) > 1e-12, b, 1.0)

    print(f"\n=== {label} ===")
    print(f"{'sector':>10} {'Z %':>8} {'E %':>8} {'M %':>8} {'pq %':>8} {'Xp %':>8}")
    for k, s in enumerate(cal.sectors):
        print(f"{s:>10} {pct(shocked['Z'], base['Z'])[k]:>8.2f} "
              f"{pct(shocked['E'], base['E'])[k]:>8.2f} "
              f"{pct(shocked['M'], base['M'])[k]:>8.2f} "
              f"{pct(shocked['pq'], base['pq'])[k]:>8.2f} "
              f"{pct(shocked['Xp'], base['Xp'])[k]:>8.2f}")
    gdp0, gdp1 = real_gdp(cal, base), real_gdp(cal, shocked)
    print(f"{'real GDP':>10} {100.0 * (gdp1 - gdp0) / gdp0:>8.3f} %   "
          f"eps {float(shocked['eps'][0]):.4f}   "
          f"factor returns {np.round(shocked['pf'], 4)}")


if __name__ == "__main__":
    cal = calibrate(NATIONAL_SAM, NATIONAL_ELAS)
    print(f"model: {len(cal.sectors)} sectors x {len(cal.factors)} factors")
    base_vars, base_res = solve(cal)
    dev = float(np.abs(np.concatenate([
        base_vars["Z"] - cal.Z0, base_vars["pq"] - 1.0])).max())
    va_r, ca_r = dropped_identity_residuals(cal, base_vars)
    print(f"benchmark replication: max deviation {dev:.2e} "
          f"({len(base_res.iterations)} iterations, PATH)")
    print(f"dropped identities at solution: VA zero-profit {va_r:.2e}, "
          f"current account {ca_r:.2e}")

    # FMD-style incursion: beef & pork export markets close by half,
    # livestock productivity takes a 5% hit.
    fmd = apply_shock(cal,
                      tfp={"beef": 0.95, "pork": 0.95},
                      export_demand={"beef": 0.5, "pork": 0.5})
    fmd_vars, fmd_res = solve(fmd)
    va_r, ca_r = dropped_identity_residuals(fmd, fmd_vars)
    report(cal, base_vars, fmd_vars,
           f"FMD-style shock (beef/pork market closure + 5% tfp cut) — "
           f"{len(fmd_res.iterations)} PATH iterations, identities "
           f"{max(va_r, ca_r):.1e}")

    # Trade-policy shock: +10pp tariff on manufactures imports.
    tar = apply_shock(cal, tariff_add={"mfg": 0.10})
    tar_vars, tar_res = solve(tar)
    report(cal, base_vars, tar_vars,
           f"+10pp tariff on mfg imports — {len(tar_res.iterations)} "
           f"PATH iterations")
