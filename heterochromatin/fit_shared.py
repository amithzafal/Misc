"""Shared-siRNA 10-repeat model: fit ura4 coupling and the epe1D change.

dh parameters = single-locus eps0=0 fit, rescaled to L repeats sharing one siRNA
pool (eps x L). Free: kappa, eps_u (ura4), epe1_writer w (fraction of writer
available at cen in epe1D), epe1_factor f (d3 multiplier in epe1D).
Targets: ura4 WT, W31G, epe1D means and dh epe1D mean.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from repeats_model import BASE, simulate, strain

single = json.load(open("results/fit_no_eps0.json"))["params"]
L = BASE["L"]
dh_mean = lambda s: pd.read_csv(f"data/{s}.csv").filter(["ch0_spots", "spot_count"]).iloc[:, 0].mean()
ura4_mean = lambda s: pd.read_csv(f"data/ura4/{s}.csv")["spot_count"].mean()
base = dict(BASE, shared=1, T=single["T"], gamma=single["gamma"], S_p=single["S_sat_proc"],
            S_m=single["S_sat_meth"], eps_s=single["eps_s"] * L, eps_m=single["eps_m"] * L,
            T_u=ura4_mean("clr4d") * BASE["d1"])
targets = [("U", "wt", ura4_mean("wt")), ("U", "w31g", ura4_mean("w31g")),
           ("U", "epe1d", ura4_mean("epe1d")), ("dh", "epe1d", dh_mean("epe1d"))]
names = ["kappa", "eps_u", "epe1_writer", "epe1_factor"]
lo, hi = np.log10([1e-3, 1e-4, 0.02, 0.05]), np.log10([1.0, 0.1, 1.0, 1.0])
x0 = np.log10([0.04, 0.003, 0.18, 0.7])
CELLS = 200


def run(x):
    v = dict(zip(names, 10 ** np.clip(x, lo, hi)))
    p = dict(base, kappa=v["kappa"], eps_u=v["eps_u"])
    out = {}
    for s in ["wt", "w31g", "epe1d"]:
        o = simulate(strain(s, p, v["epe1_factor"], v["epe1_writer"]), n_cells=CELLS, seed=1)
        out[s] = dict(dh=o["dh"].mean(), U=o["U"].mean(), Uoff=(o["U"] <= 2).mean())
    return p, v, out


def obj(x):
    _, _, m = run(x)
    return sum(np.log((m[s][k] + 0.5) / (t + 0.5)) ** 2 for k, s, t in targets)


best = minimize(obj, x0, method="Nelder-Mead",
                options=dict(maxiter=90, xatol=1e-2, fatol=1e-4,
                             initial_simplex=x0 + np.vstack([np.zeros(4), 0.25 * np.eye(4)])))
p, v, m = run(best.x)
print("score", round(best.fun, 4))
for k, s, t in targets:
    print(f"  {k:2s} {s:6s} data {t:6.2f} model {m[s][k]:6.2f}")
print("  ura4 P(U<=2):", {s: round(m[s]["Uoff"], 2) for s in m})
print("  dh means:", {s: round(m[s]["dh"], 2) for s in m})
print("fitted", {k: round(x, 5) for k, x in v.items()})
json.dump(dict(score=best.fun, params=p, fitted=v), open("results/fit_shared.json", "w"), indent=1)
