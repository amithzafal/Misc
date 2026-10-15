"""Fit the ura4 coupling (kappa, eps_u) and the epe1D d3 factor to ura4 smFISH means.
dh parameters are fixed at the eps0 = 0 best fit; T_u = d1 * <U>_clr4D."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ura4_model import DH, URA4, simulate, strain

data = {s: pd.read_csv(f"data/ura4/{s}.csv")["spot_count"].to_numpy()
        for s in ["wt", "w31g", "epe1d", "clr4d"]}
targets = {s: data[s].mean() for s in ["wt", "w31g", "epe1d"]}
base = {**DH, **URA4, "T_u": data["clr4d"].mean() * DH["d1"]}
names = ["kappa", "eps_u", "epe1_factor"]
lo, hi = np.array([-4, -4, -2.5]), np.array([0, 0, 0])
CELLS = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def run(x):
    kappa, eps_u, f = 10 ** np.clip(x, lo, hi)
    p = {**base, "kappa": kappa, "eps_u": eps_u}
    return {s: simulate(strain(s, p, f), n_cells=CELLS, seed=1)["U"].mean() for s in targets}


def obj(x):
    m = run(x)
    return sum(np.log((m[s] + 0.5) / (t + 0.5)) ** 2 for s, t in targets.items())


rng = np.random.default_rng(0)
X = lo + (hi - lo) * rng.random((150, 3))
F = np.array([obj(x) for x in X])
print("random best", F.min(), flush=True)
best = min((minimize(obj, X[i], method="Nelder-Mead",
                     options=dict(maxiter=120, xatol=1e-2, fatol=1e-4)) for i in np.argsort(F)[:3]),
           key=lambda r: r.fun)
x = np.clip(best.x, lo, hi)
fit = dict(zip(names, (10 ** x).tolist()))
print("score", best.fun, fit)
print("targets", {s: round(v, 2) for s, v in targets.items()}, "model", {s: round(v, 2) for s, v in run(x).items()})
json.dump(dict(score=best.fun, fit=fit, T_u=base["T_u"]), open("results/fit_ura4.json", "w"), indent=1)
