"""Fit the 10-repeat dh + ura4 model to smFISH means (dh total and ura4).

Stage 1 (dh):   python fit_repeats.py dh
    free: gamma eps_s eps_m S_p S_m epe1_factor ; targets dh WT/R406H/W31G/epe1D
Stage 2 (ura4): python fit_repeats.py ura4 [--n0]
    free: kappa eps_u [N0] ; targets ura4 WT/W31G/epe1D ; dh parameters from stage 1
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from repeats_model import BASE, simulate, strain

stage = sys.argv[1]
use_n0 = "--n0" in sys.argv
CELLS = 150
dh_data = {s: pd.read_csv(f"data/{s}.csv")["ch0_spots"].mean() for s in ["wt", "r406h", "w31g", "clr4d"]}
dh_data["epe1d"] = pd.read_csv("data/epe1d.csv")["spot_count"].mean()
ura4_data = {s: pd.read_csv(f"data/ura4/{s}.csv")["spot_count"].mean() for s in ["wt", "w31g", "epe1d", "clr4d"]}

base = dict(BASE, T=dh_data["clr4d"] * BASE["d1"], T_u=ura4_data["clr4d"] * BASE["d1"])
if stage == "dh":
    free = {"gamma": (-3, 1), "eps_s": (-6, -1), "eps_m": (-4, 0), "S_p": (-1, 2),
            "S_m": (-1, 2), "epe1_factor": (-2, 0)}
    targets = [("dh", s, dh_data[s]) for s in ["wt", "r406h", "w31g", "epe1d"]]
else:
    prev = json.load(open("results/fit_repeats_dh.json"))
    base.update(prev["params"])
    free = {"kappa": (-4, 0), "eps_u": (-4, 0)}
    if use_n0:
        free["N0"] = (0.5, np.log10(30))
    free["epe1_factor"] = (np.log10(prev["epe1_factor"]),) * 2   # fixed from stage 1
    targets = [("U", s, ura4_data[s]) for s in ["wt", "w31g", "epe1d"]]
names = list(free)
lo = np.array([free[k][0] for k in names])
hi = np.array([free[k][1] for k in names])


def params(x):
    v = dict(zip(names, 10 ** np.clip(x, lo, hi)))
    f = v.pop("epe1_factor")
    if "N0" in v:
        v["N0"] = round(v["N0"])
    return dict(base, **v), f


def means(x):
    p, f = params(x)
    out = {}
    for s in sorted({t[1] for t in targets}):
        o = simulate(strain(s, p, f), n_cells=CELLS, seed=1)
        out[s] = (o["dh"].mean(), o["U"].mean())
    return out


def obj(x):
    m = means(x)
    return sum(np.log((m[s][0 if key == "dh" else 1] + 0.5) / (t + 0.5)) ** 2 for key, s, t in targets)


rng = np.random.default_rng(0)
X = lo + (hi - lo) * rng.random((60 if stage == "dh" else 40, len(names)))
F = np.array([obj(x) for x in X])
print("random best", round(F.min(), 4), flush=True)
best = min((minimize(obj, X[i], method="Nelder-Mead", options=dict(maxiter=70, xatol=1e-2, fatol=1e-4))
            for i in np.argsort(F)[:2]), key=lambda r: r.fun)
p, f = params(best.x)
m = means(best.x)
print("score", round(best.fun, 4))
for key, s, t in targets:
    print(f"  {key} {s:6s} data {t:6.2f} model {m[s][0 if key == 'dh' else 1]:6.2f}")
print("  (other signal)", {s: (round(v[0], 2), round(v[1], 2)) for s, v in m.items()})
fitted = {k: p[k] for k in names if k != "epe1_factor"}
print("fitted", fitted, "epe1_factor", round(f, 4))
out = dict(score=best.fun, params={k: p[k] for k in BASE}, epe1_factor=f, fitted=fitted)
json.dump(out, open(f"results/fit_repeats_{stage}{'_n0' if use_n0 else ''}.json", "w"), indent=1)
