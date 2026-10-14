"""Fit the model to the *shift* of the smFISH distributions (mean RNA per mutant).

Objective: sum over mutants of (log((<R>_model + 0.5) / (<R>_data + 0.5)))^2.
T is fixed by the clr4D mean (no methylation -> <R> = T/d1), so clr4D is exact
and the remaining mutants (WT, R406H, W31G, and epe1D if a target is given)
constrain the free parameters. The shape of the distributions is not fitted.

Search: random sampling of the log10 box, then Nelder-Mead from the best
points, using common random numbers (fixed seed) to smooth the SSA objective.

Examples
    python fit.py                                     # saturating model
    python fit.py --free gamma eps0 eps_s eps_m       # original mass-action model
    python fit.py --epe1-target 3                     # also use the epe1D estimate
"""
import argparse
import json
import time

import numpy as np
from scipy.optimize import minimize

from analysis import load_dataset, parse_set
from model import Params, mutant_params, simulate

BOUNDS = {  # log10 bounds
    "gamma": (-3, 1), "eps0": (-7, -3), "eps_s": (-5, -1), "eps_m": (-4, 0),
    "S_sat_proc": (-1, 2), "S_sat_meth": (-1, 2.5), "d3": (-3, -1),
}

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data")
ap.add_argument("--free", nargs="*",
                default=["gamma", "eps0", "eps_s", "eps_m", "S_sat_proc", "S_sat_meth"])
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--epe1-target", type=float, help="mean RNA for epe1D (no smFISH yet)")
ap.add_argument("--epe1-factor", type=float, default=0.5)
ap.add_argument("--n-random", type=int, default=300)
ap.add_argument("--n-polish", type=int, default=3)
ap.add_argument("--maxiter", type=int, default=200)
ap.add_argument("--cells", type=int, default=300)
ap.add_argument("--out", default="results/fit.json")
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()

data = load_dataset(a.data_dir)
if "clr4d" not in data:
    raise SystemExit("need clr4d data to fix T")
targets = {m: float(c.mean()) for m, c in data.items() if m != "clr4d"}
if a.epe1_target is not None:
    targets["epe1d"] = a.epe1_target
base = Params().with_(**parse_set(a.set))
base = base.with_(T=float(data["clr4d"].mean() * base.d1))
print("targets (mean RNA):", {m: round(v, 2) for m, v in targets.items()})
print("fixed T =", round(base.T, 4), " free:", a.free)

SSA = dict(n_cells=a.cells, t_burn=1e4, t_sample=np.arange(0, 5000, 500))
lo = np.array([BOUNDS[k][0] for k in a.free])
hi = np.array([BOUNDS[k][1] for k in a.free])


def make(x):
    return base.with_(**{k: 10 ** v for k, v in zip(a.free, x)})


def sim_means(p):
    out = {}
    for m in targets:
        q = mutant_params(p, m, epe1_factor=a.epe1_factor)
        # every strain starts silenced (fully methylated) and relaxes
        tr = simulate(q, init=(0, 20, q.N), seed=a.seed, **SSA)
        out[m] = float(tr[:, :, 0].mean())
    return out


def objective(x):
    means = sim_means(make(np.clip(x, lo, hi)))
    return sum(np.log((means[m] + 0.5) / (t + 0.5)) ** 2 for m, t in targets.items())


t0 = time.time()
rng = np.random.default_rng(a.seed)
X = lo + (hi - lo) * rng.random((a.n_random, len(a.free)))
F = np.array([objective(x) for x in X])
print(f"random search: best {F.min():.4f} ({time.time() - t0:.0f}s)")

best = None
for i in np.argsort(F)[: a.n_polish]:
    simplex = X[i] + np.vstack([np.zeros(len(lo)), 0.3 * np.eye(len(lo))])
    r = minimize(objective, X[i], method="Nelder-Mead",
                 options=dict(maxiter=a.maxiter, xatol=1e-2, fatol=1e-4,
                              initial_simplex=simplex))
    print(f"  polish from {F[i]:.4f} -> {r.fun:.4f}")
    if best is None or r.fun < best.fun:
        best = r

xb = np.clip(best.x, lo, hi)
p = make(xb)
means = sim_means(p)
print(f"\nbest score {best.fun:.4f}  ({time.time() - t0:.0f}s)")
print(f"  {'clr4d':6s} data {data['clr4d'].mean():6.2f}  model {p.T / p.d1:6.2f} (exact)")
for m, t in targets.items():
    print(f"  {m:6s} data {t:6.2f}  model {means[m]:6.2f}")
fitted = {k: float(10 ** v) for k, v in zip(a.free, xb)}
print("fitted:", {k: f"{v:.4g}" for k, v in fitted.items()})
with open(a.out, "w") as fh:
    json.dump(dict(score=best.fun, targets=targets, model_means=means, free=a.free,
                   fitted=fitted, params={k: (None if v == np.inf else v)
                                          for k, v in vars(p).items()}), fh, indent=1)
set_args = " ".join(f"{k}={v:.6g}" for k, v in vars(p).items()
                    if k != "gamma" and v != getattr(Params(), k))
print("saved", a.out)
print(f"plot: python run_mutants.py --set {set_args} --gamma {p.gamma:.6g} "
      f"--epe1-factor {a.epe1_factor} --data-dir {a.data_dir}")
