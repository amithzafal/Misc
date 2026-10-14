"""2D phase diagram over any two parameters (Gillespie + ODE).

For every grid point it records, for each mutant, the stochastic mean RNA,
the fraction of silent cells (M > K), and the number of stable deterministic
steady states (>1 = bistable). It also scores the point against the target
mean RNA levels (WT 1, R406H 3, clr4D 20, epe1D 3).

Axis spec: name:lo:hi:n[:lin]   (log spacing by default)
'epe1_factor' is accepted as an axis (d3 multiplier in epe1D).

Examples
    python phase_diagram.py                                     # gamma vs eps_m
    python phase_diagram.py --x eps_m:1e-4:0.1:12 --y epe1_factor:0.01:1:12 --calibrate
    python phase_diagram.py --x eps_s:1e-4:0.05:10 --y N:10:100:10:lin --set eps_m=0.01
"""
import argparse
import time

import matplotlib
import numpy as np

from analysis import evaluate_mutants, score, ssa_calibrate_gamma, parse_set
from model import Params, TARGET_MEAN_R

SCAN_MUTANTS = ["wt", "r406h", "epe1d", "w31g"]   # clr4D is analytic: R = T/d1

ap = argparse.ArgumentParser()
ap.add_argument("--x", default="eps_m:1e-4:0.1:12")
ap.add_argument("--y", default="gamma:3e-3:1:12")
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--epe1-factor", type=float, default=0.1)
ap.add_argument("--calibrate", action="store_true",
                help="re-fit gamma at each point so that <R>_WT = 1 (slower)")
ap.add_argument("--cells", type=int, default=200)
ap.add_argument("--out", default="phase")
ap.add_argument("--show", action="store_true", help="open the figure window")
a = ap.parse_args()
if not a.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend chosen above)


def axis(spec):
    f = spec.split(":")
    name, lo, hi, n = f[0], float(f[1]), float(f[2]), int(f[3])
    lin = len(f) > 4 and f[4] == "lin"
    v = np.linspace(lo, hi, n) if lin else np.logspace(np.log10(lo), np.log10(hi), n)
    if name == "N":
        v = np.unique(np.round(v).astype(int))
    return name, v, not lin


xn, xs, xlog = axis(a.x)
yn, ys, ylog = axis(a.y)
base = Params().with_(**parse_set(a.set))
if a.calibrate and "gamma" in (xn, yn):
    raise SystemExit("--calibrate fits gamma; do not also scan it")

shape = (len(ys), len(xs))
R = {m: np.full(shape, np.nan) for m in SCAN_MUTANTS}
sil = {m: np.full(shape, np.nan) for m in SCAN_MUTANTS}
nst = np.zeros(shape, int)
sc = np.full(shape, np.nan)
gam = np.full(shape, np.nan)

t0 = time.time()
for i, yv in enumerate(ys):
    for j, xv in enumerate(xs):
        kw, e1 = {}, a.epe1_factor
        for name, v in ((xn, xv), (yn, yv)):
            if name == "epe1_factor":
                e1 = float(v)
            else:
                kw[name] = int(v) if name == "N" else float(v)
        p = base.with_(**kw)
        if a.calibrate:
            g = ssa_calibrate_gamma(p, n_cells=a.cells, iters=10)
            if g is None:
                continue
            p = p.with_(gamma=g)
        gam[i, j] = p.gamma
        res = evaluate_mutants(p, mutants=SCAN_MUTANTS, epe1_factor=e1, n_cells=a.cells)
        for m in SCAN_MUTANTS:
            R[m][i, j] = res[m]["ssa"]["R_mean"]
            sil[m][i, j] = res[m]["ssa"]["frac_silent"]
        nst[i, j] = res["wt"]["n_stable"]
        res["clr4d"] = {"ssa": {"R_mean": p.T / p.d1}}
        sc[i, j] = score(res)
    print(f"row {i + 1}/{len(ys)} done ({time.time() - t0:.0f}s)", flush=True)

np.savez(a.out + ".npz", x=xs, y=ys, xname=xn, yname=yn, score=sc, gamma=gam,
         n_stable_wt=nst, **{f"R_{m}": R[m] for m in SCAN_MUTANTS},
         **{f"silent_{m}": sil[m] for m in SCAN_MUTANTS})

# ---- plot ----
panels = [(f"<R> {m}" + (f" (target {TARGET_MEAN_R[m]:g})" if TARGET_MEAN_R[m] else ""),
           np.log10(R[m] + 0.05), "log10 <R>", "viridis", (-1.3, 1.35)) for m in SCAN_MUTANTS]
panels.append(("WT fraction silent (M>K)", sil["wt"], "fraction", "magma", (0, 1)))
panels.append(("score: sum sq. log error", np.log10(sc), "log10 score", "RdYlBu", None))
fig, axs = plt.subplots(2, 3, figsize=(14, 8.5))
X, Y = np.meshgrid(xs, ys)
for ax, (title, Z, lab, cmap, lim) in zip(axs.flat, panels):
    pc = ax.pcolormesh(X, Y, Z, shading="nearest", cmap=cmap,
                       vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
    fig.colorbar(pc, ax=ax, label=lab)
    if (nst > 1).any():  # deterministic WT bistability
        ax.contour(X, Y, (nst > 1).astype(float), levels=[0.5], colors="w",
                   linestyles="--", linewidths=1)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(xn)
    ax.set_ylabel(yn)
    ax.set_title(title, fontsize=10)
k = np.unravel_index(np.nanargmin(sc), sc.shape)
for ax in axs.flat:
    ax.plot(xs[k[1]], ys[k[0]], "r*", ms=12)
fig.suptitle(f"base: {base}" +
             ("" if "epe1_factor" in (xn, yn) else f"  epe1_factor={a.epe1_factor}") +
             ("  [gamma calibrated per point]" if a.calibrate else ""), fontsize=8)
fig.tight_layout()
fig.savefig(a.out + ".png", dpi=120)
print(f"best: {xn}={xs[k[1]]:.4g}, {yn}={ys[k[0]]:.4g}"
      + ("" if "gamma" in (xn, yn) else f", gamma={gam[k]:.4g}") + f", score={sc[k]:.3f}")
print(" ".join(f"{m}={R[m][k]:.2f}" for m in SCAN_MUTANTS))
print("saved", a.out + ".png", a.out + ".npz")
if a.show:
    plt.show()
