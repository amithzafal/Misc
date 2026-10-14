"""Calibrate gamma to WT <R> = 1 and compare all mutants (ODE vs Gillespie).

Examples
    python run_mutants.py                         # defaults, SSA calibration of gamma
    python run_mutants.py --set eps_m=0.01 N=50   # change any parameter
    python run_mutants.py --gamma 0.05            # skip calibration
    python run_mutants.py --data-dir smfish/      # overlay smFISH (wt.txt, r406h.txt, ...)
"""
import argparse
import os

import matplotlib
import numpy as np

from analysis import (evaluate_mutants, score, ssa_calibrate_gamma, parse_set,
                      load_counts, log_likelihood)
from model import MUTANTS, TARGET_MEAN_R, Params, calibrate_gamma, rna_histogram

ap = argparse.ArgumentParser()
ap.add_argument("--set", nargs="*", default=[], help="param=value overrides")
ap.add_argument("--gamma", type=float, help="fix gamma instead of calibrating")
ap.add_argument("--calib", choices=["ssa", "ode"], default="ssa")
ap.add_argument("--epe1-factor", type=float, default=0.1, help="d3 multiplier in epe1D")
ap.add_argument("--cells", type=int, default=1000)
ap.add_argument("--data-dir", help="folder with <mutant>.txt per-cell counts")
ap.add_argument("--out", default="mutants.png")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--show", action="store_true", help="open the figure window")
a = ap.parse_args()
if not a.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend chosen above)

p = Params().with_(**parse_set(a.set))
if a.gamma is not None:
    g = a.gamma
elif a.calib == "ode":
    g = calibrate_gamma(p)
else:
    g = ssa_calibrate_gamma(p, seed=a.seed, n_cells=a.cells)
if g is None:
    raise SystemExit("No gamma reaches <R>_WT = 1 for these parameters.")
p = p.with_(gamma=g)
print("Parameters:", p)

res = evaluate_mutants(p, epe1_factor=a.epe1_factor, seed=a.seed, n_cells=a.cells)

hdr = f"{'mutant':7s} {'target':>6s} | {'ODE R':>7s} {'ODE S':>7s} {'ODE M':>6s} {'#stab':>5s} | " \
      f"{'SSA R':>6s} {'Fano':>5s} {'P(R=0)':>6s} {'SSA S':>6s} {'SSA M':>6s} {'silent':>6s}"
print(hdr)
print("-" * len(hdr))
for m in MUTANTS:
    e = res[m]
    o, s = e["ode_silent"], e["ssa"]
    tgt = TARGET_MEAN_R[m]
    print(f"{m:7s} {('%.1f' % tgt) if tgt is not None else '-':>6s} | "
          f"{o['R']:7.2f} {o['S']:7.2f} {o['M']:6.3f} {e['n_stable']:5d} | "
          f"{s['R_mean']:6.2f} {s['R_fano']:5.2f} {s['frac_R0']:6.2f} "
          f"{s['S_mean']:6.2f} {s['M_mean']:6.3f} {s['frac_silent']:6.2f}")
print(f"score (sum sq. log error vs targets): SSA {score(res):.3f}   ODE {score(res, key='ode'):.3f}")

fig, axs = plt.subplots(1, len(MUTANTS), figsize=(3.2 * len(MUTANTS), 3), sharey=True)
for ax, m in zip(axs, MUTANTS):
    tr = res[m]["traj"]
    h = rna_histogram(tr, r_max=max(40, int(tr[:, :, 0].max())))
    ax.bar(np.arange(h.size), h, width=1, color="0.6", label="SSA")
    title = f"{m}  <R>={res[m]['ssa']['R_mean']:.2f}"
    if a.data_dir:
        f = os.path.join(a.data_dir, f"{m}.txt")
        if os.path.exists(f):
            c = load_counts(f)
            hd = np.bincount(c, minlength=h.size)[: h.size] / c.size
            ax.step(np.arange(hd.size), hd, where="mid", color="C3", label="smFISH")
            ll = log_likelihood(c, tr)
            print(f"{m}: data mean {c.mean():.2f} (n={c.size}), log-lik/cell {ll:.3f}")
            title += f"\ndata={c.mean():.2f}"
    if TARGET_MEAN_R[m] is not None:
        ax.axvline(TARGET_MEAN_R[m], color="k", ls="--", lw=1, label="target mean")
    ax.axvline(res[m]["ssa"]["R_mean"], color="0.3", lw=1.5, label="SSA mean")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("dh RNA / cell")
    ax.set_xlim(-0.5, 40)
axs[0].set_ylabel("probability")
axs[0].legend(fontsize=8)
fig.suptitle(", ".join(f"{k}={v:.3g}" for k, v in vars(p).items()), fontsize=7)
fig.tight_layout()
fig.savefig(a.out, dpi=130)
print("saved", a.out)
if a.show:
    plt.show()
