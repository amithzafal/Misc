"""Calibration, mutant evaluation and data comparison built on model.py."""
from __future__ import annotations

import numpy as np

from model import (MUTANTS, TARGET_MEAN_R, Params, mutant_params, silent_state,
                   active_state, fixed_points, simulate, summarize, rna_histogram, initial_state,
                   calibrate_gamma)

# Default SSA settings: burn-in long compared with the slowest rate (1/d2 = 200 min)
SSA_DEFAULTS = dict(n_cells=300, t_burn=1e4, t_sample=np.arange(0, 5000, 500))


def parse_set(items):
    """['eps_m=0.01', 'N=50'] -> dict with the right types."""
    out = {}
    for it in items or []:
        k, v = it.split("=")
        if k not in Params.__dataclass_fields__:
            raise ValueError(f"unknown parameter {k}")
        out[k] = int(float(v)) if k == "N" else float(v)
    return out


def ssa_mean_R(p: Params, init_from: Params | None = None, seed=0, **ssa):
    kw = {**SSA_DEFAULTS, **ssa}
    tr = simulate(p, init_from=init_from, seed=seed, **kw)
    return summarize(tr, p), tr


def ssa_calibrate_gamma(p: Params, target_R=1.0, lo=1e-3, hi=10.0, iters=12,
                        seed=0, **ssa):
    """Bisection in log(gamma) on the stochastic WT mean RNA.
    Common random numbers (fixed seed) keep the objective close to monotone.
    Returns None if even gamma=hi cannot reach target_R."""
    def f(lg):
        s, _ = ssa_mean_R(p.with_(gamma=10 ** lg), seed=seed, **ssa)
        return s["R_mean"] - target_R

    a, b = np.log10(lo), np.log10(hi)
    fa, fb = f(a), f(b)
    if fb > 0:
        return None
    if fa < 0:
        return lo
    for _ in range(iters):
        m = 0.5 * (a + b)
        if f(m) > 0:
            a = m
        else:
            b = m
    return 10 ** (0.5 * (a + b))


def evaluate_mutants(p: Params, mutants=MUTANTS, epe1_factor=0.1, ssa=True,
                     seed=0, **ssa_kw):
    """Return {mutant: dict(ode=..., ssa=..., traj=...)}.
    Every mutant starts from the WT silent steady state (mutation introduced
    in a silenced background), then relaxes to its own stationary state."""
    res = {}
    init = initial_state(p, "silent")
    for m in mutants:
        q = mutant_params(p, m, epe1_factor=epe1_factor)
        fps = fixed_points(q)
        stab = [f for f in fps if f["stable"]]
        entry = dict(params=q, n_stable=len(stab),
                     ode_silent=silent_state(q, fps), ode_active=active_state(q, fps))
        if ssa:
            n0 = int(round(init[2] / p.N * q.N))
            entry["ssa"], entry["traj"] = ssa_mean_R(q, init=(init[0], init[1], n0),
                                                     seed=seed, **ssa_kw)
        res[m] = entry
    return res


def score(res, targets=TARGET_MEAN_R, key="ssa"):
    """Sum of squared log errors of mean RNA vs targets (pseudocount 0.5)."""
    tot = 0.0
    for m, tgt in targets.items():
        if tgt is None or m not in res:
            continue
        if key == "ssa":
            r = res[m]["ssa"]["R_mean"]
        else:
            fp = res[m]["ode_silent"]
            r = fp["R"] if fp else np.nan
        tot += np.log((r + 0.5) / (tgt + 0.5)) ** 2
    return tot


# --------------------------------------------------------------------------
# smFISH data (to be supplied): per-cell RNA counts for each mutant
# --------------------------------------------------------------------------
def load_counts(path, column="ch0_spots"):
    """Per-cell dh RNA counts from a CSV with a header containing `column`
    (e.g. cell_id,ch0_spots,ch1_spots), or a plain file with one count per line."""
    import csv
    with open(path) as fh:
        first = fh.readline()
        fh.seek(0)
        if column in first:
            return np.array([int(float(r[column])) for r in csv.DictReader(fh)], int)
        vals = []
        for line in fh:
            try:
                vals.append(int(round(float(line.strip().split(",")[0]))))
            except ValueError:
                continue
    return np.array(vals, int)


def load_dataset(folder, mutants=MUTANTS):
    """{mutant: counts} for every <mutant>.csv / <mutant>.txt present in folder."""
    import os
    out = {}
    for m in mutants:
        for ext in (".csv", ".txt"):
            f = os.path.join(folder, m + ext)
            if os.path.exists(f):
                out[m] = load_counts(f)
                break
    return out


def log_likelihood(counts, traj, pseudo=1e-4):
    """Log-likelihood per cell of observed smFISH counts under the simulated
    stationary RNA distribution (all cells x samples pooled)."""
    r_max = max(int(counts.max()), int(traj[:, :, 0].max())) + 1
    h = rna_histogram(traj, r_max) + pseudo
    h /= h.sum()
    return np.mean(np.log(h[np.minimum(counts, r_max)]))
