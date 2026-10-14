# %% [markdown]
# dh RNA / siRNA / H3K9me model (no RNAi-independent nucleation, eps0 = 0),
# Gillespie simulation vs smFISH for each strain. Self-contained notebook cell.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange

# ---- best-fit parameters (fit to smFISH means; eps0 = 0) -------------------
P = dict(
    T=1.141,              # max transcription, min^-1  (= d1 * <R>_clr4D)
    K=0.2, k=4.0,         # repression threshold, Hill coefficient
    gamma=0.06658,        # RNA -> siRNA processing
    S_p=4.921,            # siRNA saturation of processing (molecules)
    nu=1e-5,              # methylation-independent siRNA production, min^-1
    d1=0.05, d2=0.005, d3=0.015,   # RNA, siRNA, H3K9me decay, min^-1
    eps_s=5.246e-05,      # siRNA (RITS)-guided methylation
    eps_m=0.009437,       # read-write (chromodomain) methylation
    S_m=12.9,             # siRNA saturation of methylation (molecules)
    N=30,                 # nucleosomes in the dh locus
)

# strain -> multiplicative changes relative to WT
STRAINS = {
    "WT":        {},
    "clr4R406H": dict(eps_s=0.1, eps_m=0.1),   # catalytic weakening, writing /10
    "clr4W31G":  dict(eps_m=0.0),              # chromodomain: no read-write
    "clr4D":     dict(eps_s=0.0, eps_m=0.0),   # no writer
}

# smFISH files (ch0_spots = dh transcripts per cell)
DATA_DIR = "."
FILES = {
    "WT":        "WTdh_ch0_ch1_spots_per_cell.csv",
    "clr4R406H": "clr4R406Hdh_ch0_ch1_spots_per_cell.csv",
    "clr4W31G":  "clr4W31Gdh_ch0_ch1_spots_per_cell.csv",
    "clr4D":     "clr4D_dh_ch0_ch1_spots_per_cell.csv",
}


# ---- Gillespie SSA ------------------------------------------------------------
# reactions: 1 transcription  R+1   T/(1+(M/K)^k)
#            2 RNA decay      R-1   d1 R
#            3 processing     R-1,S+1  gamma R M S/(1+S/S_p)
#            4 indep. siRNA   S+1   nu R
#            5 siRNA decay    S-1   d2 S
#            6 methylation    n+1   (N-n) R (eps_s + eps_m M) S/(1+S/S_m)
#            7 demethylation  n-1   d3 n                         (M = n/N)
@njit(cache=True)
def _ssa_cell(par, R, S, n, t_sample, seed):
    T, K, k, gamma, S_p, nu, d1, d2, d3, eps_s, eps_m, S_m, Nf = par
    N = int(Nf)
    np.random.seed(seed)
    out = np.empty(t_sample.shape[0])
    a = np.empty(7)
    t, j = 0.0, 0
    while j < t_sample.shape[0]:
        M = n / N
        a[0] = T / (1.0 + (M / K) ** k)
        a[1] = d1 * R
        a[2] = gamma * R * M * S / (1.0 + S / S_p)
        a[3] = nu * R
        a[4] = d2 * S
        a[5] = (N - n) * R * (eps_s + eps_m * M) * S / (1.0 + S / S_m)
        a[6] = d3 * n
        a0 = a.sum()
        t += np.random.exponential(1.0 / a0)
        while j < t_sample.shape[0] and t_sample[j] < t:
            out[j] = R
            j += 1
        r = np.random.random() * a0
        i, c = 0, a[0]
        while r >= c:
            i += 1
            c += a[i]
        if i == 0:
            R += 1
        elif i == 1:
            R -= 1
        elif i == 2:
            R -= 1
            S += 1
        elif i == 3:
            S += 1
        elif i == 4:
            S -= 1
        elif i == 5:
            n += 1
        else:
            n -= 1
    return out


@njit(parallel=True, cache=True)
def _ssa_many(par, n0, t_sample, seeds):
    out = np.empty((seeds.shape[0], t_sample.shape[0]))
    for c in prange(seeds.shape[0]):
        # each cell starts silenced (fully methylated, some siRNA) and relaxes
        out[c] = _ssa_cell(par, 0, 20, n0, t_sample, seeds[c])
    return out


def simulate(p, n_cells=1000, t_burn=1e4, t_sample=np.arange(0, 5000, 500), seed=0):
    """RNA counts, one snapshot per cell per sample time (pooled)."""
    keys = ["T", "K", "k", "gamma", "S_p", "nu", "d1", "d2", "d3", "eps_s", "eps_m", "S_m", "N"]
    par = np.array([float(p[x]) for x in keys])
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, n_cells)
    return _ssa_many(par, int(p["N"]), t_sample + t_burn, seeds).ravel()


def strain_params(name):
    p = dict(P)
    for key, factor in STRAINS[name].items():
        p[key] = P[key] * factor
    return p


# ---- simulate and compare --------------------------------------------------------
sim = {s: simulate(strain_params(s)) for s in STRAINS}
data = {s: pd.read_csv(f"{DATA_DIR}/{f}")["ch0_spots"].to_numpy() for s, f in FILES.items()}

fig, axs = plt.subplots(1, len(STRAINS), figsize=(4 * len(STRAINS), 3.2))
for ax, s in zip(axs, STRAINS):
    d, m = data[s], sim[s]
    xmax = int(np.percentile(np.concatenate([d, m]), 99)) + 2
    bins = np.arange(-0.5, xmax + 1)
    ax.hist(m, bins=bins, density=True, color="0.75", label="model")
    ax.hist(d, bins=bins, density=True, histtype="step", lw=1.8, color="C3", label="smFISH")
    ax.axvline(m.mean(), color="0.35", lw=1.2)
    ax.axvline(d.mean(), color="C3", lw=1.2, ls="--")
    ax.set_title(f"{s}\ndata {d.mean():.2f}  |  model {m.mean():.2f}", fontsize=10)
    ax.set_xlabel("dh RNA / cell")
axs[0].set_ylabel("probability")
axs[0].legend(frameon=False)
fig.tight_layout()
plt.show()

print(pd.DataFrame({s: dict(data_mean=data[s].mean(), model_mean=sim[s].mean(),
                            data_P0=(data[s] == 0).mean(), model_P0=(sim[s] == 0).mean())
                    for s in STRAINS}).T.round(2))
