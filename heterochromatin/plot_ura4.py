"""ura4 (and dh) distributions of the coupled model vs smFISH."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ura4_model import DH, URA4, simulate, strain

fit = json.load(open("results/fit_ura4.json"))
p = {**DH, **URA4, "T_u": fit["T_u"], "kappa": fit["fit"]["kappa"], "eps_u": fit["fit"]["eps_u"]}
f = fit["fit"]["epe1_factor"]
S = ["wt", "w31g", "epe1d", "clr4d"]
ura4 = {s: pd.read_csv(f"data/ura4/{s}.csv")["spot_count"].to_numpy() for s in S}
dh = {s: pd.read_csv(f"data/{s}.csv")["ch0_spots"].to_numpy() for s in ["wt", "w31g", "clr4d"]}
sim = {s: simulate(strain(s, p, f), n_cells=1000) for s in S}

fig, axs = plt.subplots(3, 4, figsize=(15, 9.5))
for j, s in enumerate(S):
    for i, (key, d, lab) in enumerate([("U", ura4.get(s), "ura4"), ("R", dh.get(s), "dh")]):
        ax, m = axs[i, j], sim[s][key]
        bins = np.arange(-0.5, 41)
        ax.hist(m, bins=bins, density=True, color="0.75", label="model")
        title = f"{s} {lab}: model {m.mean():.2f}"
        if d is not None:
            ax.hist(d, bins=bins, density=True, histtype="step", lw=1.8, color="C3", label="smFISH")
            title += f" | data {d.mean():.2f}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(f"{lab} RNA / cell")
    ax = axs[2, j]
    ax.scatter(sim[s]["M"][::3] + np.random.uniform(-.01, .01, sim[s]["M"][::3].size),
               sim[s]["U"][::3], s=3, alpha=0.3)
    ax.set_xlabel("dh methylation M (same cell)")
    ax.set_ylabel("ura4 RNA")
    ax.set_xlim(-0.03, 1)
axs[0, 0].legend(frameon=False)
fig.suptitle(f"coupled dh-ura4 model: kappa={p['kappa']:.3g}, eps_u={p['eps_u']:.3g}, "
             f"N_u={p['N_u']}, T_u={p['T_u']:.3g}, epe1 d3 factor={f:.3g}", fontsize=10)
fig.tight_layout()
fig.savefig("results/ura4_fit.png", dpi=110)
for s in S:
    U, M = sim[s]["U"], sim[s]["M"]
    print(f"{s:6s} ura4 model {U.mean():5.2f} P(U<=2) {np.mean(U <= 2):.2f} | data P(<=2) {np.mean(ura4[s] <= 2):.2f}"
          f" | dh model {sim[s]['R'].mean():5.2f}  P(dh M<0.05) {np.mean(M < 0.05):.2f}")
