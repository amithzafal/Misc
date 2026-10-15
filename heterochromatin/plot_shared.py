"""All dh and ura4 distributions: shared-siRNA 10-repeat model (results/fit_shared.json) vs smFISH."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from repeats_model import simulate, strain

fit = json.load(open("results/fit_shared.json"))
p, v = fit["params"], fit["fitted"]
col = lambda df: df["ch0_spots"] if "ch0_spots" in df else df["spot_count"]
dh = {s: col(pd.read_csv(f"data/{s}.csv")).to_numpy() for s in ["wt", "r406h", "w31g", "epe1d", "clr4d"]}
ura4 = {s: pd.read_csv(f"data/ura4/{s}.csv")["spot_count"].to_numpy() for s in ["wt", "w31g", "epe1d", "clr4d"]}
S = ["wt", "r406h", "w31g", "epe1d", "clr4d"]
sim = {s: simulate(strain(s, p, v["epe1_factor"], v["epe1_writer"]), n_cells=600, seed=3) for s in S}

fig, axs = plt.subplots(2, 5, figsize=(19, 7))
for j, s in enumerate(S):
    for i, (key, d, lab) in enumerate([("dh", dh, "dh"), ("U", ura4, "ura4")]):
        ax, m = axs[i, j], sim[s][key]
        bins = np.arange(-0.5, 41)
        ax.hist(m, bins=bins, density=True, color="0.75", label="model")
        t = f"{s} {lab}: model {m.mean():.2f}"
        if s in d:
            ax.hist(d[s], bins=bins, density=True, histtype="step", lw=1.8, color="C3", label="smFISH")
            t += f" | data {d[s].mean():.2f}"
        ax.set_title(t, fontsize=9)
        ax.set_xlabel(f"{lab} RNA / cell")
axs[0, 0].legend(frameon=False)
fig.suptitle(f"10 dh repeats, shared siRNA pool; ura4 coupled to cen1 repeat.  kappa={v['kappa']:.3g}, "
             f"eps_u={v['eps_u']:.3g};  epe1D: d3 x {v['epe1_factor']:.2f}, writer x {v['epe1_writer']:.2f}", fontsize=10)
fig.tight_layout()
fig.savefig("results/fit_shared.png", dpi=110)
for s in S:
    print(f"{s:6s} dh {sim[s]['dh'].mean():5.2f} (data {dh[s].mean():5.2f}) | ura4 {sim[s]['U'].mean():5.2f}"
          + (f" (data {ura4[s].mean():5.2f}, P<=2 model {np.mean(sim[s]['U'] <= 2):.2f} data {np.mean(ura4[s] <= 2):.2f})" if s in ura4 else ""))
