"""Mean dh RNA per mutant: smFISH vs fitted models (reads results/fit_*.json)."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import load_dataset

fits = sys.argv[1:] or ["results/fit_massaction.json", "results/fit_saturating.json"]
data = load_dataset("data")
order = ["wt", "r406h", "w31g", "clr4d"]
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(order))
ax.plot(x, [data[m].mean() for m in order], "o", color="C3", ms=11, label="smFISH", zorder=3)
for i, f in enumerate(fits):
    r = json.load(open(f))
    T, d1 = r["params"]["T"], r["params"]["d1"]
    y = [r["model_means"].get(m, T / d1 if m == "clr4d" else np.nan) for m in order]
    name = f.split("fit_")[-1].replace(".json", "")
    ax.plot(x + 0.12 * (i + 1), y, "s", ms=8, label=f"{name} (score {r['score']:.2f})")
ax.set_xticks(x + 0.1, order)
ax.set_yscale("log")
ax.set_ylabel("mean dh RNA / cell")
ax.legend(frameon=False)
ax.set_title("Shift of the RNA distribution per mutant", fontsize=10)
fig.tight_layout()
fig.savefig("results/fit_shift.png", dpi=130)
print("saved results/fit_shift.png")
