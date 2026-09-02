#!/usr/bin/env python3
"""End-to-end run. Defaults to synthetic data so you can check the plumbing.

    python run_pipeline.py                    # synthetic, no torch needed
    python run_pipeline.py --train            # also trains (requires torch)
    python run_pipeline.py --no-bf-contrast   # NULL TEST: brightfield carries
                                              # no nuclear signal. Whatever the
                                              # model scores here is chance.

To use real data, replace ``load_data`` with your own loader returning
(brightfield, gfp, cell_labels) as (T, [Z,] H, W) / (T, H, W) / (T, H, W).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pombe_nucseg import (
    LabelGenConfig, DatasetConfig, TrainConfig,
    generate_labels_movie, fit_area_scaling, geometric_baseline,
    per_cell_metrics, compare_methods, build_crops, assign_weights,
    weight_report, split_by_position,
)
from pombe_nucseg.infer import rank_for_annotation

pd.set_option("display.width", 200)


def load_data(args):
    """Replace this with your own loader (tifffile / nd2 / zarr)."""
    from pombe_nucseg.synthetic import synth_movie
    mv = synth_movie(
        n_frames=args.frames, n_cells=args.cells, anaphase_frac=0.25,
        bf_nucleus_contrast=0.0 if args.no_bf_contrast else 0.3,
        seed=args.seed,
    )
    return mv["brightfield"], mv["gfp"], mv["cell_labels"], mv["nucleus_gt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--cells", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bf-contrast", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--pixel-size", type=float, default=0.1)
    args = ap.parse_args()

    bf, gfp, cells, nuc_gt = load_data(args)
    print(f"loaded {bf.shape[0]} frames, {cells.max()} cells in frame 0\n")

    # -- 1. auto labels ----------------------------------------------------
    print("[1] generating labels from GFP")
    lcfg = LabelGenConfig(pixel_size_um=args.pixel_size)
    nuc_labels, table = generate_labels_movie(bf, gfp, cells, lcfg, progress=False)
    print(table["condition"].value_counts().to_string())
    rej = table.loc[table["reject_reason"] != "", "reject_reason"]
    if len(rej):
        print("rejects:", rej.value_counts().to_dict())
    print(f"kept {table['qc_pass'].sum()} / {len(table)} cells\n")

    # -- 2. THE BASELINE TEST (run this before training anything) ----------
    print("[2] geometric baseline vs GFP-derived labels")
    scaling = fit_area_scaling(table)
    print(f"    A_nuc = {scaling['k']:.3f} * A_cell^{scaling['b']:.3f}   "
          f"(R2={scaling['r2']:.3f}, n={scaling['n']})")

    mets = {"baseline_circle": [], "auto_labels_vs_gt": []}
    for t in range(bf.shape[0]):
        tf = table[(table["frame"] == t) & table["qc_pass"]]
        ids = tf["cell_id"].to_numpy()
        base = geometric_baseline(cells[t], scaling)
        m = per_cell_metrics(base, nuc_labels[t], cells[t], args.pixel_size, ids)
        m["frame"], m["position"] = t, "pos0"
        mets["baseline_circle"].append(m)
        # sanity check of the auto labels themselves against synthetic truth
        m2 = per_cell_metrics(nuc_labels[t], nuc_gt[t], cells[t],
                              args.pixel_size, ids)
        m2["frame"], m2["position"] = t, "pos0"
        mets["auto_labels_vs_gt"].append(m2)
    mets = {k: pd.concat(v, ignore_index=True) for k, v in mets.items()}
    print(compare_methods(mets, table).to_string(index=False))
    print("\n    -> your model must beat baseline_circle, per condition.\n")

    # -- 3. crops and weights ---------------------------------------------
    print("[3] building weighted crop dataset")
    dcfg = DatasetConfig()
    X, Y, meta = build_crops(bf, nuc_labels, cells, table, dcfg)
    meta = assign_weights(meta, dcfg)
    print(f"    X={X.shape}  Y={Y.shape}  positive pixel fraction={Y.mean():.4f}")
    print(weight_report(meta).to_string(index=False))

    # -- 4. what to hand-annotate -----------------------------------------
    print("\n[4] top cells to hand-annotate")
    print(rank_for_annotation(table, mets["baseline_circle"], n=8)
          .to_string(index=False))

    # -- 5. train ----------------------------------------------------------
    if args.train:
        print("\n[5] training")
        from pombe_nucseg.train import train_model, evaluate_by_condition
        n = len(meta)
        rs = np.random.default_rng(0)
        val = rs.choice(n, size=max(8, n // 5), replace=False)
        tr = np.setdiff1d(np.arange(n), val)
        # NOTE: on real data split by position, not at random:
        #   tr, val = split_by_position(meta, val_positions=["pos07", "pos08"])
        tcfg = TrainConfig(in_channels=X.shape[1], epochs=args.epochs)
        model, hist = train_model(X, Y, meta, tr, val, tcfg)
        print(evaluate_by_condition(model, X, Y, meta, val).to_string(index=False))
    else:
        print("\n[5] skipped (pass --train)")


if __name__ == "__main__":
    main()
