#!/usr/bin/env python3
"""Dry run of the label pipeline on real (T, H, W) arrays, no GPU needed.

    python dry_run.py --masks masks.npy --gfp gfp.npy --bf bf.npy \
        --pixel-size 0.065 --frames 0:10 --out dryrun

Writes to --out:
    report.txt          everything printed below
    table.csv           per-cell table (all gate columns)
    to_annotate.csv     ranked annotation list
    nuc_labels.npz      auto nucleus labels for the frames used
    panels/*.png        accepted / rejected / borderline cells with overlays
    subset.npz          (--subset-out) the frames used, compressed, for upload

Add --train for a short CPU training + inference plumbing check (needs torch).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

from pombe_nucseg import (
    LabelGenConfig, DatasetConfig, TrainConfig, generate_labels_movie,
    fit_area_scaling, geometric_baseline, per_cell_metrics, compare_methods,
    build_crops, assign_weights, weight_report,
)
from pombe_nucseg.dataset import suggest_manual_multiplier
from pombe_nucseg.infer import rank_for_annotation

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


class Tee:
    def __init__(self, path):
        self.fh = open(path, "w")

    def __call__(self, *a, **k):
        print(*a, **k)
        print(*a, **k, file=self.fh)
        self.fh.flush()


def parse_frames(spec: str, T: int):
    if spec is None:
        return np.arange(T)
    if ":" in spec:
        a, b = spec.split(":")
        a = int(a) if a else 0
        b = int(b) if b else T
        return np.arange(a, min(b, T))
    if "," in spec:
        return np.array([int(x) for x in spec.split(",")])
    return np.arange(min(int(spec), T))


def load_stack(path, name, log):
    arr = np.load(path, mmap_mode="r")
    log(f"{name}: {path}\n    shape={arr.shape} dtype={arr.dtype}")
    return arr


def to_THW(arr, name, T_ref, HW_ref, log, reduce="mid"):
    """Coerce a stack to (T, H, W) [or (T, Z, H, W) for brightfield]."""
    a = np.asarray(arr)
    a = np.squeeze(a)
    if a.ndim == 2:
        a = a[None]
    if a.ndim == 4:
        # find the axis that is neither T nor H, W
        if a.shape[-2:] == HW_ref and a.shape[0] == T_ref:
            z_axis = 1
        elif a.shape[-2:] == HW_ref and a.shape[1] == T_ref:
            a = np.moveaxis(a, 1, 0); z_axis = 1
        else:
            raise ValueError(f"{name}: cannot interpret shape {a.shape} against "
                             f"T={T_ref}, HW={HW_ref}")
        if reduce == "keep":
            log(f"    {name}: {a.shape[z_axis]} planes kept as channels")
            return a
        if reduce == "max":
            a = a.max(axis=z_axis); log(f"    {name}: max-projected over {z_axis}")
        else:
            a = np.take(a, a.shape[z_axis] // 2, axis=z_axis)
            log(f"    {name}: mid-plane taken")
    if a.ndim != 3:
        raise ValueError(f"{name}: expected 3-D after coercion, got {a.shape}")
    if a.shape[-2:] != HW_ref and a.shape[-2:][::-1] == HW_ref:
        a = np.swapaxes(a, -1, -2); log(f"    {name}: transposed H/W")
    if a.shape[-2:] != HW_ref:
        raise ValueError(f"{name}: HW {a.shape[-2:]} != masks {HW_ref}")
    return a


def q(s, ps=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    return s.quantile(ps).round(3).to_dict()


def save_panels(out, bf, gfp, cells, nl, table, n_each=8, size=96, rng=None):
    """Side-by-side BF | GFP | overlay for a few cells of each kind."""
    from skimage.io import imsave
    from skimage.segmentation import find_boundaries
    rng = rng or np.random.default_rng(0)
    os.makedirs(os.path.join(out, "panels"), exist_ok=True)

    def norm(x):
        lo, hi = np.percentile(x, (1, 99.5))
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)

    groups = {
        "accepted_interphase": table[table.qc_pass & (table.condition == "interphase")],
        "accepted_bridge": table[table.qc_pass & (table.condition == "anaphase_bridge")],
        "accepted_two_nuclei": table[table.qc_pass & (table.condition == "two_nuclei")],
        "borderline": table[table.get("split_borderline", False).astype(bool)],
    }
    for reason in table.loc[table.reject_reason != "", "reject_reason"].unique():
        groups[f"rejected_{reason}"] = table[table.reject_reason == reason]

    half = size // 2
    for name, g in groups.items():
        if g.empty:
            continue
        g = g.sample(min(n_each, len(g)), random_state=int(rng.integers(1 << 30)))
        rows = []
        for r in g.itertuples():
            t = int(r.frame); cy, cx = int(round(r.cell_y)), int(round(r.cell_x))
            H, W = cells[t].shape
            y0, x0 = max(0, cy - half), max(0, cx - half)
            y1, x1 = min(H, y0 + size), min(W, x0 + size)
            b = norm(np.asarray(bf[t][y0:y1, x0:x1], dtype=np.float32))
            gg = norm(np.asarray(gfp[t][y0:y1, x0:x1], dtype=np.float32))
            cm = cells[t][y0:y1, x0:x1] == r.cell_id
            nm = nl[t][y0:y1, x0:x1] > 0
            ov = np.stack([b, b, b], -1)
            ov[find_boundaries(cm, mode="inner")] = (1, 1, 0)
            ov[find_boundaries(nm & cm, mode="inner")] = (1, 0, 1)
            gc = np.stack([np.zeros_like(gg), gg, np.zeros_like(gg)], -1)
            gc[find_boundaries(cm, mode="inner")] = (1, 1, 0)
            tile = np.concatenate([np.stack([b] * 3, -1), gc, ov], axis=1)
            pad = np.zeros((size, 3 * size, 3), np.float32)
            pad[:tile.shape[0], :tile.shape[1]] = tile
            rows.append(pad)
        img = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
        imsave(os.path.join(out, "panels", f"{name}.png"), img, check_contrast=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--gfp", required=True)
    ap.add_argument("--bf", required=True)
    ap.add_argument("--pixel-size", type=float, required=True, help="um per pixel")
    ap.add_argument("--frames", default=None, help="'0:10', '0,5,9' or a count")
    ap.add_argument("--gfp-z", default="mid", choices=["mid", "max"])
    ap.add_argument("--bf-z", default="mid", choices=["mid", "max", "keep"])
    ap.add_argument("--out", default="dryrun")
    ap.add_argument("--crop-size", type=int, default=None, help="default: fit longest cell")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--subset-out", default=None, help="save the frames used as .npz")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = Tee(os.path.join(args.out, "report.txt"))
    log(f"pombe_nucseg dry run\n{' '.join(sys.argv)}\n")

    # -- load ----------------------------------------------------------------
    masks_raw = load_stack(args.masks, "masks", log)
    gfp_raw = load_stack(args.gfp, "gfp", log)
    bf_raw = load_stack(args.bf, "brightfield", log)

    m = np.squeeze(np.asarray(masks_raw))
    if m.ndim == 2:
        m = m[None]
    if m.ndim != 3:
        raise ValueError(f"masks must be (T, H, W); got {m.shape}")
    T, H, W = m.shape
    frames = parse_frames(args.frames, T)
    log(f"\nusing frames {frames.tolist()} of {T}; frame size {H}x{W}; "
        f"pixel size {args.pixel_size} um -> field {H * args.pixel_size:.0f} x {W * args.pixel_size:.0f} um")

    cells = m[frames].astype(np.int32)
    gfp = to_THW(gfp_raw, "gfp", T, (H, W), log, reduce=args.gfp_z)[frames].astype(np.float32)
    bf = to_THW(bf_raw, "brightfield", T, (H, W), log, reduce=args.bf_z)[frames].astype(np.float32)
    log(f"cells per frame: {[int(len(np.unique(c)) - 1) for c in cells]}")
    log(f"gfp intensity  : min {gfp.min():.1f}  median {np.median(gfp):.1f}  p99.9 {np.percentile(gfp, 99.9):.1f}")
    log(f"bf intensity   : min {bf.min():.1f}  median {np.median(bf):.1f}  max {bf.max():.1f}")
    if gfp.max() <= 1.0 or gfp.dtype == np.uint8:
        log("WARNING: gfp looks normalised / 8-bit; gates were designed for raw counts")

    if args.subset_out:
        np.savez_compressed(args.subset_out, masks=cells, gfp=gfp_raw[frames] if gfp_raw.ndim == 3 else gfp,
                            bf=bf_raw[frames] if bf_raw.ndim == 3 else bf,
                            frames=frames, pixel_size=args.pixel_size)
        log(f"subset saved: {args.subset_out} ({os.path.getsize(args.subset_out) / 1e6:.1f} MB)")

    # -- labels --------------------------------------------------------------
    log("\n[1] label generation")
    lcfg = LabelGenConfig(pixel_size_um=args.pixel_size)
    sc = lcfg.scaled()
    log(f"    derived: smooth {sc['smooth_sigma_px']:.2f} px, erode {sc['erode_cell_px']} px, "
        f"nucleus area {sc['min_nuc_area_px']:.0f}-{sc['max_nuc_area_px']:.0f} px")
    nl, table = generate_labels_movie(bf, gfp, cells, lcfg, progress=False)
    table.to_csv(os.path.join(args.out, "table.csv"), index=False)
    np.savez_compressed(os.path.join(args.out, "nuc_labels.npz"), nuc_labels=nl, frames=frames)

    log(f"    cells: {len(table)}   qc_pass: {int(table.qc_pass.sum())}")
    log("    condition:\n" + table.condition.value_counts().to_string())
    rej = table.loc[table.reject_reason != "", "reject_reason"].value_counts()
    log("    reject reasons:\n" + (rej.to_string() if len(rej) else "    none"))
    log("\n    cell geometry (all cells):")
    log(f"      length um  {q(table.cell_length_um)}")
    log(f"      width  um  {q(table.cell_width_um)}")
    log(f"      area   um2 {q(table.cell_area_um2)}")
    log("\n    gate columns, all cells vs qc-pass (compare with LabelGenConfig thresholds):")
    for col, thr in [("gfp_snr", lcfg.min_gfp_snr), ("nuc_cyto_ratio", lcfg.min_nuc_cyto_ratio),
                     ("outside_ratio", lcfg.max_outside_ratio), ("focus_score", lcfg.min_focus_score)]:
        log(f"      {col:15s} thr={thr}\n        all : {q(table[col].dropna())}\n"
            f"        pass: {q(table.loc[table.qc_pass, col].dropna())}")
    ok = table[table.qc_pass]
    log("\n    nucleus morphology (qc-pass):")
    log(f"      nuc_area_um2   {q(ok.nuc_area_um2)}")
    log(f"      nuc/cell area  {q(ok.nuc_area_px / ok.cell_area_px)}")
    log(f"      nuc_aspect     {q(ok.nuc_aspect)}   (anaphase_aspect_min={lcfg.anaphase_aspect_min})")
    log(f"      nuc_solidity   {q(ok.nuc_solidity)}   (anaphase_solidity_max={lcfg.anaphase_solidity_max})")
    log(f"      nuc_neck_ratio {q(ok.nuc_neck_ratio)}   (borderline band {lcfg.neck_borderline})")
    hist, edges = np.histogram(ok.nuc_neck_ratio.dropna(), bins=np.linspace(0, 1, 11))
    log("      neck ratio histogram: " + " ".join(f"[{a:.1f}-{b:.1f}]:{n}" for a, b, n in zip(edges[:-1], edges[1:], hist)))
    log(f"      split_borderline: {int(ok.split_borderline.sum())} of {len(ok)}")
    log("      neck ratio by condition:\n" + ok.groupby("condition").nuc_neck_ratio.describe()[["count", "mean", "min", "50%", "max"]].round(2).to_string())

    # -- baseline ------------------------------------------------------------
    log("\n[2] geometric baseline (the model must beat this, per condition)")
    try:
        scaling = fit_area_scaling(table)
        log(f"    A_nuc = {scaling['k']:.3f} * A_cell^{scaling['b']:.3f}   R2={scaling['r2']:.3f}  n={scaling['n']}")
        mets = []
        for i, t in enumerate(range(len(frames))):
            tf = table[(table.frame == t) & table.qc_pass]
            base = geometric_baseline(cells[t], scaling)
            mm = per_cell_metrics(base, nl[t], cells[t], args.pixel_size, tf.cell_id.to_numpy())
            mm["frame"], mm["position"] = t, "pos0"
            mets.append(mm)
        base_m = pd.concat(mets, ignore_index=True)
        log(compare_methods({"baseline_circle": base_m}, table).to_string(index=False))
    except ValueError as e:
        base_m = None
        log(f"    skipped: {e}")

    # -- crops ---------------------------------------------------------------
    log("\n[3] crops and weights")
    longest = float(table.loc[table.qc_pass, "cell_length_um"].max())
    fit = int(np.ceil((longest + 2.0) / args.pixel_size / 32) * 32)
    crop = args.crop_size or fit
    log(f"    longest qc-pass cell {longest:.1f} um -> crop that fits {fit} px; using {crop}")
    dcfg = DatasetConfig(crop_size=(crop, crop))
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        X, Y, meta = build_crops(bf, nl, cells, table, dcfg)
        meta = assign_weights(meta, dcfg)
        rep = weight_report(meta)
    for w in wl:
        log(f"    WARNING: {w.message}")
    log(f"    X={X.shape} Y={Y.shape}  {X.nbytes / 1e6:.0f} MB   positive pixel fraction {Y.mean():.4f}")
    log(f"    clipped cells: {int(meta.cell_clipped.sum())}")
    log(rep.to_string(index=False))
    log(f"    suggest_manual_multiplier (no manual labels yet): {suggest_manual_multiplier(meta, dcfg)}")
    est = len(table) / len(frames)
    log(f"    at {est:.0f} cells/frame, {T} frames -> ~{est * T:,.0f} crops, "
        f"~{est * T * X[0].nbytes / 1e9:.1f} GB at this crop size")

    # -- annotation ranking --------------------------------------------------
    log("\n[4] annotation ranking (top 15; full list in to_annotate.csv)")
    rank = rank_for_annotation(table, base_m, n=500, frame_stride=1)
    rank.to_csv(os.path.join(args.out, "to_annotate.csv"), index=False)
    log(rank.head(15).to_string(index=False))

    # -- panels --------------------------------------------------------------
    save_panels(args.out, bf if bf.ndim == 3 else bf[:, bf.shape[1] // 2], gfp, cells, nl, table)
    log(f"\npanels written to {os.path.join(args.out, 'panels')}  (columns: BF | GFP | BF + cell (yellow) + nucleus (magenta))")

    # -- optional train ------------------------------------------------------
    if args.train:
        log("\n[5] training plumbing check (CPU, few epochs; not a trained model)")
        from pombe_nucseg.train import train_model, evaluate_by_condition
        from pombe_nucseg.infer import predict_frame
        n = len(meta)
        val = meta.index[meta.frame == meta.frame.max()].to_numpy()   # hold out the last frame
        tr = np.setdiff1d(np.arange(n), val)
        tcfg = TrainConfig(in_channels=X.shape[1], epochs=args.epochs, num_workers=0,
                           batch_size=16, base_filters=8, depth=3)
        model, hist = train_model(X, Y, meta, tr, val, tcfg,
                                  out_path=os.path.join(args.out, "unet_dryrun.pt"))
        log(hist.to_string(index=False))
        log(evaluate_by_condition(model, X, Y, meta, val).to_string(index=False))
        t = len(frames) - 1
        pred = predict_frame(model, bf[t], cells[t], table[(table.frame == t)], dcfg)
        log(f"    predict_frame on frame {frames[t]}: {int(len(np.unique(pred)) - 1)} cells labelled")

    log("\ndone")


if __name__ == "__main__":
    main()
