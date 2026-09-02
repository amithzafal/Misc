"""Validation utilities.

The critical function here is :func:`geometric_baseline`. In S. pombe the
interphase nucleus is actively centred by microtubule pushing forces and its
volume scales with cell volume, so "a circle at the cell centroid, sized from
the cell" is a strong predictor built entirely from the cell mask you already
have. Any learned model must clearly beat it, in the conditions you care
about, or it is not extracting nuclear information from brightfield.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops

__all__ = [
    "fit_area_scaling",
    "geometric_baseline",
    "per_cell_metrics",
    "stratified_report",
    "compare_methods",
]


# --------------------------------------------------------------------------
def fit_area_scaling(table: pd.DataFrame,
                     conditions: Tuple[str, ...] = ("interphase",)) -> Dict[str, float]:
    """Fit nuclear vs cell cross-sectional area: A_nuc = k * A_cell^b.

    Fit on interphase only - anaphase nuclei are mid-division and off-relation.
    Returns k, b, the simple ratio, and R^2 of the log-log fit.
    """
    d = table[table["qc_pass"] & table["condition"].isin(conditions)]
    d = d[(d["nuc_area_px"] > 0) & (d["cell_area_px"] > 0)]
    if len(d) < 10:
        raise ValueError(f"only {len(d)} cells available for the scaling fit")

    x = np.log(d["cell_area_px"].to_numpy())
    y = np.log(d["nuc_area_px"].to_numpy())
    b, loga = np.polyfit(x, y, 1)
    resid = y - (b * x + loga)
    r2 = 1.0 - np.var(resid) / np.var(y) if np.var(y) > 0 else 0.0
    return {
        "k": float(np.exp(loga)),
        "b": float(b),
        "ratio": float(np.mean(d["nuc_area_px"] / d["cell_area_px"])),
        "r2": float(r2),
        "n": int(len(d)),
    }


def geometric_baseline(
    cell_labels: np.ndarray,
    scaling: Dict[str, float],
    oracle_areas: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Circle at each cell centroid, sized from the cell (or from an oracle).

    ``oracle_areas`` maps cell_id -> true nuclear area in pixels. Supplying it
    isolates how much of the baseline's error is position rather than size; the
    gap between the two baselines is the most your model can win on sizing.
    """
    out = np.zeros_like(cell_labels, dtype=np.int32)
    yy, xx = np.mgrid[0:cell_labels.shape[0], 0:cell_labels.shape[1]]
    for prop in regionprops(cell_labels):
        cid = prop.label
        if oracle_areas is not None:
            area = oracle_areas.get(cid, np.nan)
            if not np.isfinite(area) or area <= 0:
                continue
        else:
            area = scaling["k"] * prop.area ** scaling["b"]
        r = np.sqrt(area / np.pi)
        cy, cx = prop.centroid
        circ = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        out[circ & (cell_labels == cid)] = cid
    return out


# --------------------------------------------------------------------------
def per_cell_metrics(
    pred_nuc: np.ndarray,
    gt_nuc: np.ndarray,
    cell_labels: np.ndarray,
    pixel_size_um: float = 0.1,
    cell_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """IoU / Dice / centroid offset / area ratio, per cell.

    Both label images are reduced to a binary mask *within each cell*, which is
    the right comparison when the cell mask already separates instances.
    """
    rows = []
    ids = cell_ids if cell_ids is not None else np.unique(cell_labels)
    for cid in ids:
        if cid == 0:
            continue
        cell = cell_labels == cid
        if not cell.any():
            continue
        p = (pred_nuc > 0) & cell
        g = (gt_nuc > 0) & cell
        inter = float(np.logical_and(p, g).sum())
        union = float(np.logical_or(p, g).sum())
        psum, gsum = float(p.sum()), float(g.sum())

        if psum > 0 and gsum > 0:
            pc = ndi.center_of_mass(p)
            gc = ndi.center_of_mass(g)
            dist = float(np.hypot(pc[0] - gc[0], pc[1] - gc[1]) * pixel_size_um)
        else:
            dist = np.nan

        rows.append(dict(
            cell_id=int(cid),
            iou=inter / union if union > 0 else np.nan,
            dice=2 * inter / (psum + gsum) if (psum + gsum) > 0 else np.nan,
            centroid_dist_um=dist,
            pred_area_px=psum,
            gt_area_px=gsum,
            area_ratio=psum / gsum if gsum > 0 else np.nan,
            detected=psum > 0,
            has_gt=gsum > 0,
        ))
    return pd.DataFrame(rows)


def stratified_report(metrics: pd.DataFrame, table: pd.DataFrame,
                      by: str = "condition") -> pd.DataFrame:
    """Join metrics to the per-cell table and summarise by condition.

    Read the anaphase rows first. A headline IoU dominated by interphase can
    look healthy while the frames you actually care about are failing.
    """
    keys = [c for c in ("position", "frame", "cell_id") if c in table.columns]
    m = metrics.copy()
    if "position" in table.columns and "position" not in m.columns:
        for k in keys[:-1]:
            m[k] = table[k].iloc[0]
    merged = m.merge(table[keys + [by]], on=keys, how="left")
    grp = merged.groupby(by)
    out = grp.agg(
        n=("iou", "size"),
        iou_mean=("iou", "mean"),
        iou_median=("iou", "median"),
        dice_mean=("dice", "mean"),
        centroid_um_median=("centroid_dist_um", "median"),
        area_ratio_median=("area_ratio", "median"),
        detect_rate=("detected", "mean"),
    ).round(3)
    return out.reset_index()


def compare_methods(methods: Dict[str, pd.DataFrame], table: pd.DataFrame,
                    by: str = "condition") -> pd.DataFrame:
    """Stack :func:`stratified_report` output for several prediction sets."""
    frames = []
    for name, met in methods.items():
        rep = stratified_report(met, table, by=by)
        rep.insert(0, "method", name)
        frames.append(rep)
    return pd.concat(frames, ignore_index=True)
