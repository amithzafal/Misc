"""Turn frames + labels into a weighted training set of per-cell crops.

Each sample is one cell, cropped and rotated so the long axis is horizontal.
Working inside the cell crop removes the instance-separation problem entirely:
the task becomes binary semantic segmentation of (usually) one blob.

Sample weights combine a per-condition weight with a multiplier for hand-drawn
labels, so manually corrected anaphase cells carry the influence you want
without you having to annotate thousands of them.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops

from .config import DatasetConfig

__all__ = [
    "extract_cell_crop",
    "build_crops",
    "assign_weights",
    "weight_report",
    "suggest_manual_multiplier",
    "apply_manual_overrides",
    "save_dataset",
    "load_dataset",
    "split_by_position",
]


# --------------------------------------------------------------------------
def _pad_slice(img: np.ndarray, cy: float, cx: float, half: int,
               order: int = 1) -> np.ndarray:
    """Square window centred on (cy, cx), reflect-padded at frame edges."""
    H, W = img.shape[-2:]
    y0, y1 = int(round(cy)) - half, int(round(cy)) + half
    x0, x1 = int(round(cx)) - half, int(round(cx)) + half
    py0, px0 = max(0, -y0), max(0, -x0)
    py1, px1 = max(0, y1 - H), max(0, x1 - W)
    sub = img[..., max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if py0 or py1 or px0 or px1:
        pad = [(0, 0)] * (sub.ndim - 2) + [(py0, py1), (px0, px1)]
        mode = "reflect" if order > 0 else "constant"
        sub = np.pad(sub, pad, mode=mode)
    return sub


def extract_cell_crop(
    channels: np.ndarray,
    masks: Sequence[np.ndarray],
    cy: float,
    cx: float,
    orientation: float,
    cfg: DatasetConfig,
) -> Tuple[np.ndarray, list]:
    """Crop and rotate one cell.

    Parameters
    ----------
    channels : (C, H, W) float array - brightfield planes (and anything else).
    masks    : sequence of (H, W) binary/int arrays, resampled nearest-neighbour.
    orientation : ``regionprops.orientation`` of the cell, in radians.

    Returns rotated, centre-cropped ``(C, h, w)`` channels and list of masks.
    """
    h, w = cfg.crop_size
    half = int(np.ceil(np.hypot(h, w) / 2)) + 4

    ch = _pad_slice(channels, cy, cx, half, order=1).astype(np.float32)
    mk = [_pad_slice(m.astype(np.float32), cy, cx, half, order=0) for m in masks]

    if cfg.align_major_axis:
        # regionprops measures orientation from the row axis; this angle puts
        # the major axis on the horizontal.
        ang = 90.0 - np.degrees(orientation)
        ch = ndi.rotate(ch, ang, axes=(-2, -1), reshape=False, order=1,
                        mode="reflect")
        mk = [ndi.rotate(m, ang, reshape=False, order=0, mode="constant")
              for m in mk]

    cy0 = ch.shape[-2] // 2
    cx0 = ch.shape[-1] // 2
    sy = slice(cy0 - h // 2, cy0 - h // 2 + h)
    sx = slice(cx0 - w // 2, cx0 - w // 2 + w)
    return ch[..., sy, sx], [m[sy, sx] for m in mk]


def _normalize(x: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    if cfg.normalize == "none":
        return x
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        v = x[c]
        if cfg.normalize == "zscore":
            s = v.std()
            out[c] = (v - v.mean()) / (s if s > 1e-6 else 1.0)
        else:
            lo, hi = np.percentile(v, cfg.norm_percentiles)
            out[c] = (v - lo) / (hi - lo if hi - lo > 1e-6 else 1.0)
    return out


# --------------------------------------------------------------------------
def build_crops(
    brightfield,
    nucleus_labels,
    cell_labels,
    table: pd.DataFrame,
    cfg: DatasetConfig,
    only_qc_pass: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build the crop tensors for one position.

    ``brightfield`` may be (T, H, W) or (T, Z, H, W); z-planes become channels,
    which is worth doing - transmitted-light nuclear contrast in pombe is
    strongly defocus-dependent and near zero at true focus.

    Returns
    -------
    X : (N, C, h, w) float32     C = n_z (+1 if the cell mask channel is on)
    Y : (N, 1, h, w) float32     binary nucleus target
    meta : DataFrame, one row per crop, aligned to X/Y
    """
    bf = np.asarray(brightfield)
    if bf.ndim == 3:                 # (T, H, W) -> (T, 1, H, W)
        bf = bf[:, None]
    T = bf.shape[0]

    rows = table[table["qc_pass"]] if only_qc_pass else table
    Xs, Ys, metas = [], [], []

    for t in range(T):
        sub = rows[rows["frame"] == t]
        if sub.empty:
            continue
        cl = np.asarray(cell_labels[t])
        nl = np.asarray(nucleus_labels[t])
        planes = bf[t].astype(np.float32)

        for r in sub.itertuples():
            cell_mask = (cl == r.cell_id)
            nuc_mask = (nl > 0) & cell_mask
            ch, (cm, nm) = extract_cell_crop(
                planes, [cell_mask, nuc_mask],
                r.cell_y, r.cell_x, r.cell_orientation, cfg,
            )
            ch = _normalize(ch, cfg)
            if cfg.include_cell_mask_channel:
                ch = np.concatenate([ch, cm[None].astype(np.float32)], axis=0)
            cmb = cm > 0.5
            clipped = bool(cmb[0].any() or cmb[-1].any()
                           or cmb[:, 0].any() or cmb[:, -1].any())
            Xs.append(ch)
            Ys.append((nm > 0.5).astype(np.float32)[None])
            metas.append({
                "position": r.position, "frame": r.frame, "cell_id": r.cell_id,
                "condition": r.condition, "source": r.source,
                "split_borderline": bool(getattr(r, "split_borderline", False)),
                "cell_area_um2": r.cell_area_um2,
                "nuc_area_um2": r.nuc_area_um2,
                "cell_clipped": clipped,
            })

    if not Xs:
        raise ValueError("no crops produced - check qc_pass and frame indices")
    meta = pd.DataFrame(metas)
    n_clip = int(meta["cell_clipped"].sum())
    if n_clip:
        # A cell that does not fit the crop loses its tips - and in anaphase
        # the daughter nuclei are at the tips. Those pixels are silently
        # absent from the label and can never be predicted at inference.
        max_len = float(table.loc[table["qc_pass"], "cell_length_um"].max()) \
            if "cell_length_um" in table.columns else float("nan")
        warnings.warn(
            f"{n_clip}/{len(meta)} cells touch the crop border (crop_size="
            f"{cfg.crop_size}, longest qc-pass cell {max_len:.1f} um). "
            "Increase DatasetConfig.crop_size so crop_size * pixel_size >= "
            "longest cell + 2 * crop_pad_um, or drop these cells "
            "(meta['cell_clipped']).", stacklevel=2)
    return (np.stack(Xs).astype(np.float32),
            np.stack(Ys).astype(np.float32),
            meta)


# --------------------------------------------------------------------------
def assign_weights(meta: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Per-sample training weight = condition weight x manual multiplier.

    Also emits ``effective_n``: the weighted share of the dataset each group
    occupies. Check it. If manual anaphase crops end up at 60% of effective
    exposure the model will overfit them and regress on interphase.
    """
    m = meta.copy()
    base = m["condition"].map(cfg.condition_weights).fillna(1.0)
    manual = m["source"].eq("manual")
    mult = np.where(manual, cfg.manual_weight_multiplier, 1.0)
    if "split_borderline" in m.columns:
        # auto labels whose bridge/two-nuclei call is a thresholding accident
        mult = np.where(m["split_borderline"].astype(bool) & ~manual,
                        cfg.borderline_auto_weight, mult)
    m["weight"] = np.minimum(base * mult, cfg.max_sample_weight).astype(float)

    total = m["weight"].sum()
    share = m.groupby(["condition", "source"])["weight"].sum() / total
    m.attrs["effective_share"] = share.round(4).to_dict()
    return m


def weight_report(meta: pd.DataFrame, max_draws_per_sample: float = 3.0,
                  max_share_ratio: float = 5.0) -> pd.DataFrame:
    """Raw counts vs weighted exposure, per condition and label source.

    ``draws_per_sample`` is how many times one crop of that group is expected
    to be drawn per epoch under ``weighting_mode="sampler"`` (and equally its
    relative loss weight under ``"loss"``). Groups with more than
    ``max_draws_per_sample``, or whose weighted share exceeds their count share
    by more than ``max_share_ratio``, get ``flag=True`` and a warning: the
    model will memorise those crops rather than learn from them.
    """
    g = meta.groupby(["condition", "source"])
    out = g.agg(n=("weight", "size"), weight_sum=("weight", "sum"),
                w_max=("weight", "max"))
    N, Wtot = len(meta), float(meta["weight"].sum())
    out["count_share"] = (out["n"] / N).round(4)
    out["weighted_share"] = (out["weight_sum"] / Wtot).round(4)
    out["draws_per_sample"] = (N * out["w_max"] / Wtot).round(2)
    out["flag"] = ((out["draws_per_sample"] > max_draws_per_sample)
                   | (out["weighted_share"] > max_share_ratio * out["count_share"]))
    out = out.drop(columns="w_max").reset_index()
    bad = out[out["flag"]]
    if len(bad):
        warnings.warn(
            "over-weighted groups (each crop drawn "
            + ", ".join(f"{r.condition}/{r.source}: {r.draws_per_sample:.1f}x per epoch"
                        for r in bad.itertuples())
            + "). Lower manual_weight_multiplier / condition_weights or annotate "
            "more cells; see suggest_manual_multiplier().", stacklevel=2)
    return out


def suggest_manual_multiplier(meta: pd.DataFrame, cfg: DatasetConfig,
                              target_manual_share: float = 0.15,
                              max_draws_per_sample: float = 3.0) -> Dict[str, float]:
    """Solve for ``manual_weight_multiplier`` instead of guessing it.

    Picks the multiplier that gives manual crops ``target_manual_share`` of the
    weighted mass, then lowers it if any single crop would be drawn more than
    ``max_draws_per_sample`` times per epoch. With few annotations the second
    cap binds and the manual share ends up below target - that is the honest
    answer: you cannot buy influence for two crops without memorising them,
    only more annotation raises the share safely.
    """
    base = meta["condition"].map(cfg.condition_weights).fillna(1.0).to_numpy()
    manual = meta["source"].eq("manual").to_numpy()
    if not manual.any():
        return {"multiplier": cfg.manual_weight_multiplier, "manual_share": 0.0,
                "max_draws": 0.0, "n_manual": 0}
    w_auto = float(base[~manual].sum())
    w_man_base = float(base[manual].sum())
    s = target_manual_share
    mult = s / (1.0 - s) * w_auto / w_man_base
    # cap: draws of the heaviest manual crop = N * base_max * mult / total
    N = len(meta)
    bmax = float(base[manual].max())
    for _ in range(50):
        total = w_auto + mult * w_man_base
        draws = N * bmax * mult / total
        if draws <= max_draws_per_sample:
            break
        mult *= 0.9
    total = w_auto + mult * w_man_base
    return {"multiplier": float(mult),
            "manual_share": float(mult * w_man_base / total),
            "max_draws": float(N * bmax * mult / total),
            "n_manual": int(manual.sum())}


# --------------------------------------------------------------------------
def apply_manual_overrides(
    nucleus_labels: np.ndarray,
    table: pd.DataFrame,
    manual_labels: Dict[Tuple[str, int], np.ndarray],
    cell_labels: Optional[np.ndarray] = None,
    conditions: Optional[Dict[Tuple[str, int, int], str]] = None,
    position: Optional[str] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Replace auto labels with hand-drawn ones where they exist.

    ``manual_labels`` maps ``(position, frame) -> label image`` keyed by cell
    ID (0 = not annotated). Draw these in napari / QuPath / Fiji on the crops
    the triage step flags; you only need the hard cases.

    ``nucleus_labels`` (and ``cell_labels``) are the stacks for ONE position;
    pass ``position`` to skip manual images belonging to other positions.

    Every nucleus pixel the auto pipeline wrote inside the cell is cleared
    before the manual mask is written - including the second daughter, which
    carries ID ``cell_id + offset``, not ``cell_id``. Without ``cell_labels``
    the clear falls back to ID arithmetic and the manual mask is not clipped
    to the cell.

    Any cell present in a manual image is overridden, marked ``source="manual"``
    and forced ``qc_pass=True`` - a human looked at it, so the automatic
    rejections no longer apply. Its condition is re-derived from the drawn
    mask (component count, then the same aspect / solidity rule as the auto
    path) unless ``conditions`` supplies one; a hand-drawn dumbbell should not
    keep an ``interphase`` weight because the auto pass called it that.
    """
    from .labelgen import _classify
    from .config import LabelGenConfig

    nl = np.array(nucleus_labels, copy=True)
    tb = table.copy()
    tb["source"] = tb["source"].astype(object)
    lcfg = LabelGenConfig()

    for (pos, frame), man in manual_labels.items():
        if position is not None and pos != position:
            continue
        man = np.asarray(man)
        ids = np.unique(man)
        ids = ids[ids > 0]
        sel_frame = (tb["position"] == pos) & (tb["frame"] == frame)
        max_cid = int(tb.loc[sel_frame, "cell_id"].max()) if sel_frame.any() else 0
        offset = 10 ** int(np.ceil(np.log10(max_cid + 1))) if max_cid else 1000
        for cid in ids:
            cid = int(cid)
            m = man == cid
            if cell_labels is not None:
                cell = cell_labels[frame] == cid
                outside = int((m & ~cell).sum())
                if outside:
                    warnings.warn(f"manual nucleus {pos}/{frame}/{cid}: {outside} px "
                                  "outside the cell mask were dropped", stacklevel=2)
                m = m & cell
                nl[frame][cell] = 0                 # every auto id in this cell
            else:
                ids_here = np.unique(nl[frame])
                mine = [i for i in ids_here if i > 0 and i % offset == cid]
                nl[frame][np.isin(nl[frame], mine)] = 0
            nl[frame][m] = cid

            sel = sel_frame & (tb["cell_id"] == cid)
            if not sel.any():
                warnings.warn(f"manual nucleus {pos}/{frame}/{cid} has no row in the "
                              "table; its pixels were written but build_crops will "
                              "not see it", stacklevel=2)
                continue
            cond = conditions.get((pos, frame, cid)) if conditions is not None else None
            if cond is None:
                comp, n_comp = ndi.label(m)
                props = sorted(regionprops(comp), key=lambda q: q.area, reverse=True)
                cond = _classify(props, lcfg, neck=0.0) if props else "reject"
            tb.loc[sel, "source"] = "manual"
            tb.loc[sel, "qc_pass"] = bool(m.any())
            tb.loc[sel, "reject_reason"] = "" if m.any() else "manual_empty"
            tb.loc[sel, "nuc_area_px"] = float(m.sum())
            tb.loc[sel, "condition"] = cond
            if "split_borderline" in tb.columns:
                tb.loc[sel, "split_borderline"] = False
    return nl, tb


# --------------------------------------------------------------------------
def save_dataset(path: str, X: np.ndarray, Y: np.ndarray,
                 meta: pd.DataFrame) -> None:
    np.savez_compressed(path, X=X, Y=Y)
    meta.to_csv(path.replace(".npz", "_meta.csv"), index=False)


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    d = np.load(path)
    meta = pd.read_csv(path.replace(".npz", "_meta.csv"))
    return d["X"], d["Y"], meta


def split_by_position(meta: pd.DataFrame, val_positions: Sequence[str]
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Index arrays for a held-out split by position.

    Split on whole positions or experiments, never random crops - consecutive
    time-lapse frames of the same cell are near-duplicates and a random split
    leaks them across the boundary.
    """
    is_val = meta["position"].isin(val_positions).to_numpy()
    return np.where(~is_val)[0], np.where(is_val)[0]
