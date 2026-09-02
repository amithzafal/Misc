"""Turn frames + labels into a weighted training set of per-cell crops.

Each sample is one cell, cropped and rotated so the long axis is horizontal.
Working inside the cell crop removes the instance-separation problem entirely:
the task becomes binary semantic segmentation of (usually) one blob.

Sample weights combine a per-condition weight with a multiplier for hand-drawn
labels, so manually corrected anaphase cells carry the influence you want
without you having to annotate thousands of them.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .config import DatasetConfig

__all__ = [
    "extract_cell_crop",
    "build_crops",
    "assign_weights",
    "weight_report",
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
            Xs.append(ch)
            Ys.append((nm > 0.5).astype(np.float32)[None])
            metas.append({
                "position": r.position, "frame": r.frame, "cell_id": r.cell_id,
                "condition": r.condition, "source": r.source,
                "cell_area_um2": r.cell_area_um2,
                "nuc_area_um2": r.nuc_area_um2,
            })

    if not Xs:
        raise ValueError("no crops produced - check qc_pass and frame indices")
    return (np.stack(Xs).astype(np.float32),
            np.stack(Ys).astype(np.float32),
            pd.DataFrame(metas))


# --------------------------------------------------------------------------
def assign_weights(meta: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Per-sample training weight = condition weight x manual multiplier.

    Also emits ``effective_n``: the weighted share of the dataset each group
    occupies. Check it. If manual anaphase crops end up at 60% of effective
    exposure the model will overfit them and regress on interphase.
    """
    m = meta.copy()
    base = m["condition"].map(cfg.condition_weights).fillna(1.0)
    mult = np.where(m["source"].eq("manual"), cfg.manual_weight_multiplier, 1.0)
    m["weight"] = np.minimum(base * mult, cfg.max_sample_weight).astype(float)

    total = m["weight"].sum()
    share = m.groupby(["condition", "source"])["weight"].sum() / total
    m.attrs["effective_share"] = share.round(4).to_dict()
    return m


def weight_report(meta: pd.DataFrame) -> pd.DataFrame:
    """Raw counts vs weighted exposure, per condition and label source."""
    g = meta.groupby(["condition", "source"])
    out = g.agg(n=("weight", "size"), weight_sum=("weight", "sum"))
    out["count_share"] = (out["n"] / out["n"].sum()).round(4)
    out["weighted_share"] = (out["weight_sum"] / out["weight_sum"].sum()).round(4)
    return out.reset_index()


# --------------------------------------------------------------------------
def apply_manual_overrides(
    nucleus_labels: np.ndarray,
    table: pd.DataFrame,
    manual_labels: Dict[Tuple[str, int], np.ndarray],
    conditions: Optional[Dict[Tuple[str, int, int], str]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Replace auto labels with hand-drawn ones where they exist.

    ``manual_labels`` maps ``(position, frame) -> label image`` keyed by cell
    ID (0 = not annotated). Draw these in napari / QuPath / Fiji on the crops
    the triage step flags; you only need the hard cases.

    Any cell present in a manual image is overridden, marked ``source="manual"``
    and forced ``qc_pass=True`` - a human looked at it, so the automatic
    rejections no longer apply.
    """
    nl = np.array(nucleus_labels, copy=True)
    tb = table.copy()
    tb["source"] = tb["source"].astype(object)

    for (pos, frame), man in manual_labels.items():
        man = np.asarray(man)
        ids = np.unique(man)
        ids = ids[ids > 0]
        sel_frame = (tb["position"] == pos) & (tb["frame"] == frame)
        for cid in ids:
            m = man == cid
            nl[frame][m] = cid
            # clear any auto pixels for this cell that fall outside the manual mask
            stale = (nl[frame] == cid) & (~m)
            nl[frame][stale] = 0

            sel = sel_frame & (tb["cell_id"] == cid)
            if not sel.any():
                continue
            cond = None
            if conditions is not None:
                cond = conditions.get((pos, frame, int(cid)))
            tb.loc[sel, "source"] = "manual"
            tb.loc[sel, "qc_pass"] = True
            tb.loc[sel, "reject_reason"] = ""
            tb.loc[sel, "nuc_area_px"] = float(m.sum())
            if cond is not None:
                tb.loc[sel, "condition"] = cond
            elif (tb.loc[sel, "condition"] == "reject").any():
                # a human annotated a cell the pipeline rejected; label it
                # from the component count so it is not silently dropped
                n_comp = ndi.label(m)[1]
                tb.loc[sel, "condition"] = (
                    "two_nuclei" if n_comp >= 2 else "anaphase_bridge"
                )
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
