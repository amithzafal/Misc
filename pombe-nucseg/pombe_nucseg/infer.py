"""Inference over a time-lapse, and triage for choosing what to hand-annotate.

At your scale - hundreds of cells x hundreds of frames - per-cell crops are the
right unit: they batch cleanly on the GPU, and the cell mask you already have
supplies the instance identity, so there is no post-hoc matching step.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .config import DatasetConfig
from .dataset import extract_cell_crop, _normalize

__all__ = ["load_model", "cell_table_from_masks", "predict_frame",
           "predict_movie", "rank_for_annotation"]


def cell_table_from_masks(cell_labels, position: str = "pos0") -> pd.DataFrame:
    """Per-cell geometry from cell masks alone - what inference needs.

    At inference there is no GFP and therefore no label table; the only
    columns ``predict_frame`` uses are the cell centroid and orientation, and
    those come from the masks.
    """
    from skimage.measure import regionprops
    rows = []
    for t in range(len(cell_labels)):
        for p in regionprops(np.asarray(cell_labels[t])):
            rows.append(dict(position=position, frame=t, cell_id=int(p.label),
                             cell_y=float(p.centroid[0]), cell_x=float(p.centroid[1]),
                             cell_orientation=float(p.orientation),
                             cell_length_um=np.nan, qc_pass=True))
    return pd.DataFrame(rows)


def load_model(checkpoint: str, in_channels: int = 2, base: int = 16,
               depth: int = 4, device: Optional[str] = None):
    import torch
    from .model import build_unet

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint, map_location=device)
    cfg = ck.get("config", {})
    model = build_unet(cfg.get("in_channels", in_channels),
                       cfg.get("base_filters", base),
                       cfg.get("depth", depth)).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def _unrotate_into(canvas: np.ndarray, mask: np.ndarray, cy: float, cx: float,
                   orientation: float, cell_mask: np.ndarray, cid: int,
                   align: bool) -> None:
    """Rotate a predicted crop back to frame coordinates and paste it in."""
    h, w = mask.shape
    half = int(np.ceil(np.hypot(h, w) / 2)) + 4
    big = np.zeros((2 * half, 2 * half), dtype=np.float32)
    y0 = half - h // 2
    x0 = half - w // 2
    big[y0:y0 + h, x0:x0 + w] = mask

    if align:
        big = ndi.rotate(big, -(90.0 - np.degrees(orientation)),
                         reshape=False, order=0, mode="constant")

    H, W = canvas.shape
    ty0, tx0 = int(round(cy)) - half, int(round(cx)) - half
    sy0, sx0 = max(0, ty0), max(0, tx0)
    sy1, sx1 = min(H, ty0 + 2 * half), min(W, tx0 + 2 * half)
    if sy1 <= sy0 or sx1 <= sx0:
        return
    sub = big[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0] > 0.5
    region = canvas[sy0:sy1, sx0:sx1]
    inside = cell_mask[sy0:sy1, sx0:sx1] == cid
    region[sub & inside] = cid


def predict_frame(model, brightfield: np.ndarray, cell_labels: np.ndarray,
                  table_frame: pd.DataFrame, cfg: DatasetConfig,
                  batch_size: int = 64, threshold: float = 0.5,
                  keep_largest: int = 2, min_area_px: int = 30) -> np.ndarray:
    """Predict nucleus labels for one frame. Returns a cell-ID-keyed label image."""
    import torch

    device = next(model.parameters()).device
    bf = np.asarray(brightfield, dtype=np.float32)
    if bf.ndim == 2:
        bf = bf[None]

    rows = list(table_frame.itertuples())
    out = np.zeros_like(cell_labels, dtype=np.int32)

    for s in range(0, len(rows), batch_size):
        chunk = rows[s:s + batch_size]
        batch = []
        for r in chunk:
            ch, (cm,) = extract_cell_crop(
                bf, [cell_labels == r.cell_id],
                r.cell_y, r.cell_x, r.cell_orientation, cfg,
            )
            ch = _normalize(ch, cfg)
            if cfg.include_cell_mask_channel:
                ch = np.concatenate([ch, cm[None].astype(np.float32)], axis=0)
            batch.append(ch)

        x = torch.from_numpy(np.stack(batch)).to(device)
        with torch.no_grad():
            p = (torch.sigmoid(model(x)) > threshold).cpu().numpy()[:, 0]

        for r, mask in zip(chunk, p):
            lab, n = ndi.label(mask)
            if n == 0:
                continue
            sizes = np.bincount(lab.ravel())[1:]
            order = np.argsort(sizes)[::-1][:keep_largest]
            keep = np.zeros(mask.shape, dtype=bool)
            for k in order:
                if sizes[k] >= min_area_px:
                    keep |= (lab == k + 1)
            _unrotate_into(out, keep.astype(np.float32), r.cell_y, r.cell_x,
                           r.cell_orientation, cell_labels, r.cell_id,
                           cfg.align_major_axis)
    return out


def predict_movie(model, brightfield, cell_labels, table: Optional[pd.DataFrame],
                  cfg: DatasetConfig, position: str = "pos0",
                  progress: bool = True, **kw) -> np.ndarray:
    """``table`` may be None: geometry is then derived from the cell masks."""
    if table is None:
        table = cell_table_from_masks(cell_labels, position)
    frames = []
    T = len(cell_labels)
    for t in range(T):
        tf = table[(table["frame"] == t) & (table["position"] == position)]
        frames.append(predict_frame(model, brightfield[t], cell_labels[t],
                                    tf, cfg, **kw))
        if progress and (t % 25 == 0 or t == T - 1):
            print(f"  inferring frame {t + 1}/{T}", flush=True)
    return np.stack(frames)


# --------------------------------------------------------------------------
def rank_for_annotation(table: pd.DataFrame, metrics: Optional[pd.DataFrame] = None,
                        n: int = 200, frame_stride: int = 1) -> pd.DataFrame:
    """Rank cells by how much a hand-drawn label would be worth.

    Annotate where the automatic pipeline is uncertain or wrong, not at random:

      1. anaphase bridges  - the threshold decides where the bridge breaks, so
         the auto label is arbitrary exactly here
      2. rejected cells with good GFP signal - the pipeline failed on a cell
         that clearly has a nucleus, so it is a real failure mode
      3. cells whose auto label disagrees most with the geometric baseline
      4. borderline bridge / two-nucleus calls (``split_borderline`` from the
         neck ratio) and cells the bleed-through guards rejected

    Pass ``metrics`` (from :func:`validate.per_cell_metrics` against the
    baseline) to enable rule 3. ``frame_stride`` keeps every k-th frame only:
    consecutive frames of one dividing cell are near-duplicates, and without
    thinning the top of the list is 200 frames of the same ten cells.
    """
    t = table.copy()
    if frame_stride > 1:
        t = t[t["frame"] % frame_stride == 0]
    t = t.reset_index(drop=True)
    priority = np.zeros(len(t))
    priority += 4.0 * (t["condition"] == "anaphase_bridge")
    priority += 2.0 * (t["condition"] == "two_nuclei")
    if "split_borderline" in t.columns:
        priority += 3.0 * t["split_borderline"].astype(bool)
    good_signal = t["gfp_snr"].fillna(0) > t["gfp_snr"].median()
    priority += 3.0 * (~t["qc_pass"].astype(bool) & good_signal)
    priority += 1.5 * t["reject_reason"].isin(
        ["too_many_nuclei", "no_nucleus", "edge_bleed", "asymmetric_nuclei"])

    if metrics is not None and "iou" in metrics.columns:
        # align by key lookup, never by merge: a merge re-orders / duplicates
        # rows when metrics lack a column of the key or carry duplicate keys,
        # and the positional priority array would then be added to the wrong
        # cells
        keys = [c for c in ("position", "frame", "cell_id") if c in metrics.columns]
        if "position" in t.columns and "position" not in keys and t["position"].nunique() > 1:
            raise ValueError("metrics has no 'position' column but table spans "
                             "several positions - ambiguous join")
        lut = metrics.drop_duplicates(keys).set_index(keys)["iou"]
        idx = pd.MultiIndex.from_frame(t[keys]) if len(keys) > 1 else t[keys[0]]
        iou = lut.reindex(idx).to_numpy(dtype=float)
        priority += 2.0 * (1.0 - np.nan_to_num(iou, nan=0.5))

    t["annotation_priority"] = priority
    cols = ["position", "frame", "cell_id", "condition", "reject_reason",
            "split_borderline", "nuc_neck_ratio", "gfp_snr", "annotation_priority"]
    cols = [c for c in cols if c in t.columns]
    return t.sort_values("annotation_priority", ascending=False).head(n)[cols]
