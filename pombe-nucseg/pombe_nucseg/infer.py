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

__all__ = ["load_model", "predict_frame", "predict_movie", "rank_for_annotation"]


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


def predict_movie(model, brightfield, cell_labels, table: pd.DataFrame,
                  cfg: DatasetConfig, position: str = "pos0",
                  progress: bool = True, **kw) -> np.ndarray:
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
                        n: int = 200) -> pd.DataFrame:
    """Rank cells by how much a hand-drawn label would be worth.

    Annotate where the automatic pipeline is uncertain or wrong, not at random:

      1. anaphase bridges  - the threshold decides where the bridge breaks, so
         the auto label is arbitrary exactly here
      2. rejected cells with good GFP signal - the pipeline failed on a cell
         that clearly has a nucleus, so it is a real failure mode
      3. cells whose auto label disagrees most with the geometric baseline
      4. borderline two-nucleus calls

    Pass ``metrics`` (from :func:`validate.per_cell_metrics` against the
    baseline) to enable rule 3.
    """
    t = table.copy()
    priority = np.zeros(len(t))
    priority += 4.0 * (t["condition"] == "anaphase_bridge")
    priority += 2.0 * (t["condition"] == "two_nuclei")
    good_signal = t["gfp_snr"].fillna(0) > t["gfp_snr"].median()
    priority += 3.0 * (~t["qc_pass"] & good_signal)
    priority += 1.5 * t["reject_reason"].isin(["too_many_nuclei", "no_nucleus"])

    if metrics is not None and "iou" in metrics.columns:
        keys = [c for c in ("position", "frame", "cell_id") if c in metrics.columns]
        t = t.merge(metrics[keys + ["iou"]], on=keys, how="left")
        priority = priority[:len(t)] + 2.0 * (1.0 - t["iou"].fillna(0.5))

    t["annotation_priority"] = priority
    cols = ["position", "frame", "cell_id", "condition", "reject_reason",
            "gfp_snr", "annotation_priority"]
    cols = [c for c in cols if c in t.columns]
    return t.sort_values("annotation_priority", ascending=False).head(n)[cols]
