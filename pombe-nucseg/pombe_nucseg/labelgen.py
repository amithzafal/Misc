"""Automated nucleus label generation from nuclear-GFP, gated by cell masks.

The cell mask you already have does most of the work: it removes the
instance-separation problem, lets each cell get its own threshold (expression
varies severalfold across a population), and gives the cell-nucleus pairing for
free.

Outputs per frame:
  * a nucleus label image, with nucleus IDs keyed to their parent cell ID
  * a per-cell table with morphology, QC flags and a condition class

Condition classes (S. pombe undergoes *closed* mitosis - the envelope does not
break down, so NLS-GFP stays nuclear throughout):
  interphase      one round nucleus
  anaphase_bridge one elongated / waisted object mid-separation
  two_nuclei      two resolved nuclei (late anaphase, septated, binucleate)
  reject          zero, too many, or failed QC
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, laplace
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import disk

from .config import LabelGenConfig

__all__ = [
    "frame_background",
    "focus_score",
    "generate_labels_frame",
    "generate_labels_movie",
]

CONDITIONS = ("interphase", "anaphase_bridge", "two_nuclei", "reject")


# --------------------------------------------------------------------------
# frame-level helpers
# --------------------------------------------------------------------------
def frame_background(gfp: np.ndarray, cell_mask: np.ndarray,
                     percentile: float = 5.0) -> Tuple[float, float]:
    """Background level and noise sigma from pixels outside every cell.

    Falls back to the whole frame if the field is packed with cells.
    """
    outside = gfp[cell_mask == 0]
    if outside.size < 100:
        outside = gfp.ravel()
    bg = float(np.percentile(outside, percentile))
    # robust sigma from the lower half of the background distribution
    lo = outside[outside <= np.percentile(outside, 50)]
    sigma = float(np.std(lo)) if lo.size > 10 else float(np.std(outside))
    return bg, max(sigma, 1e-6)


def focus_score(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Variance-of-Laplacian focus metric. Higher is sharper.

    Scale is arbitrary and detector-dependent - calibrate a cutoff on a stack
    you have stepped through focus, then set ``min_focus_score`` in the config.
    """
    img = image.astype(np.float32)
    rng = np.ptp(img)
    if rng > 0:
        img = (img - img.min()) / rng
    lap = laplace(img)
    if mask is not None and mask.any():
        lap = lap[mask.astype(bool)]
    return float(np.var(lap))


def _remove_small(binary: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components smaller than ``min_size`` pixels."""
    if min_size <= 1 or not binary.any():
        return binary
    lab, n = ndi.label(binary)
    if n == 0:
        return binary
    counts = np.bincount(lab.ravel())
    keep = np.zeros(counts.size, dtype=bool)
    keep[1:] = counts[1:] >= min_size
    return keep[lab]


# --------------------------------------------------------------------------
# per-cell segmentation
# --------------------------------------------------------------------------
def _threshold_cell(gfp_cell: np.ndarray, cfg: LabelGenConfig,
                    bg: float) -> float:
    """Threshold for one cell's GFP crop (values already background-subtracted)."""
    vals = gfp_cell[np.isfinite(gfp_cell)]
    if vals.size < 10:
        return np.inf
    if cfg.threshold_method == "fraction":
        return float(cfg.threshold_fraction * np.percentile(vals, 99.5))
    # Otsu on the in-cell distribution. Unimodal input (no nucleus, dead cell)
    # yields a meaningless split - the SNR gate upstream catches those.
    try:
        return float(threshold_otsu(vals))
    except ValueError:
        return np.inf


def _classify(props: List, cfg: LabelGenConfig) -> str:
    """Assign a condition class from the accepted nuclear components."""
    n = len(props)
    if n == 0:
        return "reject"
    if n >= 2:
        return "two_nuclei"
    p = props[0]
    minor = max(p.axis_minor_length, 1e-6)
    aspect = p.axis_major_length / minor
    if aspect >= cfg.anaphase_aspect_min or p.solidity <= cfg.anaphase_solidity_max:
        return "anaphase_bridge"
    return "interphase"


def generate_labels_frame(
    brightfield: np.ndarray,
    gfp: np.ndarray,
    cell_labels: np.ndarray,
    cfg: LabelGenConfig,
    frame: int = 0,
    position: str = "pos0",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Generate nucleus labels for one frame.

    Parameters
    ----------
    brightfield : (H, W) or (Z, H, W) array
        Transmitted-light image(s). Only used here for the focus metric; the
        z-stack is carried through to the dataset builder.
    gfp : (H, W) array
        Nuclear-GFP, already registered onto the brightfield frame.
    cell_labels : (H, W) int array
        Instance labels from your cell segmentation. 0 = background.
    cfg : LabelGenConfig
    frame, position : identifiers written into the output table.

    Returns
    -------
    nuc_labels : (H, W) int array
        Nucleus instances. IDs match the parent cell ID for single-nucleus
        cells; for two-nucleus cells the second gets ``cell_id + offset`` where
        offset is 10^ceil(log10(max_cell_id+1)), so parentage stays recoverable
        via the returned table.
    table : DataFrame, one row per cell.
    """
    bf = np.asarray(brightfield, dtype=np.float32)
    bf2d = bf if bf.ndim == 2 else bf[bf.shape[0] // 2]
    gfp = np.asarray(gfp, dtype=np.float32)
    cell_labels = np.asarray(cell_labels)

    if gfp.shape != bf2d.shape or gfp.shape != cell_labels.shape:
        raise ValueError(
            f"shape mismatch: gfp {gfp.shape}, bf {bf2d.shape}, "
            f"cells {cell_labels.shape}"
        )

    sc = cfg.scaled()
    H, W = gfp.shape
    bg, bg_sigma = frame_background(gfp, cell_labels, cfg.background_percentile)

    gfp_sub = gfp - bg
    if sc["smooth_sigma_px"] > 0.3:
        gfp_smooth = ndi.gaussian_filter(gfp_sub, sc["smooth_sigma_px"])
    else:
        gfp_smooth = gfp_sub

    nuc_labels = np.zeros_like(cell_labels, dtype=np.int32)
    max_cell_id = int(cell_labels.max()) if cell_labels.size else 0
    offset = 10 ** int(np.ceil(np.log10(max_cell_id + 1))) if max_cell_id else 1000

    rows: List[Dict] = []
    selem = disk(sc["erode_cell_px"]) if sc["erode_cell_px"] > 0 else None

    for cprop in regionprops(cell_labels):
        cid = cprop.label
        minr, minc, maxr, maxc = cprop.bbox
        cell_crop = cprop.image  # bool, bbox-shaped
        g = gfp_smooth[minr:maxr, minc:maxc]

        row: Dict = {
            "position": position,
            "frame": frame,
            "cell_id": cid,
            "cell_area_px": float(cprop.area),
            "cell_area_um2": float(cprop.area * cfg.pixel_size_um ** 2),
            "cell_length_um": float(cprop.axis_major_length * cfg.pixel_size_um),
            "cell_width_um": float(cprop.axis_minor_length * cfg.pixel_size_um),
            "cell_y": float(cprop.centroid[0]),
            "cell_x": float(cprop.centroid[1]),
            "cell_orientation": float(cprop.orientation),
            "bbox": (minr, minc, maxr, maxc),
            "gfp_bg": bg,
            "gfp_bg_sigma": bg_sigma,
            "focus_score": focus_score(bf2d[minr:maxr, minc:maxc]),
            "n_nuclei": 0,
            "nuc_area_px": np.nan,
            "nuc_area_um2": np.nan,
            "nuc_y": np.nan,
            "nuc_x": np.nan,
            "nuc_solidity": np.nan,
            "nuc_aspect": np.nan,
            "gfp_snr": np.nan,
            "condition": "reject",
            "qc_pass": False,
            "reject_reason": "",
            "source": "auto",
        }

        # -- QC gates that do not need the nucleus ---------------------------
        touches_border = (
            minr < cfg.border_margin_px or minc < cfg.border_margin_px
            or maxr > H - cfg.border_margin_px or maxc > W - cfg.border_margin_px
        )
        if touches_border:
            row["reject_reason"] = "border"
            rows.append(row)
            continue
        if cprop.area < sc["min_cell_area_px"]:
            row["reject_reason"] = "cell_too_small"
            rows.append(row)
            continue
        if cfg.min_focus_score is not None and row["focus_score"] < cfg.min_focus_score:
            row["reject_reason"] = "out_of_focus"
            rows.append(row)
            continue

        interior = (ndi.binary_erosion(cell_crop, selem)
                    if selem is not None else cell_crop)
        if not interior.any():
            interior = cell_crop

        gvals = g[interior]
        snr = float(np.percentile(gvals, 99.5) / bg_sigma) if gvals.size else 0.0
        row["gfp_snr"] = snr
        if snr < cfg.min_gfp_snr:
            row["reject_reason"] = "low_gfp_snr"
            rows.append(row)
            continue

        # -- threshold and extract components --------------------------------
        thr = _threshold_cell(gvals, cfg, bg)
        binary = (g > thr) & interior
        if cfg.fill_holes:
            binary = ndi.binary_fill_holes(binary)
        binary = _remove_small(binary, int(sc["min_nuc_area_px"]))

        comp = cc_label(binary)
        props = [
            p for p in regionprops(comp)
            if sc["min_nuc_area_px"] <= p.area <= sc["max_nuc_area_px"]
            and p.area <= cfg.max_nuc_area_frac * cprop.area
        ]
        props.sort(key=lambda p: p.area, reverse=True)

        row["n_nuclei"] = len(props)
        if len(props) == 0:
            row["reject_reason"] = "no_nucleus"
            rows.append(row)
            continue
        if len(props) > cfg.max_nuclei_per_cell:
            row["reject_reason"] = "too_many_nuclei"
            rows.append(row)
            continue

        condition = _classify(props, cfg)
        total_area = float(sum(p.area for p in props))
        big = props[0]
        row.update(
            condition=condition,
            qc_pass=True,
            nuc_area_px=total_area,
            nuc_area_um2=total_area * cfg.pixel_size_um ** 2,
            nuc_y=float(np.mean([p.centroid[0] for p in props]) + minr),
            nuc_x=float(np.mean([p.centroid[1] for p in props]) + minc),
            nuc_solidity=float(big.solidity),
            nuc_aspect=float(big.axis_major_length / max(big.axis_minor_length, 1e-6)),
        )

        # -- write instance labels -------------------------------------------
        sub = nuc_labels[minr:maxr, minc:maxc]
        for k, p in enumerate(props):
            nid = cid if k == 0 else cid + offset * k
            sub[comp == p.label] = nid
        rows.append(row)

    table = pd.DataFrame(rows)
    return nuc_labels, table


def generate_labels_movie(
    brightfield_frames,
    gfp_frames,
    cell_label_frames,
    cfg: LabelGenConfig,
    position: str = "pos0",
    progress: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Run :func:`generate_labels_frame` over a time-lapse.

    Accepts any indexable sequence (numpy stack, list of arrays, lazy loader).
    """
    n = len(gfp_frames)
    label_stack, tables = [], []
    for t in range(n):
        nl, tb = generate_labels_frame(
            brightfield_frames[t], gfp_frames[t], cell_label_frames[t],
            cfg, frame=t, position=position,
        )
        label_stack.append(nl)
        tables.append(tb)
        if progress and (t % 50 == 0 or t == n - 1):
            print(f"  [{position}] frame {t + 1}/{n}", flush=True)
    return np.stack(label_stack), pd.concat(tables, ignore_index=True)
