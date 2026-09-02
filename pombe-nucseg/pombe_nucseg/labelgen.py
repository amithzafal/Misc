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
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, laplace
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import disk

from .config import LabelGenConfig

__all__ = [
    "frame_background",
    "focus_score",
    "neck_ratio",
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
    # robust sigma: 1.4826 * MAD. (The std of the lower half of a Gaussian is
    # ~0.6 sigma, so a half-distribution std understates noise and inflates
    # every SNR by ~1.65x.)
    med = np.median(outside)
    sigma = 1.4826 * float(np.median(np.abs(outside - med)))
    if sigma <= 0:
        sigma = float(np.std(outside))
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


def neck_ratio(g: np.ndarray, mask: np.ndarray, min_peak_sep_px: int,
               rel_peak: float = 0.3, iters: int = 12) -> float:
    """Saddle-to-peak intensity ratio between the two brightest GFP maxima.

    1.0 = one blob (no second maximum, or no waist); 0.0 = fully separated.
    Found by binary-searching the threshold at which the two maxima become
    one connected component, i.e. the intensity of the mountain pass between
    them. Threshold-free, so it does not inherit Otsu's arbitrariness: on
    synthetic dumbbells it tracks bridge intensity with sd ~0.01 while the
    Otsu component count flips between 1 and 2 under a +/-20% threshold change.
    """
    gm = np.where(mask, g, 0.0).astype(np.float32)
    if not mask.any() or gm.max() <= 0:
        return 1.0
    pk = peak_local_max(gm, min_distance=max(int(min_peak_sep_px), 1),
                        num_peaks=2, labels=mask.astype(np.int32),
                        threshold_abs=rel_peak * float(gm.max()))
    if len(pk) < 2:
        return 1.0
    a, b = tuple(pk[0]), tuple(pk[1])
    lo_peak = float(min(gm[a], gm[b]))
    if lo_peak <= 0:
        return 1.0
    lo, hi = 0.0, lo_peak
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lab, _ = ndi.label((gm > mid) & mask)
        if lab[a] > 0 and lab[a] == lab[b]:
            lo = mid            # still connected: the saddle is higher
        else:
            hi = mid
    return float(lo / lo_peak)


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


def _classify(props: List, cfg: LabelGenConfig, neck: float = 0.0) -> str:
    """Assign a condition class from the accepted nuclear components.

    Two components whose saddle intensity is still above the borderline band
    were cut by the threshold, not by biology: call them a bridge.
    """
    n = len(props)
    if n == 0:
        return "reject"
    if n >= 2:
        return "anaphase_bridge" if neck >= cfg.neck_borderline[1] else "two_nuclei"
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
            "nuc_neck_ratio": np.nan,
            "split_borderline": False,
            "nuc_cyto_ratio": np.nan,
            "outside_ratio": np.nan,
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
        if len(props) == 2 and props[1].area < cfg.min_daughter_area_ratio * props[0].area:
            row["reject_reason"] = "asymmetric_nuclei"
            rows.append(row)
            continue

        nuc = np.isin(comp, [p.label for p in props])

        # -- bleed-through guards --------------------------------------------
        # (a) nucleus vs rest-of-cell contrast. Haze from a bright neighbour
        #     is a shallow gradient, so Otsu's "foreground" is barely brighter
        #     than its "background".
        cyto = interior & ~nuc
        nuc_med = float(np.median(g[nuc]))
        cyto_med = float(np.median(g[cyto])) if cyto.any() else 0.0
        row["nuc_cyto_ratio"] = nuc_med / max(cyto_med, bg_sigma)
        if row["nuc_cyto_ratio"] < cfg.min_nuc_cyto_ratio:
            row["reject_reason"] = "low_nuc_contrast"
            rows.append(row)
            continue
        # (b) does the signal continue across the cell boundary? A real
        #     nucleus sits >= 1 um inside the cortex, so the ring just outside
        #     the cell next to it is empty or at background. A false nucleus
        #     hugging the side facing a bright neighbour has haze there.
        reach = sc["erode_cell_px"] + 3
        r0, c0 = max(0, minr - reach), max(0, minc - reach)
        r1, c1 = min(H, maxr + reach), min(W, maxc + reach)
        big_nuc = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        big_nuc[minr - r0:maxr - r0, minc - c0:maxc - c0] = nuc
        ring = (ndi.binary_dilation(big_nuc, iterations=reach)
                & (cell_labels[r0:r1, c0:c1] == 0))
        if ring.any():
            row["outside_ratio"] = float(np.median(gfp_smooth[r0:r1, c0:c1][ring])) / max(nuc_med, 1e-6)
            if row["outside_ratio"] > cfg.max_outside_ratio:
                row["reject_reason"] = "edge_bleed"
                rows.append(row)
                continue
        else:
            row["outside_ratio"] = 0.0

        # -- split stability -------------------------------------------------
        neck = neck_ratio(g, interior, cfg.neck_min_peak_sep_um / cfg.pixel_size_um)
        row["nuc_neck_ratio"] = neck
        lo_b, hi_b = cfg.neck_borderline
        row["split_borderline"] = bool(lo_b <= neck <= hi_b)

        condition = _classify(props, cfg, neck)
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
