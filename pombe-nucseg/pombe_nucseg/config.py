"""Configuration objects for the pombe nuclear-segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple
import json


@dataclass
class LabelGenConfig:
    """Parameters for turning nuclear-GFP into per-cell nucleus labels.

    Defaults are set for S. pombe at ~0.1 um/px (100x, sCMOS). The nucleus is
    ~2 um across, so ~3.1 um^2 in cross-section; cells are ~3.5 x 7-14 um.
    """

    # ---- geometry / calibration -------------------------------------------
    pixel_size_um: float = 0.1

    # ---- GFP preprocessing -------------------------------------------------
    smooth_sigma_um: float = 0.08          # mild denoise before thresholding
    erode_cell_um: float = 0.15            # pull cell mask in off the boundary
    background_percentile: float = 5.0     # bg from non-cell pixels of the frame

    # ---- thresholding ------------------------------------------------------
    threshold_method: str = "otsu"         # {"otsu", "fraction"}
    threshold_fraction: float = 0.5        # used when method == "fraction"
    fill_holes: bool = True

    # ---- object acceptance -------------------------------------------------
    min_nuc_area_um2: float = 0.8
    max_nuc_area_um2: float = 12.0
    max_nuc_area_frac: float = 0.55        # nucleus vs cell cross-section
    max_nuclei_per_cell: int = 2

    # ---- QC ----------------------------------------------------------------
    min_gfp_snr: float = 3.0               # (p99.5 - bg) / bg_std inside cell
    min_focus_score: Optional[float] = None   # set after calibrating on your data
    min_cell_area_um2: float = 3.0
    border_margin_px: int = 2              # reject cells touching frame edge

    # ---- condition classification -----------------------------------------
    # A single elongated / waisted object = anaphase bridge (closed mitosis).
    anaphase_aspect_min: float = 1.7
    anaphase_solidity_max: float = 0.92

    def scaled(self) -> Dict[str, float]:
        """Micron-valued parameters converted to pixels."""
        p = self.pixel_size_um
        return {
            "smooth_sigma_px": self.smooth_sigma_um / p,
            "erode_cell_px": max(0, int(round(self.erode_cell_um / p))),
            "min_nuc_area_px": self.min_nuc_area_um2 / (p * p),
            "max_nuc_area_px": self.max_nuc_area_um2 / (p * p),
            "min_cell_area_px": self.min_cell_area_um2 / (p * p),
        }


@dataclass
class DatasetConfig:
    """Crop extraction and sample weighting."""

    crop_size: Tuple[int, int] = (128, 128)   # after rotation, in pixels
    crop_pad_um: float = 1.0                  # context beyond the cell bbox
    align_major_axis: bool = True             # rotate cell to horizontal
    include_cell_mask_channel: bool = True
    normalize: str = "percentile"             # {"percentile", "zscore", "none"}
    norm_percentiles: Tuple[float, float] = (1.0, 99.0)

    # Sampling weight by condition. Anaphase is rare and is where the
    # geometric prior fails, so it is deliberately over-represented.
    condition_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "interphase": 1.0,
            "anaphase_bridge": 6.0,
            "two_nuclei": 4.0,
        }
    )
    # Extra multiplier for hand-drawn labels on top of the condition weight.
    manual_weight_multiplier: float = 4.0

    # Cap so a handful of manual anaphase crops cannot dominate a batch.
    max_sample_weight: float = 40.0


@dataclass
class TrainConfig:
    in_channels: int = 2            # brightfield (n_z) + cell mask
    base_filters: int = 16
    depth: int = 4
    epochs: int = 60
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.0       # prefer explicit position-level splits
    num_workers: int = 4
    amp: bool = True
    seed: int = 0

    # How sample weights enter training.
    #   "sampler"  -> WeightedRandomSampler, uniform loss weights  (recommended)
    #   "loss"     -> uniform sampling, per-sample loss scaling
    #   "both"     -> sampler + loss (effect multiplies; use with care)
    weighting_mode: str = "sampler"
    dice_weight: float = 0.5        # loss = bce + dice_weight * soft_dice
    pos_weight: Optional[float] = None   # BCE positive-class weight


def save_config(cfg, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
