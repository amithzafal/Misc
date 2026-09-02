"""Synthetic S. pombe frames for testing the pipeline end to end.

Rod-shaped cells with a centred nucleus, a configurable fraction in anaphase.
Brightfield nuclear contrast is a tunable knob (``bf_nucleus_contrast``) - set
it to 0.0 to build the null dataset that tells you what your model scores when
there is genuinely no nuclear information in the transmitted-light channel.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage as ndi


def _capsule(shape, cy, cx, length_px, width_px, angle) -> np.ndarray:
    """Rod (stadium) shape: distance to a line segment <= radius."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    r = width_px / 2.0
    half = max(length_px / 2.0 - r, 0.1)
    dy, dx = np.sin(angle), np.cos(angle)
    py, px = yy - cy, xx - cx
    t = np.clip(py * dy + px * dx, -half, half)
    d2 = (py - t * dy) ** 2 + (px - t * dx) ** 2
    return d2 <= r * r


def _disk(shape, cy, cx, radius) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius


def synth_frame(
    shape: Tuple[int, int] = (512, 512),
    n_cells: int = 40,
    pixel_size_um: float = 0.1,
    anaphase_frac: float = 0.15,
    bf_nucleus_contrast: float = 0.25,
    gfp_expression_cv: float = 0.45,
    noise_sigma: float = 0.015,
    rng: np.random.Generator | None = None,
) -> Dict:
    """Return dict with brightfield, gfp, cell_labels, and ground-truth info."""
    rng = rng or np.random.default_rng(0)
    H, W = shape
    p = pixel_size_um

    bf = np.full(shape, 0.55, dtype=np.float32)
    gfp = np.zeros(shape, dtype=np.float32)
    cells = np.zeros(shape, dtype=np.int32)
    nuc_gt = np.zeros(shape, dtype=np.int32)
    truth: List[Dict] = []

    placed = 0
    tries = 0
    while placed < n_cells and tries < n_cells * 60:
        tries += 1
        length_um = rng.uniform(7.0, 14.0)
        width_um = rng.uniform(3.2, 3.8)
        L, Wd = length_um / p, width_um / p
        cy = rng.uniform(L / 2 + 4, H - L / 2 - 4)
        cx = rng.uniform(L / 2 + 4, W - L / 2 - 4)
        ang = rng.uniform(0, np.pi)

        body = _capsule(shape, cy, cx, L, Wd, ang)
        if (cells[ndi.binary_dilation(body, np.ones((5, 5)))] > 0).any():
            continue

        placed += 1
        cid = placed
        cells[body] = cid

        # nuclear volume scales with cell volume in pombe
        nuc_r_um = 0.95 * (length_um / 10.5) ** (1 / 3)
        nuc_r = nuc_r_um / p
        expr = float(np.clip(rng.lognormal(0.0, gfp_expression_cv), 0.25, 4.0))

        is_ana = rng.random() < anaphase_frac
        if is_ana:
            sep_um = rng.uniform(1.2, 4.5)   # centre-to-centre separation
            d = sep_um / p / 2.0
            dy, dx = np.sin(ang), np.cos(ang)
            r2 = nuc_r / (2 ** (1 / 3))      # two daughters, conserved volume
            n1 = _disk(shape, cy - d * dy, cx - d * dx, r2)
            n2 = _disk(shape, cy + d * dy, cx + d * dx, r2)
            nucleus = (n1 | n2) & body
            if sep_um < 2.4:   # still a connected bridge
                bridge = _capsule(shape, cy, cx, 2 * d, r2 * 0.9, ang)
                nucleus = (nucleus | (bridge & body))
            cond = "anaphase_bridge" if sep_um < 2.4 else "two_nuclei"
        else:
            jitter = rng.normal(0, 0.25 / p, 2)   # centring is tight but noisy
            nucleus = _disk(shape, cy + jitter[0], cx + jitter[1], nuc_r) & body
            cond = "interphase"

        nuc_gt[nucleus] = cid
        gfp[nucleus] += 0.9 * expr
        bf[body] -= 0.12
        if bf_nucleus_contrast:
            bf[nucleus] -= bf_nucleus_contrast * 0.12
        truth.append(dict(cell_id=cid, condition=cond, expr=expr,
                          nuc_area_px=int(nucleus.sum())))

    bf = ndi.gaussian_filter(bf, 1.2) + rng.normal(0, noise_sigma, shape)
    gfp = ndi.gaussian_filter(gfp, 1.0) + 0.05
    gfp = gfp + rng.normal(0, noise_sigma * 1.5, shape)
    return dict(
        brightfield=bf.astype(np.float32),
        gfp=np.clip(gfp, 0, None).astype(np.float32),
        cell_labels=cells,
        nucleus_gt=nuc_gt,
        truth=truth,
    )


def synth_movie(n_frames: int = 5, seed: int = 0, **kw) -> Dict:
    rng = np.random.default_rng(seed)
    frames = [synth_frame(rng=rng, **kw) for _ in range(n_frames)]
    return dict(
        brightfield=np.stack([f["brightfield"] for f in frames]),
        gfp=np.stack([f["gfp"] for f in frames]),
        cell_labels=np.stack([f["cell_labels"] for f in frames]),
        nucleus_gt=np.stack([f["nucleus_gt"] for f in frames]),
        truth=[f["truth"] for f in frames],
    )
