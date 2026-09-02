"""Training with per-sample weights.

Two ways to make rare, manually corrected anaphase cells count:

  weighting_mode="sampler"  WeightedRandomSampler draws them more often; every
                            sample contributes equally to the loss once drawn.
                            Gradients stay well scaled. This is the default.

  weighting_mode="loss"     Uniform sampling, per-sample loss scaling. Keeps
                            the epoch a true pass over the data, but a weight
                            of 24 on one crop produces a correspondingly large
                            gradient step (``grad_clip`` bounds it). Weights
                            are normalised by the mean weight of the training
                            set, a constant, so a sample's influence does not
                            depend on which batch it lands in.

  weighting_mode="both"     Effects multiply. Rarely what you want.

Whichever you pick, look at ``weight_report`` before training. If manual
anaphase crops occupy most of the weighted mass, the model will overfit them
and regress on interphase - and interphase is most of your data.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import TrainConfig
from .model import build_unet, count_parameters


# --------------------------------------------------------------------------
def make_dataloaders(X, Y, weights, train_idx, val_idx, cfg: TrainConfig,
                     augment: bool = True, mask_channel: bool = True):
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    class CropDS(Dataset):
        def __init__(self, X, Y, w, idx, aug, mask_ch):
            self.X, self.Y, self.w, self.idx, self.aug = X, Y, w, idx, aug
            self.mask_ch = mask_ch

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            x = np.array(self.X[j], dtype=np.float32)
            y = np.array(self.Y[j], dtype=np.float32)
            if self.aug:
                # Rod cells are near-symmetric about both axes once aligned,
                # so flips are label-preserving. No rotation: orientation is
                # already normalised and re-rotating would undo that.
                if np.random.rand() < 0.5:
                    x, y = x[:, ::-1], y[:, ::-1]
                if np.random.rand() < 0.5:
                    x, y = x[..., ::-1], y[..., ::-1]
                # intensity jitter on image planes only - the cell-mask
                # channel is exactly {0, 1} at inference and must stay so
                nimg = x.shape[0] - 1 if self.mask_ch else x.shape[0]
                x[:nimg] = x[:nimg] * np.float32(np.random.uniform(0.9, 1.1))
                x[:nimg] = x[:nimg] + np.float32(np.random.normal(0, 0.02))
            return (torch.from_numpy(np.ascontiguousarray(x)),
                    torch.from_numpy(np.ascontiguousarray(y)),
                    torch.tensor(float(self.w[j])))

    w = np.asarray(weights, dtype=np.float64)
    tr = CropDS(X, Y, w, train_idx, augment, mask_channel)
    va = CropDS(X, Y, w, val_idx, False, mask_channel) if len(val_idx) else None

    if cfg.weighting_mode in ("sampler", "both"):
        sampler = WeightedRandomSampler(
            weights=w[train_idx], num_samples=len(train_idx), replacement=True
        )
        tl = DataLoader(tr, batch_size=cfg.batch_size, sampler=sampler,
                        num_workers=cfg.num_workers, drop_last=True)
    else:
        tl = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True)

    vl = (DataLoader(va, batch_size=cfg.batch_size, shuffle=False,
                     num_workers=cfg.num_workers) if va is not None else None)
    return tl, vl


# --------------------------------------------------------------------------
def _loss_fn(logits, target, sample_w, cfg: TrainConfig, w_norm: float = 1.0):
    """BCE + soft Dice per sample, optionally scaled by ``sample_w / w_norm``.

    ``w_norm`` must be a constant (the mean training weight). Normalising by
    the *batch* mean makes a batch of only high-weight crops indistinguishable
    from a batch of only low-weight ones, and gives an interphase crop 0.26x
    or 1.0x depending on whether a manual crop shares its batch.
    """
    import torch
    import torch.nn.functional as F

    logits = logits.float()          # dice sums in fp16 lose precision under AMP
    target = target.float()
    pos_weight = (torch.tensor(cfg.pos_weight, device=logits.device)
                  if cfg.pos_weight else None)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight
    ).mean(dim=(1, 2, 3))

    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    dice = 1.0 - (2 * inter + 1.0) / (denom + 1.0)

    per_sample = bce + cfg.dice_weight * dice
    if cfg.weighting_mode in ("loss", "both"):
        per_sample = per_sample * (sample_w.float() / max(float(w_norm), 1e-6))
    return per_sample.mean()


def _evaluate(model, loader, device, threshold: float = 0.5) -> Dict[str, float]:
    import torch
    model.eval()
    ious, dices, n = [], [], 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            p = (torch.sigmoid(model(x)) > threshold).float()
            inter = (p * y).sum(dim=(1, 2, 3))
            union = ((p + y) > 0).float().sum(dim=(1, 2, 3))
            ious.append(((inter + 1e-6) / (union + 1e-6)).cpu().numpy())
            dices.append(((2 * inter + 1e-6) /
                          (p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + 1e-6)
                          ).cpu().numpy())
            n += x.shape[0]
    model.train()
    return {"iou": float(np.concatenate(ious).mean()),
            "dice": float(np.concatenate(dices).mean()), "n": n}


def train_model(
    X, Y, meta: pd.DataFrame, train_idx, val_idx,
    cfg: TrainConfig, out_path: str = "nucleus_unet.pt",
    verbose: bool = True,
):
    """Train and checkpoint on best validation IoU (or last epoch if no val)."""
    import torch

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if len(meta) != len(X):
        raise ValueError(f"meta has {len(meta)} rows but X has {len(X)} crops; "
                         "they must be positionally aligned")
    weights = meta["weight"].to_numpy() if "weight" in meta.columns \
        else np.ones(len(meta))
    w_norm = float(np.mean(weights[train_idx])) if len(train_idx) else 1.0
    tl, vl = make_dataloaders(X, Y, weights, train_idx, val_idx, cfg,
                              mask_channel=(X.shape[1] > 1))

    model = build_unet(cfg.in_channels, cfg.base_filters, cfg.depth).to(device)
    if verbose:
        print(f"device={device}  params={count_parameters(model):,}  "
              f"train={len(train_idx)}  val={len(val_idx)}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_amp = cfg.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -inf, not -1: with no validation set the score is -loss, and BCE+Dice
    # starts above 1, so nothing was checkpointed until loss < 1 - and a
    # stale file from an earlier run would then be loaded instead.
    best, history = -np.inf, []
    for epoch in range(cfg.epochs):
        running = 0.0
        for x, y, w in tl:
            x, y, w = x.to(device), y.to(device), w.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=use_amp):
                logits = model(x)
            loss = _loss_fn(logits, y, w, cfg, w_norm)
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        sched.step()

        rec = {"epoch": epoch, "loss": running / max(len(tl), 1)}
        if vl is not None:
            rec.update({f"val_{k}": v for k, v in
                        _evaluate(model, vl, device).items()})
            score = rec["val_iou"]
        else:
            score = -rec["loss"]
        history.append(rec)

        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(),
                        "config": cfg.__dict__, "epoch": epoch}, out_path)
        if verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
            msg = f"epoch {epoch:3d}  loss {rec['loss']:.4f}"
            if vl is not None:
                msg += f"  val_iou {rec['val_iou']:.4f}"
            print(msg, flush=True)

    return model, pd.DataFrame(history)


def evaluate_by_condition(model, X, Y, meta: pd.DataFrame, idx,
                          threshold: float = 0.5,
                          batch_size: int = 64) -> pd.DataFrame:
    """Per-condition IoU on a held-out index set - the number that matters."""
    import torch
    if len(meta) != len(X):
        raise ValueError(f"meta has {len(meta)} rows but X has {len(X)} crops")
    device = next(model.parameters()).device
    model.eval()
    rows = []
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            j = idx[s:s + batch_size]
            x = torch.from_numpy(np.asarray(X[j], dtype=np.float32)).to(device)
            y = torch.from_numpy(np.asarray(Y[j], dtype=np.float32)).to(device)
            p = (torch.sigmoid(model(x)) > threshold).float()
            inter = (p * y).sum(dim=(1, 2, 3))
            union = ((p + y) > 0).float().sum(dim=(1, 2, 3))
            iou = ((inter + 1e-6) / (union + 1e-6)).cpu().numpy()
            area_ratio = ((p.sum(dim=(1, 2, 3)) + 1e-6) /
                          (y.sum(dim=(1, 2, 3)) + 1e-6)).cpu().numpy()
            for k, jj in enumerate(j):
                rows.append(dict(idx=int(jj), iou=float(iou[k]),
                                 area_ratio=float(area_ratio[k])))
    model.train()
    df = pd.DataFrame(rows)
    df = df.join(meta.iloc[df["idx"]][["condition", "source"]].reset_index(drop=True))
    return df.groupby("condition").agg(
        n=("iou", "size"), iou_mean=("iou", "mean"),
        iou_median=("iou", "median"), area_ratio_median=("area_ratio", "median"),
    ).round(3).reset_index()
