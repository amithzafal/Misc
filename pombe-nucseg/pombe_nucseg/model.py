"""A small U-Net, sized for the task rather than for a benchmark.

Inside a cropped, cell-masked, orientation-normalised ROI the problem is binary
semantic segmentation of one blob in a 128x128 image. That does not need a ViT
backbone. With ~10^4-10^5 auto-labelled crops a ~1M-parameter U-Net saturates
the task, trains in minutes on one GPU, and runs fast enough for hundreds of
cells x hundreds of frames - which is where a SAM-class encoder would hurt.

Torch is imported lazily so the label-generation and validation modules stay
usable in environments without it.
"""

from __future__ import annotations


def _torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for model/training. "
            "Install with: pip install torch"
        ) from exc


def build_unet(in_channels: int = 2, base: int = 16, depth: int = 4):
    """Return a UNet instance (constructed lazily to avoid a hard torch import)."""
    torch, nn = _torch()

    class ConvBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class UNet(nn.Module):
        def __init__(self, in_channels, base, depth):
            super().__init__()
            self.depth = depth
            self.downs = nn.ModuleList()
            self.ups = nn.ModuleList()
            self.upconvs = nn.ModuleList()

            c = in_channels
            feats = []
            for d in range(depth):
                f = base * 2 ** d
                self.downs.append(ConvBlock(c, f))
                feats.append(f)
                c = f
            self.bottleneck = ConvBlock(c, c * 2)
            c = c * 2
            for f in reversed(feats):
                self.upconvs.append(nn.ConvTranspose2d(c, f, 2, stride=2))
                self.ups.append(ConvBlock(f * 2, f))
                c = f
            self.head = nn.Conv2d(c, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            skips = []
            for down in self.downs:
                x = down(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottleneck(x)
            for upconv, up, skip in zip(self.upconvs, self.ups, reversed(skips)):
                x = upconv(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    x = nn.functional.interpolate(x, size=skip.shape[-2:],
                                                  mode="nearest")
                x = up(torch.cat([skip, x], dim=1))
            return self.head(x)          # logits

    return UNet(in_channels, base, depth)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
