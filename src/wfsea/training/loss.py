"""Latitude-weighted losses. Owner: Hieu (Proposal 3 spectral term: Tung + Hieu)."""
from __future__ import annotations

import numpy as np
import torch


def lat_weights_grid(lats: np.ndarray, n_lon: int) -> torch.Tensor:
    """w_i ∝ cos φ_i, normalised to mean 1, flattened to (G,)."""
    w = np.cos(np.deg2rad(lats)).astype(np.float32)
    w = w / w.mean()
    return torch.from_numpy(np.repeat(w, n_lon))


def weighted_rmse(pred: torch.Tensor, target: torch.Tensor, w_grid: torch.Tensor,
                  channel_w: torch.Tensor | None = None) -> torch.Tensor:
    """sqrt(mean_{b,g,c} w_g · (pred-target)²) — same definition as the notebooks (fp32)."""
    se = (pred.float() - target.float()) ** 2
    se = se * w_grid.view(1, -1, 1)
    if channel_w is not None:
        se = se * channel_w.view(1, 1, -1)
    return torch.sqrt(se.mean())


def spectral_loss(pred: torch.Tensor, target: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Proposal 3: penalise the log power-spectrum gap (fights blurring at long leads).
    pred/target: (B, G, C) with G = H*W."""
    B, G, C = pred.shape
    p = pred.float().permute(0, 2, 1).reshape(B, C, H, W)
    t = target.float().permute(0, 2, 1).reshape(B, C, H, W)
    sp = torch.fft.rfft2(p, norm="ortho").abs().pow(2)
    st = torch.fft.rfft2(t, norm="ortho").abs().pow(2)
    return (torch.log1p(sp) - torch.log1p(st)).abs().mean()
