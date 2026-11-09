"""Satellite fusion for GraphCast-SEA (thesis gap 5 / advisor's Proposal 4, "Eye–Brain").

Baseline ("concat"): the Tb columns are simply part of the per-cell input vector, so a cell only
sees the satellite value of its own 0.25° box, mixed into one MLP with 80+ other numbers.

Here the satellite input gets its own branch ("Eye") and is fused into the atmospheric grid
embedding ("Brain") after the grid encoder:

  Eye   : satellite columns (Tb, mask, sub-grid min/std/cold fractions, at t and t-1) as an image
          (B, n_sat, H, W) → small CNN (3×3, dilated 3×3, 3×3 → receptive field 9×9 cells ≈ 2.25°)
          → e (B, G, eye_hidden): cloud features WITH spatial context (cluster shape, nearby convection).
  gated : h ← h + σ(W_g[h, e]) ⊙ W_e e
          the atmospheric state decides, per cell and per feature, how much satellite to let in.
  xattn : local cross-attention: each cell's state h_i (query) attends over the k×k neighbouring
          eye features (keys/values) with a learned relative-position bias per head:
          h_i ← h_i + W_o · MHA(q = W_q h_i, K/V = W_k,v e_{N(i)})
          i.e. the state chooses WHICH nearby clouds matter (e.g. upwind convection), not just how much.

In both modes the satellite columns are removed from the Brain input, so the fusion path is the only
way satellite information enters → "no satellite / concat / gated / xattn" is a clean ablation.
The fusion output projection starts at zero: the model starts exactly as the satellite-free model
and learns to use the satellite, which keeps the start of training stable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EyeCNN(nn.Module):
    def __init__(self, n_sat: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_sat, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )

    def forward(self, sat: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """sat (B, G, n_sat) → (B, hidden, H, W)."""
        B, G, S = sat.shape
        img = sat.transpose(1, 2).reshape(B, S, *hw)
        return self.net(img)


class SatelliteFusion(nn.Module):
    def __init__(self, n_sat: int, hidden: int, grid_shape: tuple[int, int], mode: str = "xattn",
                 eye_hidden: int = 32, window: int = 5, heads: int = 4, attn_dim: int = 64):
        super().__init__()
        if mode not in ("gated", "xattn"):
            raise ValueError(f"fusion mode {mode!r}: use gated or xattn")
        if window % 2 == 0:
            raise ValueError("fusion window must be odd")
        self.mode, self.hw, self.window, self.heads = mode, tuple(grid_shape), window, heads
        self.eye = EyeCNN(n_sat, eye_hidden)
        if mode == "gated":
            self.gate = nn.Linear(hidden + eye_hidden, hidden)
            self.proj = nn.Linear(eye_hidden, hidden)
            nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        else:
            if attn_dim % heads:
                raise ValueError("attn_dim must be divisible by heads")
            self.q = nn.Linear(hidden, attn_dim)
            self.k = nn.Linear(eye_hidden, attn_dim)
            self.v = nn.Linear(eye_hidden, attn_dim)
            self.o = nn.Linear(attn_dim, hidden)
            nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)
            self.pos_bias = nn.Parameter(torch.zeros(heads, window * window))   # relative position per head
            self.norm_q = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, sat: torch.Tensor) -> torch.Tensor:
        """h (B, G, hidden) grid embedding; sat (B, G, n_sat) → fused (B, G, hidden)."""
        e_img = self.eye(sat.to(h.dtype) if sat.dtype != h.dtype else sat, self.hw)       # (B, E, H, W)
        B, E = e_img.shape[:2]
        G = h.size(1)
        if self.mode == "gated":
            e = e_img.flatten(2).transpose(1, 2)                                            # (B, G, E)
            g = torch.sigmoid(self.gate(torch.cat([h, e], dim=-1)))
            return h + g * self.proj(e)
        k2 = self.window * self.window
        nb = F.unfold(e_img, self.window, padding=self.window // 2)                         # (B, E*k2, G)
        nb = nb.view(B, E, k2, G).permute(0, 3, 2, 1)                                       # (B, G, k2, E)
        d = self.q.out_features // self.heads
        q = self.q(self.norm_q(h)).view(B, G, self.heads, d)                               # (B, G, h, d)
        k = self.k(nb).view(B, G, k2, self.heads, d)
        v = self.v(nb).view(B, G, k2, self.heads, d)
        att = torch.einsum("bghd,bgkhd->bghk", q, k) / d ** 0.5 + self.pos_bias            # (B, G, h, k2)
        # cells outside the domain were zero-padded by unfold: mask them out
        pad_mask = self._pad_mask(h.device)                                                  # (G, k2) bool
        att = att.masked_fill(~pad_mask.view(1, G, 1, k2), float("-inf"))
        att = att.softmax(dim=-1)
        out = torch.einsum("bghk,bgkhd->bghd", att, v).reshape(B, G, -1)
        return h + self.o(out)

    def _pad_mask(self, device) -> torch.Tensor:
        if getattr(self, "_mask_cache", None) is None or self._mask_cache.device != device:
            ones = torch.ones(1, 1, *self.hw, device=device)
            m = F.unfold(ones, self.window, padding=self.window // 2)[0]                    # (k2, G)
            self._mask_cache = (m > 0.5).t().contiguous()
        return self._mask_cache
