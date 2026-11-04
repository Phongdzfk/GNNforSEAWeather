"""U-Net baseline with the same interface as GraphCastSEA (ported from nb4-unet-baseline)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class UNetSEA(nn.Module):
    """forward(x (B, G, n_in)) → (B, G, n_out) via residual ΔX — fair comparison with the GNN."""

    def __init__(self, n_in: int, n_out: int, H: int, W: int, base: int = 48, depth: int = 3, delta_scale=None):
        super().__init__()
        ds = torch.ones(n_out) if delta_scale is None else torch.as_tensor(delta_scale, dtype=torch.float32)
        self.register_buffer("delta_scale", ds)
        self.n_in, self.n_out, self.H, self.W = n_in, n_out, H, W
        mult = 2 ** depth
        self.Hp, self.Wp = -(-H // mult) * mult, -(-W // mult) * mult
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.inc = DoubleConv(n_in, chs[0])
        self.downs = nn.ModuleList([DoubleConv(chs[i], chs[i + 1]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2) for i in range(depth, 0, -1)])
        self.ups = nn.ModuleList([DoubleConv(chs[i], chs[i - 1]) for i in range(depth, 0, -1)])
        self.outc = nn.Conv2d(chs[0], n_out, 1)
        nn.init.zeros_(self.outc.weight); nn.init.zeros_(self.outc.bias)

    def forward(self, x_grid: torch.Tensor) -> torch.Tensor:
        B = x_grid.size(0)
        base = x_grid[..., : self.n_out]
        x = x_grid.permute(0, 2, 1).reshape(B, self.n_in, self.H, self.W)
        x = F.pad(x, (0, self.Wp - self.W, 0, self.Hp - self.H), mode="replicate")
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(self.pool(skips[-1])))
        h = skips[-1]
        for i, (upc, up) in enumerate(zip(self.upconvs, self.ups)):
            h = up(torch.cat([upc(h), skips[-(i + 2)]], dim=1))
        out = self.outc(h)[:, :, : self.H, : self.W]
        return base + out.reshape(B, self.n_out, self.H * self.W).permute(0, 2, 1) * self.delta_scale.to(out.dtype)
