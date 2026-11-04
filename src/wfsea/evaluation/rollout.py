"""Autoregressive rollout on a materialised split."""
from __future__ import annotations

import numpy as np
import torch

from wfsea.data.dataset import assemble_input
from wfsea.data.features import ForcingBuilder


@torch.no_grad()
def rollout(model, cache, spec, t0: int, n_steps: int, tb_policy: str = "persist", device="cpu",
            bmask: np.ndarray | None = None) -> np.ndarray:
    """Forecast X(t0+1..t0+n_steps) from X(t0-1), X(t0). Returns normalised (n_steps, G, C).

    bmask: boundary band forced with the ERA5 state at every step (when that state exists in the cache).
    """
    fb = ForcingBuilder(spec, cache, tb_policy)
    st = cache.mm["state"]
    bm = torch.from_numpy(bmask).to(device).view(1, -1, 1) if bmask is not None and bmask.any() else None
    dyn_t = torch.from_numpy(st[t0].astype(np.float32))[None].to(device)
    dyn_tm1 = torch.from_numpy(st[t0 - 1].astype(np.float32))[None].to(device)
    out = []
    for k in range(n_steps):
        forcing = torch.from_numpy(fb(t0, k))[None].to(device)
        pred = model(assemble_input(dyn_t, dyn_tm1, forcing)).float()
        if bm is not None and t0 + k + 1 < cache.T:
            truth = torch.from_numpy(st[t0 + k + 1].astype(np.float32))[None].to(device)
            pred = torch.where(bm, truth, pred)
        out.append(pred[0].cpu().numpy())
        dyn_tm1, dyn_t = dyn_t, pred
    return np.stack(out)


def evenly_spaced_starts(valid: np.ndarray, n_starts: int) -> np.ndarray:
    """Deterministic start list spread over the split (the review asks for a fixed, published list)."""
    if len(valid) <= n_starts:
        return valid
    return valid[np.linspace(0, len(valid) - 1, n_starts).round().astype(int)]
