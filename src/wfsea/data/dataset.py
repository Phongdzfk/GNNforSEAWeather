"""PyTorch dataset over a materialised split (Phong → Hieu hand-off contract).

Each item:
  dyn_t    (G, C)      X(t), normalised
  dyn_tm1  (G, C)      X(t-1)
  forcing  (K, G, F)   non-dynamic inputs for steps k = 0..K-1 (Tb policy applied)
  target   (K, G, C)   X(t+1) … X(t+K)
K = train.rollout_steps (1 for the baseline). `assemble_input` turns these into
the model input in the legacy channel order.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from wfsea.data.features import ForcingBuilder, InputSpec
from wfsea.data.memmap import SplitCache


def assemble_input(dyn_t: torch.Tensor, dyn_tm1: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
    """(B,G,C), (B,G,C), (B,G,F) → (B,G,2C+F). X(t) first: the model uses x[..., :C] as residual base."""
    return torch.cat([dyn_t, dyn_tm1, forcing], dim=-1)


class WeatherDataset(Dataset):
    def __init__(self, cache: SplitCache | str, spec: InputSpec | None = None, n_steps: int = 1,
                 tb_policy: str = "persist", indices: np.ndarray | None = None):
        from wfsea.data.memmap import open_cache
        self.cache = cache if isinstance(cache, SplitCache) else open_cache(cache)
        self.spec = spec or InputSpec(n_dyn=self.cache.C, n_static=self.cache.static.shape[1])
        self.n_steps = n_steps
        self.forcing = ForcingBuilder(self.spec, self.cache, tb_policy)
        self.indices = self.cache.valid_starts(n_future=n_steps) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        t = int(self.indices[i])
        st = self.cache.mm["state"]
        dyn_t = torch.from_numpy(st[t].astype(np.float32))
        dyn_tm1 = torch.from_numpy(st[t - 1].astype(np.float32))
        forcing = torch.from_numpy(np.stack([self.forcing(t, k) for k in range(self.n_steps)]))
        target = torch.from_numpy(st[t + 1:t + 1 + self.n_steps].astype(np.float32))
        return dyn_t, dyn_tm1, forcing, target
