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
from torch.utils.data import Dataset, Sampler

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


class InterleavedBlockSampler(Sampler):
    """Random order that is cheap to read from a memmap on a slow disk.

    Valid starts are cut into blocks of `block_len` consecutive 6-h steps; blocks are shuffled and
    taken `batch_size` at a time; batch j of a group holds step j of each of its blocks. So every batch
    still has `batch_size` independent random times, but the next batch needs only ONE new state per
    sample (X(t-1), X(t) were read for the previous batch and sit in the OS page cache) → ~3× fewer
    bytes, read sequentially. Every start is used exactly once per epoch. Use with shuffle=False and the
    same batch_size; call set_epoch(e) for a new order each epoch.
    """

    def __init__(self, n: int, batch_size: int, block_len: int = 8, seed: int = 0):
        self.n, self.B, self.L, self.seed, self.epoch = int(n), int(batch_size), int(block_len), int(seed), 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def order(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch)
        off = int(rng.integers(self.L))                      # vary block boundaries between epochs
        cuts = list(range(off, self.n, self.L)) if off else list(range(self.L, self.n, self.L))
        blocks = np.split(np.arange(self.n), cuts)
        blocks = [blocks[i] for i in rng.permutation(len(blocks))]
        out = []
        for g in range(0, len(blocks), self.B):
            grp = blocks[g:g + self.B]
            for j in range(max(len(b) for b in grp)):
                out.extend(int(b[j]) for b in grp if j < len(b))
        return np.asarray(out)

    def __iter__(self):
        return iter(self.order().tolist())

    def __len__(self) -> int:
        return self.n
