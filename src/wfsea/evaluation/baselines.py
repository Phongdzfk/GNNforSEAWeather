"""Persistence and climatology baselines.

FIX vs nb5: the notebooks used the TEST-set time mean as "climatology" (for the
climatology baseline and as the ACC reference). That uses 2026 data to forecast
2026. Here climatology is computed from the TRAIN split, per (month, hour UTC).
Owner: Phong (data) — consumed by Tung's evaluator.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from wfsea.data.memmap import SplitCache


def month_hour_keys(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times).astype("datetime64[h]")
    month = (t.astype("datetime64[M]").astype(int) % 12)
    hour = (t - t.astype("datetime64[D]")).astype(int) // 6
    return month, hour


def compute_climatology(train_cache: SplitCache, block: int = 200) -> np.ndarray:
    """(12, 4, G, C) float32 mean of the normalised state on the train split."""
    st = train_cache.mm["state"]
    month, hour = month_hour_keys(train_cache.times)
    acc = np.zeros((12, 4, train_cache.G, train_cache.C), np.float64)
    cnt = np.zeros((12, 4), np.int64)
    for t0 in range(0, train_cache.T, block):
        t1 = min(t0 + block, train_cache.T)
        a = st[t0:t1].astype(np.float32)
        for i in range(t1 - t0):
            m, h = month[t0 + i], hour[t0 + i]
            acc[m, h] += a[i]; cnt[m, h] += 1
    clim = acc / np.maximum(cnt, 1)[..., None, None]
    empty = cnt == 0
    if empty.any():
        # Only happens with a partial train split (e.g. smoke tests): fall back to the hour-of-day mean.
        warnings.warn(f"{int(empty.sum())} empty (month, hour) climatology bins — filled with hour mean")
        for h in range(4):
            have = ~empty[:, h]
            fill = (acc[have, h].sum(0) / cnt[have, h].sum()) if have.any() else acc.sum((0, 1)) / max(cnt.sum(), 1)
            clim[empty[:, h], h] = fill
    return clim.astype(np.float32)


def save_climatology(clim: np.ndarray, path: str | Path, years, channels=None) -> None:
    extra = {"channels": np.array([f"{v}|{'' if l is None else l}" for v, l in channels])} if channels else {}
    np.savez_compressed(path, clim=clim, years=np.array(years), **extra)


def load_climatology(path: str | Path, channels=None) -> np.ndarray:
    """(12, 4, G, C). With `channels`, picks those channels by name (e.g. B0's 14 of the 41 stored)."""
    z = np.load(path)
    clim = z["clim"]
    if channels is None:
        return clim
    want = [f"{v}|{'' if l is None else l}" for v, l in channels]
    if "channels" in z.files:
        have = list(z["channels"])
        if have == want:
            return clim
        missing = [w for w in want if w not in have]
        if missing:
            raise ValueError(f"climatology lacks channels {missing}")
        return clim[..., [have.index(w) for w in want]]
    if clim.shape[-1] != len(want):
        raise ValueError(f"climatology has {clim.shape[-1]} channels (no names stored), need {len(want)}")
    return clim


class Climatology:
    def __init__(self, clim: np.ndarray):
        self.clim = clim

    def at(self, time: np.datetime64) -> np.ndarray:
        m, h = month_hour_keys(np.array([time]))
        return self.clim[m[0], h[0]]


def compute_diff_std(cache: SplitCache, block: int = 200) -> np.ndarray:
    """Std of the 6-h tendency X(t+1) − X(t) per channel, in normalised units, over valid pairs.

    Used to normalise the residual target (GraphCast-style): without it the loss is dominated
    by fast variables (tp, u10, q) and nearly ignores slow ones (z500, msl).
    """
    st = cache.mm["state"]
    ok = np.diff(cache.times) == np.timedelta64(6, "h")
    s = np.zeros(cache.C); ss = np.zeros(cache.C); n = 0
    for t0 in range(0, cache.T - 1, block):
        t1 = min(t0 + block, cache.T - 1)
        a = st[t0:t1 + 1].astype(np.float32)
        d = (a[1:] - a[:-1])[ok[t0:t1]]
        s += d.sum((0, 1)); ss += (d.astype(np.float64) ** 2).sum((0, 1)); n += d.shape[0] * d.shape[1]
    mean = s / n
    return np.sqrt(np.maximum(ss / n - mean ** 2, 1e-12)).astype(np.float32)
