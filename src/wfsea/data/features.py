"""Input-channel schema: what goes into the model besides the dynamic state.

Model input at step k of a rollout started at t0 (legacy-compatible order):

  [ X(t), X(t-1) | Tb_cur, mask_cur, Tb_prev, mask_prev | lsm, z_sfc | sinφ cosφ sinλ cosλ | time(4) | solar(1) ]
    2 × 14 dyn      satellite (4)                          static (2)   position (4)          optional  optional

With the default (legacy) flags this is 28 + 4 + 2 + 4 = 38 channels, matching
the 538,382-parameter checkpoint. NOTE for the thesis: the legacy baseline has
NO time features — the "3 time channels" in docs/architecture.md do not exist
in the code; Proposal 2 adds them (features.time / features.solar).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

S0 = 1361.0  # W m-2


def toa_insolation(times: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Instantaneous top-of-atmosphere insolation / S0, shape (len(times), G). Proposal 2."""
    t = np.asarray(times).astype("datetime64[m]")
    doy = (t - t.astype("datetime64[Y]")).astype("timedelta64[D]").astype(float) + 1
    hour = (t - t.astype("datetime64[D]")).astype("timedelta64[m]").astype(float) / 60.0
    lat_g, lon_g = np.meshgrid(np.deg2rad(lats), lons, indexing="ij")
    lat_g, lon_g = lat_g.reshape(-1), lon_g.reshape(-1)
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0)          # (T,)
    ecc = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)                           # (d0/d)^2
    hour_angle = np.deg2rad(15.0 * (hour[:, None] + lon_g[None, :] / 15.0 - 12.0))
    cosz = (np.sin(lat_g)[None] * np.sin(decl)[:, None]
            + np.cos(lat_g)[None] * np.cos(decl)[:, None] * np.cos(hour_angle))
    return (ecc[:, None] * np.clip(cosz, 0, None)).astype(np.float32)


def time_harmonics(times: np.ndarray) -> np.ndarray:
    """(T, 4): sin/cos hour-of-day, sin/cos day-of-year (UTC)."""
    t = np.asarray(times).astype("datetime64[m]")
    doy = (t - t.astype("datetime64[Y]")).astype("timedelta64[D]").astype(float)
    hour = (t - t.astype("datetime64[D]")).astype("timedelta64[m]").astype(float) / 60.0
    a, b = 2 * np.pi * hour / 24.0, 2 * np.pi * doy / 365.25
    return np.stack([np.sin(a), np.cos(a), np.sin(b), np.cos(b)], axis=-1).astype(np.float32)


def position_features(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    la, lo = np.meshgrid(np.deg2rad(lats), np.deg2rad(lons), indexing="ij")
    return np.stack([np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)], axis=-1).reshape(-1, 4).astype(np.float32)


@dataclass
class InputSpec:
    n_dyn: int
    n_static: int
    satellite: bool = True
    position: bool = True
    static: bool = True
    time: bool = False
    solar: bool = False
    n_past: int = 2
    tb_subgrid: bool = False      # + 4 sub-grid Tb stats per lag (cache built with data.tb_subgrid)

    @classmethod
    def from_config(cls, cfg, n_dyn: int, n_static: int) -> "InputSpec":
        f = cfg.get("features", {})
        return cls(n_dyn=n_dyn, n_static=n_static, satellite=f.get("satellite", True),
                   position=f.get("position", True), static=f.get("static", True),
                   time=f.get("time", False), solar=f.get("solar", False),
                   n_past=cfg.get("data", {}).get("n_past", 2), tb_subgrid=f.get("tb_subgrid", False))

    @property
    def n_forcing(self) -> int:
        return ((4 + 8 * self.tb_subgrid) * self.satellite + self.n_static * self.static + 4 * self.position
                + 4 * self.time + 1 * self.solar)

    @property
    def n_in(self) -> int:
        return self.n_dyn * self.n_past + self.n_forcing

    def names(self, channel_names) -> list[str]:
        out = [f"{c}(t)" for c in channel_names] + [f"{c}(t-1)" for c in channel_names][: self.n_dyn * (self.n_past - 1)]
        if self.satellite:
            for lag in ("t", "t-1"):
                out += [f"Tb({lag})", f"Tb_mask({lag})"]
                if self.tb_subgrid:
                    out += [f"{k}({lag})" for k in ("Tb_min", "Tb_std", "Tb_f235", "Tb_f208")]
        if self.static: out += ["lsm", "geopotential_surf"][: self.n_static]
        if self.position: out += ["sin_lat", "cos_lat", "sin_lon", "cos_lon"]
        if self.time: out += ["sin_hour", "cos_hour", "sin_doy", "cos_doy"]
        if self.solar: out += ["toa_solar"]
        return out

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def satellite_slice(self) -> slice:
        """Columns of the full model input that hold satellite features (for the fusion module)."""
        start = self.n_dyn * self.n_past
        return slice(start, start + (4 + 8 * self.tb_subgrid) * self.satellite)


def tb_source_index(t0: int, k: int, lag: int, policy: str) -> tuple[int, bool]:
    """Which observed Tb index to use for input time t0+k-lag, and whether it is available.

    persist: never look past t0 (operationally honest)
    oracle : use the real future observation (legacy notebooks — optimistic!)
    mask   : future Tb treated as missing
    """
    idx = t0 + k - lag
    if idx <= t0 or policy == "oracle":
        return idx, True
    if policy == "persist":
        return t0, True
    if policy == "mask":
        return idx, False
    raise ValueError(policy)


class ForcingBuilder:
    """Builds the non-dynamic part of the input for any (t0, k)."""

    def __init__(self, spec: InputSpec, cache, tb_policy: str = "persist"):
        self.spec, self.cache, self.tb_policy = spec, cache, tb_policy
        parts = []
        if spec.static and spec.n_static:
            parts.append(cache.static[:, : spec.n_static])
        if spec.position:
            parts.append(position_features(cache.lats, cache.lons))
        self._const = np.concatenate(parts, axis=1).astype(np.float32) if parts else np.zeros((cache.G, 0), np.float32)
        self._harm = time_harmonics(cache.times) if spec.time else None

    def __call__(self, t0: int, k: int) -> np.ndarray:
        G = self.cache.G
        cols = []
        if self.spec.satellite:
            mm = self.cache.mm
            nx = 4 if self.spec.tb_subgrid else 0
            if nx and "tbx" not in mm:
                raise ValueError("features.tb_subgrid=true but the cache has no sub-grid Tb (tbx.f16): rebuild "
                                 "the caches with data.tb_subgrid=true")
            for lag in (0, 1):
                idx, avail = tb_source_index(t0, k, lag, self.tb_policy)
                if avail:
                    cols += [mm["tb"][idx].astype(np.float32)[:, None], mm["tbmask"][idx].astype(np.float32)[:, None]]
                    if nx:
                        cols.append(np.asarray(mm["tbx"][idx], dtype=np.float32))
                else:
                    cols += [np.zeros((G, 2 + nx), np.float32)]
        cols.append(self._const)
        t = t0 + k
        if self.spec.time:
            cols.append(np.broadcast_to(self._harm[t], (G, 4)))
        if self.spec.solar:
            cols.append(toa_insolation(self.cache.times[t:t + 1], self.cache.lats, self.cache.lons)[0][:, None])
        return np.concatenate(cols, axis=1).astype(np.float32)
