"""External benchmark: operational ECMWF forecasts scored on OUR grid against OUR truth (ERA5).

Source: dynamical.org analysis-ready archives (Zarr/Icechunk on AWS Open Data), opened with
`dynamical_catalog` (pip install "dynamical-catalog>=0.8.0"; needs internet on Kaggle):

  aifs         ecmwf-aifs-single-forecast                  ECMWF's data-driven model, deterministic,
                                                          init every 6 h since 2024-04, 0.25°, 6-hourly leads
  ifs_control  ecmwf-ifs-ens-forecast-15-day-0-25-degree   physics NWP (IFS), ensemble CONTROL member 0,
                                                          00 UTC inits only, 3-hourly leads up to 144 h

Unit conversions to our channels: °C → K; geopotential height (m) → geopotential (m² s⁻²);
precipitation_surface is the MEAN RATE since the previous output step (kg m⁻² s⁻¹ = mm s⁻¹) →
6-h accumulation in metres = Σ rate·Δt / 1000 over the steps inside (t-6 h, t].

Caveats for the thesis (write them next to the numbers):
  * these models start from ECMWF's operational analysis, ours from ERA5; all are scored against ERA5,
    which slightly favours ours at the first steps (same "truth" as our initial state);
  * AIFS/IFS are global models run on supercomputers — the comparison is a reference, not a contest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

G0 = 9.80665
SOURCES = {
    "aifs": dict(dataset="ecmwf-aifs-single-forecast", member=None),
    "ifs_control": dict(dataset="ecmwf-ifs-ens-forecast-15-day-0-25-degree", member=0),
}
# our channel name → (their variable, conversion)
_K = lambda x: x + 273.15
_GPH = lambda x: x * G0
VARS = {
    "t2m": ("temperature_2m", _K), "d2m": ("dew_point_temperature_2m", _K),
    "u10": ("wind_u_10m", None), "v10": ("wind_v_10m", None),
    "msl": ("pressure_reduced_to_mean_sea_level", None),
    "z500": ("geopotential_height_500hpa", _GPH), "z850": ("geopotential_height_850hpa", _GPH),
    "z925": ("geopotential_height_925hpa", _GPH),
    "t850": ("temperature_850hpa", _K), "t925": ("temperature_925hpa", _K),
    "tp": ("precipitation_surface", None),
}


def open_source(name: str):
    import dynamical_catalog
    return dynamical_catalog.open(SOURCES[name]["dataset"], chunks=None)


class ExternalForecast:
    """Callable: (init_time, n_steps) → (n_steps, G, C) physical array on our grid, NaN for channels the
    source does not provide; None if that init time is not in the archive. Results are cached on disk."""

    def __init__(self, name: str, lats, lons, channel_names, step_hours: int = 6, ds=None,
                 cache_dir: str | Path | None = None, accum_hours: int = 6):
        self.name, self.channels = name, list(channel_names)
        self.member = SOURCES.get(name, {}).get("member")
        self.ds = ds if ds is not None else open_source(name)
        self.step, self.accum = int(step_hours), int(accum_hours)
        self.lats, self.lons = np.asarray(lats, float), np.asarray(lons, float)
        lat_src = self.ds["latitude"].values
        lon_src = self.ds["longitude"].values
        lon_q = (self.lons + 180.0) % 360.0 - 180.0                            # any convention → their [−180, 180)
        self.ilat = _match(lat_src, self.lats, "latitude")
        self.ilon = _match(lon_src, lon_q, "longitude")
        self.inits = {np.datetime64(t, "ns"): i for i, t in enumerate(self.ds["init_time"].values)}
        self.leads_h = (self.ds["lead_time"].values / np.timedelta64(1, "h")).astype(float) \
            if np.issubdtype(self.ds["lead_time"].dtype, np.timedelta64) else self.ds["lead_time"].values / 3600.0
        self.cache_dir = Path(cache_dir) / name if cache_dir else None

    def available(self, init_time) -> bool:
        return np.datetime64(init_time, "ns") in self.inits

    def __call__(self, init_time, n_steps: int) -> np.ndarray | None:
        it = np.datetime64(init_time, "ns")
        if it not in self.inits:
            return None
        key = str(it)[:13].replace(":", "")
        if self.cache_dir and (self.cache_dir / f"{key}_{n_steps}.npy").exists():
            return np.load(self.cache_dir / f"{key}_{n_steps}.npy")
        G = len(self.lats) * len(self.lons)
        out = np.full((n_steps, G, len(self.channels)), np.nan, np.float32)
        leads = [(k + 1) * self.step for k in range(n_steps)]
        for c, ch in enumerate(self.channels):
            if ch not in VARS or VARS[ch][0] not in self.ds:
                continue
            var, conv = VARS[ch]
            if ch == "tp":
                out[:, :, c] = self._accum(var, it, leads)
                continue
            li = [self._lead_index(h) for h in leads]
            if any(i is None for i in li):
                continue
            a = self._read(var, it, li)                                          # (n_steps, H, W)
            out[:, :, c] = (conv(a) if conv else a).reshape(n_steps, G)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_dir / f"{key}_{n_steps}.npy", out)
        return out

    # ── helpers ───────────────────────────────────────────────────────────────
    def _lead_index(self, hours: float):
        i = np.where(np.isclose(self.leads_h, hours))[0]
        return int(i[0]) if len(i) else None

    def _read(self, var, it, lead_idx):
        da = self.ds[var].isel(init_time=self.inits[it], lead_time=lead_idx)
        if "ensemble_member" in da.dims:
            da = da.isel(ensemble_member=self.member or 0)
        da = da.isel(latitude=self.ilat, longitude=self.ilon)
        return np.asarray(da.transpose("lead_time", "latitude", "longitude").values, np.float64)

    def _accum(self, var, it, leads):
        """6-h accumulation (m) ending at each lead from mean rates over the output intervals inside it."""
        res = []
        for L in leads:
            idx, dts = [], []
            prev = L - self.accum
            inside = [h for h in self.leads_h if prev < h <= L]
            for h in sorted(inside):
                j = self._lead_index(h)
                lower = max([x for x in self.leads_h if x < h], default=None)
                if j is None or lower is None:
                    return np.full((len(leads), len(self.lats) * len(self.lons)), np.nan, np.float32)
                idx.append(j); dts.append((h - lower) * 3600.0)
            if not idx or abs(sum(dts) - self.accum * 3600.0) > 1:
                return np.full((len(leads), len(self.lats) * len(self.lons)), np.nan, np.float32)
            r = self._read(var, it, idx)                                          # (n, H, W) mm/s
            res.append((np.clip(r, 0, None) * np.asarray(dts)[:, None, None]).sum(0) / 1000.0)
        return np.stack(res).reshape(len(leads), -1).astype(np.float32)


def _match(src: np.ndarray, want: np.ndarray, what: str) -> np.ndarray:
    """Indices of `want` in `src` (0.25° grids: exact up to rounding)."""
    order = np.argsort(src)
    pos = np.searchsorted(src[order], want)
    pos = np.clip(pos, 0, len(src) - 1)
    lo = np.clip(pos - 1, 0, len(src) - 1)
    best = np.where(np.abs(src[order][pos] - want) <= np.abs(src[order][lo] - want), pos, lo)
    idx = order[best]
    if np.abs(src[idx] - want).max() > 1e-3:
        raise ValueError(f"{what}: target grid is not on the source grid (max offset "
                         f"{np.abs(src[idx] - want).max():.4f}°)")
    return idx
