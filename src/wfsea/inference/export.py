"""Produce a physical-unit forecast file (for API / dashboard / later the assistant). Owner: Hieu → Tung.

forecast_<init>.nc  dims (lead, lat, lon), coords valid_time(lead), with
  raw channels in SI units and derived, human-friendly fields:
  t2m_c (°C), rh2m (%), wind10_speed (m/s), wind10_dir (deg, meteorological),
  tp_mm (mm in the hour ending at valid time — see note), msl_hpa (hPa).

NOTE (precipitation): ERA5 `tp` sampled at 00/06/12/18 from the hourly product is
the accumulation of the PRECEDING HOUR, not of 6 hours. The assistant reports it
as "mm/h at that time" until Phong confirms the download semantics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from wfsea.constants import TIME_STEP_HOURS, channel_name
from wfsea.evaluation.rollout import rollout


def derived_fields(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    if "t2m" in raw:
        out["t2m_c"] = raw["t2m"] - 273.15
    if "t2m" in raw and "d2m" in raw:
        # Magnus formula
        t, td = raw["t2m"] - 273.15, raw["d2m"] - 273.15
        es = 6.112 * np.exp(17.62 * t / (243.12 + t)); e = 6.112 * np.exp(17.62 * td / (243.12 + td))
        out["rh2m"] = np.clip(100 * e / es, 0, 100)
    if "u10" in raw and "v10" in raw:
        out["wind10_speed"] = np.hypot(raw["u10"], raw["v10"])
        out["wind10_dir"] = (np.degrees(np.arctan2(-raw["u10"], -raw["v10"])) + 360) % 360
    if "tp" in raw:
        out["tp_mm"] = np.clip(raw["tp"] * 1000.0, 0, None)
    if "msl" in raw:
        out["msl_hpa"] = raw["msl"] / 100.0
    return out


def forecast_to_dataset(fc_phys: np.ndarray, channels, lats, lons, init_time, model_name="graphcast_sea") -> xr.Dataset:
    """fc_phys (L, G, C) physical units → xarray Dataset."""
    L = fc_phys.shape[0]
    H, W = len(lats), len(lons)
    raw = {channel_name(v, l): fc_phys[..., i].reshape(L, H, W) for i, (v, l) in enumerate(channels)}
    fields = {**raw, **derived_fields(raw)}
    lead_h = (np.arange(L) + 1) * TIME_STEP_HOURS
    init = np.datetime64(init_time, "ns")
    ds = xr.Dataset({k: (("lead", "lat", "lon"), v.astype(np.float32)) for k, v in fields.items()},
                    coords={"lead": lead_h, "lat": lats, "lon": lons,
                            "valid_time": ("lead", init + lead_h.astype("timedelta64[h]"))})
    ds.attrs.update({"init_time": str(init)[:16], "model": model_name, "lead_units": "hours",
                     "tp_note": "tp_mm = precipitation in the hour ending at valid_time (ERA5 hourly accumulation)"})
    return ds


def export_forecast(model, cache, spec, stats, t0: int, n_steps: int, out_dir, tb_policy="persist", device="cpu",
                    bmask=None) -> Path:
    channels = cache.channels
    fc = rollout(model, cache, spec, t0, n_steps, tb_policy, device, bmask)
    fc = stats.denorm(fc, channels)
    ds = forecast_to_dataset(fc, channels, cache.lats, cache.lons, cache.times[t0])
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"forecast_{str(cache.times[t0])[:13].replace('-', '').replace('T', '_')}.nc"
    ds.to_netcdf(path)
    return path
