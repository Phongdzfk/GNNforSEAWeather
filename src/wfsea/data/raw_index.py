"""Find raw ERA5 / MERGIR files by their CONTENT, not by folder or file names.

Why: the team's download notebooks name files like
  era5_data/data_stream-oper_stepType-instant_3.nc       (single levels, month 3)
  era5_data_pressure/data_stream-oper_stepType-instant_3.nc   (pressure levels!)
with the year only in the Kaggle notebook name, hourly time steps, and sometimes a
hard-coded year that differs from the notebook. Guessing from names is fragile, so every
.nc file under the input roots is opened (metadata only) and indexed by
  kind (pressure / single), variables, pressure levels, time range, time step.

Rules when assembling one year:
  * only 00/06/12/18 UTC are kept (hourly downloads are subsampled);
  * tp from hourly files is summed over the preceding `tp_accum_hours` (6 → true 6-h rain);
  * if several file sets provide the same (variable, level) — e.g. an old 2-level download
    and a new 7-level one — the set with MORE pressure levels wins, then the one with more files.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wfsea.config import ON_KAGGLE

JUNK = ("expver", "number", "step")
TIME_NAMES = ("valid_time", "time")
LEVEL_NAMES = ("pressure_level", "level", "isobaricInhPa")
SYNOPTIC = (0, 6, 12, 18)


@dataclass(frozen=True)
class RawFile:
    path: str
    kind: str                     # "pressure" | "single"
    vars: tuple[str, ...]
    levels: tuple[int, ...] | None
    t0: np.datetime64
    t1: np.datetime64
    step_h: float

    @property
    def group(self) -> str:       # files of one download set live in one folder
        return str(Path(self.path).parent)


def default_roots(roots=None) -> list[str]:
    if roots:
        return [str(r) for r in (roots if isinstance(roots, (list, tuple)) else [roots])]
    return ["/kaggle/input", "/kaggle/working"] if ON_KAGGLE else ["data/raw"]


def _walk(roots, suffix: str):
    for root in roots:
        if not Path(root).exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.endswith(".zarr") and not d.startswith(".")]
            for f in filenames:
                if f.endswith(suffix):
                    yield os.path.join(dirpath, f)


def _describe(path: str) -> RawFile | None:
    try:
        with xr.open_dataset(path, engine="h5netcdf", decode_times=True) as ds:
            tname = next((t for t in TIME_NAMES if t in ds.coords), None)
            if tname is None or not {"latitude", "longitude"} & set(ds.coords) | {"lat", "lon"} & set(ds.coords):
                return None
            t = ds[tname].values
            lname = next((l for l in LEVEL_NAMES if l in ds.coords), None)
            vars_ = tuple(v for v in ds.data_vars if v not in JUNK)
            levels = tuple(int(x) for x in np.atleast_1d(ds[lname].values)) if lname else None
            step = float(np.median(np.diff(t)).astype("timedelta64[m]").astype(int) / 60) if len(t) > 1 else 0.0
            return RawFile(path, "pressure" if lname else "single", vars_, levels, t.min(), t.max(), step)
    except Exception as e:  # unreadable / not ERA5
        print(f"[raw_index] skip {path}: {type(e).__name__}")
        return None


def scan_era5(roots=None, exclude: tuple[str, ...] = ()) -> list[RawFile]:
    return _scan_cached(tuple(default_roots(roots)), tuple(exclude))


@lru_cache(maxsize=8)
def _scan_cached(roots: tuple[str, ...], exclude: tuple[str, ...]) -> list[RawFile]:
    files = [p for p in _walk(roots, ".nc") if not any(x in p for x in exclude)]
    out = [d for d in map(_describe, sorted(files)) if d is not None]
    print(f"[raw_index] {len(out)} ERA5 NetCDF files indexed under {list(roots)}")
    return out


def summarize(index: list[RawFile]) -> pd.DataFrame:
    """One row per download set (folder): what it contains and which years."""
    rows = []
    for f in index:
        rows.append({"group": f.group, "kind": f.kind, "vars": ",".join(f.vars),
                     "levels": ",".join(map(str, f.levels or ())), "step_h": f.step_h,
                     "start": str(f.t0)[:13], "end": str(f.t1)[:13]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.groupby(["group", "kind", "vars", "levels", "step_h"])
              .agg(files=("start", "size"), start=("start", "min"), end=("end", "max")).reset_index())


@lru_cache(maxsize=4096)
def _open(path: str) -> xr.Dataset:
    ds = xr.open_dataset(path, engine="h5netcdf", chunks={})
    ren = {k: v for k, v in {"valid_time": "time", "latitude": "lat", "longitude": "lon", "level": "pressure_level",
                             "isobaricInhPa": "pressure_level"}.items() if k in ds.dims or k in ds.coords}
    ds = ds.rename(ren) if ren else ds
    return ds.drop_vars([v for v in JUNK if v in ds.variables], errors="ignore")


def _overlaps(f: RawFile, t0, t1) -> bool:
    return not (f.t1 < t0 or f.t0 > t1)


def _coverage(fs: list[RawFile], year: int) -> float:
    """Hours of `year` covered by a file set (approximate: union of file spans)."""
    y0, y1 = np.datetime64(f"{year}-01-01T00:00"), np.datetime64(f"{year}-12-31T23:59")
    spans = sorted((max(f.t0, y0), min(f.t1, y1)) for f in fs if not (f.t1 < y0 or f.t0 > y1))
    total, end = 0.0, None
    for a, b in spans:
        a = a if end is None or a > end else end
        if b > a:
            total += float((b - a) / np.timedelta64(1, "h")); end = b
    return total


def _rank_groups(cands: list[RawFile], year: int) -> list[list[RawFile]]:
    """Download sets (folders) best first: more pressure levels, then more of `year` covered.
    (A set that only touches the year — e.g. last year's file reaching 31 Dec — ranks low.)"""
    groups: dict[str, list[RawFile]] = {}
    for f in cands:
        groups.setdefault(f.group, []).append(f)
    return sorted(groups.values(), key=lambda fs: (max(len(f.levels or ()) for f in fs), _coverage(fs, year),
                                                    len(fs)), reverse=True)


def _pick_group(cands: list[RawFile], year: int | None = None) -> list[RawFile]:
    if year is None:
        year = int(str(max(f.t1 for f in cands))[:4])
    return _rank_groups(cands, year)[0]


def _series(files: list[RawFile], var: str, t0, t1, level: int | None) -> xr.DataArray:
    parts = []
    for f in sorted(files, key=lambda f: f.t0):
        da = _open(f.path)[var]
        if level is not None:
            da = da.sel(pressure_level=level)
        da = da.sel(time=slice(t0, t1))
        if da.sizes["time"]:
            parts.append(da)
    if not parts:
        raise FileNotFoundError(f"no data for {var}{'' if level is None else level} in {t0}..{t1}")
    da = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    _, keep = np.unique(da.time.values, return_index=True)            # overlapping downloads
    return da.isel(time=np.sort(keep)).sortby("time")


def _synoptic(da: xr.DataArray, year: int) -> xr.DataArray:
    t = pd.DatetimeIndex(da.time.values)
    keep = (t.year == year) & t.hour.isin(SYNOPTIC) & (t.minute == 0)
    return da.isel(time=np.where(keep)[0])


def open_era5_year_indexed(year: int, roots=None, pressure_vars=("z", "q", "u", "v"), levels=(850, 500),
                           surface_vars=("t2m", "d2m", "u10", "v10", "msl", "tp"),
                           static_vars=("lsm", "geopotential_surf"), tp_accum_hours: int = 6,
                           exclude: tuple[str, ...] = ()) -> xr.Dataset:
    """Lazy dataset of one year at 00/06/12/18 UTC: pressure vars (time, pressure_level, lat, lon),
    surface + static vars (time, lat, lon). Names match the rest of the pipeline."""
    index = scan_era5(roots, exclude)
    t0 = np.datetime64(f"{year - 1}-12-31T00:00")      # previous evening needed for 6-h tp sums
    t1 = np.datetime64(f"{year}-12-31T23:59")
    out: dict[str, xr.DataArray] = {}

    def merged(cands, var, level, accumulate):
        """Series of `var` for `year` from ALL download sets that have it, best set first: each set is
        processed on its own (tp 6-h sums use that set's own time step), then gaps in the best set are
        filled from the next ones — e.g. 2016 = Jan–Feb from one notebook + Mar–Dec from another."""
        parts = []
        for fs in _rank_groups(cands, year):
            da = _series(fs, var, t0, t1, level)
            step = min(f.step_h for f in fs)
            if accumulate and tp_accum_hours > 1 and 0 < step < tp_accum_hours:
                # ERA5 hourly tp at t = rain in (t-1h, t]; sum of the last k hours = rain in (t-6h, t]
                k = int(round(tp_accum_hours / step))
                da = da.rolling(time=k, min_periods=k).sum()
            da = _synoptic(da, year)
            if da.sizes["time"]:
                parts.append(da)
        if not parts:
            raise FileNotFoundError(f"{year}: no data for {var}{'' if level is None else level}")
        da = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
        _, keep = np.unique(da.time.values, return_index=True)        # first = best set wins
        return da.isel(time=np.sort(keep)).sortby("time")

    for var in pressure_vars:
        per_level = []
        for lev in levels:
            cands = [f for f in index if f.kind == "pressure" and var in f.vars and f.levels and lev in f.levels
                     and _overlaps(f, t0, t1)]
            if not cands:
                raise FileNotFoundError(f"{year}: no pressure file with {var} at {lev} hPa")
            per_level.append(merged(cands, var, lev, False).expand_dims(pressure_level=[lev]))
        out[var] = xr.concat(per_level, dim="pressure_level", join="inner").transpose("time", "pressure_level", ...)

    want_single = {**{v: v for v in surface_vars}, **{v: v for v in static_vars}}
    want_single["geopotential_surf"] = "z"            # surface geopotential is called z in single-level files
    for name, src in want_single.items():
        if name not in surface_vars and name not in static_vars:
            continue
        cands = [f for f in index if f.kind == "single" and src in f.vars and _overlaps(f, t0, t1)]
        if not cands:
            raise FileNotFoundError(f"{year}: no single-level file with {src}")
        da = merged(cands, src, None, name == "tp")
        if name == "tp":
            da.attrs["accumulation"] = f"{tp_accum_hours} h"
        out[name] = da.rename(name)

    # keep only time steps present in EVERY variable (e.g. 2026: single levels stop earlier than pressure)
    common = None
    for da in out.values():
        t = pd.DatetimeIndex(da.time.values)
        common = t if common is None else common.intersection(t)
    lost = {k: int(v.sizes["time"] - len(common)) for k, v in out.items() if v.sizes["time"] > len(common)}
    if lost:
        print(f"[raw_index] {year}: {len(common)} common 6-h steps; dropped steps missing elsewhere: {lost}")
    ds = xr.Dataset({k: v.sel(time=common) for k, v in out.items()})
    ds.attrs["tp_accumulation_hours"] = tp_accum_hours
    return ds


# ── MERGIR (NASA) ─────────────────────────────────────────────────────────────

_MERG = re.compile(r"merg_(\d{10})")


def nasa_files_indexed(year: int, roots=None) -> list[str]:
    """MERGIR files of `year` at 00/06/12/18 UTC, found anywhere under the roots by file name."""
    hits = {}
    for p in _walk(default_roots(roots), ".nc4"):
        m = _MERG.search(os.path.basename(p))
        if m and m.group(1)[:4] == str(year) and int(m.group(1)[8:10]) in SYNOPTIC:
            hits.setdefault(m.group(1), p)          # first copy wins if a file was downloaded twice
    return [hits[k] for k in sorted(hits)]
