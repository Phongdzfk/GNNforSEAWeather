"""Merge raw ERA5 files, align NASA MERGIR to the ERA5 grid, normalise, write Zarr.

Ported from nb3-2025.ipynb. Two ways to use it:
  * `normalized_year` + `wfsea materialize` with data.zarr_layout=raw  → raw NetCDF straight into the
    fp16 memmap cache (default; no intermediate Zarr, fits Kaggle's 20 GB output for 2015–2024).
  * `preprocess_year` → per-year Zarr stores (v1 layout), if a Zarr copy is wanted.
Fixes vs nb3:
  1. No `shift(time=-1)` label store: labels are taken from the next state at
     load time, so year boundaries no longer silently drop / mis-pair samples.
  2. Satellite regridding is named honestly: `bilinear` (what nb3 did, via
     xarray.interp) or `box_mean` (average of all 4 km pixels inside each 0.25°
     cell — the area-conserving choice for a field this much finer than the grid).
  3. Normalisation stats are loaded from the train-only stats files and checked.
Owner: Phong.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from wfsea.config import ON_KAGGLE

JUNK_COORDS = ("expver", "step", "number")


def _drop_junk(ds: xr.Dataset) -> xr.Dataset:
    return ds.drop_vars([v for v in JUNK_COORDS if v in ds.variables], errors="ignore")


def standardise_coords(ds: xr.Dataset) -> xr.Dataset:
    ren = {k: v for k, v in {"valid_time": "time", "latitude": "lat", "longitude": "lon",
                             "level": "pressure_level"}.items() if k in ds.dims or k in ds.coords}
    return ds.rename(ren) if ren else ds


def open_era5_year(raw_root, year: int, chunks=None, schema: dict | None = None) -> xr.Dataset:
    """One year of ERA5 at 00/06/12/18 UTC, found by file content (see wfsea.data.raw_index).

    `schema` = {"pressure_vars", "levels", "surface_vars", "static_vars", "tp_accum_hours", "exclude"};
    defaults are the 14-channel baseline.
    """
    from wfsea.data.raw_index import open_era5_year_indexed
    return open_era5_year_indexed(year, raw_root, **(schema or {}))


def nasa_files(raw_root, year: int) -> list[str]:
    from wfsea.data.raw_index import nasa_files_indexed
    files = nasa_files_indexed(year, raw_root)
    if not files:
        raise FileNotFoundError(f"No MERGIR files (merg_{year}*.nc4) under {raw_root or 'default roots'}")
    return files


def open_nasa_year(raw_root, year: int, chunks=None) -> xr.Dataset:
    paths = nasa_files(raw_root, year)
    ds = xr.open_mfdataset(paths, combine="by_coords", chunks=chunks or {"time": 100}, engine="h5netcdf")
    if "Tb" not in ds.data_vars and "IR" in ds.data_vars:
        ds = ds.rename({"IR": "Tb"})
    return ds


def tb_sums(years, raw_root=None, file_stride: int = 5) -> tuple[float, float, int]:
    """(sum, sum of squares, count) of raw 4 km Tb over `years`, reading every `file_stride`-th file."""
    s = ss = 0.0; n = 0
    for y in years:
        for f in nasa_files(raw_root, y)[::file_stride]:
            with xr.open_dataset(f, engine="h5netcdf") as ds:
                da = ds["Tb" if "Tb" in ds else "IR"]
                if "time" in da.dims:               # each file has :00 and :30 — use the slot we train on
                    da = da.sel(time=da.time.dt.minute == 0)
                v = da.values.astype(np.float64)
            v = v[np.isfinite(v)]
            s += v.sum(); ss += (v * v).sum(); n += v.size
    return s, ss, n


def tb_stats_streaming(years, raw_root=None, file_stride: int = 5) -> tuple[float, float, int]:
    """Mean/std of raw 4 km Tb over the train years, reading every `file_stride`-th file.

    Reading all ~14,600 files of 10 years is I/O-bound for hours; a stride of 5 cycles through
    the four synoptic hours and keeps the estimate unbiased. Streaming float64 sums, no dask.
    """
    s, ss, n = tb_sums(years, raw_root, file_stride)
    mean = s / n
    return float(mean), float(np.sqrt(max(ss / n - mean * mean, 1e-12))), int(n)


TB_SUBGRID = ("Tb_min", "Tb_std", "Tb_f235", "Tb_f208")   # extra per-cell features from the 4 km pixels
TB_COLD = {"Tb_f235": 235.0, "Tb_f208": 208.0}            # deep convection / overshooting-top thresholds (K)


def regrid_tb(tb: xr.DataArray, lats: np.ndarray, lons: np.ndarray, method: str = "box_mean",
              subgrid: bool = False):
    """MERGIR 4 km → target 0.25° grid.

    subgrid=True (box_mean only) also returns, per 0.25° cell, statistics of the ~50 pixels inside it:
    min Tb (coldest cloud top), std (texture), and the fraction of pixels colder than 235 K / 208 K
    (deep convection). They keep the sub-grid convective structure that a plain mean averages away,
    at 4 extra channels instead of storing the 4 km images (~50 GB for 12 years). Returns a Dataset
    (Tb + TB_SUBGRID) in that case, else the mean as a DataArray.
    """
    if method == "bilinear":
        if subgrid:
            raise ValueError("sub-grid Tb statistics need regrid=box_mean")
        return tb.interp(lat=lats, lon=lons, method="linear")
    if method != "box_mean":
        raise ValueError(method)
    res_lat = abs(float(lats[1] - lats[0]))
    res_lon = abs(float(lons[1] - lons[0]))
    # Index of the target cell that each source pixel centre falls into (−1 = outside).
    def _bin(src, tgt, res):
        idx = np.rint((src - tgt[0]) / (tgt[1] - tgt[0])).astype(int)
        ok = (idx >= 0) & (idx < len(tgt)) & (np.abs(src - tgt[np.clip(idx, 0, len(tgt) - 1)]) <= res / 2 + 1e-9)
        return np.where(ok, idx, -1)
    il, jl = _bin(tb.lat.values, lats, res_lat), _bin(tb.lon.values, lons, res_lon)
    ii, jj = np.meshgrid(il, jl, indexing="ij")
    valid_pix = (ii >= 0) & (jj >= 0)
    flat = np.where(valid_pix, ii * len(lons) + jj, 0).ravel()
    n_cells = len(lats) * len(lons)

    def _one(frame: np.ndarray):
        v = frame.ravel()
        good = valid_pix.ravel() & np.isfinite(v)
        idx, vg = flat[good], v[good].astype(np.float64)
        c = np.bincount(idx, minlength=n_cells)
        s = np.bincount(idx, weights=vg, minlength=n_cells)
        has = c > 0
        mean = np.full(n_cells, np.nan, dtype=np.float32)
        np.divide(s, c, out=mean, where=has)
        if not subgrid:
            return mean.reshape(len(lats), len(lons)), None
        ss = np.bincount(idx, weights=vg * vg, minlength=n_cells)
        var = np.where(has, ss / np.maximum(c, 1) - np.nan_to_num(mean.astype(np.float64)) ** 2, np.nan)
        mn = np.full(n_cells, np.inf); np.minimum.at(mn, idx, vg)
        ex = {"Tb_min": np.where(has, mn, np.nan), "Tb_std": np.sqrt(np.clip(var, 0, None))}
        for k, thr in TB_COLD.items():
            ex[k] = np.where(has, np.bincount(idx, weights=(vg < thr).astype(np.float64), minlength=n_cells)
                             / np.maximum(c, 1), np.nan)
        return mean.reshape(len(lats), len(lons)), {k: a.astype(np.float32).reshape(len(lats), len(lons))
                                                     for k, a in ex.items()}

    res = [_one(tb.isel(time=t).values) for t in range(tb.sizes["time"])]
    coords = {"time": tb.time.values, "lat": lats, "lon": lons}
    mean = xr.DataArray(np.stack([r[0] for r in res]), dims=("time", "lat", "lon"), coords=coords, name="Tb")
    if not subgrid:
        return mean
    out = xr.Dataset({"Tb": mean})
    for k in TB_SUBGRID:
        out[k] = (("time", "lat", "lon"), np.stack([r[1][k] for r in res]))
    return out


def save_zarr(ds: xr.Dataset, path: str | Path, chunks=None) -> None:
    for v in ds.variables:
        ds[v].encoding = {}
    if chunks:
        ds = ds.chunk({k: v for k, v in chunks.items() if k in ds.dims})
    if os.path.exists(path):
        shutil.rmtree(path)
    ds.to_zarr(path, mode="w", consolidated=True)


def normalized_year(year: int, stats, era5_root=None, nasa_root=None, regrid: str = "box_mean",
                    schema: dict | None = None, tb_subgrid: bool = False):
    """Lazy normalised ERA5 dataset + eager normalised Tb on the ERA5 grid for one year (no files written)."""
    from wfsea.data.stats import tp_forward
    ds = open_era5_year(era5_root, year, schema=schema)
    if stats.tp_transform and "tp" in ds:          # same transform the stats were computed with
        ds["tp"] = tp_forward(ds["tp"], stats.tp_transform)
    norm = {}
    for v in ds.data_vars:
        da = ds[v]
        if "pressure_level" in da.dims:
            levs = da.pressure_level.values
            m = xr.DataArray([stats.mean[f"{v}_{int(l)}hpa"] for l in levs], coords=[("pressure_level", levs)])
            sd = xr.DataArray([stats.std[f"{v}_{int(l)}hpa"] for l in levs], coords=[("pressure_level", levs)])
        else:
            m, sd = stats.mean[v], stats.std[v]
        norm[v] = ((da - m) / sd).astype("float32")
    brain = xr.Dataset(norm)
    tb = open_nasa_year(nasa_root, year)["Tb"]
    tb = tb.sel(time=tb.time.isin(brain.time.values))     # on-the-hour slots only
    if tb_subgrid:
        r = regrid_tb(tb, brain.lat.values, brain.lon.values, regrid, subgrid=True)
        eye = xr.Dataset({
            "Tb": ((r["Tb"] - stats.tb_mean) / stats.tb_std).astype("float32"),
            "Tb_min": ((r["Tb_min"] - stats.tb_mean) / stats.tb_std).astype("float32"),   # same scale as Tb
            "Tb_std": (r["Tb_std"] / stats.tb_std).astype("float32"),
            "Tb_f235": r["Tb_f235"].astype("float32"), "Tb_f208": r["Tb_f208"].astype("float32"),  # already 0..1
        })
        return brain, eye
    eye = regrid_tb(tb, brain.lat.values, brain.lon.values, regrid)
    eye = ((eye - stats.tb_mean) / stats.tb_std).astype("float32").to_dataset(name="Tb")
    return brain, eye


def preprocess_year(year: int, era5_root, nasa_root, stats, out_root, regrid: str = "box_mean") -> dict:
    """Write brain_<year>.zarr (normalised ERA5 incl. statics) and eye_<year>.zarr (normalised Tb)."""
    from wfsea.constants import STATIC_VARS
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    brain, eye = normalized_year(year, stats, era5_root, nasa_root, regrid)
    save_zarr(brain, out_root / f"brain_{year}.zarr", {"time": 100, "lat": 50, "lon": 50})
    eye.attrs["regrid"] = regrid
    save_zarr(eye, out_root / f"eye_{year}.zarr", {"time": 100, "lat": 50, "lon": 50})
    return {"year": year, "n_time_era5": int(brain.sizes["time"]), "n_time_tb": int(eye.sizes["time"]),
            "static_present": [v for v in STATIC_VARS if v in brain.data_vars], "regrid": regrid}
