"""One-time Zarr → float16 memmap decode (the "11 h → 6 h" optimisation), made reproducible.

Cache layout for one split (e.g. <scratch>/test_2026):
  state.f16   (T, G, C)  normalised dynamic channels (order in meta.json; default DYNAMIC_CHANNELS)
  tb.f16      (T, G)     normalised Tb, NaN → 0
  tbmask.u8   (T, G)     1 = Tb observed, 0 = missing
  tbx.f16     (T, G, 4)  optional sub-grid Tb stats (min, std, frac<235 K, frac<208 K), see preprocess.TB_SUBGRID
  time.npy    (T,)       datetime64[ns]
  static.npy  (G, S)     normalised lsm, geopotential_surf
  meta.json   shapes, channel list, lat/lon, years, gaps, fp16 error report

Unlike the notebooks, there is no separate label file: the target of sample t is
state[t+1], and a sample is only valid if t-1, t, t+1 are exactly 6 h apart.
Owner: Phong (data contract → Hieu).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from wfsea.config import find_dir
from wfsea.constants import DYNAMIC_CHANNELS, STATIC_VARS, TIME_STEP_HOURS, channel_name

FORMAT_VERSION = 2


def _level_dim(da: xr.DataArray) -> str | None:
    return next((d for d in ("pressure_level", "level") if d in da.dims), None)


def _channel(ds: xr.Dataset, var: str, lev: int | None) -> xr.DataArray:
    da = ds[var]
    dim = _level_dim(da)
    if lev is not None:
        if dim is None:
            raise KeyError(f"{var} has no pressure level dim")
        da = da.sel({dim: lev})
    return da


def _read_block(ds: xr.Dataset, t0: int, t1: int, channels=DYNAMIC_CHANNELS) -> np.ndarray:
    """(t1-t0, G, C) float32. Each variable is read ONCE with all its levels: the raw files store the
    levels of a variable together, so reading level by level would decompress the same chunks 7×."""
    out = np.empty((t1 - t0, ds.sizes["lat"] * ds.sizes["lon"], len(channels)), np.float32)
    by_var: dict[str, list[tuple[int, int | None]]] = {}
    for i, (v, l) in enumerate(channels):
        by_var.setdefault(v, []).append((i, l))
    for v, items in by_var.items():
        da = ds[v].isel(time=slice(t0, t1))
        dim = _level_dim(da)
        if dim is None:
            a = da.values.astype(np.float32).reshape(t1 - t0, -1)
            for i, l in items:
                if l is not None:
                    raise KeyError(f"{v} has no pressure level dim")
                out[..., i] = a
            continue
        levs = [l for _, l in items]
        a = da.sel({dim: levs}).transpose("time", dim, ...).values.astype(np.float32).reshape(t1 - t0, len(levs), -1)
        for j, (i, _) in enumerate(items):
            out[..., i] = a[:, j]
    return out


def open_year(zarr_root: str | Path | None, year: int, layout: str, raw: dict | None = None
              ) -> tuple[xr.Dataset, xr.Dataset | None, xr.Dataset]:
    """Return (state_ds, extra_last_state_ds_or_None, eye_ds) for one year.

    layout: raw    = raw ERA5/MERGIR NetCDF, normalised on the fly (needs raw["stats"])
            legacy = Zarr written by the old nb3 notebooks (normalised with 2020–2024 stats!)
            v1     = Zarr written by `preprocess_year`
    """
    if layout == "raw":
        from wfsea.data.preprocess import normalized_year
        raw = raw or {}
        if raw.get("stats") is None:
            raise ValueError("layout=raw needs normalisation stats (run `wfsea stats` first)")
        brain, eye = normalized_year(year, raw["stats"], raw.get("era5_root"), raw.get("nasa_root"),
                                     raw.get("regrid", "box_mean"), raw.get("schema"),
                                     tb_subgrid=bool(raw.get("tb_subgrid")))
        return brain, None, eye
    def _p(name):
        if zarr_root is not None and (Path(zarr_root) / name).exists():
            return Path(zarr_root) / name
        return find_dir(name, [str(zarr_root)] if zarr_root else None)

    if layout == "legacy":
        x = xr.open_zarr(_p(f"brain_{year}_x.zarr"), consolidated=True)
        lab = xr.open_zarr(_p(f"brain_{year}_label.zarr"), consolidated=True)
        eye = xr.open_zarr(_p(f"eye_{year}_norm.zarr"), consolidated=True)
        # nb3 built labels with ds.shift(time=-1): label[i] holds the state at x.time[i] + 6 h but keeps
        # the time coordinate x.time[i]. So the only state missing from x is label[-1], valid 6 h later.
        extra = lab.isel(time=[-1])
        extra = extra.assign_coords(time=extra.time.values + np.timedelta64(TIME_STEP_HOURS, "h"))
        return x, extra, eye
    if layout == "v1":
        return (xr.open_zarr(_p(f"brain_{year}.zarr"), consolidated=True), None,
                xr.open_zarr(_p(f"eye_{year}.zarr"), consolidated=True))
    raise ValueError(layout)


def materialize_split(years, cache_dir: str | Path, zarr_root=None, layout: str = "legacy",
                      block: int = 100, force: bool = False, raw: dict | None = None,
                      channels=DYNAMIC_CHANNELS) -> Path:
    channels = tuple(tuple(c) for c in channels)
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if (meta.get("format_version") == FORMAT_VERSION and meta.get("years") == list(map(int, years))
                and meta.get("layout") == layout and [tuple(c) for c in meta["channels"]] == list(channels)
                and bool(meta.get("tb_subgrid")) == bool((raw or {}).get("tb_subgrid"))):
            print(f"[memmap] cache hit: {cache_dir}")
            return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    opened = [open_year(zarr_root, y, layout, raw) for y in years]
    first = opened[0][0]
    lats, lons = first.lat.values, first.lon.values
    G, C = len(lats) * len(lons), len(channels)
    T = sum(o[0].sizes["time"] + (o[1].sizes["time"] if o[1] is not None else 0) for o in opened)

    state = np.memmap(cache_dir / "state.f16", dtype=np.float16, mode="w+", shape=(T, G, C))
    tb = np.memmap(cache_dir / "tb.f16", dtype=np.float16, mode="w+", shape=(T, G))
    tbm = np.memmap(cache_dir / "tbmask.u8", dtype=np.uint8, mode="w+", shape=(T, G))
    from wfsea.data.preprocess import TB_SUBGRID
    sub = list(TB_SUBGRID) if all(k in opened[0][2].data_vars for k in TB_SUBGRID) else []
    tbx = (np.memmap(cache_dir / "tbx.f16", dtype=np.float16, mode="w+", shape=(T, G, len(sub)))
           if sub else None)
    times = np.empty(T, dtype="datetime64[ns]")
    nan_count = np.zeros(C, dtype=np.int64)
    q_err = np.zeros(C, dtype=np.float64); q_ref = np.zeros(C, dtype=np.float64)

    # RAM: a block is read as float32 plus a float16 copy and an error buffer (~3× block bytes).
    # Keep one block ≲ 0.5 GB so 41 channels on Kaggle (~30 GB RAM) stay far from the limit.
    block = max(8, min(block, int(5e8 // (G * C * 4))))
    pos = 0
    for (x, extra, eye), year in zip(opened, years):
        if not (np.array_equal(x.lat.values, lats) and np.array_equal(x.lon.values, lons)):
            raise ValueError(f"Grid of {year} differs from {years[0]}")
        eye_tb = eye["Tb"].reindex(time=x.time.values)   # missing Tb → NaN → mask 0
        eye_x = eye[sub].reindex(time=x.time.values) if sub else None
        parts = [(x, eye_tb, eye_x)]
        if extra is not None:
            parts.append((extra, eye["Tb"].reindex(time=extra.time.values),
                          eye[sub].reindex(time=extra.time.values) if sub else None))
        for ds_part, tb_part, x_part in parts:
            n = ds_part.sizes["time"]
            for t0 in range(0, n, block):
                t1 = min(t0 + block, n)
                a = _read_block(ds_part, t0, t1, channels)
                nan_count += np.isnan(a).sum(axis=(0, 1))
                a = np.nan_to_num(a, copy=False)
                a16 = a.astype(np.float16)
                # fp16 quantisation error (normalised units) — required by the review (q, tp!)
                q_err += ((a16.astype(np.float32) - a) ** 2).sum(axis=(0, 1))
                q_ref += (a ** 2).sum(axis=(0, 1))
                state[pos + t0:pos + t1] = a16
                tbr = tb_part.isel(time=slice(t0, t1)).values.astype(np.float32).reshape(t1 - t0, -1)
                tbm[pos + t0:pos + t1] = np.isfinite(tbr)
                tb[pos + t0:pos + t1] = np.nan_to_num(tbr, nan=0.0).astype(np.float16)
                if sub:
                    xb = np.stack([x_part[k].isel(time=slice(t0, t1)).values.astype(np.float32)
                                   .reshape(t1 - t0, -1) for k in sub], axis=-1)
                    tbx[pos + t0:pos + t1] = np.nan_to_num(xb, nan=0.0).astype(np.float16)
                print(f"  [memmap {year}] {pos + t1}/{T}", end="\r", flush=True)
            times[pos:pos + n] = ds_part.time.values
            pos += n
    print()
    order = np.argsort(times, kind="stable")
    if not np.all(order == np.arange(T)):
        raise ValueError("Years must be given in chronological order")
    state.flush(); tb.flush(); tbm.flush()
    if tbx is not None:
        tbx.flush()
    np.save(cache_dir / "time.npy", times)

    # Statics (time-invariant) — taken from the first timestep.
    ds0 = first.isel(time=0)
    stat_names = [v for v in STATIC_VARS if v in ds0.data_vars]
    static = (np.stack([ds0[v].values.reshape(-1) for v in stat_names], axis=1).astype(np.float32)
              if stat_names else np.zeros((G, 0), np.float32))
    np.save(cache_dir / "static.npy", np.nan_to_num(static))

    dt = np.diff(times).astype("timedelta64[h]").astype(int)
    gaps = [{"after": str(times[i])[:16], "hours": int(dt[i])} for i in np.where(dt != TIME_STEP_HOURS)[0]]
    meta = {
        "format_version": FORMAT_VERSION, "years": list(map(int, years)), "layout": layout,
        "norm_stats_years": (raw or {}).get("stats").meta.get("years") if layout == "raw" else None,
        "regrid": (raw or {}).get("regrid", "box_mean") if layout == "raw" else "bilinear (nb3)",
        "T": int(T), "G": int(G), "C": int(C), "n_lat": len(lats), "n_lon": len(lons),
        "lats": lats.tolist(), "lons": lons.tolist(),
        "channels": [[v, l] for v, l in channels], "static_vars": stat_names,
        "time_start": str(times[0])[:16], "time_end": str(times[-1])[:16],
        "gaps": gaps, "nan_count": dict(zip([channel_name(v, l) for v, l in channels], nan_count.tolist())),
        "tb_coverage": float(np.asarray(tbm).mean()),
        "tb_subgrid": sub,
        "fp16_relative_rmse": dict(zip([channel_name(v, l) for v, l in channels],
                                       np.sqrt(q_err / np.maximum(q_ref, 1e-12)).tolist())),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[memmap] {cache_dir}: T={T} ({meta['time_start']} → {meta['time_end']}), gaps={len(gaps)}")
    return cache_dir


class SplitCache:
    """Read-only view of a materialised split. Opens memmaps lazily (fork-safe)."""

    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.T, self.G, self.C = self.meta["T"], self.meta["G"], self.meta["C"]
        self.lats = np.asarray(self.meta["lats"]); self.lons = np.asarray(self.meta["lons"])
        self.times = np.load(self.dir / "time.npy")
        self.static = np.load(self.dir / "static.npy")
        self.channels = [(v, l) for v, l in self.meta["channels"]]
        self._mm = None

    @property
    def mm(self) -> dict:
        if self._mm is None:
            self._mm = {
                "state": np.memmap(self.dir / "state.f16", dtype=np.float16, mode="r", shape=(self.T, self.G, self.C)),
                "tb": np.memmap(self.dir / "tb.f16", dtype=np.float16, mode="r", shape=(self.T, self.G)),
                "tbmask": np.memmap(self.dir / "tbmask.u8", dtype=np.uint8, mode="r", shape=(self.T, self.G)),
            }
            if self.tb_subgrid:
                self._mm["tbx"] = np.memmap(self.dir / "tbx.f16", dtype=np.float16, mode="r",
                                            shape=(self.T, self.G, len(self.tb_subgrid)))
        return self._mm

    @property
    def tb_subgrid(self) -> list[str]:
        return list(self.meta.get("tb_subgrid") or [])

    def __getstate__(self):
        d = self.__dict__.copy(); d["_mm"] = None
        return d

    def valid_starts(self, n_future: int = 1, n_past: int = 2) -> np.ndarray:
        """Indices t with t-(n_past-1) … t+n_future all present and 6 h apart."""
        step = np.timedelta64(TIME_STEP_HOURS, "h")
        ok = np.diff(self.times) == step                    # ok[i]: i → i+1 is one step
        need = n_past - 1 + n_future
        if self.T <= need:
            return np.array([], dtype=int)
        run = np.ones(self.T - need, dtype=bool)
        for k in range(need):
            run &= ok[k:k + self.T - need]
        return np.where(run)[0] + (n_past - 1)


# ── Multi-part caches (Kaggle: one notebook output ≤ 20 GB) ─────────────────

class _Concat:
    """Read-only view over several memmaps along axis 0; supports a[i] and a[i:j]."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.offsets = np.cumsum([0] + [len(a) for a in arrays])
        self.shape = (int(self.offsets[-1]),) + tuple(arrays[0].shape[1:])

    def __len__(self):
        return self.shape[0]

    def _loc(self, i):
        p = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return p, i - self.offsets[p]

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            k = int(key) + (self.shape[0] if key < 0 else 0)
            p, j = self._loc(k)
            return self.arrays[p][j]
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            if step != 1:
                raise ValueError("step slicing not supported")
            out = []
            while start < stop:
                p, j = self._loc(start)
                n = min(stop - start, len(self.arrays[p]) - j)
                out.append(np.asarray(self.arrays[p][j:j + n]))
                start += n
            return np.concatenate(out) if len(out) > 1 else (out[0] if out else np.empty((0,) + self.shape[1:]))
        if key is Ellipsis or (isinstance(key, tuple) and key == ()):
            return self[0:self.shape[0]]
        raise TypeError(key)

    def __array__(self, dtype=None, copy=None):
        a = self[0:self.shape[0]]
        return a.astype(dtype) if dtype is not None else a


class MultiSplitCache(SplitCache):
    """Several consecutive SplitCache parts (e.g. train 2015–2018, 2019–2022, 2023–2024) seen as one."""

    def __init__(self, dirs):
        self.parts = [SplitCache(d) for d in dirs]
        self.parts.sort(key=lambda c: c.times[0])
        p0 = self.parts[0]
        for c in self.parts[1:]:
            if c.channels != p0.channels or c.G != p0.G:
                raise ValueError("cache parts have different channels or grids")
            if c.tb_subgrid != p0.tb_subgrid:
                raise ValueError("cache parts differ in sub-grid Tb channels (rebuild with the same data.tb_subgrid)")
        self.dir = p0.dir
        self.meta = dict(p0.meta)
        self.meta["years"] = sum((c.meta["years"] for c in self.parts), [])
        self.meta["parts"] = [str(c.dir) for c in self.parts]
        self.T = sum(c.T for c in self.parts); self.G, self.C = p0.G, p0.C
        self.lats, self.lons, self.static, self.channels = p0.lats, p0.lons, p0.static, p0.channels
        self.times = np.concatenate([c.times for c in self.parts])
        self._mm = None

    @property
    def mm(self) -> dict:
        if self._mm is None:
            keys = ["state", "tb", "tbmask"] + (["tbx"] if self.tb_subgrid else [])
            self._mm = {k: _Concat([c.mm[k] for c in self.parts]) for k in keys}
        return self._mm


def open_cache(path_or_paths) -> SplitCache:
    if isinstance(path_or_paths, (list, tuple)):
        return SplitCache(path_or_paths[0]) if len(path_or_paths) == 1 else MultiSplitCache(path_or_paths)
    return SplitCache(path_or_paths)


def part_years(years, n_channels: int, setting="auto", budget_bytes: float = 18e9,
               n_tb_extra: int = 0) -> list[list[int]]:
    """Split years into consecutive groups whose cache fits one Kaggle output (≤ ~17 GB)."""
    from wfsea.constants import N_LAT, N_LON
    years = [int(y) for y in years]
    if setting in (None, 0, "none"):
        return [years]
    if setting == "auto":
        per_year = 1461 * N_LAT * N_LON * (2 * n_channels + 3 + 2 * n_tb_extra)
        k = max(1, int(budget_bytes // per_year))
    else:
        k = int(setting)
    return [years[i:i + k] for i in range(0, len(years), k)]
