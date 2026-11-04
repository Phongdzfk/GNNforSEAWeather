"""Synthetic data in the exact legacy Kaggle layout, so the whole pipeline is testable on a laptop."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

LATS = np.arange(3.0, -0.01, -0.25)        # descending like ERA5 (13)
LONS = np.arange(100.0, 103.01, 0.25)      # 13
LEVELS_FILE_ORDER = [500, 850]             # deliberately NOT the canonical order
SFC = ["tp", "u10", "v10", "d2m", "t2m", "msl"]
PRS = ["z", "q", "u", "v"]
PHYS = {"t2m": (299.0, 3.0), "d2m": (294.0, 3.0), "u10": (0.0, 4.0), "v10": (0.0, 4.0), "msl": (100900.0, 300.0),
        "tp": (0.0002, 0.0006), "lsm": (0.4, 0.45), "geopotential_surf": (2000.0, 4000.0),
        "z_500hpa": (57500.0, 300.0), "z_850hpa": (14900.0, 150.0), "q_500hpa": (0.002, 0.001),
        "q_850hpa": (0.012, 0.003), "u_500hpa": (0.0, 6.0), "u_850hpa": (0.0, 6.0),
        "v_500hpa": (0.0, 5.0), "v_850hpa": (0.0, 5.0)}


def _series(year: int, n_days: int, rng) -> xr.Dataset:
    times = pd.date_range(f"{year}-01-01", periods=n_days * 4 + 1, freq="6h")
    T = len(times)
    la, lo = np.meshgrid(LATS, LONS, indexing="ij")
    ph = np.arange(T)[:, None, None] * 0.4
    base = np.sin(la[None] + ph) * np.cos(lo[None] / 2 - ph / 3)
    vars_ = {}
    for i, v in enumerate(SFC):
        vars_[v] = (("time", "lat", "lon"), (base * (1 + 0.1 * i) + 0.05 * rng.standard_normal((T, *la.shape))).astype("f4"))
    for i, v in enumerate(PRS):
        arr = np.stack([base * (1 - 0.1 * i) + 0.1 * j for j in range(2)], axis=1)
        vars_[v] = (("time", "pressure_level", "lat", "lon"), arr.astype("f4"))
    vars_["lsm"] = (("time", "lat", "lon"), np.broadcast_to((lo > 101.5).astype("f4") * 2 - 1, (T, *la.shape)).copy())
    vars_["geopotential_surf"] = (("time", "lat", "lon"), np.broadcast_to((la / 3).astype("f4"), (T, *la.shape)).copy())
    return xr.Dataset(vars_, coords={"time": times, "pressure_level": LEVELS_FILE_ORDER, "lat": LATS, "lon": LONS})


@pytest.fixture(scope="session")
def legacy_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("kaggle_input")
    rng = np.random.default_rng(0)
    for year, days in ((2020, 6), (2021, 6), (2022, 3), (2023, 4)):
        s = _series(year, days, rng)
        x = s.isel(time=slice(0, -1))
        label = s.shift(time=-1).dropna("time", how="all")    # exactly what nb3 did
        assert np.array_equal(label.time.values, x.time.values)
        tb = (np.sin(np.arange(x.sizes["time"]))[:, None, None] * np.ones((1, len(LATS), len(LONS)))).astype("f4")
        tb[3, :2, :2] = np.nan
        eye = xr.Dataset({"Tb": (("time", "lat", "lon"), tb)}, coords={"time": x.time, "lat": LATS, "lon": LONS})
        x.to_zarr(root / f"brain_{year}_x.zarr", consolidated=True)
        label.to_zarr(root / f"brain_{year}_label.zarr", consolidated=True)
        eye.to_zarr(root / f"eye_{year}_norm.zarr", consolidated=True)
        s.to_zarr(root / f"_truth_{year}.zarr", consolidated=True)
    stats = {k: {"mean": m, "std": sd} for k, (m, sd) in PHYS.items()}
    stats["_meta"] = {"years": [2020, 2021]}
    (root / "era5_normalization_stats.json").write_text(json.dumps(stats))
    (root / "nasa_normalization_stats.json").write_text(json.dumps({"mean": 260.0, "std": 25.0}))
    return root


@pytest.fixture(scope="session")
def cfg(legacy_data, tmp_path_factory):
    from wfsea.config import load_config, resolve_paths
    out = tmp_path_factory.mktemp("out")
    c = load_config("smoke.yaml", [
        f"paths.zarr_root={legacy_data}", f"paths.stats_dir={legacy_data}",
        f"paths.out={out}", f"paths.scratch={out / 'cache'}", "data.zarr_layout=legacy",
        "data.train_years=[2020,2021]", "data.val_years=[2022]", "data.test_years=[2023]",
        "graph.mesh_res=1.0", "graph.mesh_lat_start=0.0", "graph.mesh_lat_stop=3.5",
        "graph.mesh_lon_start=100.0", "graph.mesh_lon_stop=103.5",
        "model.hidden=32", "model.n_layers=2", "model.unet_base=8", "model.unet_depth=2",
        "eval.n_starts=3", "eval.n_steps=8",
    ])
    return resolve_paths(c)
