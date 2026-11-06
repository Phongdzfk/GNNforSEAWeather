"""Raw ERA5/MERGIR NetCDF → train-only stats → memmap (the default 2015–2024 path)."""
import json

import numpy as np
import pandas as pd
import xarray as xr

from wfsea.constants import DYNAMIC_CHANNELS
from wfsea.data.memmap import SplitCache, materialize_split
from wfsea.data.raw_index import nasa_files_indexed, scan_era5
from wfsea.data.stats import compute_stats_raw, load_stats, save_stats

LAT = np.arange(3.0, -0.01, -0.25)
LON = np.arange(100.0, 103.01, 0.25)
PHYS = {"t2m": 300, "d2m": 295, "u10": 1, "v10": -1, "msl": 101000, "tp": 0.0003, "z": 1, "q": 1, "u": 2, "v": 3}


def _write_year(folder, year, rng):
    folder.mkdir(parents=True)
    t = pd.date_range(f"{year}-01-01", periods=12, freq="6h")
    sh = (len(t), len(LAT), len(LON))
    co = {"valid_time": t, "latitude": LAT, "longitude": LON}
    f = lambda base, sc: (base + sc * rng.standard_normal(sh)).astype("f4")
    lev = [500, 850]
    for part, sl in ((1, slice(0, 6)), (2, slice(6, 12))):
        p = xr.Dataset({v: (("valid_time", "pressure_level", "latitude", "longitude"),
                            np.stack([f(PHYS[v] * (1 + i), 1.0) for i in range(2)], 1)) for v in "zquv"},
                       coords={**co, "pressure_level": lev}).isel(valid_time=sl)
        p.to_netcdf(folder / f"era5_pressure_{year}_part_{part}.nc", engine="h5netcdf")
    xr.Dataset({"tp": (("valid_time", "latitude", "longitude"), np.abs(f(0.0003, 0.0002)))}, coords=co) \
        .to_netcdf(folder / "data_stream-oper_stepType-accum.nc", engine="h5netcdf")
    inst = {v: (("valid_time", "latitude", "longitude"), f(PHYS[v], 2.0)) for v in ("t2m", "d2m", "u10", "v10", "msl")}
    inst["z"] = (("valid_time", "latitude", "longitude"), np.broadcast_to(np.linspace(0, 9000, len(LON)), sh).astype("f4"))
    inst["lsm"] = (("valid_time", "latitude", "longitude"), np.broadcast_to((LON > 101.5).astype("f4"), sh).copy())
    xr.Dataset(inst, coords=co).to_netcdf(folder / "data_stream-oper_stepType-instant.nc", engine="h5netcdf")


def _write_nasa(folder, year):
    folder.mkdir(parents=True)
    lat = np.arange(-0.2, 3.2, 0.0364); lon = np.arange(99.8, 103.2, 0.0364)
    for ts in pd.date_range(f"{year}-01-01", periods=12, freq="6h"):
        tb = np.full((2, len(lat), len(lon)), 250.0 + ts.hour, "f4")      # value encodes the hour
        tb[1] = 999.0                                                     # :30 slot must be ignored
        xr.Dataset({"Tb": (("time", "lat", "lon"), tb)},
                   coords={"time": [ts, ts + pd.Timedelta("30min")], "lat": lat, "lon": lon}) \
            .to_netcdf(folder / f"merg_{ts:%Y%m%d%H}_4km-pixel.nc4", engine="h5netcdf")


def test_raw_stats_and_memmap(tmp_path):
    rng = np.random.default_rng(1)
    era, nasa = tmp_path / "in", tmp_path / "in"
    _write_year(era / "era5-2015-2019" / "era5_data" / "year_2016", 2016, rng)   # different notebooks
    _write_year(era / "era5-2020-2026" / "era5_data" / "year_2020", 2020, rng)
    _write_year(era / "era5-2020-2026" / "era5_data" / "year_2025", 2025, rng)
    for y in (2016, 2020, 2025):
        _write_nasa(nasa / "nasa" / f"part{y}" / f"year_{y}", y)
    idx = scan_era5(era)
    assert {f.kind for f in idx} == {"pressure", "single"} and any("era5-2015-2019" in f.path for f in idx)
    assert len(nasa_files_indexed(2020, nasa)) == 12

    st = compute_stats_raw([2016, 2020], era, nasa, tb_file_stride=1)
    assert st.meta["years"] == [2016, 2020]
    assert abs(st.mean["t2m"] - 300) < 0.5 and abs(st.mean["z_850hpa"] - 2.0) < 0.2
    assert abs(st.tb_mean - 259.0) < 1e-3          # only the :00 slot (250 + hour); the :30 slot (999) is ignored
    save_stats(st, tmp_path / "stats"); st = load_stats(tmp_path / "stats"); st.check_train_only([2016, 2020])

    raw = {"stats": st, "era5_root": era, "nasa_root": nasa, "regrid": "box_mean"}
    c = SplitCache(materialize_split([2016, 2020], tmp_path / "cache_tr", layout="raw", raw=raw))
    assert c.T == 24 and c.meta["norm_stats_years"] == [2016, 2020] and len(c.meta["gaps"]) == 1
    phys = c.mm["state"][:, :, DYNAMIC_CHANNELS.index(("t2m", None))].astype(np.float32) * st.std["t2m"] + st.mean["t2m"]
    assert abs(phys.mean() - 300) < 0.5
    # Tb: only the on-the-hour slot, box-averaged, normalised with train stats
    tb = c.mm["tb"][:4].astype(np.float32) * st.tb_std + st.tb_mean
    np.testing.assert_allclose(tb.mean(1), [250, 256, 262, 268], atol=0.5)
    assert c.mm["tbmask"][:].min() == 1
    assert c.meta["static_vars"] == ["lsm", "geopotential_surf"]


def test_stats_partials_resume_and_split(tmp_path):
    """Two half-runs (+ resume from a copied partial dir, parallel workers) == one full run."""
    import shutil
    from wfsea.data.stats import compute_year_partials, merge_partials
    rng = np.random.default_rng(1)
    era = nasa = tmp_path / "in"
    for y in (2016, 2020):
        _write_year(era / f"era5-{y}" / "era5_data" / f"year_{y}", y, rng)
        _write_nasa(nasa / "nasa" / f"part{y}" / f"year_{y}", y)
    full = compute_stats_raw([2016, 2020], era, nasa, tb_file_stride=1, out_dir=tmp_path / "full")
    a = tmp_path / "a"; b = tmp_path / "b"
    compute_year_partials([2016], a, era, nasa, tb_file_stride=1)
    shutil.copytree(a, tmp_path / "inputs" / "a")              # like attaching notebook A's output
    parts = compute_year_partials([2016, 2020], b, era, nasa, tb_file_stride=1, workers=2,
                                  search_dirs=[tmp_path / "inputs"])
    assert (b / "stats_partial" / "year_2016.json").exists()   # carried over, not recomputed
    st = merge_partials(parts, [2016, 2020])
    for k in full.mean:
        assert abs(st.mean[k] - full.mean[k]) < 1e-9 * max(1, abs(full.mean[k]))
        assert abs(st.std[k] - full.std[k]) < 1e-9 * max(1, abs(full.std[k]))
    assert abs(st.tb_mean - full.tb_mean) < 1e-9


def test_year_picks_own_notebook_and_fills_gaps(tmp_path):
    """Year Y must come from notebook Y, not from notebook Y-1 (which also touches 31 Dec of Y-1),
    and a year split over two notebooks (Jan–Feb + rest) is merged."""
    from wfsea.data import raw_index as ri
    ri._scan_cached.cache_clear(); ri._open.cache_clear()
    rng = np.random.default_rng(3)
    root = tmp_path / "in"
    _write_year(root / "nb-2015" / "era5_data", 2015, rng)
    _write_year(root / "nb-2016" / "era5_data", 2016, rng)
    ds = ri.open_era5_year_indexed(2016, [str(root)], pressure_vars=("z",), levels=(850,),
                                   surface_vars=("t2m", "tp"), static_vars=())
    t = pd.DatetimeIndex(ds.time.values)
    assert t.year.unique().tolist() == [2016] and len(t) > 10


def test_regrid_tb_subgrid_stats():
    """Sub-grid Tb stats per 0.25° cell: min, std, cold-pixel fractions."""
    from wfsea.data.preprocess import regrid_tb
    lats, lons = np.array([0.0, 0.25]), np.array([100.0, 100.25])
    plat = np.arange(-0.1, 0.36, 0.05); plon = np.arange(99.9, 100.36, 0.05)     # ~5×5 pixels per cell
    v = np.full((1, len(plat), len(plon)), 280.0, "f4")
    li = np.where(np.abs(plat - 0.0) <= 0.125)[0]; lj = np.where(np.abs(plon - 100.0) <= 0.125)[0]
    v[0, li[0], lj[0]] = 200.0                                                    # one overshooting top in cell (0,0)
    v[0, li[1], lj[1]] = 230.0                                                    # one deep-convection pixel
    tb = xr.DataArray(v, dims=("time", "lat", "lon"), coords={"time": [np.datetime64("2020-01-01")],
                                                              "lat": plat, "lon": plon})
    r = regrid_tb(tb, lats, lons, subgrid=True)
    n = len(li) * len(lj)
    assert float(r["Tb_min"][0, 0, 0]) == 200.0 and float(r["Tb_min"][0, 1, 1]) == 280.0
    np.testing.assert_allclose(float(r["Tb_f235"][0, 0, 0]), 2 / n, rtol=1e-6)
    np.testing.assert_allclose(float(r["Tb_f208"][0, 0, 0]), 1 / n, rtol=1e-6)
    assert float(r["Tb_std"][0, 1, 1]) < 1e-3 and float(r["Tb_std"][0, 0, 0]) > 5
    np.testing.assert_allclose(float(r["Tb"][0, 0, 0]), (280 * (n - 2) + 200 + 230) / n, rtol=1e-6)


def test_cache_with_tb_subgrid_and_features(tmp_path):
    from wfsea.data.features import ForcingBuilder, InputSpec
    rng = np.random.default_rng(2)
    era = nasa = tmp_path / "in"
    _write_year(era / "e" / "year_2020", 2020, rng)
    _write_nasa(nasa / "n" / "year_2020", 2020)
    st = compute_stats_raw([2020], era, nasa, tb_file_stride=1)
    raw = {"stats": st, "era5_root": era, "nasa_root": nasa, "regrid": "box_mean", "tb_subgrid": True}
    c = SplitCache(materialize_split([2020], tmp_path / "c", layout="raw", raw=raw))
    assert c.tb_subgrid == ["Tb_min", "Tb_std", "Tb_f235", "Tb_f208"]
    x = c.mm["tbx"][:].astype(np.float32)
    assert x.shape == (c.T, c.G, 4) and np.isfinite(x).all()
    assert (x[..., 2:] == 0).all()                  # synthetic Tb 250–268 K: no pixels colder than 235 K
    spec = InputSpec(n_dyn=c.C, n_static=2, tb_subgrid=True)
    f = ForcingBuilder(spec, c)(0, 0)
    assert f.shape == (c.G, spec.n_forcing) and spec.n_forcing == 12 + 2 + 4
    assert len(spec.names([f"c{i}" for i in range(c.C)])) == spec.n_in
    sl = spec.satellite_slice(); assert sl.stop - sl.start == 12
    # a cache WITHOUT sub-grid stats must refuse a sub-grid spec
    c2 = SplitCache(materialize_split([2020], tmp_path / "c2", layout="raw", raw={**raw, "tb_subgrid": False}))
    import pytest
    with pytest.raises(ValueError):
        ForcingBuilder(spec, c2)(0, 0)


def test_tb_binner_matches_regrid_tb():
    from wfsea.data.preprocess import TbBinner, regrid_tb
    rng = np.random.default_rng(5)
    plat = np.arange(-0.6, 1.4, 0.0364); plon = np.arange(99.4, 101.6, 0.0364)
    v = (200 + 90 * rng.random((1, len(plat), len(plon)))).astype("f4"); v[0, 3:9, 5:20] = np.nan
    lats, lons = np.arange(-0.5, 1.26, 0.25), np.arange(99.5, 101.51, 0.25)
    tb = xr.DataArray(v, dims=("time", "lat", "lon"), coords={"time": [np.datetime64("2020-01-01")], "lat": plat, "lon": plon})
    ref = regrid_tb(tb, lats, lons, subgrid=True)
    b = TbBinner(plat, plon, lats, lons)
    got = b(v[0][b.rows, b.cols])
    for k in ("Tb", "Tb_min", "Tb_std", "Tb_f235", "Tb_f208"):
        np.testing.assert_allclose(got[k], ref[k].values[0], rtol=1e-5, atol=1e-4, equal_nan=True)
