"""Raw files exactly as the team's Kaggle download notebooks write them (hourly, month-suffixed names,
pressure files called ...instant_<m>.nc in era5_data_pressure/), plus a new 6-hourly multi-level download."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wfsea.data.memmap import MultiSplitCache, materialize_split, open_cache, part_years
from wfsea.data.raw_index import nasa_files_indexed, open_era5_year_indexed, scan_era5, summarize
from wfsea.data.stats import compute_stats_raw

LAT = np.arange(3.0, -0.01, -0.25)
LON = np.arange(100.0, 103.01, 0.25)


def _nc(path, data_vars, times, levels=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    co = {"valid_time": times, "latitude": LAT, "longitude": LON}
    if levels is not None:
        co["pressure_level"] = levels
    xr.Dataset(data_vars, coords=co).to_netcdf(path, engine="h5netcdf")


def _team_year(root, year, months=(1, 2)):
    """notebook-1-era5-<year>/era5_data(+_pressure)/data_stream-oper_stepType-*_<m>.nc, hourly."""
    nb = root / f"notebook-1-era5-{year}"
    for m in months:
        t = pd.date_range(f"{year}-{m:02d}-01", f"{year}-{m:02d}-{pd.Timestamp(year, m, 1).days_in_month} 23:00", freq="h")
        sh = (len(t), len(LAT), len(LON))
        f = lambda val: (np.full(sh, val, "f4"))
        sfc = {v: (("valid_time", "latitude", "longitude"), f(val)) for v, val in
               (("t2m", 300.0), ("d2m", 295.0), ("u10", 1.0), ("v10", -1.0), ("msl", 101000.0), ("z", 500.0), ("lsm", 1.0))}
        _nc(nb / "era5_data" / f"data_stream-oper_stepType-instant_{m}.nc", sfc, t)
        _nc(nb / "era5_data" / f"data_stream-oper_stepType-accum_{m}.nc",
            {"tp": (("valid_time", "latitude", "longitude"), f(1e-4))}, t)            # 0.1 mm every hour
        old = {v: (("valid_time", "pressure_level", "latitude", "longitude"), np.full((len(t), 2, *sh[1:]), 1.0, "f4"))
               for v in "zquv"}
        _nc(nb / "era5_data_pressure" / f"data_stream-oper_stepType-instant_{m}.nc", old, t, [500, 850])


def _new_pressure(root, year, months=(1, 2), levels=(850, 700, 500)):
    for m in months:
        t = pd.date_range(f"{year}-{m:02d}-01", f"{year}-{m:02d}-{pd.Timestamp(year, m, 1).days_in_month} 18:00", freq="6h")
        new = {v: (("valid_time", "pressure_level", "latitude", "longitude"),
                   np.full((len(t), len(levels), len(LAT), len(LON)), 2.0, "f4")) for v in "ztquv"}
        _nc(root / "era5-pl-new" / "era5_pressure" / f"year_{year}" / f"era5_pressure_{year}_{m:02d}.nc", new, t,
            list(levels))


def _nasa(root, year, months=(1, 2)):
    lat = np.arange(-0.2, 3.2, 0.1); lon = np.arange(99.8, 103.2, 0.1)
    for ts in pd.date_range(f"{year}-01-01", f"{year}-{max(months):02d}-28 18:00", freq="6h"):
        p = root / "nasa" / f"merg_{ts:%Y%m%d%H}_4km-pixel.nc4"
        p.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset({"Tb": (("time", "lat", "lon"), np.full((1, len(lat), len(lon)), 250.0, "f4"))},
                   coords={"time": [ts], "lat": lat, "lon": lon}).to_netcdf(p, engine="h5netcdf")


@pytest.fixture(scope="module")
def team_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("kaggle_input_team")
    for y in (2018, 2019):
        _team_year(root, y)
        _new_pressure(root, y)
        _nasa(root, y)
    return root


def test_index_and_year_assembly(team_root):
    idx = scan_era5(team_root)
    s = summarize(idx)
    assert len(s) == 2 * 4               # per year: sfc-instant, sfc-accum, old pressure, new pressure
    ds = open_era5_year_indexed(2018, team_root, pressure_vars=("z", "t", "q", "u", "v"), levels=(850, 700, 500))
    t = pd.DatetimeIndex(ds.time.values)
    assert set(t.hour) == {0, 6, 12, 18} and len(t) == (31 + 28) * 4 and t.year.unique().tolist() == [2018]
    # the 3-level new download wins over the old 2-level one for 850/500
    assert float(ds["z"].sel(pressure_level=850).mean()) == 2.0
    # hourly tp summed over the last 6 h: 6 × 0.1 mm; the very first step has no previous evening → NaN
    tp = ds["tp"].values
    assert np.isnan(tp[0]).all() and np.allclose(tp[1:], 6e-4)
    assert float(ds["geopotential_surf"].mean()) == 500.0 and "lsm" in ds


def test_missing_level_is_a_clear_error(team_root):
    with pytest.raises(FileNotFoundError, match="1000 hPa"):
        open_era5_year_indexed(2018, team_root, pressure_vars=("t",), levels=(1000,))


def test_nasa_index_by_filename(team_root):
    files = nasa_files_indexed(2019, team_root)
    assert len(files) == (31 + 28) * 4 and all("merg_2019" in f for f in files)


def test_streaming_stats_and_multipart_cache(team_root, tmp_path):
    sc = {"pressure_vars": ("z", "t", "q", "u", "v"), "levels": (850, 700, 500)}
    st = compute_stats_raw([2018], team_root, team_root, tb_file_stride=1, schema=sc)
    assert st.mean["t_700hpa"] == pytest.approx(2.0) and st.mean["t2m"] == pytest.approx(300.0)
    assert st.meta["tp_accumulation_hours"] == 6 and st.meta["pressure_levels"] == [850, 700, 500]
    from wfsea.constants import make_channels
    ch = make_channels(pressure_vars=sc["pressure_vars"], levels=sc["levels"])
    raw = {"stats": st, "era5_root": team_root, "nasa_root": team_root, "schema": sc}
    groups = part_years([2018, 2019], len(ch), 1)
    assert groups == [[2018], [2019]]
    dirs = [materialize_split(g, tmp_path / f"part{g[0]}", layout="raw", raw=raw, channels=ch) for g in groups]
    c = open_cache(dirs)
    assert isinstance(c, MultiSplitCache) and c.C == 6 + 15 and c.T == 2 * 59 * 4
    a = c.mm["state"][230:240]                                     # slice across the part boundary
    assert a.shape == (10, c.G, c.C)
    np.testing.assert_array_equal(a[6], np.asarray(c.parts[1].mm["state"][0]))
    starts = c.valid_starts()
    assert 236 not in starts and 237 in starts                     # 2018-02-28 18:00 → 2019-01-01 gap
    assert part_years(list(range(2015, 2025)), 41) == [[2015, 2016, 2017, 2018], [2019, 2020, 2021, 2022], [2023, 2024]]
