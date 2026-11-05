"""Two-phase CDS download logic with a fake client (no network)."""
import json

from wfsea.config import load_config, resolve_paths
from wfsea.data import download as dl


class FakeRemote:
    def __init__(self, client, rid):
        self.client, self.request_id = client, rid

    @property
    def status(self):
        return self.client.status.get(self.request_id, "running")

    @property
    def request(self):
        return self.client.requests[self.request_id]

    def get_results(self):
        raise RuntimeError("400 Client Error: too many queued requests")

    def download(self, target):
        open(target, "wb").write(b"CDF-fake")
        return target


class FakeJobs:
    def __init__(self, ids):
        self.request_ids = ids


class FakeClient:
    def __init__(self):
        self.submitted, self.status, self.requests = [], {}, {}

    def submit(self, dataset, req):
        rid = f"id-{req['year'][0]}-{req['month'][0]}-{len(self.submitted)}"
        self.submitted.append(rid); self.requests[rid] = req; self.status[rid] = "accepted"
        return FakeRemote(self, rid)

    def get_remote(self, rid):
        return FakeRemote(self, rid)

    def get_jobs(self, **kw):
        return FakeJobs([r for r in self.submitted if self.status[r] in ("accepted", "running", "successful")])


def test_out_dir_is_never_none(tmp_path, monkeypatch):
    cfg = resolve_paths(load_config(None, [f"paths.scratch={tmp_path}"]))
    assert cfg["paths"]["out"].endswith("outputs") and cfg["paths"]["out"].startswith("/")


def test_submit_once_then_fetch_incrementally(tmp_path, monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(dl, "_client", lambda: fake)
    reqs = dl.pressure_requests([2018], ["z", "t"], [850, 500])
    assert len(reqs) == 12 and reqs["2018-03"]["time"] == ["00:00", "06:00", "12:00", "18:00"]
    state = tmp_path / "cds_requests.json"
    dl.cds_submit(reqs, state, max_new=5)
    dl.cds_submit(reqs, state, max_new=50)                 # second run only submits the remaining 7
    assert len(fake.submitted) == 12

    for rid in fake.submitted[:2]:
        fake.status[rid] = "successful"
    st = dl.cds_fetch(state, tmp_path / "out")
    assert st["2018-01"]["status"] == "downloaded" and st["2018-03"]["status"] == "accepted"
    assert (tmp_path / "out" / "year_2018" / "era5_pressure_2018_01.nc").exists()

    other = tmp_path / "old.json"
    other.write_text(json.dumps({"2018-04": {"request_id": "x", "status": "downloaded"}}))
    merged = dl.merge_states([state, other])
    assert merged["2018-04"]["status"] == "downloaded" and merged["2018-01"]["status"] == "downloaded"


def test_lost_state_adopts_queued_jobs_and_retries_rejected(tmp_path, monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(dl, "_client", lambda: fake)
    reqs = dl.pressure_requests([2015], ["z"], [850])
    dl.cds_submit(reqs, tmp_path / "lost.json")             # first run — its state file is then lost
    fake.status[fake.submitted[2]] = "rejected"              # CDS rejected 2015-03
    n_before = len(fake.submitted)

    state = tmp_path / "new.json"
    st = dl.cds_submit(reqs, state)                          # new run, empty state
    assert sum(1 for v in st.values() if v.get("adopted")) == 11
    assert len(fake.submitted) == n_before + 1               # only the rejected month is sent again
    assert st["2015-03"]["request_id"] != fake.submitted[2]

    fake.status[st["2015-04"]["request_id"]] = "rejected"
    out = dl.cds_fetch(state, tmp_path / "out", reqs=reqs)   # fetch re-submits it once
    assert out["2015-04"]["status"] in ("accepted", "submitted") and out["2015-04"].get("tries") == 2


def test_gcs_download_layout(tmp_path, monkeypatch):
    """Fake ARCO store (hourly, levels, lat descending, lon 0..360) → monthly CDS-like NetCDF read by raw_index.
    A small box keeps the fake store tiny (the real domain would need GBs of RAM)."""
    import numpy as np, pandas as pd, xarray as xr
    from wfsea.data import download as dl
    from wfsea.data.raw_index import scan_era5
    LAT_MAX, LAT_MIN, LON_MIN, LON_MAX = 10.0, 8.0, 100.0, 102.5
    for k, v in dict(LAT_MAX=LAT_MAX, LAT_MIN=LAT_MIN, LON_MIN=LON_MIN, LON_MAX=LON_MAX).items():
        monkeypatch.setattr(dl, k, v)

    t = pd.date_range("2019-12-31T00", "2020-02-01T00", freq="1h")
    lat = np.arange(LAT_MAX + 1.0, LAT_MIN - 1.0 - 1e-9, -0.25)
    lon = np.arange(LON_MIN - 1.0, LON_MAX + 1.0 + 1e-9, 0.25)
    lev = np.array([100, 200, 300, 500, 700, 850, 925, 1000])
    shape = (len(t), len(lev), len(lat), len(lon))
    hours = ((t - t[0]) / pd.Timedelta("1h")).values.astype("float32")
    base = np.broadcast_to(hours[:, None, None, None], shape)
    store = xr.Dataset({n: (("time", "level", "latitude", "longitude"), (base + i).astype("float32"))
                        for i, n in enumerate(dl.GCS_NAMES.values())},
                       coords={"time": t, "level": lev, "latitude": lat, "longitude": lon},
                       attrs={"valid_time_stop": "2020-01-31T23:00", "valid_time_stop_era5t": "2020-01-31T23:00"})
    url = str(tmp_path / "arco.zarr")
    store.chunk({"time": 1}).to_zarr(url)

    out = tmp_path / "era5_pressure"
    levels = [1000, 925, 850, 700, 500, 300, 200]
    dl.gcs_pressure_download([2020], ["z", "t", "q", "u", "v"], levels, out, url=url, workers=4)
    f = out / "year_2020" / "era5_pressure_gcs_2020-01.nc"
    assert f.exists() and not (out / "year_2020" / "era5_pressure_gcs_2020-02.nc").exists()  # Feb > stop
    with xr.open_dataset(f, engine="h5netcdf") as d:
        assert d.sizes == {"valid_time": 124, "pressure_level": 7, "latitude": 9, "longitude": 11}
        assert list(d.pressure_level.values) == levels
        # value = hours since store start (+ var offset) → checks the time step mapping
        h0 = (pd.Timestamp("2020-01-01T06") - t[0]) / pd.Timedelta("1h")
        assert float(d["t"].isel(valid_time=1, pressure_level=0, latitude=3, longitude=5)) == h0 + 1
    # second run skips the finished month
    mtime = f.stat().st_mtime
    dl.gcs_pressure_download([2020], ["z", "t", "q", "u", "v"], levels, out, url=url, workers=4)
    assert f.stat().st_mtime == mtime
    idx = [r for r in scan_era5([str(out)]) if r.kind == "pressure"]
    assert idx and set(idx[0].vars) == {"z", "t", "q", "u", "v"} and idx[0].step_h == 6.0


def test_gcs_single_tp_sum_and_index(tmp_path, monkeypatch):
    """Single levels from a fake ARCO store: 6-h tp = sum of the 6 previous hourly values;
    raw_index reads the files as a 6-h set and does not sum again."""
    import numpy as np, pandas as pd, xarray as xr
    from wfsea.data import download as dl
    from wfsea.data import raw_index as ri
    for k, v in dict(LAT_MAX=10.0, LAT_MIN=8.0, LON_MIN=100.0, LON_MAX=102.5).items():
        monkeypatch.setattr(dl, k, v)
    t = pd.date_range("2019-12-31T00", "2020-02-01T00", freq="1h")
    lat = np.arange(11.0, 7.0 - 1e-9, -0.25); lon = np.arange(99.0, 104.0 + 1e-9, 0.25)
    hrs = ((t - t[0]) / pd.Timedelta("1h")).values.astype("float32")
    f3 = lambda base: (("time", "latitude", "longitude"),
                       np.broadcast_to(base[:, None, None], (len(t), len(lat), len(lon))).astype("float32"))
    vars_ = {n: f3(hrs + i) for i, n in enumerate(dl.GCS_SINGLE.values())}
    vars_["total_precipitation"] = f3(np.ones(len(t), "float32") * 1e-3)
    vars_["temperature"] = (("time", "level", "latitude", "longitude"),
                            np.zeros((len(t), 1, len(lat), len(lon)), "float32"))
    store = xr.Dataset(vars_, coords={"time": t, "level": [850], "latitude": lat, "longitude": lon},
                       attrs={"valid_time_stop": "2020-01-31T23:00", "valid_time_stop_era5t": "2020-01-31T23:00"})
    url = str(tmp_path / "arco.zarr"); store.chunk({"time": 24}).to_zarr(url)
    out = tmp_path / "era5_single"
    dl.gcs_single_download([2020], out, url=url, workers=4)
    f = out / "year_2020" / "era5_single_gcs_2020-01.nc"
    with xr.open_dataset(f, engine="h5netcdf") as d:
        assert d.sizes["valid_time"] == 124 and set(d.data_vars) == set(dl.GCS_SINGLE)
        assert np.allclose(d["tp"].values, 6e-3)                          # 6 hours × 1 mm
        h = (pd.Timestamp("2020-01-02T12") - t[0]) / pd.Timedelta("1h")
        assert float(d["t2m"].sel(valid_time="2020-01-02T12").isel(latitude=0, longitude=0)) == h + 3
    ri._scan_cached.cache_clear(); ri._open.cache_clear()
    idx = [r for r in ri.scan_era5([str(out)]) if r.kind == "single"]
    assert idx and idx[0].step_h == 6.0
