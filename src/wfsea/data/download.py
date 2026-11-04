"""ERA5 (CDS API) and NASA GPM MERGIR (GES DISC OPeNDAP) downloaders.

Ported from nb1-era5-2020-2026.ipynb and nb2-nasa-v2.ipynb. Credentials are
read from environment variables or Kaggle Secrets — never commit them.

  ERA5 : CDS_URL, CDS_KEY            (Kaggle secrets "url", "key")
  NASA : EARTHDATA_USER, EARTHDATA_PASS

Owner: Phong. Pressure levels: two-phase CDS download (`cds_submit` → `cds_fetch`) that
survives queue delays, or `gcs_pressure_download` (Google ARCO-ERA5, public) that has no queue.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

from wfsea.constants import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN

ERA5_SURFACE_REQUEST = [
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "2m_dewpoint_temperature", "2m_temperature",
    "mean_sea_level_pressure", "total_precipitation",
    "geopotential", "land_sea_mask",
]
AREA = [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX]   # N, W, S, E
TIMES = ["00:00", "06:00", "12:00", "18:00"]

# MERGIR 4 km pixel indices covering the domain (from the original OPeNDAP query).
MERGIR_SUBSET = "?Tb[0:1][1346:2432][7476:8851],time,lat[1346:2432],lon[7476:8851]"
MERGIR_URL = ("https://disc2.gesdisc.eosdis.nasa.gov/opendap/MERGED_IR/GPM_MERGIR.1/"
              "{y}/{doy}/merg_{y}{m}{d}{h}_4km-pixel.nc4.nc4")


def _secret(name: str, kaggle_name: str | None = None) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        return UserSecretsClient().get_secret(kaggle_name or name)
    except Exception:
        return None


PRESSURE_NAMES = {"z": "geopotential", "t": "temperature", "q": "specific_humidity",
                  "u": "u_component_of_wind", "v": "v_component_of_wind", "w": "vertical_velocity"}


def _client():
    """ECMWF Data Stores client (installed with cdsapi ≥ 0.7). Kaggle secrets: 'url', 'key'."""
    from ecmwf.datastores import Client
    url = _secret("CDS_URL", "url") or "https://cds.climate.copernicus.eu/api"
    return Client(url=url, key=_secret("CDS_KEY", "key"), timeout=120, retry_after=60, maximum_tries=10)


def pressure_requests(years, variables, levels, times=TIMES) -> dict[str, dict]:
    """One request per (year, month): 6-hourly only (the model never uses the other 20 hours),
    regional area. 7 levels × 5 variables × 4 times × 31 days ≈ 4,300 items — far below the
    per-request limit, and ~6× cheaper than the hourly requests the old notebooks sent."""
    reqs = {}
    for y in map(int, years):
        for m in range(1, 13):
            reqs[f"{y}-{m:02d}"] = {
                "product_type": ["reanalysis"], "variable": [PRESSURE_NAMES[v] for v in variables],
                "year": [str(y)], "month": [f"{m:02d}"], "day": [f"{d:02d}" for d in range(1, 32)],
                "time": list(times), "pressure_level": [str(l) for l in levels],
                "data_format": "netcdf", "download_format": "unarchived", "area": AREA,
            }
    return reqs


def _load_state(path: Path) -> dict:
    import json
    return json.loads(path.read_text()) if path.exists() else {}


def _save_state(path: Path, state: dict) -> None:
    import json
    path.write_text(json.dumps(state, indent=1))


def cds_submit(reqs: dict[str, dict], state_path: str | Path, dataset: str = "reanalysis-era5-pressure-levels",
               max_new: int = 48) -> dict:
    """Phase 1: SUBMIT requests and record their IDs — returns immediately, no waiting in the queue.

    The old notebooks blocked on client.retrieve() for each month; when the CDS queue was slow the
    HTTP poll timed out, the loop moved on, and re-running re-submitted the same month (making the
    queue longer). Here each month is submitted once and its request ID is kept in a JSON file that
    survives between Kaggle sessions (keep it in /kaggle/working and re-attach it).
    """
    state_path = Path(state_path)
    state = _load_state(state_path)
    client, n = _client(), 0
    adopted = adopt_server_jobs(client, reqs, state)
    if adopted:
        print(f"[cds] re-using {len(adopted)} request(s) already queued on CDS: {sorted(adopted)[:6]}...")
        _save_state(state_path, state)
    for key, req in reqs.items():
        if key in state and state[key].get("status") not in ("failed", "rejected", "deleted", "dismissed"):
            continue
        if n >= max_new:
            break
        try:
            rid = client.submit(dataset, req).request_id
            state[key] = {"request_id": rid, "status": "submitted"}
            n += 1
            print(f"[cds] submitted {key}: {rid}")
        except Exception as e:
            state[key] = {"status": "submit_error", "error": str(e)[:300]}
            print(f"[cds] submit {key} failed: {e}")
        _save_state(state_path, state)
    return state


def _req_key(req: dict) -> tuple:
    one = lambda v: tuple(sorted(map(str, v if isinstance(v, (list, tuple)) else [v])))
    return (one(req.get("year", [])), one(req.get("month", [])), one(req.get("variable", [])),
            one(req.get("pressure_level", [])), one(req.get("time", [])))


def adopt_server_jobs(client, reqs: dict[str, dict], state: dict, limit: int = 200) -> list[str]:
    """Find identical requests already queued/finished on CDS (e.g. from a run whose state file was lost)
    and record their IDs instead of submitting duplicates. Returns the adopted keys."""
    wanted = {_req_key(r): k for k, r in reqs.items()
              if k not in state or state[k].get("status") in ("failed", "rejected", "submit_error")}
    if not wanted:
        return []
    try:
        ids = client.get_jobs(limit=limit, sortby="-created", status=["accepted", "running", "successful"]).request_ids
    except Exception as e:
        print(f"[cds] could not list existing jobs ({e}); submitting normally")
        return []
    adopted = []
    for rid in ids:
        try:
            key = wanted.get(_req_key(client.get_remote(rid).request))
        except Exception:
            continue
        if key and key not in adopted:
            state[key] = {"request_id": rid, "status": "submitted", "adopted": True}
            adopted.append(key)
    return adopted


def merge_states(paths) -> dict:
    """Combine request-state files from several Kaggle runs (attached as inputs) + the current one."""
    order = {"downloaded": 4, "successful": 3, "running": 2, "accepted": 1, "submitted": 1}
    merged: dict = {}
    for p in paths:
        for k, st in _load_state(Path(p)).items():
            if k not in merged or order.get(st.get("status"), 0) > order.get(merged[k].get("status"), 0):
                merged[k] = st
    return merged


def cds_fetch(state_path: str | Path, out_dir: str | Path, prefix: str = "era5_pressure",
              wait_hours: float = 0.0, poll_minutes: float = 10.0, reqs: dict | None = None,
              max_tries: int = 4) -> dict:
    """Phase 2: download every request that is ready. With wait_hours > 0 keep polling until all are
    downloaded or the time is up (use ~11 h in a Kaggle "Save & Run All" session).
    Nothing is ever re-submitted here."""
    import time as _time
    deadline = _time.time() + wait_hours * 3600
    while True:
        state = _fetch_once(Path(state_path), Path(out_dir), prefix)
        if reqs:   # CDS sometimes rejects when a user has too many queued jobs → retry those later
            retry = {k: reqs[k] for k, st in state.items() if k in reqs
                     and st.get("status") in ("failed", "rejected", "submit_error") and st.get("tries", 1) < max_tries}
            if retry:
                for k in retry:
                    state[k]["tries"] = state[k].get("tries", 1) + 1
                _save_state(Path(state_path), state)
                tries = {k: state[k]["tries"] for k in retry}
                state = cds_submit(retry, state_path)
                for k, t in tries.items():
                    state[k]["tries"] = t
                _save_state(Path(state_path), state)
        retryable = lambda st: reqs and st.get("tries", 1) < max_tries
        pending = [k for k, st in state.items()
                   if st.get("status") not in ("downloaded", "dismissed", "deleted")
                   and (st.get("status") not in ("failed", "rejected", "submit_error") or retryable(st))]
        if not pending or _time.time() + poll_minutes * 60 > deadline:
            if pending:
                print(f"[cds] {len(pending)} still pending — run fetch again later: {pending[:6]}...")
            return state
        _time.sleep(poll_minutes * 60)


def _fetch_once(state_path: Path, out_dir: Path, prefix: str) -> dict:
    state = _load_state(state_path)
    client = _client()
    counts: dict[str, int] = {}
    for key, st in sorted(state.items()):
        if st.get("status") == "downloaded" or "request_id" not in st:
            counts[st.get("status", "?")] = counts.get(st.get("status", "?"), 0) + 1
            continue
        year = key[:4]
        target = out_dir / f"year_{year}" / f"{prefix}_{key.replace('-', '_')}.nc"
        try:
            remote = client.get_remote(st["request_id"])
            status = remote.status
            st["status"] = status
            if status in ("failed", "rejected") and "error" not in st:
                try:
                    remote.get_results()
                except Exception as e:      # the API puts the reason in the error body
                    st["error"] = str(e)[:500]
                print(f"[cds] {key} {status}: {st.get('error', '(no reason given)')}")
            if status == "successful":
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = remote.download(str(target) + ".part")
                _unpack(Path(tmp), target)
                st["status"], st["path"] = "downloaded", str(target)
                print(f"[cds] downloaded {key} → {target.name}")
        except Exception as e:
            st["error"] = str(e)[:300]
            print(f"[cds] {key}: {e}")
        counts[st["status"]] = counts.get(st["status"], 0) + 1
        _save_state(state_path, state)
    print("[cds] status:", counts)
    bad = sorted(k for k, st in state.items() if st.get("status") in ("failed", "rejected", "submit_error"))
    if bad:
        print(f"[cds] {len(bad)} month(s) to re-submit with `download-submit` (done automatically next run): {bad}")
    return state


def _unpack(tmp: Path, target: Path) -> None:
    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as zf:
            names = [n for n in zf.namelist() if n.endswith(".nc")]
            for i, n in enumerate(names):
                dest = target if len(names) == 1 else target.with_name(f"{target.stem}_{i}.nc")
                with zf.open(n) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        tmp.unlink()
    else:
        tmp.rename(target)


GCS_ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
GCS_NAMES = {"z": "geopotential", "t": "temperature", "q": "specific_humidity",
             "u": "u_component_of_wind", "v": "v_component_of_wind", "w": "vertical_velocity"}


def _open_gcs(url: str):
    import xarray as xr
    so = dict(token="anon") if url.startswith("gs://") else None
    return xr.open_zarr(url, chunks=None, storage_options=so, decode_timedelta=False)


def _gcs_index(ds, levels):
    """Integer indices of our box / levels in the global ARCO grid (lat 90→-90, lon 0→359.75)."""
    import numpy as np
    lat, lon = ds["latitude"].values, ds["longitude"].values
    iy = np.where((lat <= LAT_MAX + 1e-6) & (lat >= LAT_MIN - 1e-6))[0]
    ix = np.where((lon >= LON_MIN - 1e-6) & (lon <= LON_MAX + 1e-6))[0]
    lv = list(ds["level"].values.astype(int))
    missing = [l for l in levels if int(l) not in lv]
    if missing:
        raise ValueError(f"levels {missing} not in ARCO store (has {lv})")
    il = [lv.index(int(l)) for l in levels]
    return slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1), il


def _read_step(ds, name, ti, il, sy, sx, tries=5):
    import time
    for k in range(tries):
        try:
            return ds[name].isel(time=ti, level=il, latitude=sy, longitude=sx).values.astype("float32")
        except Exception as e:                      # transient GCS errors
            if k == tries - 1:
                raise
            print(f"[gcs] retry {name} t={ti}: {type(e).__name__}", flush=True)
            time.sleep(5 * (k + 1))


def gcs_pressure_download(years, variables, levels, out_dir: str | Path, url: str = GCS_ARCO_URL,
                          workers: int = 16, max_hours: float | None = None, era5t: bool = True,
                          bench: bool = False) -> None:
    """No queue, no key: read Google's public ARCO-ERA5 Zarr (0.25°, hourly, 37 levels) anonymously.

    Each Zarr chunk is ONE time step of one variable, global, all 37 levels (~100 MB compressed),
    so reading is network-bound; `workers` threads fetch different (variable, time) chunks at once
    (workers=0 → measure 8/16/32 for ~2 min and keep the fastest).
    Output: one NetCDF per month, era5_pressure/year_<y>/era5_pressure_gcs_<y>-<mm>.nc, laid out like
    the CDS files (valid_time, pressure_level, latitude, longitude; vars z,t,q,u,v) so raw_index
    picks it up unchanged. Finished months are skipped → re-running continues where it stopped.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import pandas as pd
    import xarray as xr
    try:
        import zarr
        zarr.config.set({"async.concurrency": max(32, 2 * workers)})
    except Exception:
        pass

    t_start = time.time()
    ds = _open_gcs(url)
    stop = pd.Timestamp(ds.attrs.get("valid_time_stop_era5t" if era5t else "valid_time_stop") or ds.time.values[-1])
    print(f"[gcs] store {url}\n[gcs] ERA5 until {ds.attrs.get('valid_time_stop')}, ERA5T until "
          f"{ds.attrs.get('valid_time_stop_era5t')} → using data up to {stop}", flush=True)
    sy, sx, il = _gcs_index(ds, levels)
    lat = ds["latitude"].values[sy]
    lon = ds["longitude"].values[sx]
    tindex = pd.DatetimeIndex(ds["time"].values)
    names = {v: GCS_NAMES[v] for v in variables}

    def month_times(y, m):
        t = pd.date_range(f"{y}-{m:02d}-01", periods=pd.Period(f"{y}-{m:02d}").days_in_month * 4, freq="6h")
        return t[t <= stop]

    def measure(w):
        y = int(sorted(years)[0])
        t = month_times(y, 1)[: max(4, w // len(names) * 2)]
        ti = tindex.get_indexer(t)
        t0 = time.time()
        with ThreadPoolExecutor(w) as p:
            list(p.map(lambda a: _read_step(ds, *a), [(names[v], int(i), il, sy, sx) for v in names for i in ti]))
        dt = (time.time() - t0) / len(t)
        print(f"[gcs] bench: {w:>2} workers → {dt:.2f} s/step ≈ {dt * 1460 / 3600:.2f} h/year", flush=True)
        return dt

    if bench or workers <= 0:
        res = {w: measure(w) for w in ([workers] if workers > 0 else (8, 16, 32))}
        if bench:
            return
        workers = min(res, key=res.get)
        print(f"[gcs] using {workers} workers", flush=True)
    pool = ThreadPoolExecutor(workers)

    for y in sorted(int(x) for x in years):
        for m in range(1, 13):
            out = Path(out_dir) / f"year_{y}" / f"era5_pressure_gcs_{y}-{m:02d}.nc"
            if out.exists():
                continue
            t = month_times(y, m)
            if len(t) == 0:
                print(f"[gcs] {y}-{m:02d}: not in store yet (stop {stop}) — skipped", flush=True)
                continue
            if max_hours and (time.time() - t_start) / 3600 > max_hours:
                print(f"[gcs] time budget {max_hours} h reached — re-run (with this output as input) to continue")
                pool.shutdown(wait=False)
                return
            if len(t) < pd.Period(f"{y}-{m:02d}").days_in_month * 4:   # month not finished in the store yet
                out = out.with_name(out.stem + "_partial.nc")          # full month is fetched on a later run
            ti = tindex.get_indexer(t)
            if (ti < 0).any():
                raise ValueError(f"{y}-{m:02d}: {int((ti < 0).sum())} time steps missing in store")
            t0 = time.time()
            data = {v: np.empty((len(t), len(il), len(lat), len(lon)), "float32") for v in names}

            def job(a):
                v, k = a
                data[v][k] = _read_step(ds, names[v], int(ti[k]), il, sy, sx)

            list(pool.map(job, [(v, k) for k in range(len(t)) for v in names]))
            dsm = xr.Dataset({v: (("valid_time", "pressure_level", "latitude", "longitude"), a)
                              for v, a in data.items()},
                             coords={"valid_time": t.values, "pressure_level": np.asarray(levels, "int64"),
                                     "latitude": lat, "longitude": lon},
                             attrs={"source": url, "note": "ARCO-ERA5 (Google), 6-hourly subset"})
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".nc.part")
            dsm.to_netcdf(tmp, engine="h5netcdf",
                          encoding={v: {"zlib": True, "complevel": 1, "shuffle": True} for v in names})
            tmp.rename(out)
            dt = time.time() - t0
            print(f"[gcs] {y}-{m:02d}: {len(t)} steps in {dt / 60:.1f} min ({dt / len(t):.2f} s/step), "
                  f"{out.stat().st_size / 1e9:.2f} GB", flush=True)
    pool.shutdown()
    print(f"[gcs] done in {(time.time() - t_start) / 3600:.2f} h", flush=True)


def mergir_urls(start: datetime, end: datetime) -> list[str]:
    urls, day = [], start
    while day <= end:
        for h in ("00", "06", "12", "18"):
            urls.append(MERGIR_URL.format(y=day.strftime("%Y"), doy=f"{day.timetuple().tm_yday:03d}",
                                          m=day.strftime("%m"), d=day.strftime("%d"), h=h) + MERGIR_SUBSET)
        day += timedelta(days=1)
    return urls


def download_mergir(start: datetime, end: datetime, out_root: str | Path) -> None:
    user, pw = _secret("EARTHDATA_USER"), _secret("EARTHDATA_PASS")
    if not (user and pw):
        raise RuntimeError("Set EARTHDATA_USER / EARTHDATA_PASS (env or Kaggle secrets)")
    netrc = Path.home() / ".netrc"
    netrc.write_text(f"machine urs.earthdata.nasa.gov login {user} password {pw}\n")
    netrc.chmod(0o600)
    cookies = Path.home() / ".urs_cookies"
    cookies.touch()
    year = start.year
    while year <= end.year:
        y0 = max(start, datetime(year, 1, 1))
        y1 = min(end, datetime(year, 12, 31))
        folder = Path(out_root) / f"year_{year}"
        folder.mkdir(parents=True, exist_ok=True)
        listing = folder / "_links.txt"
        listing.write_text("\n".join(mergir_urls(y0, y1)) + "\n")
        subprocess.run(["wget", f"--load-cookies={cookies}", f"--save-cookies={cookies}",
                        "--keep-session-cookies", "-nc", "-i", str(listing), "-P", str(folder)], check=False)
        for f in folder.glob("*?*"):
            clean = folder / f.name.split("?")[0]
            if not clean.exists():
                f.rename(clean)
        listing.unlink()
        year += 1
