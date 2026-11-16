"""External benchmark reader (AIFS / IFS) on a fake dynamical.org-like dataset."""
import numpy as np
import pandas as pd
import xarray as xr

from wfsea.evaluation.external import G0, ExternalForecast


def _fake(lead_h, member=False):
    lat = np.arange(-2.0, 2.01, 0.25)                      # ascending like the source
    lon = np.arange(-180.0, 180.0, 0.25)
    init = pd.date_range("2026-01-01", periods=4, freq="6h")
    lead = pd.to_timedelta(lead_h, unit="h")
    shp = (len(init), len(lead)) + ((2,) if member else ()) + (len(lat), len(lon))
    dims = ("init_time", "lead_time") + (("ensemble_member",) if member else ()) + ("latitude", "longitude")
    lt = np.asarray(lead_h, float).reshape((1, -1) + (1,) * (len(shp) - 2))
    base = np.broadcast_to(lt, shp).astype("f4")
    ds = xr.Dataset({
        "temperature_2m": (dims, base + 20.0),                                # °C = 20 + lead_h
        "geopotential_height_500hpa": (dims, np.full(shp, 5800.0, "f4")),
        "precipitation_surface": (dims, np.full(shp, 1e-3, "f4")),            # 1e-3 mm/s
        "wind_u_10m": (dims, np.ones(shp, "f4")),
    }, coords={"init_time": init, "lead_time": lead, "latitude": lat, "longitude": lon})
    if member:
        ds["temperature_2m"][dict(ensemble_member=1)] = 999.0                 # control (0) must be used
    return ds


def test_aifs_like_6h():
    lats = np.arange(1.0, -1.01, -0.25); lons = np.arange(100.0, 101.01, 0.25)
    ext = ExternalForecast("aifs", lats, lons, ["t2m", "u10", "z500", "tp", "q850"],
                           ds=_fake([0, 6, 12, 18, 24]))
    assert ext(np.datetime64("2025-12-31T00"), 4) is None                     # not in archive
    f = ext(np.datetime64("2026-01-01T06"), 4)
    assert f.shape == (4, len(lats) * len(lons), 5)
    np.testing.assert_allclose(f[:, 0, 0], [20 + 6 + 273.15, 20 + 12 + 273.15, 20 + 18 + 273.15, 20 + 24 + 273.15], rtol=1e-6)
    np.testing.assert_allclose(f[:, :, 2], 5800.0 * G0, rtol=1e-6)
    np.testing.assert_allclose(f[:, :, 3], 1e-3 * 21600 / 1000, rtol=1e-6)    # 6 h of 1e-3 mm/s = 21.6 mm
    assert np.isnan(f[:, :, 4]).all()                                         # no q850 in the source
    assert np.isnan(ext(np.datetime64("2026-01-01T06"), 5)[4]).all()          # lead 30 h not in archive


def test_ifs_like_3h_control_member(tmp_path):
    lats = np.array([0.0, -0.25]); lons = np.array([179.75, 180.0])           # crosses the date line in 0..360
    ext = ExternalForecast("ifs_control", lats, lons, ["t2m", "tp"], ds=_fake([0, 3, 6, 9, 12], member=True),
                           cache_dir=tmp_path)
    f = ext(np.datetime64("2026-01-01T00"), 2)
    np.testing.assert_allclose(f[:, 0, 0], [26 + 273.15, 32 + 273.15], rtol=1e-6)   # member 0, not 999
    np.testing.assert_allclose(f[:, :, 1], 1e-3 * 21600 / 1000, rtol=1e-6)          # two 3-h intervals
    assert list((tmp_path / "ifs_control").glob("*.npy"))                           # cached on disk
