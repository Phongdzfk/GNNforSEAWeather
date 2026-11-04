import json

import numpy as np
import xarray as xr

from wfsea.constants import DYNAMIC_CHANNELS
from wfsea.data.dataset import WeatherDataset
from wfsea.data.features import InputSpec, tb_source_index
from wfsea.data.manifest import build_manifest
from wfsea.data.memmap import SplitCache, materialize_split


def _cache(cfg, split):
    years = cfg["data"][f"{split}_years"]
    return materialize_split(years, f"{cfg['paths']['scratch']}/{split}", cfg["paths"]["zarr_root"], "legacy")


def test_memmap_reconstructs_full_series_in_canonical_order(cfg, legacy_data):
    c = SplitCache(_cache(cfg, "test"))
    truth = xr.open_zarr(legacy_data / "_truth_2023.zarr")
    assert c.T == truth.sizes["time"]                     # x + last label = every state
    assert np.array_equal(c.times, truth.time.values)     # last-label time shifted by +6 h
    for ci, (v, lev) in enumerate(DYNAMIC_CHANNELS):      # canonical order regardless of file order
        ref = truth[v] if lev is None else truth[v].sel(pressure_level=lev)
        got = np.asarray(c.mm["state"][:, :, ci], dtype=np.float32)
        np.testing.assert_allclose(got, ref.values.reshape(c.T, -1), atol=2e-3)
    assert c.mm["tbmask"][3].min() == 0 and c.mm["tbmask"][-1].max() == 0   # NaN Tb and missing last Tb masked


def test_year_boundary_is_not_a_valid_sample(cfg):
    c = SplitCache(_cache(cfg, "train"))                  # 2020 (6 days) + 2021 (6 days): big gap between
    meta = json.loads((c.dir / "meta.json").read_text())
    assert len(meta["gaps"]) == 1
    starts = c.valid_starts(n_future=1)
    step = np.timedelta64(6, "h")
    assert all(c.times[t] - c.times[t - 1] == step and c.times[t + 1] - c.times[t] == step for t in starts)
    assert len(starts) == c.T - 2 - 2                     # lose 1 at each end of each year chunk


def test_dataset_shapes_and_legacy_input_width(cfg):
    c = SplitCache(_cache(cfg, "val"))
    spec = InputSpec(n_dyn=14, n_static=2)
    assert spec.n_in == 38                                # 538,382-param checkpoint layout
    ds = WeatherDataset(c, spec, n_steps=3)
    dyn_t, dyn_tm1, forcing, target = ds[0]
    assert dyn_t.shape == (c.G, 14) and forcing.shape == (3, c.G, spec.n_forcing) and target.shape == (3, c.G, 14)


def test_tb_policy_never_peeks_when_persist():
    assert tb_source_index(10, 0, 0, "persist") == (10, True)
    assert tb_source_index(10, 3, 0, "persist") == (10, True)
    assert tb_source_index(10, 3, 0, "oracle") == (13, True)
    assert tb_source_index(10, 3, 1, "mask") == (12, False)
    assert tb_source_index(10, 0, 1, "mask") == (9, True)


def test_manifest_detects_no_overlap(cfg, tmp_path):
    caches = {s: _cache(cfg, s) for s in ("train", "val", "test")}
    m = build_manifest(caches, tmp_path / "m.json", checksum=True)
    assert m["split_overlap"] == []
    assert set(m["splits"]["test"]["sha256"]) >= {"state.f16", "tb.f16", "time.npy"}
