"""End-to-end smoke: materialize → train (1 epoch) → evaluate → export. Mirrors the Kaggle notebooks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wfsea.cli import main
from wfsea.evaluation.baselines import month_hour_keys
from wfsea.evaluation.metrics import acc_from_terms, acc_terms, wrmse


def _args(cfg):
    p = cfg["paths"]
    return [f"paths.zarr_root={p['zarr_root']}", f"paths.stats_dir={p['stats_dir']}", f"paths.out={p['out']}",
            f"paths.scratch={p['scratch']}", "data.zarr_layout=legacy", "data.train_years=[2020,2021]", "data.val_years=[2022]",
            "data.test_years=[2023]", "graph.mesh_res=1.0", "graph.mesh_lat_start=0.0", "graph.mesh_lat_stop=3.5",
            "graph.mesh_lon_start=100.0", "graph.mesh_lon_stop=103.5", "model.hidden=32", "model.n_layers=2",
            "eval.n_starts=3", "eval.n_steps=8"]


def test_cli_end_to_end(cfg):
    a = _args(cfg)
    main(["train", "--config", "smoke.yaml", *a])
    run = Path(cfg["paths"]["out"]) / "smoke"
    hist = json.loads((run / "history.json").read_text())
    assert len(hist["train_loss"]) == 1 and np.isfinite(hist["best_val"])
    assert (run / "graph_stats.json").exists()

    main(["evaluate", "--config", "smoke.yaml", "--ckpt", str(run / "best.pt"), *a])
    ev = run / "eval_persist"
    df = pd.read_csv(ev / "metrics_raw.csv")
    assert set(df.model) == {"graphcast", "persistence", "climatology"}
    assert set(df.lead_h) == {6, 12, 18, 24, 30, 36, 42, 48}
    assert df.channel.nunique() == 14 and set(df.region) >= {"all", "land", "sea"}
    assert len(json.loads((ev / "start_times.json").read_text())) == 3
    rm = pd.read_csv(ev / "summary_rmse.csv")
    assert np.isfinite(rm.select_dtypes("number").values).all()

    out_fc = Path(cfg["paths"]["out"]) / "fc"
    main(["export-forecast", "--config", "smoke.yaml", "--ckpt", str(run / "best.pt"), "--split", "test",
          "--forecast-dir", str(out_fc), *a])
    ds = xr.open_dataset(next(out_fc.glob("forecast_*.nc")))
    assert {"t2m_c", "rh2m", "wind10_speed", "tp_mm", "msl_hpa"} <= set(ds.data_vars)
    assert ds.sizes["lead"] == 8 and float(ds.rh2m.max()) <= 100


def test_metrics_basic():
    w = np.ones(4)
    t = np.array([[1.0], [2.0], [3.0], [4.0]])
    assert wrmse(t, t, w)[0] == 0
    np.testing.assert_allclose(wrmse(t + 1, t, w), [1.0])
    num, ff, oo = acc_terms(t, t, np.full_like(t, 2.5), w)
    np.testing.assert_allclose(acc_from_terms(num, ff, oo), [1.0])


def test_month_hour_keys():
    m, h = month_hour_keys(np.array(["2024-03-05T18:00", "2024-12-31T00:00"], dtype="datetime64[ns]"))
    assert list(m) == [2, 11] and list(h) == [3, 0]


def test_resume_continues_training(cfg):
    a = _args(cfg) + ["run_name=resume_test", "train.epochs=2"]
    main(["train", "--config", "smoke.yaml", *a, "train.epochs=1"])
    run = Path(cfg["paths"]["out"]) / "resume_test"
    assert (run / "last.pt").exists()
    main(["train", "--config", "smoke.yaml", *a, f"train.resume={run / 'last.pt'}"])
    hist = json.loads((run / "history.json").read_text())
    assert len(hist["val_loss"]) == 2          # epoch 1 from the first session + epoch 2 after resume


def test_channel_subset_reads_bigger_cache(cfg, tmp_path):
    """B0 (fewer channels) reads a cache built with more channels — no second cache."""
    import json
    import numpy as np
    from wfsea.data.memmap import CacheDirs, ChannelSubset, open_cache
    from wfsea.evaluation.baselines import compute_diff_std, load_climatology, save_climatology
    from wfsea.cli import _ensure_cache
    d = _ensure_cache(cfg, "train")[0]
    full = open_cache(d)
    want = [full.channels[3], full.channels[0]]
    sub = open_cache(CacheDirs([d], want))
    assert isinstance(sub, ChannelSubset) and sub.C == 2 and sub.channels == want
    np.testing.assert_array_equal(sub.mm["state"][5], np.asarray(full.mm["state"][5])[:, [3, 0]])
    np.testing.assert_array_equal(sub.mm["state"][2:4], np.asarray(full.mm["state"][2:4])[..., [3, 0]])
    np.testing.assert_allclose(compute_diff_std(sub), compute_diff_std(full)[[3, 0]], rtol=1e-5)
    clim = np.random.rand(12, 4, full.G, full.C).astype("f4")
    save_climatology(clim, tmp_path / "c.npz", [2020], full.channels)
    np.testing.assert_array_equal(load_climatology(tmp_path / "c.npz", want), clim[..., [3, 0]])
    import pickle; assert pickle.loads(pickle.dumps(sub)).mm["state"].shape == (full.T, full.G, 2)


def test_ensure_cache_falls_back_to_superset(cfg, monkeypatch):
    import copy
    import wfsea.cli as cli
    from wfsea.data.memmap import ChannelSubset, open_cache
    full = cli._ensure_cache(cfg, "train")
    orig = cli._channels
    monkeypatch.setattr(cli, "_channels", lambda c: list(orig(c))[:5] if c.get("_small") else orig(c))
    small = copy.deepcopy(cfg); small["_small"] = True
    small["data"]["cache_part_years"] = 1            # different part grouping than the stored cache
    dirs = cli._ensure_cache(small, "train")
    assert [str(d) for d in dirs] == [str(d) for d in full]           # reused, nothing built
    c = open_cache(dirs)
    assert isinstance(c, ChannelSubset) and c.channels == [tuple(x) for x in list(orig(cfg))[:5]]


def test_train_with_interleaved_sampler(cfg):
    a = _args(cfg) + ["run_name=interleaved_test", "train.epochs=1", "train.sampler=interleaved", "train.block_len=4"]
    main(["train", "--config", "smoke.yaml", *a])
    hist = json.loads((Path(cfg["paths"]["out"]) / "interleaved_test" / "history.json").read_text())
    assert len(hist["val_loss"]) == 1 and "data_wait_sec" in hist


def test_fusion_train_and_evaluate(cfg):
    """xattn fusion: trains, saves, reloads from the checkpoint and evaluates."""
    a = _args(cfg) + ["run_name=xattn_test", "train.epochs=1", "model.fusion=xattn", "model.fusion_window=3",
                      "model.eye_hidden=8", "model.fusion_dim=16", "features.time=true", "features.solar=true"]
    main(["train", "--config", "smoke.yaml", *a])
    run = Path(cfg["paths"]["out"]) / "xattn_test"
    main(["evaluate", "--config", "smoke.yaml", "--ckpt", str(run / "best.pt"), *a])
    df = pd.read_csv(run / "eval_persist" / "metrics_raw.csv")
    assert np.isfinite(df.rmse).all() and "graphcast" in set(df.model)   # (raw ACC is NaN for empty regions)
