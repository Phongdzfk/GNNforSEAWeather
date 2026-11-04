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
