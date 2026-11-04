"""v2/v3/v4 improvements: tendency-normalised loss, time/solar, geo edges, rollout fine-tune, boundary forcing."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wfsea.cli import main
from wfsea.data.boundary import boundary_mask
from wfsea.data.features import toa_insolation
from wfsea.data.memmap import SplitCache
from wfsea.data.stats import NormStats, tp_forward, tp_inverse
from wfsea.evaluation.baselines import compute_diff_std
from wfsea.models.graphcast_sea import GraphCastSEA
from wfsea.graph.mesh import build_graph

from test_pipeline import _args


def test_tp_transform_roundtrip_and_denorm():
    tp = np.array([0.0, 1e-5, 1e-3, 0.02])
    np.testing.assert_allclose(tp_inverse(tp_forward(tp, "log1p"), "log1p"), tp, rtol=1e-6, atol=1e-12)
    st = NormStats({"tp": 1.0, "t2m": 290.0}, {"tp": 2.0, "t2m": 5.0}, 0, 1, {"tp_transform": "log1p"})
    y = st.denorm(np.array([[0.0, 1.0]]), [("t2m", None), ("tp", None)])
    np.testing.assert_allclose(y, [[290.0, np.expm1(3.0) * 1e-4]], rtol=1e-6)


def test_solar_is_zero_at_night_and_peaks_near_noon():
    lats, lons = np.array([10.0]), np.array([105.0])          # Hanoi-ish longitude: noon ≈ 05 UTC
    t = np.array(["2024-03-20T05:00", "2024-03-20T17:00"], dtype="datetime64[ns]")
    s = toa_insolation(t, lats, lons)[:, 0]
    assert s[0] > 0.9 and s[1] == 0


def test_boundary_mask_band():
    lats, lons = np.arange(3.0, -0.01, -0.25), np.arange(100.0, 103.01, 0.25)
    m = boundary_mask(lats, lons, 0.5).reshape(len(lats), len(lons))
    assert m[0].all() and m[:, -1].all() and not m[6, 6] and m[2, 6] == False and m[1, 6] == True


def test_delta_scale_keeps_persistence_init_and_scales_output():
    lats, lons = np.arange(3.0, -0.01, -0.25), np.arange(100.0, 103.01, 0.25)
    g = build_graph(lats, lons, 1.0, 0.0, 3.5, 100.0, 103.5, edge_features="geo")
    ds = torch.linspace(0.1, 1.4, 14)
    m = GraphCastSEA(38, 14, g, hidden=16, n_layers=2, delta_scale=ds)
    x = torch.randn(1, len(lats) * len(lons), 38)
    torch.testing.assert_close(m(x), x[..., :14])
    torch.nn.init.constant_(m.decoder.bias, 1.0)
    torch.testing.assert_close(m(x) - x[..., :14], ds.expand(1, len(lats) * len(lons), 14))
    assert "delta_scale" in m.state_dict()


def test_diff_std_matches_numpy(cfg):
    from test_data import _cache
    c = SplitCache(_cache(cfg, "val"))
    st = np.asarray(c.mm["state"], dtype=np.float32)
    ref = (st[1:] - st[:-1]).reshape(-1, c.C).std(0)          # val has no gaps
    np.testing.assert_allclose(compute_diff_std(c), ref, rtol=1e-3)


def test_v2_then_rollout_finetune_then_boundary_eval(cfg):
    a = _args(cfg)
    v2 = ["features.time=true", "features.solar=true", "graph.edge_features=geo", "train.loss_norm=delta"]
    main(["train", "--config", "smoke.yaml", *a, *v2, "run_name=t_v2"])
    out = Path(cfg["paths"]["out"])
    assert (out / "t_v2" / "best.pt").exists()
    ck = torch.load(out / "t_v2" / "best.pt", weights_only=False)
    assert ck["input_spec"]["time"] and ck["input_spec"]["solar"] and ck["model_state"]["delta_scale"].std() > 0

    main(["train", "--config", "smoke.yaml", *a, *v2, "run_name=t_v3", f"train.init_from={out / 't_v2' / 'best.pt'}",
          "train.rollout_steps=3", "train.grad_checkpoint=true", "boundary.width_deg=0.5"])
    hist = json.loads((out / "t_v3" / "history.json").read_text())
    assert np.isfinite(hist["best_val"])

    main(["evaluate", "--config", "smoke.yaml", "--ckpt", str(out / "t_v3" / "best.pt"), *a])
    ev = out / "t_v3" / "eval_persist"
    meta = json.loads((ev / "eval_config.json").read_text())
    assert meta["boundary_forcing"].startswith("ERA5")
    df = pd.read_csv(ev / "metrics_raw.csv")
    assert np.isfinite(df.rmse).all() and df[df.channel == "tp"].rmse.max() < 0.05   # physical metres
