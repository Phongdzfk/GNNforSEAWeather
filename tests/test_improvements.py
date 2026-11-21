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


def test_interleaved_sampler_covers_all_and_streams():
    import numpy as np
    from wfsea.data.dataset import InterleavedBlockSampler
    s = InterleavedBlockSampler(103, batch_size=4, block_len=8, seed=1)
    for e in (0, 1):
        s.set_epoch(e)
        o = s.order()
        assert sorted(o.tolist()) == list(range(103))               # each start exactly once
    o = s.order()
    b = o[:len(o) // 4 * 4].reshape(-1, 4)
    assert len(set(b[0])) == 4                                      # a batch = 4 different streams
    assert (np.abs(np.diff(b[0])) > 1).any()                        # ... not 4 consecutive times
    assert (b[1] - b[0] == 1).mean() >= 0.75                        # next batch = next step of the same streams
    s.set_epoch(2); assert not np.array_equal(s.order(), o)         # new order each epoch


def test_tp_inverse_is_capped():
    import numpy as np
    from wfsea.data.stats import TP_MAX_M, tp_forward, tp_inverse
    x = np.array([0.0, 0.001, 0.05, 0.2])
    np.testing.assert_allclose(tp_inverse(tp_forward(x, "log1p"), "log1p"), x, rtol=1e-6, atol=1e-9)
    assert tp_inverse(np.array([50.0]), "log1p")[0] == np.float64(TP_MAX_M) or \
        abs(tp_inverse(np.array([50.0]), "log1p")[0] - TP_MAX_M) < 1e-9          # no exp blow-up
    assert tp_inverse(np.array([-3.0]), "log1p")[0] == 0


def test_sat_dropout_hides_only_satellite(cfg):
    import numpy as np, torch
    from wfsea.cli import _ensure_cache
    from wfsea.data.dataset import WeatherDataset
    from wfsea.data.features import InputSpec
    from wfsea.data.memmap import open_cache
    c = open_cache(_ensure_cache(cfg, "train"))
    spec = InputSpec(n_dyn=c.C, n_static=c.static.shape[1], time=True)
    keep = WeatherDataset(c, spec)
    drop = WeatherDataset(c, spec, sat_dropout=1.0)
    _, _, fk, _ = keep[3]; _, _, fd, _ = drop[3]
    assert drop.n_sat == 4 and fk[..., :4].abs().sum() > 0
    assert (fd[..., :4] == 0).all() and torch.equal(fd[..., 4:], fk[..., 4:])
    torch.manual_seed(0)
    half = WeatherDataset(c, spec, sat_dropout=0.5)
    hidden = [bool((half[i][2][..., :4] == 0).all()) for i in range(40)]
    assert 8 < sum(hidden) < 32


def test_degenerate_static_borrowed_from_sibling_cache(tmp_path):
    """A cache whose statics came out constant (NaN source read) borrows them from a cache on the same grid."""
    import json as _json
    import numpy as _np
    from wfsea.data.memmap import SplitCache, static_is_degenerate
    G = 12
    meta = {"T": 1, "G": G, "C": 1, "lats": [0.0, 1.0, 2.0], "lons": [0.0, 1.0, 2.0, 3.0],
            "channels": [["t2m", None]], "static_vars": ["lsm", "geopotential_surf"]}
    good = _np.random.default_rng(0).normal(size=(G, 2)).astype("float32")
    for name, st in [("val_2025", good), ("test_2026", _np.zeros((G, 2), "float32"))]:
        d = tmp_path / name; d.mkdir()
        (d / "meta.json").write_text(_json.dumps(meta)); _np.save(d / "static.npy", st)
        _np.save(d / "time.npy", _np.array(["2026-01-01T00"], "datetime64[ns]"))
    assert static_is_degenerate(_np.zeros((G, 2))) and not static_is_degenerate(good)
    c = SplitCache(tmp_path / "test_2026")
    assert _np.allclose(c.static, good)
    _np.save(tmp_path / "val_2025" / "static.npy", _np.zeros((G, 2), "float32"))
    import pytest as _pt
    with _pt.raises(ValueError):
        SplitCache(tmp_path / "test_2026")
