import numpy as np
import torch

from wfsea.config import load_config
from wfsea.graph.mesh import build_graph
from wfsea.models import ChannelAdapter, build_model, count_params
from wfsea.models.graphcast_sea import GraphCastSEA

ERA5_LATS = np.arange(28.5, -11.01, -0.25)
ERA5_LONS = np.arange(92.0, 142.01, 0.25)


def test_legacy_graph_matches_documented_numbers():
    g = build_graph(ERA5_LATS, ERA5_LONS)
    assert (len(ERA5_LATS), len(ERA5_LONS)) == (159, 201)
    s = g.stats()
    assert s["n_mesh"] == 500 and s["mesh_shape"] == [20, 25]
    assert s["n_g2m"] == 127_836 and s["n_m2m"] == 3_734 and s["n_m2g"] == 127_836
    assert s["mesh_type"].startswith("regular")


def test_legacy_param_count_is_538382():
    g = build_graph(ERA5_LATS, ERA5_LONS)
    m = GraphCastSEA(n_in=38, n_out=14, graph=g, hidden=128, n_layers=4)
    assert count_params(m) == 538_382


def test_geo_edges_change_only_edge_mlps():
    g = build_graph(ERA5_LATS, ERA5_LONS, edge_features="geo")
    assert g.edge_dim == 4
    # edge dim 1→4 adds 3 inputs to g2m, m2g and each of the 4 processor message MLPs
    assert count_params(GraphCastSEA(38, 14, g)) == 538_382 + 3 * 128 * (2 + 4)


def _tiny_graph():
    lats, lons = np.arange(3.0, -0.01, -0.25), np.arange(100.0, 103.01, 0.25)
    return lats, lons, build_graph(lats, lons, 1.0, 0.0, 3.5, 100.0, 103.5)


def test_zero_init_decoder_is_persistence_and_grads_flow():
    lats, lons, g = _tiny_graph()
    m = GraphCastSEA(38, 14, g, hidden=32, n_layers=2)
    x = torch.randn(2, len(lats) * len(lons), 38)
    y = m(x)
    torch.testing.assert_close(y, x[..., :14])
    y.pow(2).mean().backward()
    assert m.decoder.weight.grad.abs().sum() > 0


def test_state_dict_roundtrip_without_graph_indices():
    lats, lons, g = _tiny_graph()
    m = GraphCastSEA(38, 14, g, hidden=32, n_layers=2)
    keys = set(m.state_dict())
    assert {"mesh_static", "m2m_deg", "m2m_ea_zero"} <= keys and "g2m_src" not in keys
    m2 = GraphCastSEA(38, 14, g, hidden=32, n_layers=2)
    m2.load_state_dict(m.state_dict())


def test_unet_same_interface():
    cfg = load_config(None, ["model.name=unet", "model.unet_base=8", "model.unet_depth=2"])
    m = build_model(cfg, 38, 14, grid_shape=(13, 13))
    x = torch.randn(1, 169, 38)
    torch.testing.assert_close(m(x), x[..., :14])


def test_channel_adapter_permutes_both_ways():
    class Echo(torch.nn.Module):            # returns its X(t) block → tests the permutation only
        def forward(self, x):
            return x[..., :3] * torch.tensor([1.0, 10.0, 100.0])
    model_ch, data_ch = [("a", None), ("b", None), ("c", None)], [("c", None), ("a", None), ("b", None)]
    ad = ChannelAdapter(Echo(), model_ch, data_ch)
    x = torch.tensor([[[3.0, 1.0, 2.0, 0, 0, 0, 9.0]]])       # data order c=3, a=1, b=2
    out = ad(x)
    torch.testing.assert_close(out, torch.tensor([[[300.0, 1.0, 20.0]]]))


def test_load_legacy_notebook_checkpoint(tmp_path):
    """A checkpoint saved by nb4-grapcastmodel-v2 (config dict + state_dict) loads and runs."""
    from wfsea.models import load_checkpoint
    lats, lons, g = _tiny_graph()
    m = GraphCastSEA(38, 14, g, hidden=32, n_layers=2)
    torch.nn.init.normal_(m.decoder.weight, std=0.01)
    legacy_vars = ["z", "q", "u", "v", "tp", "u10", "v10", "d2m", "t2m", "msl"]   # Zarr data_vars order
    torch.save({"epoch": 3, "model_state": m.state_dict(), "val_loss": 0.1,
                "config": {"n_in": 38, "n_out": 14, "n_dyn": 14, "n_mesh": g.n_mesh, "hidden": 32, "n_layers": 2,
                           "dynamic_vars": legacy_vars, "static_vars": ["lsm", "geopotential_surf"],
                           "mesh_res": 1.0, "n_past": 2, "residual": True}}, tmp_path / "old.pt")
    cfg = load_config(None, ["graph.mesh_res=1.0", "graph.mesh_lat_start=0.0", "graph.mesh_lat_stop=3.5",
                             "graph.mesh_lon_start=100.0", "graph.mesh_lon_stop=103.5"])
    model, _, spec, channels, _ = load_checkpoint(tmp_path / "old.pt", cfg, lats, lons, legacy_levels=(500, 850))
    assert spec.n_in == 38 and channels[:2] == [("z", 500), ("z", 850)] and channels[8] == ("tp", None)
    x = torch.randn(1, len(lats) * len(lons), 38)
    torch.testing.assert_close(model(x), m(x))


def test_mesh_hidden_widens_only_mesh_side():
    g = build_graph(ERA5_LATS, ERA5_LONS, edge_features="geo")
    from wfsea.config import load_config
    from wfsea.models import build_model
    sizes = {}
    for c in ("v2.yaml", "v2_wide.yaml", "v2_wider.yaml"):
        cfg = load_config(c)
        m = build_model(cfg, 43, 14, g)
        sizes[c] = count_params(m)
        assert m.grid_encoder.net[0].out_features == 128
    assert sizes["v2.yaml"] < 2e6 < sizes["v2_wide.yaml"] < 8e6 < sizes["v2_wider.yaml"] < 2e7
    lats, lons, tg = _tiny_graph()
    m = GraphCastSEA(38, 14, tg, hidden=16, n_layers=2, mesh_hidden=48)
    x = torch.randn(2, len(lats) * len(lons), 38)
    torch.testing.assert_close(m(x), x[..., :14])


def test_satellite_fusion_modes():
    import numpy as np, torch
    from wfsea.data.features import InputSpec
    from wfsea.graph.mesh import build_graph
    from wfsea.models import build_model, count_params
    lats = np.arange(3.0, -0.01, -0.25); lons = np.arange(100.0, 103.01, 0.25)
    g = build_graph(lats, lons, mesh_res=1.0, mesh_lat_start=0.0, mesh_lat_stop=3.5, mesh_lon_start=100.0,
                    mesh_lon_stop=103.5, edge_features="geo")
    spec = InputSpec(n_dyn=5, n_static=2, time=True, solar=True, tb_subgrid=True)
    G = len(lats) * len(lons)
    sl = spec.satellite_slice()
    x = torch.randn(2, G, spec.n_in)
    sizes = {}
    for mode in ("concat", "gated", "xattn"):
        cfg = {"model": {"name": "graphcast", "hidden": 16, "n_layers": 2, "fusion": mode, "eye_hidden": 8,
                         "fusion_window": 3, "fusion_heads": 2, "fusion_dim": 8}}
        torch.manual_seed(0)
        m = build_model(cfg, spec.n_in, 5, g, (len(lats), len(lons)), spec=spec)
        torch.nn.init.normal_(m.decoder.weight, std=0.1)          # make outputs depend on the input
        y = m(x)
        assert y.shape == (2, G, 5) and torch.isfinite(y).all()
        sizes[mode] = count_params(m)
        if mode != "concat":
            x2 = x.clone(); x2[..., sl] += 5.0
            assert torch.allclose(m(x2), y, atol=1e-5)            # fusion starts closed: satellite ignored
            y.sum().backward()
            eye = m.fusion.eye.net[0].weight.grad
            out = (m.fusion.proj if mode == "gated" else m.fusion.o).weight.grad
            assert out is not None and out.abs().sum() > 0         # ... but it can learn to open
            assert eye is not None                                 # eye is in the graph
    assert sizes["gated"] > sizes["concat"] and sizes["xattn"] > sizes["concat"]


def test_xattn_ignores_padding_and_uses_neighbours():
    import torch
    from wfsea.models.fusion import SatelliteFusion
    torch.manual_seed(0)
    f = SatelliteFusion(n_sat=2, hidden=8, grid_shape=(16, 16), mode="xattn", eye_hidden=4, window=3, heads=2, attn_dim=8)
    torch.nn.init.normal_(f.o.weight, std=0.5)
    h = torch.randn(1, 256, 8); sat = torch.zeros(1, 256, 2)
    base = f(h, sat)
    sat2 = sat.clone(); sat2[0, 8 * 16 + 8] = 10.0                # a cloud in cell (8,8)
    d = (f(h, sat2) - base).abs().sum(-1).view(16, 16)
    # reach = eye CNN receptive field (radius 4) + attention window (radius 1) = 5 cells
    assert d[8, 8] > 0 and d[5, 5] > 0 and d[13, 8] > 0
    assert d[0, 0] == 0 and d[8, 15] == 0 and d[2, 8] == 0
    assert torch.isfinite(f(h, sat2)).all()                       # edge cells (padding masked) are fine
