"""Model factory + checkpoint I/O (the Hieu → Tung inference contract)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import torch

from wfsea.models.graphcast_sea import GraphCastSEA
from wfsea.models.unet import UNetSEA


def build_model(cfg, n_in: int, n_out: int, graph=None, grid_shape=None, delta_scale=None,
                spec=None) -> torch.nn.Module:
    m = cfg.get("model", {})
    name = m.get("name", "graphcast")
    if name == "graphcast":
        if graph is None:
            raise ValueError("graphcast needs a graph")
        fusion = m.get("fusion") or "concat"
        kw = {}
        if fusion != "concat":
            if spec is None:
                raise ValueError("model.fusion needs the input spec (satellite columns)")
            kw = dict(fusion=fusion, sat_slice=spec.satellite_slice(), grid_shape=tuple(grid_shape),
                      fusion_kw=dict(eye_hidden=m.get("eye_hidden", 32), window=m.get("fusion_window", 5),
                                     heads=m.get("fusion_heads", 4), attn_dim=m.get("fusion_dim", 64)))
        return GraphCastSEA(n_in, n_out, graph, hidden=m.get("hidden", 128), n_layers=m.get("n_layers", 4),
                            delta_scale=delta_scale, mesh_hidden=m.get("mesh_hidden"), **kw)
    if name == "unet":
        H, W = grid_shape
        return UNetSEA(n_in, n_out, H, W, base=m.get("unet_base", 48), depth=m.get("unet_depth", 3),
                       delta_scale=delta_scale)
    raise ValueError(name)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def save_checkpoint(path, model, cfg, spec, channels, extra: dict | None = None) -> None:
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save({
        "format": "wfsea-v1",
        "model_state": core.state_dict(),
        "cfg": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg),
        "input_spec": spec.to_dict(),
        "channels": [list(c) for c in channels],
        "n_params": count_params(core),
        "git_commit": git_commit(),
        **(extra or {}),
    }, path)


def load_checkpoint(path, cfg=None, grid_lats=None, grid_lons=None, map_location="cpu",
                    legacy_levels=(850, 500)):
    """Load a wfsea-v1 checkpoint, or a legacy nb4 checkpoint (needs grid_lats/lons + cfg).

    Legacy checkpoints store only variable names; their channel order followed the
    Zarr data_vars order and the file's pressure_level order. `legacy_levels` must
    match that file order (check `ds.pressure_level.values` of brain_2026_x.zarr).
    The returned `channels` list is in MODEL order; wrap the model with
    `ChannelAdapter` when the data cache uses a different order.
    """
    from wfsea.config import Config, load_config
    from wfsea.data.features import InputSpec
    from wfsea.graph.mesh import build_graph_from_config

    ck = torch.load(path, map_location=map_location, weights_only=False)
    if ck.get("format") == "wfsea-v1":
        cfg = Config(ck["cfg"])
        spec = InputSpec(**ck["input_spec"])
        channels = [tuple(c) for c in ck["channels"]]
    else:  # legacy notebook checkpoint
        cfg = cfg or load_config()
        c = ck["config"]
        spec = InputSpec(n_dyn=c["n_dyn"], n_static=len(c.get("static_vars", [])))
        if spec.n_in != c["n_in"]:
            raise ValueError(f"Legacy checkpoint n_in={c['n_in']} but spec gives {spec.n_in}")
        from wfsea.constants import PRESSURE_VARS
        channels = [(v, lev) for v in c["dynamic_vars"]
                    for lev in (legacy_levels if v in PRESSURE_VARS else (None,))]
        cfg.setdefault("model", {}).update({"hidden": c["hidden"], "n_layers": c["n_layers"]})
    lats = grid_lats if grid_lats is not None else ck.get("grid_lats")
    lons = grid_lons if grid_lons is not None else ck.get("grid_lons")
    graph = build_graph_from_config(cfg, lats, lons) if cfg.get("model", {}).get("name", "graphcast") == "graphcast" else None
    model = build_model(cfg, spec.n_in, spec.n_dyn, graph, (len(lats), len(lons)), spec=spec)
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    missing = [k for k in missing if k != "delta_scale"]   # legacy checkpoints: delta_scale = 1
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval(), cfg, spec, channels, ck


class ChannelAdapter(torch.nn.Module):
    """Run a model trained with channel order `model_channels` on data in `data_channels` order."""

    def __init__(self, model, model_channels, data_channels):
        super().__init__()
        self.model = model
        model_channels = [tuple(c) for c in model_channels]
        data_channels = [tuple(c) for c in data_channels]
        C = len(data_channels)
        to_model = torch.tensor([data_channels.index(c) for c in model_channels])
        to_data = torch.tensor([model_channels.index(c) for c in data_channels])
        self.register_buffer("to_model", to_model, persistent=False)
        self.register_buffer("to_data", to_data, persistent=False)
        self.C = C

    def forward(self, x):
        C = self.C
        dyn = torch.cat([x[..., :C][..., self.to_model], x[..., C:2 * C][..., self.to_model]], dim=-1)
        y = self.model(torch.cat([dyn, x[..., 2 * C:]], dim=-1))
        return y[..., self.to_data]
