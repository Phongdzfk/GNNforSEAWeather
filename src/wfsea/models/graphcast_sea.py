"""GraphCast-SEA Encoder–Processor–Decoder GNN. Owner: Hieu.

Parameter names and shapes are identical to nb4-grapcastmodel-v2, so the
existing Kaggle checkpoint `graphcast_sea_best.pt` loads into this class.
Difference: graph index tensors live in (non-persistent) buffers, so
forward(x) needs only the input and DataParallel works without extra plumbing.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from wfsea.graph.mesh import Graph


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None, layers: int = 2):
        super().__init__()
        hidden = hidden or out_dim
        net: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(layers - 2):
            net += [nn.Linear(hidden, hidden), nn.SiLU()]
        net += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*net)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.net(x))


def scatter_mean_like(msg: torch.Tensor, dst: torch.Tensor, n: int, deg: torch.Tensor | None) -> torch.Tensor:
    """Sum messages (B,E,D) into (B,n,D); divide by in-degree if given."""
    B, _, D = msg.shape
    out = torch.zeros(B, n, D, device=msg.device, dtype=msg.dtype)
    out.scatter_add_(1, dst.view(1, -1, 1).expand(B, -1, D), msg)
    return out / (deg.to(msg.dtype) + 1e-6) if deg is not None else out


class GNNLayer(nn.Module):
    """Interaction-network layer, vectorised over the batch."""

    def __init__(self, node_dim: int, edge_dim: int = 1):
        super().__init__()
        self.msg_mlp = MLP(node_dim * 2 + edge_dim, node_dim)
        self.node_mlp = MLP(node_dim * 2, node_dim)

    def forward(self, x, edge_src, edge_dst, edge_attr, deg):
        B = x.size(0)
        ea = edge_attr.unsqueeze(0).expand(B, -1, -1).to(x.dtype)
        msg = self.msg_mlp(torch.cat([x[:, edge_src], x[:, edge_dst], ea], dim=-1))
        agg = scatter_mean_like(msg.to(x.dtype), edge_dst, x.size(1), deg)
        return x + self.node_mlp(torch.cat([x, agg], dim=-1))


class GraphCastSEA(nn.Module):
    def __init__(self, n_in: int, n_out: int, graph: Graph, hidden: int = 128, n_layers: int = 4,
                 delta_scale=None, mesh_hidden: int | None = None):
        """hidden = width on the 31,959-node grid (expensive: 127,836 grid↔mesh edges);
        mesh_hidden = width on the 500 mesh nodes (cheap). Widening the mesh relieves the
        grid→mesh bottleneck at little cost. mesh_hidden=None → same as hidden (legacy)."""
        super().__init__()
        self.n_in, self.n_out, self.n_mesh = n_in, n_out, graph.n_mesh
        # Decoder predicts ΔX in units of the tendency std; delta_scale converts back (ones = legacy).
        ds = torch.ones(n_out) if delta_scale is None else torch.as_tensor(delta_scale, dtype=torch.float32)
        self.register_buffer("delta_scale", ds)
        t = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt)
        # persistent (legacy names, present in old checkpoints)
        self.register_buffer("mesh_static", t(graph.mesh_static, torch.float32))
        deg = torch.bincount(t(graph.m2m_dst), minlength=graph.n_mesh).float().view(1, -1, 1)
        self.register_buffer("m2m_deg", deg)
        self.register_buffer("m2m_ea_zero", t(graph.m2m_attr, torch.float32))
        # graph topology (rebuilt from config, not stored in state_dict)
        for name in ("g2m_src", "g2m_dst", "m2m_src", "m2m_dst", "m2g_src", "m2g_dst"):
            self.register_buffer(name, t(getattr(graph, name)), persistent=False)
        self.register_buffer("g2m_ea", t(graph.g2m_attr, torch.float32), persistent=False)
        self.register_buffer("m2g_ea", t(graph.m2g_attr, torch.float32), persistent=False)

        e_g, e_m = graph.g2m_attr.shape[1], graph.m2m_attr.shape[1]
        hg, hm = hidden, mesh_hidden or hidden
        self.grid_encoder = MLP(n_in, hg)
        self.mesh_encoder = MLP(4, hm)
        self.g2m_mlp = MLP(hg + hm + e_g, hm)          # grid → mesh message, in mesh width
        self.processors = nn.ModuleList([GNNLayer(hm, edge_dim=e_m) for _ in range(n_layers)])
        self.m2g_mlp = MLP(hm + hg + e_g, hg)          # mesh → grid message, in grid width
        self.decoder = nn.Linear(hg, n_out)
        nn.init.zeros_(self.decoder.weight)   # starts as persistence
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x_grid: torch.Tensor) -> torch.Tensor:
        """x_grid (B, G, n_in) → next state (B, G, n_out) = X(t) + ΔX."""
        B = x_grid.size(0)
        base = x_grid[..., : self.n_out]
        h_grid = self.grid_encoder(x_grid)
        h_mesh = self.mesh_encoder(self.mesh_static).unsqueeze(0).expand(B, -1, -1)

        ea = self.g2m_ea.unsqueeze(0).expand(B, -1, -1).to(h_grid.dtype)
        msg = self.g2m_mlp(torch.cat([h_grid[:, self.g2m_src], h_mesh[:, self.g2m_dst], ea], dim=-1))
        h_mesh = h_mesh + scatter_mean_like(msg.to(h_mesh.dtype), self.g2m_dst, self.n_mesh, None)

        for proc in self.processors:
            h_mesh = proc(h_mesh, self.m2m_src, self.m2m_dst, self.m2m_ea_zero, self.m2m_deg)

        ea2 = self.m2g_ea.unsqueeze(0).expand(B, -1, -1).to(h_grid.dtype)
        msg2 = self.m2g_mlp(torch.cat([h_mesh[:, self.m2g_src], h_grid[:, self.m2g_dst], ea2], dim=-1))
        h_grid = h_grid + scatter_mean_like(msg2.to(h_grid.dtype), self.m2g_dst, h_grid.size(1), None)
        return base + self.decoder(h_grid) * self.delta_scale.to(h_grid.dtype)
