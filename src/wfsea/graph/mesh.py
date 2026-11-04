"""Grid–Mesh–Grid graph construction. Owner: Hieu.

Terminology (fixes the review finding): the baseline mesh is a REGULAR
2° × 2° latitude–longitude lattice with 8-neighbour (Moore) connectivity.
It is not icosahedral. Proposal 1 (multi-level 1°/2°/4° mesh) plugs in here.

Legacy settings reproduce the notebook graph exactly:
  500 mesh nodes, 127,836 G→M edges (k=4), 3,734 M→M edges, M→G = reverse(G→M).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlat, dlon = p2 - p1, np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


@dataclass
class Graph:
    mesh_lats: np.ndarray        # (M,)
    mesh_lons: np.ndarray
    mesh_shape: tuple[int, int]
    mesh_static: np.ndarray      # (M, 4) sin/cos lat/lon
    g2m_src: np.ndarray          # grid index
    g2m_dst: np.ndarray          # mesh index
    g2m_attr: np.ndarray         # (E, d)
    m2m_src: np.ndarray
    m2m_dst: np.ndarray
    m2m_attr: np.ndarray
    m2g_src: np.ndarray          # mesh index
    m2g_dst: np.ndarray          # grid index
    m2g_attr: np.ndarray
    edge_features: str

    @property
    def n_mesh(self) -> int:
        return len(self.mesh_lats)

    @property
    def edge_dim(self) -> int:
        return self.g2m_attr.shape[1]

    def stats(self) -> dict:
        d = haversine_km(self.mesh_lats[self.m2m_src], self.mesh_lons[self.m2m_src],
                         self.mesh_lats[self.m2m_dst], self.mesh_lons[self.m2m_dst])
        deg = np.bincount(self.m2m_dst, minlength=self.n_mesh)
        return {
            "mesh_type": "regular lat-lon, 8-neighbour",
            "mesh_shape": list(self.mesh_shape), "n_mesh": int(self.n_mesh),
            "n_g2m": int(len(self.g2m_src)), "n_m2m": int(len(self.m2m_src)), "n_m2g": int(len(self.m2g_src)),
            "edge_features": self.edge_features, "edge_dim": int(self.edge_dim),
            "m2m_length_km": {"min": float(d.min()), "mean": float(d.mean()), "max": float(d.max())},
            "m2m_in_degree": {"min": int(deg.min()), "max": int(deg.max())},
            "mesh_lat_range": [float(self.mesh_lats.min()), float(self.mesh_lats.max())],
            "mesh_lon_range": [float(self.mesh_lons.min()), float(self.mesh_lons.max())],
        }

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "graph.npz", **{k: v for k, v in self.__dict__.items()
                                                       if isinstance(v, np.ndarray)},
                            mesh_shape=np.array(self.mesh_shape), edge_features=np.array(self.edge_features))
        (out_dir / "graph_stats.json").write_text(json.dumps(self.stats(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Graph":
        z = np.load(path)
        kw = {k: z[k] for k in z.files if k not in ("mesh_shape", "edge_features")}
        return cls(mesh_shape=tuple(int(x) for x in z["mesh_shape"]), edge_features=str(z["edge_features"]), **kw)


def _edge_attr(kind, src_lat, src_lon, dst_lat, dst_lon):
    if kind == "legacy":
        # Euclidean distance in degree space, z-scored over the edge set (what nb4 did).
        d = np.sqrt((src_lat - dst_lat) ** 2 + (src_lon - dst_lon) ** 2)
        return ((d - d.mean()) / (d.std() + 1e-6)).astype(np.float32)[:, None]
    if kind == "geo":
        d = haversine_km(src_lat, src_lon, dst_lat, dst_lon)
        scale = d.max() + 1e-6
        dlat = (dst_lat - src_lat) * 111.2 / scale
        dlon = (dst_lon - src_lon) * 111.2 * np.cos(np.deg2rad((src_lat + dst_lat) / 2)) / scale
        return np.stack([d / scale, dlat, dlon, np.cos(np.deg2rad(dst_lat))], axis=1).astype(np.float32)
    raise ValueError(kind)


def build_graph(grid_lats: np.ndarray, grid_lons: np.ndarray, mesh_res: float = 2.0,
                mesh_lat_start: float = -10.0, mesh_lat_stop: float = 28.5,
                mesh_lon_start: float = 92.0, mesh_lon_stop: float = 142.0,
                k_g2m: int = 4, edge_features: str = "legacy") -> Graph:
    mlat1 = np.arange(mesh_lat_start, mesh_lat_stop, mesh_res)
    mlon1 = np.arange(mesh_lon_start, mesh_lon_stop, mesh_res)
    mlat, mlon = (a.reshape(-1) for a in np.meshgrid(mlat1, mlon1, indexing="ij"))
    nla, nlo = len(mlat1), len(mlon1)
    lr, lo = np.deg2rad(mlat), np.deg2rad(mlon)
    mesh_static = np.stack([np.sin(lr), np.cos(lr), np.sin(lo), np.cos(lo)], axis=1).astype(np.float32)

    glat, glon = (a.reshape(-1) for a in np.meshgrid(grid_lats, grid_lons, indexing="ij"))
    _, idx = cKDTree(np.stack([mlat, mlon], 1)).query(np.stack([glat, glon], 1), k=k_g2m)
    g2m_src = np.repeat(np.arange(len(glat)), k_g2m)
    g2m_dst = idx.reshape(-1)
    g2m_attr = _edge_attr(edge_features, glat[g2m_src], glon[g2m_src], mlat[g2m_dst], mlon[g2m_dst])

    src, dst = [], []
    for i in range(nla):
        for j in range(nlo):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if (di or dj) and 0 <= i + di < nla and 0 <= j + dj < nlo:
                        src.append(i * nlo + j); dst.append((i + di) * nlo + j + dj)
    m2m_src, m2m_dst = np.array(src), np.array(dst)
    if edge_features == "legacy":
        m2m_attr = np.zeros((len(m2m_src), 1), np.float32)   # nb4 fed zeros on mesh edges
    else:
        m2m_attr = _edge_attr(edge_features, mlat[m2m_src], mlon[m2m_src], mlat[m2m_dst], mlon[m2m_dst])

    return Graph(mesh_lats=mlat, mesh_lons=mlon, mesh_shape=(nla, nlo), mesh_static=mesh_static,
                 g2m_src=g2m_src, g2m_dst=g2m_dst, g2m_attr=g2m_attr,
                 m2m_src=m2m_src, m2m_dst=m2m_dst, m2m_attr=m2m_attr,
                 m2g_src=g2m_dst.copy(), m2g_dst=g2m_src.copy(), m2g_attr=g2m_attr.copy(),
                 edge_features=edge_features)


def build_graph_from_config(cfg, grid_lats, grid_lons) -> Graph:
    g = cfg.get("graph", {})
    return build_graph(grid_lats, grid_lons, g.get("mesh_res", 2.0), g.get("mesh_lat_start", -10.0),
                       g.get("mesh_lat_stop", 28.5), g.get("mesh_lon_start", 92.0), g.get("mesh_lon_stop", 142.0),
                       g.get("k_g2m", 4), g.get("edge_features", "legacy"))
