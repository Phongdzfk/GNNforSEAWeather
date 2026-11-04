"""Lateral boundary forcing for the regional model (improvement B7).

A limited-area model cannot know what weather enters the domain from outside, so errors
grow inward from the edges over 72 h. Regional ML models (e.g. Neural-LAM) overwrite a band
of boundary nodes with the driving analysis/forecast at every step. Here the band is forced
with ERA5 (research setting — label results "ERA5 boundary"); operationally it would come
from a global forecast (GFS/IFS). Metrics are then computed on the interior only.
"""
from __future__ import annotations

import numpy as np


def boundary_mask(lats: np.ndarray, lons: np.ndarray, width_deg: float) -> np.ndarray:
    """(G,) True for grid nodes within `width_deg` of the domain edge."""
    if not width_deg:
        return np.zeros(len(lats) * len(lons), bool)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    edge = ((la - lats.min() < width_deg) | (lats.max() - la < width_deg) |
            (lo - lons.min() < width_deg) | (lons.max() - lo < width_deg))
    return edge.reshape(-1)
