"""Verification metrics. Owner: Tung.

All functions take PHYSICAL-unit fields shaped (G, C) unless noted, and the
latitude weights w (G,) normalised to mean 1.
"""
from __future__ import annotations

import numpy as np


def wrmse(pred: np.ndarray, truth: np.ndarray, w: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Latitude-weighted RMSE per channel → (C,)."""
    se = (pred - truth) ** 2 * w[:, None]
    if mask is not None:
        return np.sqrt(se[mask].sum(0) / w[mask].sum())
    return np.sqrt(se.sum(0) / w.sum())


def acc_terms(pred, truth, clim, w):
    """Accumulators for ACC = Σ w f'o' / sqrt(Σ w f'^2 · Σ w o'^2) (anomalies vs train climatology)."""
    fa, oa = pred - clim, truth - clim
    ww = w[:, None]
    return (ww * fa * oa).sum(0), (ww * fa * fa).sum(0), (ww * oa * oa).sum(0)


def acc_from_terms(num, ff, oo):
    return num / np.sqrt(ff * oo + 1e-30)


def region_masks(static_phys: dict[str, np.ndarray], mountain_quantile: float = 0.75) -> dict[str, np.ndarray]:
    """all / land / sea / mountain (top-quartile land elevation) / plain."""
    G = next(iter(static_phys.values())).shape[0]
    masks = {"all": np.ones(G, bool)}
    if "lsm" in static_phys:
        land = static_phys["lsm"] > 0.5
        masks["land"], masks["sea"] = land, ~land
        if "geopotential_surf" in static_phys:
            elev = static_phys["geopotential_surf"] / 9.80665
            thr = np.quantile(elev[land], mountain_quantile) if land.any() else np.inf
            masks["mountain"] = land & (elev >= thr)
            masks["plain"] = land & (elev < thr)
    return masks


def radial_spectrum(field2d: np.ndarray) -> np.ndarray:
    """Isotropic power spectrum of a 2-D field (for spectral-blurring diagnostics)."""
    F = np.fft.fftshift(np.fft.fft2(field2d - field2d.mean()))
    P = np.abs(F) ** 2
    ny, nx = field2d.shape
    y, x = np.indices((ny, nx))
    r = np.sqrt((x - nx // 2) ** 2 + (y - ny // 2) ** 2).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def skill_score(rmse_model: np.ndarray, rmse_ref: np.ndarray) -> np.ndarray:
    """1 − RMSE_model / RMSE_ref (positive = better than reference)."""
    return 1.0 - rmse_model / np.maximum(rmse_ref, 1e-12)
