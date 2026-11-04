"""The single evaluation runner: N starts × 12 leads × 14 channels × regions. Owner: Tung.

Outputs (in <out>/<run_name>/eval/):
  start_times.json       the exact start list (publish it in the thesis appendix)
  metrics_raw.csv        one row per (model, start, lead, channel, region) — the evidence base
  summary_rmse.csv       mean RMSE per (model, channel, lead), physical units
  summary_acc.csv        ACC per (model, channel, lead), vs TRAIN climatology
  spectrum.csv           radial spectrum of t2m at +48 h (truth vs forecast)
  eval_config.json       tb_policy, checkpoint path, n_params, git commit
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wfsea.constants import TIME_STEP_HOURS, channel_name
from wfsea.data.memmap import SplitCache
from wfsea.evaluation.baselines import Climatology
from wfsea.evaluation.metrics import acc_from_terms, acc_terms, radial_spectrum, region_masks, wrmse
from wfsea.evaluation.rollout import evenly_spaced_starts, rollout


def evaluate(models: dict, spec, cache: SplitCache, stats, clim: np.ndarray, out_dir: str | Path,
             n_starts: int = 30, n_steps: int = 12, tb_policy: str = "persist",
             regions=("all", "land", "sea", "mountain", "plain"), device="cpu", extra_meta=None,
             bmask: np.ndarray | None = None) -> pd.DataFrame:
    """`models`: {"gnn": nn.Module, ...}. Persistence and climatology are always added."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    channels = cache.channels
    names = [channel_name(v, l) for v, l in channels]
    phys = lambda a: stats.denorm(np.asarray(a, dtype=np.float32), channels)
    w = np.repeat(np.cos(np.deg2rad(cache.lats)), len(cache.lons)).astype(np.float64)
    w /= w.mean()
    static_phys = {v: cache.static[:, i] * stats.std[v] + stats.mean[v] for i, v in enumerate(cache.meta["static_vars"])}
    masks = {k: m for k, m in region_masks(static_phys).items() if k in regions}
    if bmask is not None and bmask.any():
        # Boundary band is forced with ERA5 → score every model on the interior only (fair comparison).
        masks = {k: m & ~bmask for k, m in masks.items()}
        w = w * (~bmask)
    clim_obj = Climatology(clim)

    starts = evenly_spaced_starts(cache.valid_starts(n_future=n_steps), n_starts)
    (out_dir / "start_times.json").write_text(json.dumps([str(cache.times[s])[:16] for s in starts], indent=2))

    rows = []
    spec_rows = {}
    st = cache.mm["state"]
    for si, t0 in enumerate(starts):
        truth = np.stack([phys(st[t0 + k + 1].astype(np.float32)) for k in range(n_steps)])
        valid_times = cache.times[t0 + 1:t0 + 1 + n_steps]
        clim_seq = np.stack([phys(clim_obj.at(t)) for t in valid_times])
        fc = {"persistence": np.repeat(phys(st[t0].astype(np.float32))[None], n_steps, 0),
              "climatology": clim_seq}
        for name, m in models.items():
            fc[name] = phys(rollout(m, cache, spec, t0, n_steps, tb_policy, device, bmask))
        for model_name, f in fc.items():
            for k in range(n_steps):
                lead = (k + 1) * TIME_STEP_HOURS
                num, ff, oo = acc_terms(f[k], truth[k], clim_seq[k], w)
                for reg, mask in masks.items():
                    r = wrmse(f[k], truth[k], w, mask)
                    for c, cname in enumerate(names):
                        rows.append((model_name, str(cache.times[t0])[:16], lead, cname, reg, float(r[c]),
                                     float(num[c]) if reg == "all" else np.nan,
                                     float(ff[c]) if reg == "all" else np.nan,
                                     float(oo[c]) if reg == "all" else np.nan))
            if "t2m" in names and n_steps >= 8:
                c = names.index("t2m"); H, W = len(cache.lats), len(cache.lons)
                spec_rows.setdefault(model_name, []).append(radial_spectrum(f[7][:, c].reshape(H, W)))
                spec_rows.setdefault("truth", []).append(radial_spectrum(truth[7][:, c].reshape(H, W)))
        print(f"[eval] start {si + 1}/{len(starts)} {str(cache.times[t0])[:16]}", flush=True)

    df = pd.DataFrame(rows, columns=["model", "start", "lead_h", "channel", "region", "rmse",
                                     "acc_num", "acc_ff", "acc_oo"])
    df.to_csv(out_dir / "metrics_raw.csv", index=False)
    a = df[df.region == "all"]
    a.groupby(["model", "channel", "lead_h"]).rmse.mean().unstack("lead_h").to_csv(out_dir / "summary_rmse.csv")
    g = a.groupby(["model", "channel", "lead_h"])[["acc_num", "acc_ff", "acc_oo"]].sum()
    acc = acc_from_terms(g.acc_num, g.acc_ff, g.acc_oo).rename("acc")
    acc.unstack("lead_h").to_csv(out_dir / "summary_acc.csv")
    if spec_rows:
        with open(out_dir / "spectrum.csv", "w", newline="") as fh:
            wr = csv.writer(fh); wr.writerow(["series", "k", "power"])
            for s, arrs in spec_rows.items():
                for k, p in enumerate(np.mean(arrs, 0)):
                    wr.writerow([s, k, float(p)])
    (out_dir / "eval_config.json").write_text(json.dumps({
        "n_starts": int(len(starts)), "n_steps": n_steps, "tb_policy": tb_policy,
        "climatology": "train split, month × hour", "regions": list(masks),
        "boundary_forcing": "ERA5 band, interior-only scores" if bmask is not None and bmask.any() else "none",
        **(extra_meta or {})}, indent=2,
        default=str))
    return df
