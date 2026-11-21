"""Efficiency numbers for the thesis table. Owner: Tung.

`wfsea bench --config <cfg> --ckpt <run>/best.pt` writes <out>/<run>/bench.json with
  - inference: seconds per 72 h forecast (12 steps, warm-up excluded), peak GPU memory during it,
  - training: read from the run's history.json next to the checkpoint (s/epoch, data-wait share,
    peak training VRAM, epochs, GPU-hours) when that file is attached.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from wfsea.evaluation.rollout import evenly_spaced_starts, rollout


def _history(ckpt: str | Path) -> dict:
    h = Path(ckpt).with_name("history.json")
    if not h.exists():
        return {}
    d = json.loads(h.read_text())
    sec = [x for x in d.get("epoch_sec", []) if x]
    wait = [x for x in d.get("data_wait_sec", []) if x is not None]
    out = {"train_epochs": len(sec), "train_sec_per_epoch": float(np.median(sec)) if sec else None,
           "train_gpu_hours": float(np.sum(sec)) / 3600 if sec else None,
           "train_vram_gb": d.get("max_vram_gb")}
    if sec and len(wait) == len(sec):
        out["data_wait_share"] = float(np.sum(wait) / np.sum(sec))
    return out


def bench(model, cache, spec, ckpt, out_dir: Path, device, n_steps: int = 12, n_runs: int = 5,
          tb_policy: str = "mask", bmask=None, n_params=None) -> dict:
    model.eval()
    starts = evenly_spaced_starts(cache.valid_starts(n_future=n_steps), n_runs + 1)
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()

    def sync():
        if cuda:
            torch.cuda.synchronize()

    rollout(model, cache, spec, int(starts[0]), n_steps, tb_policy, device, bmask)   # warm-up
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    times = []
    for t0 in starts[1:]:
        sync(); t = time.perf_counter()
        rollout(model, cache, spec, int(t0), n_steps, tb_policy, device, bmask)
        sync(); times.append(time.perf_counter() - t)
    res = {"checkpoint": str(ckpt), "n_params": n_params, "device": torch.cuda.get_device_name(0) if cuda else "cpu",
           "forecast_hours": n_steps * 6, "infer_sec_per_forecast": float(np.median(times)),
           "infer_sec_per_step": float(np.median(times)) / n_steps,
           "infer_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else None, **_history(ckpt)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res
