"""Training loop (GNN or U-Net). Owner: Hieu.

Same recipe as the notebooks: AdamW, cosine LR, AMP, grad-clip 1.0, early stopping
on val loss, DataParallel on Kaggle T4×2. Added: seeding, optional multi-step
rollout loss (Proposal 3), timing/VRAM log, and a checkpoint that records config,
input schema, channel order and git commit (evidence ledger requirement).
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from torch.utils.data import DataLoader

from wfsea.config import Config, resolve_paths
from wfsea.data.dataset import WeatherDataset, assemble_input
from wfsea.data.features import InputSpec
from wfsea.data.memmap import SplitCache
from wfsea.graph.mesh import build_graph_from_config
from wfsea.models import build_model, count_params, save_checkpoint
from wfsea.training.loss import lat_weights_grid, spectral_loss, weighted_rmse


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Persistence(torch.nn.Module):
    def __init__(self, n_out):
        super().__init__(); self.n_out = n_out

    def forward(self, x):
        return x[..., : self.n_out]


def rollout_loss(net, batch, w_grid, device, amp, spectral_w=0.0, HW=None, channel_w=None,
                 bmask=None, grad_ckpt=False):
    """Mean over K autoregressive steps of the latitude-weighted RMSE.

    channel_w : 1/diff_std² (tendency-normalised loss) × optional per-variable weights
    bmask     : (G,) bool boundary band — forced with the truth after every step, excluded from the loss
    grad_ckpt : recompute each step in backward (needed for K ≥ 4 on a 16 GB T4)
    """
    dyn_t, dyn_tm1, forcing, target = (b.to(device, non_blocking=True) for b in batch)
    K = target.size(1)
    loss = 0.0
    for k in range(K):
        x = assemble_input(dyn_t, dyn_tm1, forcing[:, k])
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            if grad_ckpt and torch.is_grad_enabled():
                pred = torch.utils.checkpoint.checkpoint(net, x, use_reentrant=False)
            else:
                pred = net(x)
        pred = pred.float()
        if bmask is not None:
            pred = torch.where(bmask.view(1, -1, 1), target[:, k], pred)
        l = weighted_rmse(pred, target[:, k], w_grid, channel_w)
        if spectral_w:
            l = l + spectral_w * spectral_loss(pred, target[:, k], *HW)
        loss = loss + l / K
        dyn_tm1, dyn_t = dyn_t, pred
    return loss


def load_diff_std(cfg, train_cache: SplitCache) -> np.ndarray:
    from wfsea.config import find_file
    from wfsea.evaluation.baselines import compute_diff_std
    want = [tuple(c) for c in train_cache.channels]
    cands = [Path(cfg["paths"]["out"]) / "diff_stats.json"]
    try:
        cands.append(find_file("diff_stats.json"))
    except FileNotFoundError:
        pass
    for cand in cands:
        if not cand.exists():
            continue
        d = json.loads(cand.read_text())
        have = [tuple(c) for c in d["channels"]]
        if all(c in have for c in want):                    # exact, or a subset (B0 from 41 channels)
            ds = np.asarray(d["diff_std"], np.float32)
            print(f"[train] tendency std from {cand}")
            return ds[[have.index(c) for c in want]]
    print("[train] diff_stats.json not found — computing tendency std from the train cache")
    return compute_diff_std(train_cache)


def make_loader(ds, cfg, shuffle: bool) -> DataLoader:
    t = cfg["train"]
    kw = dict(batch_size=t["batch_size"], shuffle=shuffle, num_workers=t["num_workers"],
              pin_memory=torch.cuda.is_available(), drop_last=shuffle)
    if t["num_workers"] > 0:
        kw.update(persistent_workers=True, prefetch_factor=t["prefetch"])
    return DataLoader(ds, **kw)


def train(cfg: Config, train_cache: str | Path, val_cache: str | Path) -> Path:
    cfg = resolve_paths(cfg)
    seed_everything(cfg["seed"])
    t = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg["paths"]["out"]) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)

    from wfsea.data.memmap import open_cache
    tr_cache, va_cache = open_cache(train_cache), open_cache(val_cache)
    spec = InputSpec.from_config(cfg, n_dyn=tr_cache.C, n_static=tr_cache.static.shape[1])
    K = int(t.get("rollout_steps", 1))
    tb_policy = cfg["eval"].get("tb_policy", "persist")
    ds_tr = WeatherDataset(tr_cache, spec, n_steps=K, tb_policy=tb_policy)
    ds_va = WeatherDataset(va_cache, spec, n_steps=K, tb_policy=tb_policy)   # validate on the same horizon
    dl_tr, dl_va = make_loader(ds_tr, cfg, True), make_loader(ds_va, cfg, False)

    graph = build_graph_from_config(cfg, tr_cache.lats, tr_cache.lons) if cfg["model"]["name"] == "graphcast" else None
    if graph is not None:
        graph.save(out)
    # ── Tendency-normalised loss (improvement A1) ──
    channel_w, delta_scale = None, None
    if t.get("loss_norm", "state") == "delta":
        diff_std = load_diff_std(cfg, tr_cache)
        delta_scale = diff_std
        cw = 1.0 / diff_std ** 2
        names = [f"{v}{l or ''}" for v, l in tr_cache.channels]
        for name, wgt in (t.get("channel_weights") or {}).items():
            cw[names.index(name)] *= float(wgt)
        channel_w = torch.from_numpy(cw / cw.mean()).to(device)
    model = build_model(cfg, spec.n_in, spec.n_dyn, graph, (len(tr_cache.lats), len(tr_cache.lons)),
                        delta_scale=delta_scale).to(device)
    if t.get("init_from"):     # e.g. rollout fine-tuning starts from the single-step best.pt
        from wfsea.config import find_file
        src = Path(t["init_from"]) if Path(t["init_from"]).exists() else find_file(t["init_from"])
        sd = torch.load(src, map_location=device, weights_only=False)["model_state"]
        model.load_state_dict(sd)
        print(f"[train] initialised weights from {src}")
    from wfsea.data.boundary import boundary_mask
    bw = cfg.get("boundary", {}).get("width_deg", 0) or 0
    bmask = torch.from_numpy(boundary_mask(tr_cache.lats, tr_cache.lons, bw)).to(device) if bw else None
    grad_ckpt = bool(t.get("grad_checkpoint", False))
    n_params = count_params(model)
    net = torch.nn.DataParallel(model) if t["multi_gpu"] and torch.cuda.device_count() > 1 else model
    print(f"[train] {cfg['model']['name']} | params={n_params:,} | n_in={spec.n_in} | train={len(ds_tr)} "
          f"val={len(ds_va)} | K={K} | device={device} x{max(torch.cuda.device_count(), 1)}")

    w_grid = lat_weights_grid(tr_cache.lats, len(tr_cache.lons)).to(device)
    if bmask is not None:
        w_grid = w_grid * (~bmask).float()                       # interior only
        w_grid = w_grid / w_grid.mean()
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    spectral_w = float(t.get("spectral_weight", 0.0))
    HW = (len(tr_cache.lats), len(tr_cache.lons))
    max_steps = t.get("max_steps_per_epoch")

    # Persistence reference on val (same loss) — a model that can't beat this is broken.
    lkw = dict(channel_w=channel_w, bmask=bmask)
    with torch.no_grad():
        pnet = Persistence(spec.n_dyn)
        pl = [rollout_loss(pnet, b, w_grid, device, False, **lkw).item() for i, b in enumerate(dl_va)
              if max_steps is None or i < max_steps]
    persist_val = float(np.mean(pl))
    hist = {"train_loss": [], "val_loss": [], "lr": [], "epoch_sec": [], "persistence_val": persist_val,
            "n_params": n_params}
    best, bad, start_epoch = math.inf, 0, 1
    ckpt = out / "best.pt"
    # ── Resume (Kaggle sessions stop at 12 h; 10 training years may need 2 sessions) ──
    resume = t.get("resume")
    if resume:
        from wfsea.config import find_file
        rp = Path(resume) if Path(resume).exists() else find_file(resume)
        st = torch.load(rp, map_location=device, weights_only=False)
        model.load_state_dict(st["model_state"]); opt.load_state_dict(st["optimizer_state"])
        sched.load_state_dict(st["scheduler_state"]); scaler.load_state_dict(st["scaler_state"])
        hist, best, bad, start_epoch = st["history"], st["best"], st["bad"], st["epoch"] + 1
        if not ckpt.exists() and (rp.parent / "best.pt").exists():
            import shutil
            shutil.copy(rp.parent / "best.pt", ckpt)
        print(f"[train] resumed from {rp} at epoch {start_epoch} (best val {best:.4f})")
    deadline = time.time() + 3600 * float(t["max_hours"]) if t.get("max_hours") else None
    for epoch in range(start_epoch, t["epochs"] + 1):
        t0 = time.time()
        net.train(); losses = []
        for i, batch in enumerate(dl_tr):
            if max_steps is not None and i >= max_steps:
                break
            opt.zero_grad(set_to_none=True)
            loss = rollout_loss(net, batch, w_grid, device, amp, spectral_w, HW, grad_ckpt=grad_ckpt, **lkw)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(opt); scaler.update()
            losses.append(loss.item())
        net.eval(); vl = []
        with torch.no_grad():
            for i, batch in enumerate(dl_va):
                if max_steps is not None and i >= max_steps:
                    break
                vl.append(rollout_loss(net, batch, w_grid, device, amp, **lkw).item())
        tr_l, va_l = float(np.mean(losses)), float(np.mean(vl))
        lr_now = sched.get_last_lr()[0]; sched.step()
        sec = time.time() - t0
        for k, v in (("train_loss", tr_l), ("val_loss", va_l), ("lr", lr_now), ("epoch_sec", sec)):
            hist[k].append(v)
        print(f"Epoch {epoch:3d} | train {tr_l:.4f} | val {va_l:.4f} (persist {persist_val:.4f} "
              f"{'✓' if va_l < persist_val else '✗'}) | lr {lr_now:.2e} | {sec:.0f}s")
        if va_l < best:
            best, bad = va_l, 0
            save_checkpoint(ckpt, model, cfg, spec, tr_cache.channels,
                            {"epoch": epoch, "val_loss": best, "persistence_val": persist_val,
                             "grid_lats": tr_cache.lats, "grid_lons": tr_cache.lons,
                             "delta_scale": None if delta_scale is None else delta_scale.tolist()})
        else:
            bad += 1
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                    "scheduler_state": sched.state_dict(), "scaler_state": scaler.state_dict(),
                    "history": hist, "best": best, "bad": bad}, out / "last.pt")
        if bad >= t["patience"]:
            print(f"[train] early stop at epoch {epoch}")
            break
        if deadline and time.time() + 1.2 * sec > deadline:
            print(f"[train] stopping before the session limit after epoch {epoch}; "
                  f"continue in a new session with train.resume={out / 'last.pt'}")
            break
    if device.type == "cuda":
        hist["max_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    hist["best_val"] = best
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    print(f"[train] best val {best:.4f} vs persistence {persist_val:.4f} → {out}")
    return ckpt
