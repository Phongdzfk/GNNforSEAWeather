"""`wfsea <command>` — one entry point used identically on Kaggle and locally.

  wfsea raw-index                          what raw ERA5 files exist (by content): vars, levels, years (Phong)
  wfsea download-submit --years 2015-2018  submit 6-hourly pressure-level requests, save request IDs
  wfsea download-fetch --wait-hours 11     download the ready ones (re-run until done)
  wfsea download-gcs --years 2015-2016     no-queue alternative: Google ARCO-ERA5 (public Zarr)
  wfsea stats                              train-only normalisation stats from raw data (Phong)
  wfsea materialize --split train          raw NetCDF (or Zarr) → fp16 memmap cache  (Phong)
  wfsea manifest                           data_manifest.json         (Phong)
  wfsea climatology                        train month×hour climatology (Phong)
  wfsea graph                              graph_stats.json           (Hieu)
  wfsea train [--config configs/kaggle.yaml] [key=value ...]          (Hieu)
  wfsea evaluate --ckpt best.pt            metrics_raw.csv + summaries (Tung)
  wfsea export-forecast --ckpt best.pt     forecast_<init>.nc (physical units, for API/dashboard)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wfsea.config import find_file, load_config, resolve_paths


def _stats_dir(cfg) -> Path:
    if cfg["paths"].get("stats_dir"):
        return Path(cfg["paths"]["stats_dir"])
    return find_file("era5_normalization_stats.json").parent


def _channels(cfg):
    from wfsea.constants import DYNAMIC_CHANNELS, channels_from_config
    # legacy Zarr stores only hold the 14 baseline channels
    return DYNAMIC_CHANNELS if cfg["data"]["zarr_layout"] == "legacy" else channels_from_config(cfg)


def _cache_groups(cfg, split: str) -> list[list[int]]:
    from wfsea.data.memmap import part_years
    years = cfg["data"][f"{split}_years"]
    if split != "train":
        return [list(years)]
    return part_years(years, len(_channels(cfg)), cfg["data"].get("cache_part_years", "auto"))


def _cache_dir(cfg, split: str, years=None) -> Path:
    years = years or cfg["data"][f"{split}_years"]
    span = f"{min(years)}" if len(years) == 1 else f"{min(years)}-{max(years)}"
    return Path(cfg["paths"]["scratch"]) / f"{split}_{span}_{cfg['data']['zarr_layout']}_c{len(_channels(cfg))}"


def _raw_opts(cfg) -> dict | None:
    if cfg["data"]["zarr_layout"] != "raw":
        return None
    from wfsea.constants import schema_from_config
    from wfsea.data.stats import load_stats
    stats = load_stats(_stats_dir(cfg))
    stats.check_train_only(cfg["data"]["train_years"])
    from wfsea.constants import stats_key
    missing = [stats_key(v, l) for v, l in _channels(cfg) if stats_key(v, l) not in stats.mean]
    if missing:
        raise SystemExit(f"Stats file has no entry for {missing}. Re-run `wfsea stats` with the same "
                         f"data.pressure_vars / data.pressure_levels as this config.")
    return {"stats": stats, "era5_root": cfg["paths"].get("era5_raw_root"),
            "nasa_root": cfg["paths"].get("nasa_raw_root"), "regrid": cfg["data"].get("regrid", "box_mean"),
            "schema": schema_from_config(cfg)}


def _ensure_cache(cfg, split: str, part: int | None = None) -> list[Path]:
    """Cache dir(s) of a split. Reuses caches attached as Kaggle input (outputs of 00b/00c), builds the rest.
    The train split may be several parts (each ≤ ~17 GB so it fits one Kaggle notebook output)."""
    from wfsea.config import ON_KAGGLE, find_dir
    from wfsea.data.memmap import materialize_split
    groups = _cache_groups(cfg, split)
    if part is not None:
        groups = [groups[part]]
    dirs, raw = [], None
    for years in groups:
        target = _cache_dir(cfg, split, years)
        found = None
        if ON_KAGGLE:
            try:
                d = find_dir(target.name, ["/kaggle/input"])
                found = d if (d / "meta.json").exists() else None
            except FileNotFoundError:
                pass
        if found:
            print(f"[cache] using attached {found}")
            dirs.append(found)
            continue
        raw = raw if raw is not None else _raw_opts(cfg)
        dirs.append(materialize_split(years, target, cfg["paths"].get("zarr_root"), cfg["data"]["zarr_layout"],
                                      cfg["data"]["memmap_block"], raw=raw, channels=_channels(cfg)))
    return dirs


def _parse_years(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wfsea")
    ap.add_argument("command")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--start", default="last", help="export-forecast: time index, ISO time, or 'last'")
    ap.add_argument("--forecast-dir", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--part", type=int, default=None, help="materialize: build only this part of the train cache")
    ap.add_argument("--years", default=None, help="download-*/stats: e.g. 2015-2018 or 2015,2016")
    ap.add_argument("--max-new", type=int, default=48, help="download-submit: max new requests this run")
    ap.add_argument("--wait-hours", type=float, default=0.0, help="download-fetch: keep polling this long")
    ap.add_argument("--workers", type=int, default=0, help="download-gcs: parallel chunk reads (0 = auto); stats: parallel years (0 = 4)")
    ap.add_argument("--max-hours", type=float, default=None, help="download-gcs/stats: stop starting new months/years after this")
    ap.add_argument("--bench", action="store_true", help="download-gcs: only measure speed, write nothing")
    ap.add_argument("--single", action="store_true", help="download-gcs: single levels (u10,v10,d2m,t2m,msl,tp,z,lsm)")
    ap.add_argument("--no-checksum", action="store_true")
    ap.add_argument("--legacy-levels", default="850,500",
                    help="pressure_level order inside the old Zarr files (only for old notebook checkpoints)")
    args, rest = ap.parse_known_args(argv)
    overrides = [r for r in rest if "=" in r]
    cfg = resolve_paths(load_config(args.config, overrides))
    out = Path(cfg["paths"]["out"])
    cmd = args.command

    if cmd == "stats":
        from wfsea.config import ON_KAGGLE
        from wfsea.constants import schema_from_config
        from wfsea.data.stats import compute_year_partials, merge_partials, save_stats
        train = [int(y) for y in cfg["data"]["train_years"]]
        years = _parse_years(args.years) if args.years else train
        bad = [y for y in years if y not in train]
        if bad:
            raise SystemExit(f"stats are train-only; {bad} not in train_years {train}")
        dest = Path(cfg["paths"].get("stats_dir") or out / "stats")
        mh = args.max_hours if args.max_hours is not None else (11.0 if ON_KAGGLE else None)
        parts = compute_year_partials(
            years, dest, cfg["paths"].get("era5_raw_root"), cfg["paths"].get("nasa_raw_root"),
            cfg["data"].get("tb_file_stride", 5), cfg["data"].get("tp_transform"), schema=schema_from_config(cfg),
            time_stride=int(cfg["data"].get("stats_time_stride", 5)), workers=args.workers or 4,
            search_dirs=["/kaggle/input"] if ON_KAGGLE else [], max_hours=mh)
        missing = [y for y in train if y not in parts]
        if missing:
            print(f"[stats] partial sums saved in {dest}/stats_partial. Still missing train years {missing}: run "
                  f"`wfsea stats` again (attach this output as input) or let a teammate run --years for them.")
        else:
            st = merge_partials(parts, train)
            save_stats(st, dest)
            print(f"saved train-only stats ({train[0]}–{train[-1]}) → {dest}")
            print(f"  Tb mean {st.tb_mean:.2f} K, std {st.tb_std:.2f} K | t2m mean {st.mean.get('t2m', float('nan')):.2f}")

    elif cmd in ("download-submit", "download-fetch", "download-gcs"):
        from wfsea.config import ON_KAGGLE
        from wfsea.constants import schema_from_config
        from wfsea.data import download as dl
        sc = schema_from_config(cfg)
        years = _parse_years(args.years) if args.years else (
            cfg["data"]["train_years"] + cfg["data"]["val_years"] + cfg["data"]["test_years"])
        state = out / "cds_requests.json"
        if cmd != "download-gcs":
            prev = [str(p) for p in Path("/kaggle/input").rglob("cds_requests.json")] if ON_KAGGLE else []
            if prev or state.exists():
                dl._save_state(state, dl.merge_states(prev + ([str(state)] if state.exists() else [])))
        dest = out / "era5_pressure"
        if cmd == "download-submit":
            reqs = dl.pressure_requests(years, sc["pressure_vars"], sc["levels"])
            st = dl.cds_submit(reqs, state, max_new=args.max_new)
            print(f"[cds] {sum(1 for k in reqs if k in st)} / {len(reqs)} months submitted; state → {state}")
        elif cmd == "download-fetch":
            reqs = dl.pressure_requests(years, sc["pressure_vars"], sc["levels"]) if args.years else None
            dl.cds_fetch(state, dest, wait_hours=args.wait_hours, reqs=reqs)
        else:
            url = cfg.get("download", {}).get("gcs_url") or dl.GCS_ARCO_URL
            mh = args.max_hours if args.max_hours is not None else (11.3 if ON_KAGGLE else None)
            if args.single:
                dl.gcs_single_download(years, out / "era5_single", url=url, workers=args.workers or 32,
                                       max_hours=mh, tp_accum_hours=sc.get("tp_accum_hours", 6))
            else:
                dl.gcs_pressure_download(years, sc["pressure_vars"], sc["levels"], dest, url=url,
                                         workers=args.workers, max_hours=mh, bench=args.bench)

    elif cmd == "raw-index":
        from wfsea.data.raw_index import scan_era5, summarize
        import pandas as pd
        pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
        df = summarize(scan_era5(cfg["paths"].get("era5_raw_root"), tuple(cfg["paths"].get("era5_exclude") or ())))
        print(df.to_string(index=False))
        df.to_csv(out / "raw_index_summary.csv", index=False)

    elif cmd == "materialize":
        splits = ["train", "val", "test"] if args.split == "all" else [args.split]
        for s in splits:
            if s == "train":
                print("[cache] train parts:", _cache_groups(cfg, "train"))
            _ensure_cache(cfg, s, args.part if s == "train" else None)

    elif cmd == "manifest":
        from wfsea.data.manifest import build_manifest
        caches = {s: _ensure_cache(cfg, s) for s in ("train", "val", "test")}
        m = build_manifest(caches, out / "data_manifest.json", checksum=not args.no_checksum)
        for s, e in m["splits"].items():
            print(f"{s:5s} {e['time_start']} → {e['time_end']}  T={e['T']}  valid={e['n_valid_samples']}  gaps={len(e['gaps'])}")
        print("split overlap:", m["split_overlap"] or "none ✓")

    elif cmd == "climatology":
        from wfsea.data.memmap import open_cache
        from wfsea.evaluation.baselines import compute_climatology, compute_diff_std, save_climatology
        tr = open_cache(_ensure_cache(cfg, "train"))
        clim = compute_climatology(tr)
        save_climatology(clim, out / "climatology_train.npz", cfg["data"]["train_years"])
        print("saved", out / "climatology_train.npz", clim.shape)
        ds = compute_diff_std(tr)
        (out / "diff_stats.json").write_text(json.dumps(
            {"channels": [list(c) for c in tr.channels], "diff_std": ds.tolist(),
             "years": cfg["data"]["train_years"]}, indent=2))
        print("saved", out / "diff_stats.json", dict(zip([f"{v}{l or ''}" for v, l in tr.channels], ds.round(3))))

    elif cmd == "graph":
        from wfsea.data.memmap import open_cache
        from wfsea.graph.mesh import build_graph_from_config
        c = open_cache(_ensure_cache(cfg, "test"))
        g = build_graph_from_config(cfg, c.lats, c.lons)
        g.save(out / "graph")
        print(json.dumps(g.stats(), indent=2))

    elif cmd == "train":
        from wfsea.training.trainer import train
        train(cfg, _ensure_cache(cfg, "train"), _ensure_cache(cfg, "val"))

    elif cmd in ("evaluate", "export-forecast"):
        import torch
        from wfsea.data.memmap import open_cache
        from wfsea.data.stats import load_stats
        from wfsea.models import ChannelAdapter, load_checkpoint
        split = cfg["eval"]["split"] if cmd == "evaluate" else args.split if args.split != "train" else "test"
        cache = open_cache(_ensure_cache(cfg, split))
        stats = load_stats(_stats_dir(cfg)); stats.check_train_only(cfg["data"]["train_years"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from wfsea.config import find_file as _ff
        if not Path(args.ckpt).exists():
            args.ckpt = str(_ff(args.ckpt))          # e.g. --ckpt v2/best.pt searched under /kaggle/input
        levels = tuple(int(x) for x in args.legacy_levels.split(","))
        model, mcfg, spec, channels, ck = load_checkpoint(args.ckpt, cfg, cache.lats, cache.lons,
                                                          legacy_levels=levels)
        if list(map(tuple, channels)) != list(map(tuple, cache.channels)):
            model = ChannelAdapter(model, channels, cache.channels)
        model = model.to(device)
        from wfsea.data.boundary import boundary_mask
        bmask = boundary_mask(cache.lats, cache.lons, (mcfg.get("boundary") or {}).get("width_deg", 0) or 0)
        if cmd == "evaluate":
            from wfsea.evaluation.baselines import compute_climatology, load_climatology, save_climatology
            cp = out / "climatology_train.npz"
            if not cp.exists():
                try:
                    cp = find_file("climatology_train.npz")   # attached output of 00c
                except FileNotFoundError:
                    cp = out / "climatology_train.npz"
            if cp.exists():
                clim = load_climatology(cp)
            else:
                clim = compute_climatology(open_cache(_ensure_cache(cfg, "train")))
                save_climatology(clim, cp, cfg["data"]["train_years"])
            from wfsea.evaluation.runner import evaluate
            e = cfg["eval"]
            evaluate({mcfg.get("model", {}).get("name", "gnn"): model}, spec, cache, stats, clim,
                     out / (mcfg.get("run_name") or cfg["run_name"]) / f"eval_{e['tb_policy']}", e["n_starts"], e["n_steps"], e["tb_policy"],
                     tuple(e["regions"]), device,
                     {"checkpoint": args.ckpt, "n_params": ck.get("n_params"), "git_commit": ck.get("git_commit"),
                      "run_name": mcfg.get("run_name")}, bmask=bmask)
        else:
            import numpy as np
            from wfsea.inference.export import export_forecast
            valid = cache.valid_starts(n_future=0)
            n = cfg["eval"]["n_steps"]
            if args.start == "last":
                t0 = int(valid[-1])
            elif args.start.isdigit():
                t0 = int(args.start)
            else:
                t0 = int(np.argmin(np.abs(cache.times - np.datetime64(args.start))))
            n = min(n, cache.T - 1 - t0) if cfg["eval"]["tb_policy"] == "oracle" else n
            p = export_forecast(model, cache, spec, stats, t0, n, args.forecast_dir or out / "forecasts",
                                cfg["eval"]["tb_policy"], device, bmask)
            print("wrote", p)

    else:
        ap.error(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
