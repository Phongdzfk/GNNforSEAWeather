"""Thesis tables from one or more evaluation folders (chapter 6).

Usage:
  python scripts/analyze_eval.py <results_dir> <out_dir> --runs v4_star v3_star v2 graphcast_baseline unet_v2 \
         [--policy persist] [--ref v4_star]

<results_dir>/<run>/eval_<policy>/metrics_raw.csv are read; every run was evaluated on the same
deterministic start list, so models are compared PER START (paired). Persistence, climatology and
external benchmarks (aifs, ifs_control) are taken from whichever run contains them.

Writes to <out_dir>:
  main_rmse.md / .csv     RMSE at +24/+48/+72 h, all region, target variables
  main_acc.csv            ACC at +24/+48/+72 h
  regions.md / .csv       RMSE by region (sea / land / plain / mountain) at +24 and +72 h
  diurnal.md / .csv       t2m RMSE by VALID local hour (UTC+7) — diurnal-cycle errors
  paired.md               Δ% vs the reference model, bootstrap 95% CI over starts, #starts better
  external.md             our models vs AIFS / IFS control on their common starts only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = ["t2m", "u10", "v10", "msl", "z500", "t850", "q850", "tp"]
LEADS = [24, 48, 72]
BASE = ("persistence", "climatology")
EXTERNAL = ("aifs", "ifs_control")


def load(results: Path, runs, policy):
    frames = []
    seen_base = set()
    for r in runs:
        f = results / r / f"eval_{policy}" / "metrics_raw.csv"
        if not f.exists():
            print(f"[skip] {f} not found")
            continue
        d = pd.read_csv(f)
        ours = [m for m in d.model.unique() if m not in BASE + EXTERNAL]
        keep = d[d.model.isin(ours)].assign(model=r)
        frames.append(keep)
        for b in BASE + EXTERNAL:
            if b not in seen_base and b in set(d.model):
                frames.append(d[d.model == b]); seen_base.add(b)
    if not frames:
        raise SystemExit("no metrics found")
    return pd.concat(frames, ignore_index=True)


def md(df: pd.DataFrame, fmt="{:.3f}") -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, row in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                                 else ("—" if isinstance(v, (float, np.floating)) else str(v)) for v in row]) + " |")
    return "\n".join(out)


def paired(d, model, ref, chs, leads, rng):
    a = d[(d.model == model) & d.channel.isin(chs) & d.lead_h.isin(leads) & (d.region == "all")]
    b = d[(d.model == ref) & d.channel.isin(chs) & d.lead_h.isin(leads) & (d.region == "all")]
    m = a.merge(b, on=["start", "lead_h", "channel"], suffixes=("_a", "_b")).dropna(subset=["rmse_a", "rmse_b"])
    if m.empty:
        return np.nan, np.nan, np.nan, "0/0"
    ps = (m.rmse_a / m.rmse_b - 1).groupby(m.start).mean()
    boots = [ps.sample(len(ps), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
    return 100 * ps.mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5), f"{(ps < 0).sum()}/{len(ps)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("out")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--policy", default="persist")
    ap.add_argument("--ref", default=None, help="reference model for paired Δ% (default: first run)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = load(Path(a.results), a.runs, a.policy)
    d["lead_h"] = d.lead_h.astype(int)
    models = [m for m in a.runs if m in set(d.model)] + [m for m in BASE + EXTERNAL if m in set(d.model)]
    allr = d[d.region == "all"]
    rng = np.random.default_rng(0)

    # main tables
    t = allr[allr.channel.isin(TARGET) & allr.lead_h.isin(LEADS)].groupby(["model", "channel", "lead_h"]).rmse.mean()
    main = t.unstack(["channel", "lead_h"]).reindex(models)
    main = main[[c for c in [(ch, L) for ch in TARGET for L in LEADS] if c in main.columns]]
    main.columns = [f"{c}@{L}h" for c, L in main.columns]
    main.index.name = "model"
    main.to_csv(out / "main_rmse.csv")
    (out / "main_rmse.md").write_text(md(main, "{:.4g}"))
    g = allr[allr.lead_h.isin(LEADS)].groupby(["model", "channel", "lead_h"])[["acc_num", "acc_ff", "acc_oo"]].sum()
    acc = (g.acc_num / np.sqrt(g.acc_ff * g.acc_oo + 1e-30)).unstack(["channel", "lead_h"]).reindex(models)
    acc.to_csv(out / "main_acc.csv")

    # regions
    reg = d[d.channel.isin(["t2m", "u10", "v10", "tp"]) & d.lead_h.isin([24, 72]) & (d.region != "all")]
    rt = reg.groupby(["model", "region", "channel", "lead_h"]).rmse.mean().unstack(["channel", "lead_h"])
    rt = rt.reindex([m for m in models if m in rt.index.get_level_values(0)], level=0)
    rt.columns = [f"{c}@{L}h" for c, L in rt.columns]
    rt.to_csv(out / "regions.csv")
    rt2 = rt.copy(); rt2.index = [f"{m} · {r}" for m, r in rt.index]; rt2.index.name = "model · region"
    (out / "regions.md").write_text(md(rt2, "{:.4g}"))

    # diurnal: t2m by valid local hour (UTC+7)
    tt = allr[allr.channel == "t2m"].copy()
    tt["valid_local"] = ((pd.to_datetime(tt.start) + pd.to_timedelta(tt.lead_h, unit="h")).dt.hour + 7) % 24
    di = tt.groupby(["model", "valid_local"]).rmse.mean().unstack("valid_local").reindex(models)
    di.columns = [f"{h:02d}h" for h in di.columns]; di.index.name = "model"
    di.to_csv(out / "diurnal.csv")
    (out / "diurnal.md").write_text("t2m RMSE (K) by valid local time (UTC+7), mean over all leads\n\n" + md(di))

    # paired vs reference
    ref = a.ref or a.runs[0]
    lines = [f"Δ% RMSE vs **{ref}** (negative = better), mean over starts, bootstrap 95% CI, #starts better\n",
             "| model | vars | +6–24 h | +48–72 h |", "|---|---|---|---|"]
    for m in models:
        if m == ref:
            continue
        for name, chs in (("t2m,u10,v10", ["t2m", "u10", "v10"]), ("all 8", TARGET)):
            s = paired(d, m, ref, chs, [6, 12, 18, 24], rng); l = paired(d, m, ref, chs, [48, 54, 60, 66, 72], rng)
            f = lambda x: "—" if not np.isfinite(x[0]) else f"{x[0]:+.1f}% [{x[1]:+.1f}, {x[2]:+.1f}] ({x[3]})"
            lines.append(f"| {m} | {name} | {f(s)} | {f(l)} |")
    (out / "paired.md").write_text("\n".join(lines))

    # external benchmarks on common starts only
    ext = [e for e in EXTERNAL if e in set(d.model)]
    if ext:
        lines = ["Ours vs operational ECMWF forecasts, on the starts the external source covers (paired).\n"]
        for e in ext:
            starts = set(allr[(allr.model == e)].dropna(subset=["rmse"]).start)
            sub = allr[allr.start.isin(starts)]
            chs = sorted(set(sub[(sub.model == e)].dropna(subset=["rmse"]).channel) & set(TARGET))
            tb = sub[sub.channel.isin(chs) & sub.lead_h.isin(LEADS)].groupby(["model", "channel", "lead_h"]).rmse.mean()
            tb = tb.unstack(["channel", "lead_h"]).reindex([m for m in models if m in set(sub.model)])
            tb.columns = [f"{c}@{L}h" for c, L in tb.columns]; tb.index.name = f"{e}: {len(starts)} starts"
            lines += [md(tb, "{:.4g}"), ""]
        (out / "external.md").write_text("\n".join(lines))
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
