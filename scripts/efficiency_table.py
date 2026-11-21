"""Collect <results>/<run>/bench.json into one markdown table for the thesis.

  python scripts/efficiency_table.py /kaggle/working [--runs graphcast_baseline v2 unet_v2 v3_star_r12]
"""
import argparse
import json
from pathlib import Path


def fmt(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--runs", nargs="*")
    a = ap.parse_args()
    files = sorted(Path(a.results).glob("*/bench.json"))
    rows = {f.parent.name: json.loads(f.read_text()) for f in files}
    order = a.runs or list(rows)
    lines = ["| model | params (M) | train s/epoch | epochs | GPU-h | data wait | train VRAM (GB) | "
             "72 h forecast (s) | infer VRAM (GB) | device |", "|" + "---|" * 10]
    for r in order:
        d = rows.get(r)
        if d is None:
            continue
        ws = d.get("data_wait_share")
        wait = "—" if ws is None else f"{100 * ws:.0f}%"
        lines.append(f"| {r} | {fmt((d.get('n_params') or 0) / 1e6, 2)} | {fmt(d.get('train_sec_per_epoch'), 0)} | "
                     f"{d.get('train_epochs') or '—'} | {fmt(d.get('train_gpu_hours'))} | "
                     f"{wait} | "
                     f"{fmt(d.get('train_vram_gb'))} | {fmt(d.get('infer_sec_per_forecast'), 2)} | "
                     f"{fmt(d.get('infer_vram_gb'), 2)} | {d.get('device')} |")
    out = "\n".join(lines)
    print(out)
    (Path(a.results) / "efficiency.md").write_text(out + "\n")


if __name__ == "__main__":
    main()
