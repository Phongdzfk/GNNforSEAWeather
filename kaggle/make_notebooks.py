"""Generate the thin Kaggle notebooks in kaggle/notebooks/.

The notebooks contain NO model/data logic — they only clone this repo and call
`wfsea <command>`. Change code in src/, not in Kaggle. Regenerate with:
    python kaggle/make_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

BOOTSTRAP = r'''# ── Bootstrap: clone the repo, install wfsea ──
# REPO: repo làm việc của nhóm (fork của Tùng). Public → không cần secret.
# Nếu fork để private: Tùng (chủ repo) tạo fine-grained token, resource owner = tuk3kCS,
# chỉ repo WeatherForecast, Contents: Read-only → cả nhóm lưu vào Kaggle Secret GITHUB_TOKEN.
import os, subprocess, sys
os.environ["GIT_TERMINAL_PROMPT"] = "0"   # never wait for a username prompt

REPO   = "tuk3kCS/WeatherForecast"  # ← repo làm việc; repo của thầy (ducnx/WeatherForecast) chỉ nhận bản theo mốc
BRANCH = "main"                    # ← nhánh đang thử nghiệm, ví dụ "hieu/geo-edges"
DEST = "/kaggle/tmp/WeatherForecast" if os.path.isdir("/kaggle/tmp") else "/tmp/WeatherForecast"
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    token = None                   # repo public: clone không cần token
url = f"https://x-access-token:{token}@github.com/{REPO}.git" if token else f"https://github.com/{REPO}.git"
if not os.path.isdir(DEST):
    r = subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, url, DEST], capture_output=True, text=True)
    if r.returncode:
        err = r.stderr.replace(token or "\x00", "***")
        if "could not read Username" in err or "Authentication failed" in err or "not found" in err.lower():
            err += ("\n→ Repo đang private (hoặc sai tên REPO) mà chưa có token hợp lệ. Cách sửa: chủ repo chuyển repo sang "
                    "Public, HOẶC tạo fine-grained token (Contents: Read-only) cho repo này rồi thêm Kaggle Secret "
                    "GITHUB_TOKEN và BẬT nó cho notebook (Add-ons → Secrets → tick vào GITHUB_TOKEN).")
        raise RuntimeError(err)
    subprocess.run(["git", "-C", DEST, "remote", "set-url", "origin", f"https://github.com/{REPO}.git"])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", DEST], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr>=2.14"], check=False)
os.chdir(DEST)
COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
print("wfsea @", REPO, BRANCH, COMMIT)'''

NOTEBOOKS = {
    "00_download_pressure": {
        "title": "00 · Tải ERA5 pressure levels 6-hourly (submit → fetch, không sợ timeout)",
        "intro": "Chạy bằng **Save & Run All** (chạy nền tối đa 12 h), Accelerator: None, Internet: On.\n"
                 "Secrets: `url`, `key` (CDS) như notebook cũ. Mỗi notebook một nhóm năm (≈ 6 GB/năm với 7 tầng × 5 biến "
                 "→ tối đa 2–3 năm/notebook để dưới 20 GB output).\n"
                 "Request chỉ gửi **một lần** và lưu ID vào `cds_requests.json`. Nếu hết giờ mà còn tháng chưa xong: tạo "
                 "version mới, gắn output version trước làm input → nó đọc lại ID và tải tiếp, không gửi lại.",
        "cells": [
            "YEARS  = '2015-2016'   # ← mỗi notebook một nhóm năm\n"
            "CONFIG = 'configs/levels7.yaml'",
            "!pip install -q 'cdsapi>=0.7.7'",
            "!wfsea download-submit --config {CONFIG} --years {YEARS}",
            "!wfsea download-fetch  --config {CONFIG} --wait-hours 11",
            "!wfsea raw-index --config {CONFIG} paths.era5_raw_root=/kaggle/working/era5_pressure",
        ],
    },
    "00a_stats": {
        "title": "00a · Kiểm kê dữ liệu thô + stats train-only (Phong)",
        "intro": "**Input:** gắn TẤT CẢ notebook/dataset dữ liệu thô ERA5 (single levels + pressure levels) và NASA, mọi năm. "
                 "Code nhận file theo **nội dung** (biến, tầng, thời gian), không theo tên file/thư mục, nên layout của các "
                 "notebook tải cũ (`era5_data/…instant_<tháng>.nc`, `era5_data_pressure/…`, dữ liệu hourly) đều đọc được.\n"
                 "Ô đầu tiên in bảng kiểm kê — xem kỹ có thiếu năm/tầng nào không trước khi tính stats. Accelerator: None.",
        "cells": [
            "CONFIG = 'configs/levels7.yaml'   # dữ liệu mới 7 tầng; dùng 'configs/kaggle.yaml' cho 14 kênh cũ",
            "!wfsea raw-index --config {CONFIG}",
            "!wfsea stats --config {CONFIG}",
            "import json; s = json.load(open('/kaggle/working/stats/era5_normalization_stats.json'))\n"
            "print(s['_meta']); print({k: v for k, v in s.items() if k in ('t2m', 'msl', 'tp', 't_850hpa', 'q_850hpa')})",
        ],
    },
    "00b_cache_train": {
        "title": "00b · Raw → fp16 memmap cache, TRAIN (một phần mỗi notebook)",
        "intro": "Chuẩn hoá bằng stats của 00a, nội suy Tb (box mean), ghi memmap fp16. Với 41 kênh, train được chia "
                 "thành 3 phần (2015–2018, 2019–2022, 2023–2024), mỗi phần ≤ ~17 GB → mỗi notebook chạy **một** `PART`.\n"
                 "RAM: đọc theo khối thời gian nhỏ (tự chỉnh theo số kênh), đỉnh ~2 GB. "
                 "**Input:** dữ liệu thô các năm của phần đó + output `00a_stats`. Accelerator: None.",
        "cells": [
            "CONFIG = 'configs/levels7.yaml'\n"
            "PART   = 0                         # 0, 1, 2 — mỗi notebook một phần",
            "!wfsea materialize --config {CONFIG} --split train --part {PART} paths.scratch=/kaggle/working",
            "import json, glob; m = json.load(open(glob.glob('/kaggle/working/train_*/meta.json')[0]))\n"
            "print(m['years'], m['time_start'], '→', m['time_end'], 'T =', m['T'], 'C =', m['C'], '| gaps:', m['gaps'][:5])\n"
            "print('Tb coverage', round(m['tb_coverage'], 4), '| fp16 rel. RMSE worst:',\n"
            "      sorted(m['fp16_relative_rmse'].items(), key=lambda kv: -kv[1])[:3])",
        ],
    },
    "00c_cache_valtest": {
        "title": "00c · Val/test cache, manifest, climatology, graph (Phong)",
        "intro": "**Input:** dữ liệu thô 2025–2026 + output `00a_stats` + output của TẤT CẢ các phần `00b` "
                 "(cần cho manifest, climatology, diff_stats). Accelerator: None.",
        "cells": [
            "CONFIG = 'configs/levels7.yaml'",
            "!wfsea materialize --config {CONFIG} --split val paths.scratch=/kaggle/working",
            "!wfsea materialize --config {CONFIG} --split test paths.scratch=/kaggle/working",
            "!wfsea manifest --config {CONFIG} paths.scratch=/kaggle/working",
            "!wfsea climatology --config {CONFIG} paths.scratch=/kaggle/working",
            "!wfsea graph --config {CONFIG} paths.scratch=/kaggle/working",
            "import json; m = json.load(open('/kaggle/working/data_manifest.json'))\n"
            "for s, e in m['splits'].items():\n"
            "    print(s, e['time_start'], '→', e['time_end'], 'valid', e['n_valid_samples'], 'gaps', e['gaps'][:3])\n"
            "print('overlap between splits:', m['split_overlap'] or 'none')",
        ],
    },
    "01_train_graphcast": {
        "title": "01 · Train (GNN) — đổi CONFIG để chạy run khác",
        "intro": "Accelerator: **GPU T4 ×2**. Input: `wfsea-stats`, `wfsea-cache-train`, `wfsea-cache-valtest` "
                 "(output của 00a/00b/00c) + output của run trước nếu fine-tune/resume.\n"
                 "Danh sách run và người chạy: `project/RUN_PLAN.md`.",
        "cells": [
            "CONFIG = 'configs/v2.yaml'      # 14 kênh: kaggle.yaml (B0) · v2 · v2_wide · v2_wider · v3_* · v4_boundary | 41 kênh: v2_l7.yaml\n"
            "EXTRA  = ''                     # ghi đè, ví dụ: 'run_name=abl_no_time features.time=false train.epochs=15'\n"
            "RESUME = ''                     # phiên trước hết 12 h? điền '<run_name>/last.pt' và gắn output phiên trước",
            "!nvidia-smi --query-gpu=name,memory.total --format=csv",
            "# Smoke 5 bước (~2–5 phút) để chắc môi trường và dữ liệu ổn trước khi chạy dài\n"
            "!wfsea train --config {CONFIG} {EXTRA} run_name=smoke train.epochs=1 train.max_steps_per_epoch=5",
            "from wfsea.config import load_config\n"
            "RUN = load_config(CONFIG, EXTRA.split())['run_name']\n"
            "res = f'train.resume={RESUME}' if RESUME else ''\n"
            "print('RUN =', RUN)\n"
            "!wfsea train --config {CONFIG} {EXTRA} {res}",
            "import json; h = json.load(open(f'/kaggle/working/{RUN}/history.json'))\n"
            "print('best val', h['best_val'], '| persistence', h['persistence_val'], '| params', h['n_params'],\n"
            "      '| VRAM GB', h.get('max_vram_gb'), '| epochs', len(h['val_loss']), '| s/epoch', h['epoch_sec'][-1])",
        ],
    },
    "02_train_unet": {
        "title": "02 · Train U-Net baseline",
        "intro": "Cùng dữ liệu, cùng đầu vào và loss với v2 — chỉ thay lõi GNN bằng U-Net (so sánh công bằng). GPU T4 ×2.",
        "cells": [
            "!wfsea train --config configs/v2.yaml model.name=unet run_name=unet_v2 train.batch_size=8",
        ],
    },
    "03_evaluate": {
        "title": "03 · Evaluate 30 starts × 72 h",
        "intro": "Input: 3 dataset dữ liệu + output của run cần đánh giá. GPU T4 ×1 là đủ (~1 h/checkpoint).\n"
                 "**Chọn mô hình/ablation trên `val` (2025); chỉ chạy `test` (2026) cho bản cuối** — tránh chọn theo tập test.",
        "cells": [
            "CKPT   = 'v2/best.pt'            # <run_name>/best.pt, tìm trong /kaggle/input\n"
            "SPLIT  = 'val'                   # 'val' khi so sánh/chọn; 'test' cho số liệu luận văn\n"
            "CONFIG = 'configs/kaggle.yaml'   # checkpoint cũ của đồ án môn học: 'configs/legacy_2020.yaml'\n"
            "EXTRA  = ''                      # checkpoint cũ: '--legacy-levels 500,850' nếu Zarr cũ xếp tầng 500 trước",
            "!wfsea evaluate --config {CONFIG} --ckpt {CKPT} {EXTRA} eval.split={SPLIT} eval.tb_policy=persist paths.scratch=/kaggle/tmp/c",
            "import pandas as pd, glob\n"
            "for f in sorted(glob.glob('/kaggle/working/*/eval_*/summary_rmse.csv')):\n"
            "    print(f); display(pd.read_csv(f).query(\"channel in ['t2m','u10','v10','msl','z500','q850','tp']\")"
            "[['model','channel','24','48','72']])",
            "# Chỉ cho bản cuối: đối chiếu với cách đánh giá cũ (dùng vệ tinh tương lai) và xuất dự báo\n"
            "# !wfsea evaluate --config {CONFIG} --ckpt {CKPT} {EXTRA} eval.split=test eval.tb_policy=oracle paths.scratch=/kaggle/tmp/c\n"
            "# !wfsea export-forecast --config {CONFIG} --ckpt {CKPT} {EXTRA} --split test paths.scratch=/kaggle/tmp/c",
        ],
    },
}


def to_ipynb(nb: dict) -> dict:
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
    code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
    cells = [md(f"# {nb['title']}\n\n{nb['intro']}"), code(BOOTSTRAP)] + [code(c) for c in nb["cells"]]
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    out = HERE / "notebooks"
    out.mkdir(exist_ok=True)
    for name, nb in NOTEBOOKS.items():
        (out / f"{name}.ipynb").write_text(json.dumps(to_ipynb(nb), indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote", out / f"{name}.ipynb")


if __name__ == "__main__":
    main()
