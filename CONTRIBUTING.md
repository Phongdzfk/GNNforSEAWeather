# Quy trình làm việc

## Môi trường local

```bash
uv venv --python 3.12
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU là đủ để chạy test
uv pip install -e ".[dev]"
pytest                             # ~1 phút, dùng dữ liệu giả đúng định dạng Kaggle
```

Train/eval thật chạy trên Kaggle — xem [`kaggle/README.md`](kaggle/README.md).

## Repo và remote

- **Làm việc hằng ngày:** `origin` = `Phongdzfk/GNNforSEAWeather` (public). Mọi nhánh, Pull Request và notebook Kaggle dùng repo này.
- **Bản nộp theo mốc:** `upstream` = `ducnx/WeatherForecast` (repo của thầy). Không ai sửa code trực tiếp trên đó.

```bash
git remote add upstream https://github.com/ducnx/WeatherForecast.git   # một lần
git remote -v                      # origin → Phongdzfk/GNNforSEAWeather, upstream → ducnx/WeatherForecast
# đến mốc (có cache dữ liệu, có v2, có v3…): một người đẩy main lên repo thầy, giữ nguyên lịch sử
git checkout main && git pull origin main
git push upstream main
# nếu thầy sửa docs/ trên repo của thầy: kéo về trước khi đẩy tiếp
git pull upstream main && git push origin main
```

## Nhánh và Pull Request

- `main` luôn chạy được và pass test. Không push thẳng lên `main`.
- Tên nhánh: `<tên>/<việc>` — ví dụ `phong/data-manifest`, `hieu/geo-edges`, `tung/eval-runner`.
- Mỗi PR: một việc, có mô tả *đã làm gì / chạy thử thế nào / số liệu nào thay đổi*, ít nhất **1 người khác review**.
- Thay đổi chạm vào hợp đồng bàn giao (thứ tự kênh, định dạng cache, định dạng checkpoint, cột `metrics_raw.csv`) phải có review của người nhận.

## Quy tắc code

1. Thứ tự kênh, miền, biến chỉ khai báo trong `src/wfsea/constants.py`.
2. Siêu tham số nằm trong `configs/*.yaml`; không hard-code trong code hay notebook.
3. Thống kê chuẩn hoá và climatology **chỉ tính trên tập train** (mặc định 2015–2024, khai báo trong `configs/base.yaml`).
4. Loss/metric luôn có trọng số theo vĩ độ (cos φ).
5. Mô hình dự báo số gia: `X(t+1) = X(t) + ΔX`.
6. Notebook Kaggle chỉ gọi `wfsea <lệnh>`; logic mới phải vào `src/` kèm test.
7. Mỗi con số đưa vào luận văn/bài báo phải có một dòng trong `project/evidence_ledger.csv` trỏ tới file kết quả, `run_name` và commit.

## Ai sở hữu gì

Xem `project/KICKOFF.md` mục 3 và `.github/CODEOWNERS`.
