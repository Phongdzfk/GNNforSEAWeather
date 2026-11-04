# WeatherForecast: WeatherForecast-SEA

> **Dự báo thời tiết khu vực Đông Nam Á bằng mạng nơ-ron đồ thị kết hợp dữ liệu tái phân tích ERA5 và ảnh vệ tinh NASA**  
> *Graph Neural Networks for Regional Weather Forecasting in Southeast Asia Fusing ERA5 Reanalysis and NASA Satellite Imagery*

---

## 🧩 Mã nguồn (`src/wfsea`) và cách chạy

| Việc | Lệnh | Phụ trách |
|---|---|---|
| Kiểm thử toàn bộ pipeline (dữ liệu giả) | `pytest` | mọi người |
| Zarr → memmap fp16, manifest, climatology | `wfsea materialize --split all` · `wfsea manifest` · `wfsea climatology` | Phong |
| Thống kê graph, train GNN / U-Net | `wfsea graph` · `wfsea train --config configs/kaggle.yaml [model.name=unet]` | Hiếu |
| Đánh giá 30 starts × 72 h, xuất dự báo | `wfsea evaluate --ckpt best.pt` · `wfsea export-forecast --ckpt best.pt` | Tùng |

- Train/eval chạy trên Kaggle bằng các notebook mỏng trong [`kaggle/notebooks/`](kaggle/) — hướng dẫn: [`kaggle/README.md`](kaggle/README.md).
- Quy trình nhánh/PR: [`CONTRIBUTING.md`](CONTRIBUTING.md). Ai chạy gì, theo thứ tự nào: [`project/RUN_PLAN.md`](project/RUN_PLAN.md). Phân công và các lỗi đã tìm thấy trong notebook cũ: [`project/KICKOFF.md`](project/KICKOFF.md).
- Mọi con số trong luận văn phải có dòng tương ứng trong [`project/evidence_ledger.csv`](project/evidence_ledger.csv).

---

## ⚡ Khởi tạo Môi trường Ảo với `uv`

Dự án sử dụng công cụ quản lý môi trường và gói siêu tốc độ [uv](https://github.com/astral-sh/uv):

```bash
# 1. Tạo môi trường ảo với Python 3.12 (hoặc 3.10+)
uv venv --python 3.12

# 2. Kích hoạt môi trường ảo
source .venv/bin/activate
# (Trên Windows dùng: .venv\Scripts\activate)

# 3. Khởi chạy máy chủ tài liệu Docsify
python3 docs/serve.py
```

Truy cập hệ thống tài liệu đầy đủ tại: **http://localhost:3000**

---

## 📚 Hệ thống Tài liệu Toàn diện (Docsify)

Toàn bộ tài liệu chi tiết của dự án được lưu trữ trong thư mục [`docs/`](docs/) bao gồm:

- [**Tổng quan & Động lực Đề tài**](docs/README.md): Bối cảnh AI khí tượng, thông số kỹ thuật (0.25° grid, 2° mesh, 538K params).
- [**Kiến trúc Mạng Nơ-ron Đồ thị**](docs/architecture.md): Cấu trúc Grid–Mesh–Grid, cơ chế kNN (k=4) và 4 tầng GNN Processor.
- [**Pipeline Dữ liệu ERA5 & Vệ tinh**](docs/data-pipeline.md): Tích hợp ERA5 (14 biến) và NASA GPM MERGIR ($T_b$ 4km), kỹ thuật Float16 Memmap giảm 45% thời gian train.
- [**Cấu hình Huấn luyện & Tối ưu**](docs/models-and-training.md): Hàm mất mát Latitude-Weighted RMSE, Cosine LR, AMP và DataParallel.
- [**Kết quả Thực nghiệm & Rollout**](docs/evaluation-results.md): 30 lượt rollout +72h trên tập test 2026, ACC > 0.6, so sánh U-Net và Persistence.
- [**Tổng quan Nghiên cứu & Khoảng trống**](docs/literature-and-novel-methods.md): Review quốc tế, 5 khoảng trống nghiên cứu và 4 đề xuất đột phá.
- [**Yêu cầu Hệ thống & Cài đặt**](docs/requirements.md): Cấu hình Kaggle GPU, thư viện PyTorch Geometric, xarray, zarr.
- [**Giao diện Tác nghiệp & API**](docs/ui-dashboard.md): Lộ trình triển khai ONNX Runtime, backend FastAPI và Streamlit Mapbox.

---

## 📄 Bài báo Khoa học (Paper LaTeX)
- Khung bài báo hoàn chỉnh chuẩn IEEE nằm tại [`docs/paper/`](docs/paper/):
  - `main.tex`, `intro.tex`, `related.tex`, `method.tex`, `exp.tex`, `conclusion.tex`, `references.bib`.

---

## 🎓 Hồ sơ Khóa luận Tốt nghiệp (PTIT Thesis Template)
- Bộ hồ sơ tốt nghiệp bám sát chuẩn Học viện PTIT nằm tại [`docs/thesis/`](docs/thesis/):
  - [`Khung_do_an.docx`](docs/thesis/Khung_do_an.docx): File Word hoàn chỉnh theo mẫu chuẩn PTIT.
  - [`khung-do-an.md`](docs/thesis/khung-do-an.md): Bản thiết kế chi tiết 7 chương đồ án dạng Markdown.
  - [`thesis-outline.md`](docs/thesis/thesis-outline.md): Đề cương học thuật tóm tắt.
  - [`roadmap.md`](docs/thesis/roadmap.md): Lộ trình triển khai đồ án 12 tuần.
