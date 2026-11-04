# Khởi động dự án WeatherForecast-SEA (02/11/2026)

Tài liệu này dành cho nhóm (Phong, Hiếu, Tùng) và thầy hướng dẫn. Nó ghi lại điểm xuất phát của code, các lỗi tìm thấy trong notebook cũ, việc cần làm ngay và những điểm cần chốt. Phân công dựa trên tài liệu *Phân tích và phân công GraphCast SEA* (ngày 11/10/2026, bản 3 người).

## 1. Hiện trạng

- Repo `ducnx/WeatherForecast` trước đó chỉ có `docs/` (đặc tả, luận văn, bài báo). Code thật nằm trong 7 notebook Kaggle ở repo cũ `FN2187-TMH/GraphCast-for-Medium-Range-Weather-Forecasting-in-Southeast-Asia`.
- Đợt này chuyển toàn bộ notebook thành package Python `src/wfsea/`, có test, chạy giống nhau trên Kaggle và trên máy cá nhân.
- **Dữ liệu thô ERA5 + NASA MERGIR đã có đủ 2015–2026** (trên Kaggle). Thiết lập mới: **train 2015–2024 (10 năm), val 2025, test 2026** — gấp đôi dữ liệu train so với đồ án môn học (2020–2024).
- Hệ quả: file stats cũ (`stat-compute`, tính trên 2020–2024) và các Zarr cũ của nb3 (chuẩn hoá bằng stats đó, chỉ có 2020–2026) **không dùng cho thiết lập mới**. Pipeline mới đọc thẳng NetCDF thô → stats train-only → cache memmap fp16 (`00a` → `00b` → `00c`), không ghi Zarr trung gian. Các Zarr cũ chỉ còn dùng để tái lập con số cũ (`configs/legacy_2020.yaml`).
- Đã kiểm chứng bằng test: graph mặc định cho đúng 500 nút mesh, 127.836 cạnh G→M, 3.734 cạnh M→M; mô hình mặc định có đúng **538.382 tham số** với 38 kênh đầu vào.

```
src/wfsea/
  constants.py        miền, biến, THỨ TỰ KÊNH chuẩn (một nguồn duy nhất)
  config.py           YAML + ghi đè key=value + tự tìm đường dẫn trên Kaggle
  data/               Phong — download, preprocess, stats, memmap, dataset, manifest
  graph/mesh.py       Hiếu  — Grid–Mesh–Grid
  models/             Hiếu  — GraphCastSEA, UNetSEA, checkpoint I/O
  training/           Hiếu  — loss theo vĩ độ, trainer (AMP, DataParallel, early stop, rollout loss)
  evaluation/         Tùng  — baselines, metrics, rollout, runner 30×12×14×vùng
  inference/export.py Hiếu→Tùng — forecast_<init>.nc theo đơn vị vật lý
kaggle/notebooks/     00a stats · 00b cache train · 00c cache val/test · 01 train GNN · 02 train U-Net · 03 evaluate
tests/                dữ liệu giả đúng định dạng Kaggle → kiểm thử end-to-end
```

## 2. Những điểm tìm thấy khi đọc lại code cũ

Các điểm này ảnh hưởng trực tiếp đến số liệu trong luận văn. Code mới đã sửa hoặc đặt thành tuỳ chọn có tên rõ ràng; **số liệu cũ cần được chạy lại trước khi đưa vào báo cáo**.

| # | Vấn đề trong notebook cũ | Ảnh hưởng | Xử lý trong code mới |
|---|---|---|---|
| 1 | "Climatology" trong nb5 là **trung bình theo thời gian của chính tập test 2026**, dùng cho cả baseline climatology lẫn tham chiếu tính ACC | Rò rỉ dữ liệu test; ACC và baseline climatology không hợp lệ | Climatology tính trên **train**, theo (tháng × giờ UTC) — `evaluation/baselines.py` |
| 2 | Khi rollout, bước k dùng **Tb vệ tinh quan sát thật ở thời điểm tương lai** | Mô hình được "nhìn trước" mây thật → kết quả 24–72 h lạc quan | `eval.tb_policy`: `persist` (mặc định, giữ Tb quan sát cuối), `oracle` (tái tạo cách cũ để đối chiếu), `mask` |
| 3 | Nhãn tạo bằng `shift(time=-1)` theo từng năm rồi nối các năm | Mẫu đầu mỗi năm ghép X(t−1) cách 12 h thay vì 6 h; mất trạng thái cuối mỗi năm | Không lưu nhãn riêng; chỉ lấy mẫu khi t−1, t, t+1 cách nhau đúng 6 h; khôi phục trạng thái cuối từ label |
| 4 | Mesh là **lưới đều 2° vĩ–kinh, 8 láng giềng**, nhưng docs gọi là icosahedral; cạnh M→M có đặc trưng = 0; G→M dùng khoảng cách Euclid theo độ (docs ghi great-circle, 5 đặc trưng) | Mô tả phương pháp sai so với code | Đặt tên đúng; `graph.edge_features=geo` (haversine, Δlat, Δlon) là hướng nâng cấp của Hiếu |
| 5 | 38 kênh = 28 trạng thái + **4 kênh vệ tinh** (Tb, mask tại t và t−1) + 2 static + 4 vị trí. **Không có kênh thời gian** | docs/architecture.md ghi "Tb 1 kênh + 3 kênh thời gian" — sai | `features.time`, `features.solar` (Đề xuất 2) bật/tắt bằng config |
| 6 | ERA5 `tp` tải theo giờ tại 00/06/12/18 là **mưa tích luỹ 1 giờ trước đó**, không phải 6 giờ | Diễn giải mưa trong docs sai đơn vị | Ghi chú trong export; Phong xác nhận lại |
| 7 | Mesh bắt đầu từ 10°S trong khi lưới bắt đầu từ 11°S | Dải biên phía nam chỉ nối tới mesh xa | Giữ mặc định để tái lập; tham số `graph.mesh_lat_start` |
| 8 | "587 mẫu test 2026": nb2 tải MERGIR đến 30/05/2026 | Tập test thực chất ≈ 01/01–27/05/2026, không phải cả năm | `wfsea manifest` in ra khoảng thời gian thật — Phong xác nhận |
| 9 | Vệ tinh dùng `xarray.interp(method="linear")` (bilinear), docs ghi "bilinear bảo toàn" | Thuật ngữ sai | `regrid=bilinear` hoặc `box_mean` (trung bình các pixel 4 km trong ô 0,25°) |
| 10 | Lưu float16 chưa từng đo sai số | Có thể ảnh hưởng q, tp | Memmap ghi `fp16_relative_rmse` theo từng kênh vào meta/manifest |

Hệ quả: bảng RMSE/ACC và con số "giảm 37%" hiện có trong docs/paper phải được coi là **chờ kiểm chứng** cho đến khi `03_evaluate` chạy lại (xem `project/evidence_ledger.csv`).

## 3. Phân công theo module

| Module | A (chịu trách nhiệm) | Ghi chú |
|---|---|---|
| `data/*`, notebook `00a/00b/00c`, baseline Persistence/Climatology, dữ liệu U-Net, DEM (sau) | **Phong** | Bàn giao cho Hiếu: stats + cache memmap + `data_manifest.json` |
| `graph/*`, `models/*`, `training/*`, `inference/*`, notebook 01/02 | **Hiếu** | Bàn giao cho Tùng: `best.pt` (đã chứa config, thứ tự kênh, commit) |
| `evaluation/*`, notebook 03, evidence ledger, API/sản phẩm cuối | **Tùng** | Mọi con số vào luận văn phải có dòng trong ledger |

Sản phẩm cuối nhóm muốn là **trợ lý AI dạng chatbot trả lời câu hỏi thời tiết** (thay/bổ sung cho dashboard). Phần này **để sau khi có model tốt nhất**. `inference/export.py` đã xuất sẵn file dự báo theo đơn vị vật lý (°C, %, m/s, mm, hPa) làm đầu vào cho trợ lý sau này.

## 4. Chạy gì, ai chạy, theo thứ tự nào

Xem [`RUN_PLAN.md`](RUN_PLAN.md): bước 0 dữ liệu (Phong) → baseline B0 / v2 / U-Net song song → rollout fine-tune + ablation → biên ERA5 + nhánh nghiên cứu → đánh giá cuối trên test 2026.

Việc cần làm ngay:
- [ ] Merge Pull Request khởi động vào `main`; mỗi người thêm Kaggle Secret `GITHUB_TOKEN` (`kaggle/README.md`).
- [ ] Phong: liệt kê các notebook/dataset Kaggle chứa `year_2015` … `year_2026` (ERA5 và NASA), rồi chạy `00a` → `00b` → `00c`; xác nhận ngữ nghĩa `tp` (1 h hay 6 h) và thứ tự `pressure_level` trong Zarr cũ.
- [ ] Hiếu, Tùng: kiểm tra môi trường bằng dữ liệu cũ trong lúc chờ (mục "Trong lúc chờ" của RUN_PLAN).

## 5. Cần chốt trong buổi họp đầu

0. **Chia tập:** train 2015–2024 / val 2025 / test 2026 (đề xuất, vì đã có dữ liệu 2015). Docs và luận văn đang ghi train 2020–2024 → cần thầy đồng ý và cập nhật đồng bộ.
1. **Thuật ngữ mesh:** dùng "lưới đều 2° vĩ–kinh độ, 8 láng giềng" trong toàn bộ luận văn/bài báo.
2. **Chính sách vệ tinh khi rollout:** đề xuất báo cáo chính thức bằng `persist`, nêu `oracle` như một cận trên.
3. **Một nhánh cải tiến ưu tiên:** tài liệu phân công khuyến nghị *solar forcing* (Đề xuất 2) — chi phí thấp, đã có sẵn cờ `features.solar/time`. Nhánh thứ hai (nếu còn GPU): rollout đa bước + spectral loss (`train.rollout_steps`, `train.spectral_weight`).
4. **Quy trình code:** mọi thay đổi qua Pull Request vào `main`, ít nhất 1 người review; thầy có quyền cài bảo vệ nhánh `main` trên GitHub.
