# Kế hoạch chạy — ai chạy gì, ở đâu, theo thứ tự nào

Tất cả chạy trên **Kaggle** (dữ liệu nằm ở đó). Code lấy từ repo GitHub; notebook trong `kaggle/notebooks/` chỉ là vỏ: đổi `CONFIG`/`EXTRA` ở ô đầu rồi *Run All*. Mỗi người dùng **tài khoản Kaggle của mình** (~30 giờ GPU/tuần, mỗi phiên tối đa 12 giờ) để ba người chạy song song.

## Chuẩn bị (mỗi người, một lần)

1. Code trên Kaggle được clone từ repo làm việc public `Phongdzfk/GNNforSEAWeather` — không cần token GitHub.
2. Import các notebook trong `kaggle/notebooks/` (*File → Import notebook*).
3. Phong chia sẻ 3 dataset dữ liệu (bước 0) cho Hiếu và Tùng: trên trang output của notebook → *New Dataset* → để Private → *Settings → Sharing* → thêm hai bạn.

## Thứ tự chạy

```
Bước 0 (CPU, chia 3 người)  00_download_pressure: tải 7 tầng × (z,t,q,u,v), mỗi notebook 2–3 năm
                         │
         (Phong, CPU)    00a kiểm kê + stats ─► 00b cache train (3 phần, 3 notebook) ─► 00c val/test + climatology
                                                  │
Bước 1 (song song, GPU)  Hiếu: B0      Tùng: v2      Phong: U-Net   ──► Tùng đánh giá trên val
                                                  │
Bước 2 (song song)       Hiếu: S1 (v2_wide)   Phong: S2 (v2_wider) + A4, A5   Tùng: A0–A3 + đánh giá
                                                  │   → chọn cỡ model tốt nhất trên val
Bước 3                   Hiếu: v3 (rollout 4 → 12) từ cỡ tốt nhất            Tùng: đánh giá
                                                  │
Bước 4                   Phong: v4 (biên ERA5)   Hiếu: nhánh nghiên cứu (mesh đa mức)
                                                  │
Chốt                     Tùng: đánh giá bản tốt nhất + baseline trên test 2026, xuất dự báo
```
Trong lúc Phong chạy bước 0 (vài giờ CPU), Hiếu và Tùng có thể kiểm tra môi trường bằng dữ liệu cũ (mục "Trong lúc chờ").

## Chi tiết từng run

| ID | Người | Notebook | Đặt ở ô đầu | Input cần gắn | GPU ước tính |
|---|---|---|---|---|---|
| **D1–D6** | Phong 2, Hiếu 2, Tùng 2 | `00_download_pressure` (Save & Run All) | `YEARS='2015-2016'`, `'2017-2018'`, … `'2025-2026'` | — (Secrets `url`, `key`) | CPU, ≤ 12 h; chưa xong thì chạy version mới gắn output cũ |
| **0a** | Phong | `00a_stats` | `CONFIG='configs/levels7.yaml'` | mọi output dữ liệu thô (single levels cũ + D1–D6 + NASA) | CPU |
| **0b** | Phong (×3) | `00b_cache_train` | `PART=0`, `1`, `2` | thô của các năm trong phần + output 0a | CPU, 3 notebook song song |
| **0c** | Phong | `00c_cache_valtest` | — | thô 2025–2026 + output 0a + cả 3 phần 0b | CPU |
| **B0** | Hiếu | `01_train_graphcast` | `CONFIG='configs/kaggle.yaml'` (14 kênh, đúng kiến trúc cũ) | dữ liệu | 1–2 phiên |
| **v2** | Tùng | `01_train_graphcast` | `CONFIG='configs/v2_l7.yaml'` (41 kênh) | dữ liệu | 1–2 phiên |
| **U** | Phong | `02_train_unet` | — | 3 dataset dữ liệu | 1–2 phiên |
| **E1** | Tùng | `03_evaluate` | `CKPT='<run>/best.pt'`, `SPLIT='val'` — lần lượt B0, v2, unet_v2 | dữ liệu + output run | ~1 giờ/run |
| **S1** | Hiếu | `01_train_graphcast` | `CONFIG='configs/v2_wide.yaml'` (mesh 256 chiều, ≈5,1M tham số) | dữ liệu | 1–2 phiên |
| **S2** | Phong | `01_train_graphcast` | `CONFIG='configs/v2_wider.yaml'` (mesh 384 chiều, 16 lớp, ≈14,8M) | dữ liệu | 1–2 phiên |
| **v3a** | Hiếu | `01_train_graphcast` | `CONFIG='configs/v3_rollout4.yaml'` | dữ liệu + output **v2** | 1 phiên |
| **v3b** | Hiếu | `01_train_graphcast` | `CONFIG='configs/v3_rollout12.yaml'` | dữ liệu + output **v3a** | 1 phiên |
| **A1** | Tùng | `01_train_graphcast` | `CONFIG='configs/v2.yaml'`, `EXTRA='run_name=abl_no_time features.time=false features.solar=false train.epochs=15'` | dữ liệu | ~½ phiên |
| **A2** | Tùng | như trên | `EXTRA='run_name=abl_no_sat features.satellite=false train.epochs=15'` | dữ liệu | ~½ phiên |
| **A3** | Tùng | như trên | `EXTRA='run_name=abl_legacy_edges graph.edge_features=legacy train.epochs=15'` | dữ liệu | ~½ phiên |
| **A4** | Phong | như trên | `EXTRA='run_name=abl_state_loss train.loss_norm=state train.epochs=15'` | dữ liệu | ~½ phiên |
| **A5** | Phong | như trên | `EXTRA='run_name=abl_4layers model.n_layers=4 train.epochs=15'` | dữ liệu | ~½ phiên |
| **A0** | Tùng | như trên | `EXTRA='run_name=v2_short train.epochs=15'` — mốc so sánh cho A1–A5 | dữ liệu | ~½ phiên |
| **v4** | Phong | `01_train_graphcast` | `CONFIG='configs/v4_boundary.yaml'` | dữ liệu + output **v3b** | 1 phiên |
| **R** | Hiếu | — | Mesh đa mức (Đề xuất 1): code mới trong `graph/`, qua Pull Request | — | 1–2 phiên |
| **F** | Tùng | `03_evaluate` | bản tốt nhất + B0 + unet_v2, `SPLIT='test'`, thêm ô `oracle` và `export-forecast` | dữ liệu + output các run | ~3 giờ |

Ghi chú:
- **Dữ liệu 41 kênh (7 tầng):** dùng `configs/levels7.yaml` cho các bước 0 và `configs/v2_l7.yaml` (hoặc thêm `EXTRA` tương ứng) cho các run train. B0 giữ 14 kênh để đúng là "kiến trúc + dữ liệu như dự án cũ". Nếu chưa tải xong tầng mới, mọi bước vẫn chạy được với 14 kênh cũ (`configs/kaggle.yaml`, `configs/v2.yaml`).
- Ước lượng dung lượng (41 kênh): tải thô ≈ 6 GB/năm; cache train ≈ 4 GB/năm → 3 phần; val ≈ 4 GB, test ≈ 3 GB.
- RAM Kaggle (~30 GB): stats tính theo luồng (đỉnh vài trăm MB), cache ghi theo khối tự co theo số kênh (đỉnh ~2 GB), train đọc memmap theo mẫu — không bước nào nạp cả năm dữ liệu vào RAM.
- "Phiên" = một lần chạy notebook (tối đa 12 giờ). Thời gian thật mỗi epoch xem trong `history.json` (`epoch_sec`) của run đầu tiên rồi chỉnh `train.epochs` cho vừa. Config Kaggle đã đặt `max_hours: 11.3`: gần hết giờ, model tự lưu `last.pt` và dừng sạch. Chạy tiếp bằng `RESUME='<run>/last.pt'` và gắn output phiên trước.
- v3 mặc định fine-tune từ `v2`. Nếu S1 hoặc S2 tốt hơn trên val thì fine-tune từ bản đó, kiến trúc phải khớp, ví dụ `EXTRA='model.mesh_hidden=256 train.init_from=v2_wide/best.pt'` (S2 thêm `model.n_layers=16`). v3b, v4 cũng thêm đúng các tham số `model.*` đó.
- Đường scaling cho luận văn: RMSE/ACC 72 h theo số tham số B0 (0,54M) → v2 (1,34M) → S1 (5,1M) → S2 (14,8M), kèm thời gian/epoch và VRAM — trả lời câu "model nhỏ có sao không".
- Ablation chạy bản ngắn (15 epoch) và so với **A0** (v2 cũng 15 epoch), không so với v2 đầy đủ.
- **Chọn mô hình trên val 2025, báo cáo trên test 2026 một lần.** Không chỉnh tham số theo kết quả test.
- Output mỗi run (`best.pt`, `history.json`, `graph_stats.json`) lưu thành Kaggle Dataset có tên trùng `run_name` và chia sẻ cho nhóm, để run sau gắn vào được (v3a cần v2, v3b cần v3a, v4 cần v3b).

## Trong lúc chờ bước 0

- **Hiếu:** chạy `01_train_graphcast` với `CONFIG='configs/legacy_2020.yaml'`, gắn các Zarr cũ (`brain_*`, `eye_*`) và `stat-compute`. Chỉ cần ô smoke, mục đích là kiểm tra GPU, clone repo và DataParallel.
- **Tùng:** nếu còn checkpoint cũ `graphcast_sea_best.pt` của đồ án môn học, đánh giá nó bằng `03_evaluate` với `--config configs/legacy_2020.yaml` (thêm `--legacy-levels 500,850` nếu Phong xác nhận thứ tự tầng trong Zarr cũ là 500 trước). Kết quả cho biết số liệu cũ khi đo đúng cách.

## Các phiên bản mô hình

| Phiên bản | Thay đổi | File config |
|---|---|---|
| **B0** | Kiến trúc cũ (538K tham số), dữ liệu mới 2015–2024. Mốc "dự án cũ" | `kaggle.yaml` |
| **v2** | + forcing thời gian & bức xạ đỉnh khí quyển · cạnh mesh có khoảng cách/hướng · loss trên ΔX chuẩn hoá theo độ lệch chuẩn xu thế 6 h · processor 12 lớp · mưa biến đổi `log1p` (từ bước 0) | `v2.yaml` |
| **S1 / S2** | v2 với mesh rộng hơn (256 / 384 chiều, 12 / 16 lớp); lưới vẫn 128 nên chi phí tăng ít | `v2_wide.yaml`, `v2_wider.yaml` |
| **v3** | cỡ tốt nhất + fine-tune rollout 4 bước rồi 12 bước (gradient checkpointing) | `v3_rollout4.yaml` → `v3_rollout12.yaml` |
| **v4** | v3 + dải biên 2° gán bằng ERA5 mỗi bước, chấm điểm vùng trong | `v4_boundary.yaml` |
| **U-Net** | Baseline CNN cùng đầu vào/loss với v2 | `v2.yaml model.name=unet` |

Kết quả luận văn = bảng B0 → v2 → v3 → v4 (mỗi bước cải tiến bao nhiêu) + đường scaling theo số tham số + bảng ablation A1–A5 + so sánh với Persistence / Climatology / U-Net. v4 phải ghi rõ là "biên từ ERA5" — đây là cận trên của việc dùng biên từ mô hình toàn cầu.
