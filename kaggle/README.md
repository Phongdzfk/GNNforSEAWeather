# Chạy trên Kaggle

Dữ liệu nằm trên Kaggle nên toàn bộ train/eval chạy ở đó. Nguyên tắc: **notebook Kaggle chỉ là vỏ mỏng** — clone repo, rồi gọi `wfsea <lệnh>`. Logic nằm trong `src/wfsea/`, được review qua Pull Request, nên notebook của ba người luôn chạy cùng một code.

## Thiết lập một lần (mỗi người)

Ô đầu mỗi notebook clone code từ **repo làm việc public `Phongdzfk/GNNforSEAWeather`** (biến `REPO`) — không cần token hay secret GitHub nào. Thử nghiệm trên nhánh khác: đổi `BRANCH` ở cùng ô.

Sau đó import các notebook trong `kaggle/notebooks/` (*File → Import notebook*). Notebook tải dữ liệu còn cần Kaggle Secrets `url`, `key` của CDS.

## Thứ tự chạy

Dữ liệu thô ERA5 + NASA MERGIR có đủ **2015–2026** → train 2015–2024 (10 năm), val 2025, test 2026.

| Notebook | Phụ trách | Accelerator | Input cần gắn | Output |
|---|---|---|---|---|
| `00_download_pressure` | chia 3 người | None (Save & Run All) | — | `era5_pressure/year_<y>/…nc`, `cds_requests.json` |
| `00a_stats` | Phong | None | toàn bộ thư mục thô `year_<YYYY>` ERA5 + NASA | `stats/` (mean/std **chỉ trên train**, có `_meta.years`) |
| `00b_cache_train` | Phong | None | thô các năm của phần + `00a` | `train_<năm>_raw_c<kênh>/` (mỗi phần ≤ ~17 GB; `PART=0,1,2` với 41 kênh) |
| `00c_cache_valtest` | Phong | None | thô 2025–2026 + `00a` + mọi phần `00b` | `val_2025_raw_c*/`, `test_2026_raw_c*/`, `data_manifest.json`, `climatology_train.npz`, `diff_stats.json`, `graph/` |
| `01_train_graphcast` | Hiếu | GPU T4 ×2 | `00a` + `00b` + `00c` | `<run>/best.pt`, `history.json`, `graph_stats.json` |
| `02_train_unet` | Phong/Hiếu | GPU T4 ×2 | như trên | `<run>/best.pt` |
| `03_evaluate` | Tùng | GPU T4 ×1 | `00a` + `00b` + `00c` + output 01/02 | `eval_persist/`, `eval_oracle/`, `forecasts/` |

- Thư mục thô có thể nằm rải rác ở nhiều notebook/dataset (ví dụ bản tải 2015–2019 và bản 2020–2026): cứ gắn tất cả, code tự tìm `year_<YYYY>` ở mọi input.
- Các bước 00 chỉ chạy lại khi dữ liệu hoặc stats thay đổi. Notebook sau tự dùng cache memmap được gắn làm input, không phải đọc lại NetCDF (đây chính là tối ưu "11 h → 6 h").
- Train cache tách riêng (00b) và tự chia phần theo số kênh để mỗi output ≤ ~17 GB (giới hạn 20 GB/notebook).
- Tái lập đúng thiết lập đồ án môn học cũ (train 2020–2024, Zarr cũ của nb3, stats `stat-compute`): thêm `--config configs/legacy_2020.yaml` và gắn các Zarr `brain_*`/`eye_*` + `stat-compute`.

## Lưu ý về các notebook tải ERA5 cũ

Pipeline mới đọc được dữ liệu đã tải bằng các notebook cũ (nhận file theo nội dung, chọn 00/06/12/18 UTC từ dữ liệu hourly, cộng `tp` 6 giờ). Khi tải thêm, tránh các lỗi đã thấy:

- **Năm bị gõ cứng:** trong `notebook-1-era5-2018`, ô pressure levels để `"year": "2015"` → notebook "2018" thực ra chứa pressure của 2015. Chạy `wfsea raw-index` để thấy năm thật của từng thư mục.
- **Tải hourly** (`time` 24 giờ) cho pressure levels: mô hình chỉ dùng 4 giờ → request nặng gấp 6, xếp hàng lâu hơn. Notebook `00_download_pressure` chỉ tải 4 giờ. (Single levels hourly vẫn có ích cho `tp` cộng dồn 6 giờ — giữ nguyên.)
- **Lỗi bị nuốt:** `try/except` chỉ in lỗi rồi đi tiếp → có tháng thiếu mà không biết. `raw-index`/manifest báo khoảng trống.
- **`client.retrieve` chờ đồng bộ:** khi hàng đợi CDS chậm thì timeout, chạy lại lại gửi request mới → hàng đợi càng dài. `00_download_pressure` gửi một lần, lưu request ID, lần sau chỉ tải.
- `geopotential`, `land_sea_mask` là trường tĩnh — chỉ cần tải một lần, không cần mỗi tháng.

## Quy ước

- Mỗi lần train ghi `run_name` kèm commit (`graphcast_baseline_<commit>`) → checkpoint lưu `git_commit`, config và thứ tự kênh. Đây là đầu vào cho evidence ledger.
- Thử nghiệm trên nhánh riêng: đổi `BRANCH = "..."` ở ô đầu tiên. Không sửa code trực tiếp trong Kaggle.
- Đừng in token ra output. Ô bootstrap đã che token trong thông báo lỗi.

## Push bằng Kaggle CLI (tuỳ chọn)

```bash
pip install kaggle   # cần ~/.kaggle/kaggle.json
cp kaggle/kernel-metadata.example.json kaggle/notebooks/kernel-metadata.json
# sửa id, code_file, dataset_sources / kernel_sources cho đúng slug dữ liệu của nhóm
kaggle kernels push -p kaggle/notebooks
```
