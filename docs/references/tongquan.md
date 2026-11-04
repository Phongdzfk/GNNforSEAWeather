## Tên đề tài: Dự báo thời tiết khu vực Đông Nam Á bằng mạng nơ-ron đồ thị kết hợp dữ liệu tái phân tích ERA5 và ảnh vệ tinh NASA

**Tên tiếng Anh:** Graph Neural Networks for Regional Weather Forecasting in Southeast Asia Fusing ERA5 Reanalysis and NASA Satellite Imagery (WeatherForecast-SEA)

## Nội dung chính của đồ án

- Xây dựng mô hình Mạng Nơ-ron Đồ thị (GNN) dạng GraphCast chuyên biệt cho khu vực Đông Nam Á ($11^\circ\text{S} - 28.5^\circ\text{N}$, $92^\circ\text{E} - 142^\circ\text{E}$), hoạt động tại độ phân giải không gian $0.25^\circ \times 0.25^\circ$ (~28 km) gồm 31,959 nút lưới quan sát và lưới ẩn động lực học $2.0^\circ$ (500 nút).
- Dự báo đồng thời 14 biến động lực học khí quyển: biến mặt đất ($t2m, d2m, u10, v10, msl, tp$), biến tầng khí áp 850 hPa và 500 hPa ($z, q, u, v$), kết hợp các trường tĩnh ($lsm, geopotential\_surf$).
- Dung hợp dữ liệu đa nguồn: Nhánh "Bộ não" (ECMWF ERA5 Reanalysis) cung cấp trường khí quyển đồng hóa toàn diện; Nhánh "Mắt thần" (Vệ tinh địa tĩnh NASA GPM MERGIR) cung cấp trường nhiệt độ bức xạ hồng ngoại đỉnh mây ($T_b$ 4 km) phát hiện sớm các ổ mây giông đối lưu nhiệt đới và bão Biển Đông.
- Thiết kế mô hình học thặng dư tự hồi quy (Residual Autoregressive): dự báo số gia $\Delta \mathbf{X}_{t+1} = \mathbf{X}_{t+1} - \mathbf{X}_t$ từ chuỗi lịch sử 2 bước thời gian $[\mathbf{X}_{t-1}, \mathbf{X}_t]$ nhằm ổn định sai số trong quá trình rollout dài hạn (+6h đến +72h).
- Tối ưu hóa hệ thống trên phần cứng phổ thông: giải quyết bài toán nghẽn I/O khi giải nén Zarr bằng kỹ thuật tiền giải mã mảng nhị phân nhúng bộ nhớ (Float16 Memory-Mapped Arrays), huấn luyện nửa độ chính xác (AMP) và song song hóa DataParallel, giảm thời gian huấn luyện từ 11h xuống 6h trên 2 GPU Kaggle mà không bị lỗi tràn bộ nhớ (OOM).
- Đánh giá thực nghiệm toàn diện qua 30 lượt rollout tự hồi quy (+72h) trên năm kiểm thử độc lập 2026: so sánh định lượng với Persistence, Climatology và mô hình tích chập U-Net; đo lường hệ số tương quan dị thường ACC, RMSE trọng số diện tích vĩ độ; phân tích sai số địa lý (hải dương vs đồng bằng vs đất liền vs miền núi) và phân tích phổ năng lượng Fourier 2D.
- Xây dựng ứng dụng Dashboard trực quan hóa bản đồ gió động lực và trường nhiệt độ thời gian thực (Streamlit + Mapbox) cùng dịch vụ backend REST API (FastAPI).

## Cơ sở dữ liệu ban đầu

- **[ECMWF ERA5 Reanalysis](https://cds.climate.copernicus.eu/):** Bộ dữ liệu tái phân tích khí hậu toàn cầu thế hệ thứ 5 của ECMWF, độ phân giải $0.25^\circ$, tần suất 6 giờ ($00, 06, 12, 18\text{ UTC}$) giai đoạn 2020–2026. Bao gồm các trường mặt đất và tầng khí áp chuẩn (850 hPa, 500 hPa).
- **[NASA GPM MERGIR](https://disc.gsfc.nasa.gov/datasets/GPM_MERGIR_1/summary):** Bộ dữ liệu nhiệt độ bức xạ hồng ngoại đỉnh mây (Brightness Temperature $T_b$) kênh $4\mu m$ từ hệ thống vệ tinh địa tĩnh toàn cầu (Himawari-8, GOES, Meteosat), độ phân giải $4\text{ km}$, tần suất 30 phút, được đồng bộ về mốc 6 giờ và nội suy lưới $0.25^\circ$.
- **Phân chia dữ liệu chống rò rỉ (Strict Anti-Leakage Split):**
  - Tập huấn luyện (Train): 2020, 2021, 2022, 2023, 2024 (7,303 mẫu).
  - Tập kiểm định (Validation): 2025 (1,459 mẫu).
  - Tập kiểm thử độc lập (Test): 2026 (587 mẫu).
  - Mọi tham số chuẩn hóa Z-score (Mean/Std) được tính toán hoàn toàn độc quyền trên tập huấn luyện 2020–2024.
