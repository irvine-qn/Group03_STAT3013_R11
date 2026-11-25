# Dữ liệu đầu vào của GAM

Thư mục này **không chứa dữ liệu gốc**. Nguồn duy nhất là bảng Level 1 chung của nhóm:
`data/processed/mekong/level1_series.csv` (21 mùa × 27 bước, diện tích ngập km² toàn ĐBSCL).

Hai file ở đây do `python src/GAM/build_dataset.py` sinh ra, **chỉ để xem**:

| File | Nội dung |
|---|---|
| `gam_rows.csv` | Mọi dòng dự báo của mùa 2003–2020 cho 3 tầm (1.224 dòng). Mỗi dòng: mùa, bước gốc `t`, ngày đích, `a_t`, `a_t1`, mức bình thường `c_target`, giá trị thật `y_target` và mục tiêu GAM học `anomaly_target = y_target − c_target`. |
| `dataset_summary.csv` | Số dòng, số mùa, trung bình và độ lệch chuẩn của anomaly theo tầm. |

Ý nghĩa các cột:

| Cột | Nghĩa |
|---|---|
| `h_steps`, `h_days` | Tầm dự báo: 1, 2, 4 bước = 8, 16, 32 ngày |
| `t` | Bước gốc (hôm nay), 1..27; bước 1 = ngày 153, mỗi bước 8 ngày |
| `doy_target` | Ngày trong năm của bước đích (t + h) |
| `c_t`, `c_target` | Mức bình thường = trung bình 5 mùa liền trước, tại bước gốc và bước đích |
| `a_t`, `a_t1` | Anomaly hôm nay và 8 ngày trước: y − c |
| `anomaly_target` | Mục tiêu GAM học |

Vì sao bước 2 (`run_model.py`) không đọc `gam_rows.csv`? Trong từng fold, mô hình phải dựng lại
dòng chỉ từ các mùa trước mùa test (hàm `floodbench.features.build_rows`). Bảng ở đây gộp mọi mùa,
nên chỉ dùng để mở ra xem dữ liệu trông như thế nào.

Mùa 2000–2002 không có dòng nào: chúng chưa có đủ 3 mùa trước để tính mức bình thường, nên chỉ được
dùng làm "lịch sử" cho climatology.
