# AnomAR1 – mốc thống kê chính

> **Bắt đầu từ [`notebooks/AnomAR1/anomar1_overview.ipynb`](../../notebooks/AnomAR1/anomar1_overview.ipynb)**:
> notebook đi qua từng bước bằng hàm thật và chỉ sang file chi tiết bên dưới.
> Bản tiếng Anh: [`anomar1_overview_en.ipynb`](../../notebooks/AnomAR1/anomar1_overview_en.ipynb).

## 1. AnomAR1 là gì

Diện tích ngập sau h bước = **mức bình thường** tại bước đích + φ^h × **độ lệch hôm nay**.

- Mức bình thường (climatology) c = trung bình 5 mùa liền trước, cùng bước.
- Độ lệch (anomaly) a = y − c, giả định là AR(1): a_t ≈ φ·a_{t−1}, nên sau h bước còn φ^h·a_t.
- φ = hồi quy qua gốc, ước lượng lại ở mỗi mùa test và mỗi tầm, chỉ từ các mùa huấn luyện.

Đây là **mốc** mà mọi mô hình khác của nhóm phải vượt. Dự báo `anom_ar1` ở đây trùng tuyệt đối với
cột `anom_ar1` trong bảng của GAM, LightGBM và Chronos-2.

## 2. Luồng dữ liệu và thứ tự đọc code

```
data/processed/level1_series.csv
        │
   anomar1_data.py         đọc thành ma trận Y: 21 mùa × 27 bước (km² ngập)
        │
   anomar1_protocol.py     climatology, gốc dự báo, mùa huấn luyện, bảng dòng, φ, AnomAR1, persistence, clim5
        ▼
   anomar1_forecast.py     (bước 1) 11 mùa test × 3 tầm → predictions.csv, phi_by_fold.csv, regression_rows.csv
        ▼
   anomar1_evaluate.py     (bước 2) rmse.csv, vs_anom_ar1.csv, rmse_by_phase.csv, rmse_by_size.csv
        ▼
   anomar1_diagnostics.py  (bước 3) φ^h so với φ trực tiếp, kiểm định phần dư, hình

   anomar1_eda.py          thống kê mô tả 21 mùa, xu hướng đỉnh lũ (độc lập với dự báo)
```

## 3. Chạy lại

```powershell
.venv\Scripts\python.exe src/AnomAR1/anomar1_forecast.py
.venv\Scripts\python.exe src/AnomAR1/anomar1_evaluate.py
.venv\Scripts\python.exe src/AnomAR1/anomar1_diagnostics.py
.venv\Scripts\python.exe src/AnomAR1/anomar1_eda.py
.venv\Scripts\python.exe -m pytest tests/AnomAR1
```

## 4. Kết quả (11 mùa test 2010–2020)

RMSE gộp, km² (`results/AnomAR1/tables/rmse.csv`):

| Mô hình | 8 ngày | 16 ngày | 32 ngày |
|---|--:|--:|--:|
| AnomAR1 | 1090 | 1375 | 1687 |
| AnomAR1, học 7 mùa | 1084 | 1355 | 1682 |
| Clim5 | 1812 | 1845 | 1916 |
| Persistence | 1312 | 2068 | 3456 |

- Persistence kém hơn AnomAR1 có ý nghĩa ở cả ba tầm (p Holm < 0,05).
- Clim5 kém hơn về số ở mọi tầm nhưng chưa có ý nghĩa sau Holm (n = 11).
- Học 7 mùa gần nhất tương đương học đủ mùa trong biên ±5% RMSE (p TOST < 0,01).

Kiểm định đầy đủ: `results/AnomAR1/tables/vs_anom_ar1.csv`.

## 5. Ghi chú

- `build_dataset.py`, `trace_1602.py` và `level3.py` đã gỡ vì đầu vào của chúng (ảnh gốc, `data/interim/`,
  `series_mean.csv`) không còn trong repo. Bảng `trace_1602.csv` là kết quả lịch sử; lấy lại script ở commit `db99215`.
- Chỉ 11 mùa test, một chuỗi toàn vùng: khoảng tin cậy rộng. "Không khác có ý nghĩa" không có nghĩa "tương đương".
