# LightGBM – cây tăng cường

> **Bắt đầu từ [`notebooks/LightGBM/lightgbm_overview.ipynb`](../../notebooks/LightGBM/lightgbm_overview.ipynb)**:
> notebook đi qua từng bước bằng hàm thật và chỉ sang file chi tiết bên dưới.
> Bản tiếng Anh: [`lightgbm_overview_en.ipynb`](../../notebooks/LightGBM/lightgbm_overview_en.ipynb).
> Chẩn đoán đầy đủ: [`thien_lightgbm.ipynb`](../../notebooks/LightGBM/thien_lightgbm.ipynb).

## 1. LightGBM là gì

Nhiều cây quyết định nhỏ, mỗi cây sửa phần sai của các cây trước (gradient boosting). Học từ các mùa trước,
mỗi dòng là một gốc dự báo. Ba biến thể **khai báo trước** khi chạy test:

| Biến thể | Đặc trưng | Mục tiêu |
|---|---|---|
| `lgbm_raw` (bản chính) | y_t, y_t1, y_t2, c_t, c_target, ngày đích | y ở đích |
| `lgbm_anomaly` | a_t, a_t1, ngày đích | anomaly ở đích, rồi cộng c_target |
| `lgbm_noclim` | y_t, y_t1, y_t2, ngày đích | y ở đích (bỏ climatology để xem lợi thế đến từ đâu) |

Mỗi (mùa test, tầm) chọn 1 trong 6 cấu hình (num_leaves, n_estimators, learning_rate) bằng 3 mùa cuối của tập học;
`min_child_samples = 10`, `subsample = 0,8`, seed 0.

## 2. Luồng dữ liệu và thứ tự đọc code

```
data/processed/level1_series.csv
        │
   lightgbm_data.py       đọc thành ma trận Y: 21 mùa × 27 bước
   lightgbm_protocol.py   rolling origin, bảng đặc trưng, chọn cấu hình, AnomAR1, clim5
   lightgbm_model.py      3 biến thể, lưới 6 cấu hình, lưới Random Forest để đối chứng
        ▼
   lightgbm_forecast.py   (bước 1) 11 mùa test × 3 tầm → predictions.csv (seed 0)
   lightgbm_evaluate.py   (bước 2) rmse.csv, vs_anom_ar1.csv, rmse_by_phase.csv, rmse_by_size.csv
   lightgbm_diag.py       (bước 3) cấu hình chọn, SHAP, 10 seed, permutation importance, so với Random Forest
```

## 3. Chạy lại

```powershell
.venv\Scripts\python.exe src/LightGBM/lightgbm_forecast.py
.venv\Scripts\python.exe src/LightGBM/lightgbm_evaluate.py
.venv\Scripts\python.exe src/LightGBM/lightgbm_diag.py            # 10 seed, khoảng 1 phút
.venv\Scripts\python.exe src/LightGBM/lightgbm_diag.py extras
.venv\Scripts\python.exe src/LightGBM/lightgbm_diag.py seasons
.venv\Scripts\python.exe -m pytest tests/LightGBM
```

## 4. Kết quả (11 mùa test 2010–2020)

RMSE gộp, km² (`results/LightGBM/tables/rmse.csv`):

| Mô hình | 8 ngày | 16 ngày | 32 ngày |
|---|--:|--:|--:|
| AnomAR1 | 1090 | 1375 | 1687 |
| LightGBM thô | 1051 | 1284 | 1688 |
| LightGBM anomaly | 1101 | 1364 | 1699 |
| LightGBM không climatology | 1059 | 1301 | 1778 |

`lgbm_raw` nhỉnh hơn AnomAR1 về số ở 8–16 ngày (thắng 9/11 mùa ở 16 ngày), chủ yếu khi nước lên và nước rút,
nhưng không tầm nào khác có ý nghĩa (`vs_anom_ar1.csv`, p Holm = 1). Kết quả ổn định qua 10 seed
(độ lệch chuẩn RMSE 0,4–1,7%).

## 5. Ghi chú

- Cây không ngoại suy: năm lũ cực trị dễ bị dự báo thấp.
- Lợi thế khi nước rút do vài mùa chi phối; mùa lũ nhỏ đều là những năm cuối (cỡ lũ lẫn với lượng dữ liệu học).
- Ghi chú công việc cũ của phần LightGBM: [`docs/lightgbm/thien_lightgbm.md`](../../docs/lightgbm/thien_lightgbm.md)
  (một số đường dẫn trong đó còn trỏ tới `floodbench` cũ).
