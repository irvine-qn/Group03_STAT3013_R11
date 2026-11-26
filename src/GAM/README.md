# GAM – mô hình cộng tổng quát

> **Bắt đầu từ [`notebooks/GAM/gam_overview.ipynb`](../../notebooks/GAM/gam_overview.ipynb)**:
> notebook đi qua từng bước bằng hàm thật và chỉ sang file chi tiết bên dưới.
> Bản tiếng Anh: [`gam_overview_en.ipynb`](../../notebooks/GAM/gam_overview_en.ipynb).

## 1. GAM là gì

Dự báo anomaly của bước đích bằng một hàm trơn theo ngày cộng hai số hạng tuyến tính:

    a_{t+h} ≈ β0 + s(ngày đích) + β1·a_t + β2·a_{t−1}        dự báo = c_{t+h} + anomaly dự báo

- c = mức bình thường (trung bình 5 mùa liền trước), a = y − c.
- s(·) là spline phạt; hai siêu tham số (n_splines, lam) chọn trong lưới 6 cấu hình bằng 3 mùa cuối của tập học.
- AnomAR1 là trường hợp riêng (không có β0, s(·), β2 và β1 = φ^h), nên GAM trả lời: phần sửa lệch theo mùa
  vụ có giúp gì không?

## 2. Luồng dữ liệu và thứ tự đọc code

```
data/processed/level1_series.csv
        │
   gam_data.py            đọc thành ma trận Y: 21 mùa × 27 bước
   gam_protocol.py        climatology, bảng dòng, rolling origin, chọn cấu hình, AnomAR1, kiểm định cấp mùa
   gam_model.py           ma trận đầu vào, lưới 6 cấu hình, học và dự báo, hệ số
   gam_common.py          hằng số và đường dẫn
        ▼
   gam_build_dataset.py   (bước 1) bảng dòng để xem → data/processed/GAM/
   gam_run_model.py       (bước 2) dự báo 11 mùa test → predictions.csv, cấu hình, hệ số, bóc tách
   gam_analysis.py        (bước 3) rmse.csv, vs_anom_ar1.csv, bảng theo điều kiện, phần dư
   gam_figures.py         (bước 4) 12 hình
```

## 3. Chạy lại

```powershell
.venv\Scripts\python.exe src/GAM/gam_build_dataset.py
.venv\Scripts\python.exe src/GAM/gam_run_model.py      # khoảng 2–4 phút
.venv\Scripts\python.exe src/GAM/gam_analysis.py
.venv\Scripts\python.exe src/GAM/gam_figures.py
.venv\Scripts\python.exe -m pytest tests/GAM
```

## 4. Kết quả (11 mùa test 2010–2020)

RMSE gộp, km² (`results/GAM/tables/rmse.csv`):

| Mô hình | 8 ngày | 16 ngày | 32 ngày |
|---|--:|--:|--:|
| AnomAR1 | 1090 | 1375 | 1687 |
| GAM | 1096 | 1372 | 1688 |

So với AnomAR1 ở cấp mùa, n = 11 (`results/GAM/tables/vs_anom_ar1.csv`): tỉ lệ RMSE 1,006 / 0,997 / 1,001,
không tầm nào khác có ý nghĩa; ở 8 ngày tương đương trong ±5% RMSE (p TOST = 0,002).
Bản học 7 mùa so với AnomAR1 học 7 mùa: `gam_w7_vs_anom_ar1_w7.csv`.

## 5. Ghi chú

- Thí nghiệm bóc tách (`gam_ablation_*`) nghĩ ra sau khi xem kết quả test: chỉ để hiểu, không vào bảng chính.
- p Holm chỉ hiệu chỉnh trong 3 tầm của GAM. Bảng theo giai đoạn và cỡ lũ chỉ để mô tả, không kiểm định.
