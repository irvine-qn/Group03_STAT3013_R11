# Comparison – đặt 4 mô hình cạnh nhau

> **Bắt đầu từ [`notebooks/Comparison/comparison_overview.ipynb`](../../notebooks/Comparison/comparison_overview.ipynb)**.
> Bản tiếng Anh: [`comparison_overview_en.ipynb`](../../notebooks/Comparison/comparison_overview_en.ipynb).

## 1. Làm gì

Không học lại mô hình nào. Đọc `predictions.csv` của 4 thư mục kết quả, kiểm tra mọi thư mục dùng chung một mốc
AnomAR1, rồi tính lại các bảng chung bằng cùng công thức với bảng của từng mô hình. Số ở đây khớp với số trong
`results/<Mô hình>/tables/` (có test kiểm tra).

Bản chính của mỗi thành viên, khai báo trước: `gam`, `lgbm_raw`, `chronos2`; mốc là `anom_ar1`.

## 2. Luồng dữ liệu

```
results/{AnomAR1,GAM,LightGBM,Chronos-2}/tables/predictions.csv
        │
   comparison_tables.py    (bước 1) kiểm tra mốc chung, rồi ghi bảng → results/Comparison/tables/
   comparison_figures.py   (bước 2) 4 hình → results/Comparison/figures/
```

| Bảng | Nội dung |
|---|---|
| `baseline_check.csv` | cột `anom_ar1` của mỗi thư mục lệch bao nhiêu so với thư mục AnomAR1 |
| `rmse_all.csv` | RMSE gộp của mọi mô hình × tầm, kèm thư mục nguồn và vai trò (`baseline`, `main`, `variant`, `simple`) |
| `vs_anom_ar1_all.csv` | kiểm định cấp mùa so với AnomAR1; `p_t_holm_main` = Holm trên 9 phép so của 3 bản chính × 3 tầm |
| `rmse_by_phase.csv` | RMSE theo giai đoạn lũ, 4 bản chính × 3 tầm (cùng dạng với bảng của từng mô hình) |
| `rmse_by_size.csv` | RMSE theo cỡ lũ của mùa test, 4 bản chính × 3 tầm |
| `season_rmse.csv` | RMSE từng mùa test và tỉ lệ so với AnomAR1 cùng mùa |

Hình: `rmse_by_horizon.png`, `season_ratio.png`, `conditions.png`, `example_seasons.png`.

## 3. Chạy lại

```powershell
.venv\Scripts\python.exe src/Comparison/comparison_tables.py
.venv\Scripts\python.exe src/Comparison/comparison_figures.py
.venv\Scripts\python.exe -m pytest tests/Comparison
```

## 4. Kết quả (11 mùa test 2010–2020)

| Mô hình | RMSE 8 ngày | 16 ngày | 32 ngày | Tỉ lệ so với AnomAR1 (8 / 16 / 32) | p Holm (9 phép so) |
|---|--:|--:|--:|--:|--:|
| AnomAR1 | 1090 | 1375 | 1687 | – | – |
| GAM | 1096 | 1372 | 1688 | 1,006 / 0,997 / 1,001 | 1 / 1 / 1 |
| LightGBM thô | 1051 | 1284 | 1688 | 0,965 / 0,933 / 1,001 | 1 / 1 / 1 |
| Chronos-2 | 1116 | 1412 | 1788 | 1,024 / 1,027 / 1,060 | 1 / 1 / **0,045** |

Không mô hình nào tốt hơn AnomAR1 có ý nghĩa. Chronos-2 zero-shot kém hơn có ý nghĩa ở 32 ngày.
GAM tương đương AnomAR1 ở 8 ngày trong biên ±5% RMSE (p TOST = 0,002).

## 5. Ghi chú

- Holm của từng thư mục mô hình chỉ hiệu chỉnh trong các phép so của mô hình đó; bảng ở đây hiệu chỉnh trên cả 9.
  Biến thể và mốc đơn giản có p thô, không vào họ hiệu chỉnh.
- Bảng theo giai đoạn và cỡ lũ chỉ để mô tả, không kiểm định.
- Muốn thêm mô hình: ghi `predictions.csv` đúng 7 cột chuẩn vào `results/<Mô hình>/tables/` (có cả cột `anom_ar1`),
  rồi thêm thư mục đó vào `SOURCES` trong `comparison_tables.py`.
