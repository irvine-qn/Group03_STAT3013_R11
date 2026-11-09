# src/GAM – Hướng dẫn đọc code và chạy lại

Phần của Bình trong đồ án STAT3013 (xem [reports/members/binh_gam.md](../../../reports/members/binh_gam.md)).
Báo cáo kết quả: [bao_cao_gam.md](bao_cao_gam.md).

## 1. Cấu trúc thư mục

```
src/GAM/
├── dataset/        bảng dữ liệu đầu vào để xem (sinh ở bước 1) + README giải thích cột
├── docs/           README.md (file này) và bao_cao_gam.md (báo cáo có hình)
├── process/        code, chạy theo thứ tự bước 1 → 4
│   ├── common.py         hằng số, đường dẫn, hàm nhỏ dùng chung
│   ├── gam_model.py      MÔ HÌNH: đọc file này trước
│   ├── build_dataset.py  bước 1: dựng bảng dữ liệu đầu vào
│   ├── run_model.py      bước 2: chạy GAM trên mọi fold, lưu dự báo, cấu hình, hệ số
│   ├── analysis.py       bước 3: RMSE, kiểm định, theo điều kiện, phần dư
│   └── figures.py        bước 4: 12 hình cho báo cáo
├── result/
│   ├── tables/     mọi bảng csv do code sinh ra
│   └── figures/    mọi hình png do code sinh ra
└── test/           pytest: khớp mô hình chung, đáp án biết trước, không rò rỉ
```

## 2. Chạy lại (từ gốc repo)

```bash
pip install -r requirements.txt && pip install -e .     # một lần
python src/GAM/process/build_dataset.py   # bước 1, vài giây
python src/GAM/process/run_model.py       # bước 2, vài giây đến 1 phút
python src/GAM/process/analysis.py        # bước 3
python src/GAM/process/figures.py         # bước 4
pytest src/GAM/test                       # 13 test
```

Bước 2 in ra bảng `reproduction_check`: độ lệch giữa dự báo ở đây và dự báo chung của nhóm
(`results/tables/probe_predictions.csv`) phải xấp xỉ 0 (lần chạy gần nhất: < 1e-9 km²).

**Thư mục này không sửa gì ngoài `src/GAM/`.** File chung của nhóm chỉ được đọc.

## 3. Đọc code theo thứ tự nào

1. **`gam_model.py`**: docstring đầu file giải thích mô hình trong 4 dòng. Bốn khối hàm:
   1. `design_matrix`, `anomaly_target`: từ bảng dòng sang X và y.
   2. `fit_gam`, `predict`, `fit_predict`: học và dự báo với một cấu hình.
   3. `gam_forecast`: quy trình đầy đủ của một fold (chọn cấu hình → học lại → dự báo).
   4. `coefficients`, `seasonal_curve`, `contributions`: đọc mô hình đã học.
2. **`run_model.py`**: hàm `run` là vòng lặp fold × tầm. Mọi thứ được lưu ở đây, nên bước 3 và 4 không học lại mô hình.
3. **`analysis.py`**: mỗi hàm sinh một bảng. Docstring đầu file liệt kê bảng nào làm gì.
4. **`figures.py`**: mỗi hàm `fig_*` vẽ một hình, đúng thứ tự trong báo cáo.

## 4. Một dự báo đi qua code như thế nào

Ví dụ: dự báo diện tích ngập ngày 249 của mùa 2016, đứng ở ngày 233 (bước 11), tầm 16 ngày.

| Bước | Hàm | Việc làm |
|---|---|---|
| 1 | `cv.rolling_origin` | Fold 2016: học mùa 2003–2015, test mùa 2016 |
| 2 | `features.build_rows` | Mỗi dòng gồm `y_t`, `y_t1`, mức bình thường `c` (TB 5 mùa trước), ngày đích, nhãn `y_target`. Dòng của mùa 2016 chỉ dùng 2011–2015 để tính `c` |
| 3 | `gam_model.design_matrix` | X = [ngày đích, a_t = y_t − c_t, a_{t−1}] |
| 4 | `tuning.pick_config` | Học 6 cấu hình trên 2003–2012, chấm trên 2013–2015, chọn cấu hình MSE nhỏ nhất |
| 5 | `gam_model.fit_gam` | Học lại cấu hình đó trên 2003–2015 |
| 6 | `gam_model.predict` | Dự báo = c(ngày 249) + β0 + s(249) + β1·a_t + β2·a_{t−1} |

Hình 3 trong báo cáo vẽ đúng ví dụ này, kèm giá trị từng phần.

## 5. Ghi chú môi trường

- Code chỉ cần `numpy`, `pandas`, `scipy`, `matplotlib`, `pyyaml`, `pygam` và gói `floodbench` của repo.
- Bốn test so sánh trực tiếp với `floodbench.models.stat` (cần `scikit-learn`) và với `statsmodels` sẽ tự **bỏ qua** nếu máy thiếu hai thư viện đó. Môi trường đủ theo `requirements.txt` thì cả 13 test đều chạy.
- Kết quả không phụ thuộc số ngẫu nhiên: pyGAM khớp mô hình xác định, không có seed.
