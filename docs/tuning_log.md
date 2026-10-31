# Nhật ký tinh chỉnh (tuning log)

Mục đích: chứng minh các mô hình được tinh chỉnh với ngân sách ngang nhau. Ghi mỗi lần chạy tìm tham số.

| Ngày | Mô hình | Lưới tham số đã thử | Số tổ hợp | Cách chọn (CV nào) | Tham số chọn |
|---|---|---|:-:|---|---|
| 2026-10-31 | GAM (pyGAM: s(doy) + l(a_t) + l(a_t-1), mục tiêu anomaly) | (n_splines, lam) ∈ {(6;0,1), (6;1), (8;0,6), (8;5), (12;1), (12;10)} | 6 | Validation trong tập học: 3 mùa cuối, fit trên các mùa trước đó; tối thiểu MSE | Chọn lại ở mỗi fold và mỗi tầm; xem `floodbench.models.stat` |
| 2026-10-31 | LightGBM (3 biến thể khai báo trước: raw, anomaly, noclim) | (num_leaves, n_estimators, lr) ∈ {(4,200,0,05), (4,100,0,05), (8,200,0,05), (4,400,0,03), (2,300,0,05), (8,100,0,1)}; cố định min_child_samples=10, subsample=0,8, seed=0 | 6 mỗi biến thể | Như trên | Chọn lại ở mỗi fold và mỗi tầm |
| 2026-10-31 | Chronos-2 (zero-shot), AnomAR1 và 4 baseline | Không có siêu tham số | 0 | Không áp dụng | amazon/chronos-2, chronos-forecasting 2.3.2, phân vị 0,5 |
