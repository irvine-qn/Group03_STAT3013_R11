# Bình – GAM, hồi quy Level 3, chương Dữ liệu và phương pháp, câu hỏi cho thầy

> File này vừa là **danh sách việc** của Bình, vừa là **nơi viết report** cho phần GAM, phần hồi quy Level 3 và nháp chương 3 (Data & Methodology) của báo cáo. Phần A là việc cần làm; phần B là khung report, viết thẳng vào các mục bên dưới (xóa dòng hướng dẫn in nghiêng khi viết xong).
> Đọc trước: [docs/tong_hop_du_an.md](../../docs/tong_hop_du_an.md) mục 0, 1, 4, 5, 7.

| Hạng mục | Nội dung |
|---|---|
| Mô hình | GAM dự báo anomaly: s(ngày đích) + l(a_t) + l(a_t−1), cộng lại climatology |
| Code | `src/floodbench/models/stat.py`, hàm `gam`, lưới `GAM_GRID`; chọn cấu hình ở `src/floodbench/models/tuning.py` |
| Tên trong kết quả | `gam`, `gam_w7` |
| Nhánh git | `feature/gam-binh` (tách từ `dev`, PR vào `dev`) |
| Review chéo | Phu Hung |
| Việc chung | Hồi quy đỉnh lũ theo cơ cấu đất (Level 3); chương 3 Data & Methodology; gửi câu hỏi cho thầy |
| Hạn gợi ý | **Gửi câu hỏi cho thầy: 11-07** · Hiểu code và chạy lại: 11-08 · Phân tích GAM và hồi quy: 11-12 · Report xong: 11-14 |

---

## Phần A. Việc cần làm

### A1. Chuẩn bị
- [ ] Cài môi trường theo mục 6.1 của file tổng hợp, chạy `pytest` thấy 45 test pass.
- [ ] Chạy lại mô hình của mình: `python -m floodbench.benchmark mekong gam`, rồi `python -m floodbench.report_tables`. Kiểm tra số không đổi (`git diff results/tables`).
- [ ] Đọc hiểu từng dòng: `gam` và `_fit_predict` (stat.py), `pick_config` (tuning.py), `build_rows` (features.py). Tự trả lời: vì sao GAM dự báo anomaly chứ không dự báo y? Cấu hình được chọn bằng dữ liệu nào?

### A2. Phân tích riêng
- [ ] In cấu hình (n_splines, lam) được chọn ở từng fold và từng tầm. Cấu hình có phân tán không? (Đợt thử cũ: phân tán, nghĩa là khác biệt giữa cấu hình nhỏ so với nhiễu.)
- [ ] Vẽ hàm trơn s(ngày đích) và hệ số của hai trễ cho một fold tiêu biểu. Diễn giải: anomaly phai nhanh hay chậm ở đầu, giữa, cuối mùa?
- [ ] Chẩn đoán phần dư: ACF trong mùa, Q–Q, phần dư theo ngày.
- [ ] Diễn giải kết quả tương đương với AnomAR1 ở 8 ngày (TOST ±5%). Vì sao GAM gần như trùng AnomAR1? Ở 16 và 32 ngày có tương đương không?
- [ ] Giải thích vì sao GAM nằm gần ranh giới thống kê/ML (thầy sẽ hỏi).
- [ ] So với cửa sổ 7 mùa: GAM có ổn định hơn LightGBM khi ít dữ liệu không?

Code phân tích để ở `notebooks/binh_gam.ipynb`, hình lưu `results/figures/gam/`. Hàm dùng lại được thì đưa vào `src/floodbench/`, kèm test.

### A3. Lấy số cho report (dán vào notebook)
```python
import pandas as pd
M, d = ["gam"], "results/tables/"
for f in ["probe_rmse", "probe_by_phase_h2", "probe_by_magnitude_h2", "probe_window7", "probe_window7_by_magnitude_h2"]:
    t = pd.read_csv(d + f + ".csv")
    print(f); print(t[t.model.str.removesuffix("_w7").isin(M + ["anom_ar1"])].to_string(index=False))
t = pd.read_csv(d + "probe_vs_anom_ar1.csv")
print(t[t.model.isin(M)].to_string(index=False))   # tỉ lệ RMSE, CI, p Holm, p TOST
```

### A4. Việc chung
- [ ] **Gửi thầy các câu hỏi mở** (mục 11 file tổng hợp) trước 11-07, ưu tiên: SEM, số phương pháp bắt buộc, biến phụ, mở rộng vùng, "mô hình truyền thống" gồm những gì, kết luận "ML không hơn thống kê" có hợp lệ không. Ghi câu trả lời (hoặc "thầy chưa trả lời" + giả định tạm) vào `docs/decisions.md` và báo cả nhóm.
- [ ] **Hồi quy Level 3** (dùng bảng Level 3 của Quỳnh, có trước 11-10): đỉnh lũ ~ năm + diện tích thủy sản + diện tích lúa 3 vụ, n = 21. Kiểm tra đủ 5 giả định (tuyến tính; độc lập: Durbin–Watson; phương sai đều: Breusch–Pagan; chuẩn: Q–Q, Shapiro; đa cộng tuyến: VIF). Biến đất tăng cùng năm nên đa cộng tuyến rất có thể cao; báo rõ và thử bỏ bớt biến. Diễn giải "phù hợp với", không nhân quả. Code ở `notebooks/binh_regression_level3.ipynb`, bảng `results/tables/regression_level3.csv`.
- [ ] **Nháp chương 3 Data & Methodology:** dataset và cách bản đồ MODIS được tạo (Sakamoto 2007, Vu 2022), pipeline cắt mặt nạ và quy lưới bằng mean, cờ chất lượng, định nghĩa mục tiêu, giao thức so sánh (rolling-origin, chống rò rỉ, ngân sách tinh chỉnh, kiểm định cấp mùa, TOST, Holm). Nguồn: mục 3–4 file tổng hợp và `docs/decisions.md`.
- [ ] Viết ở Phần C, D, E bên dưới.

### A5. Nộp
- [ ] Report viết xong ở Phần B, mọi số ghi rõ file nguồn trong `results/tables/`.
- [ ] Nếu đổi lưới: vẫn 6 cấu hình, ghi `docs/tuning_log.md`, chạy lại và báo cả nhóm.
- [ ] PR vào `dev`, Phu Hung review.

---

## Phần B. Report: GAM

### B1. Mô hình là gì
*Giải thích dễ hiểu (hồi quy cộng tính, mỗi biến một hàm trơn), công thức, ví dụ.*

### B2. Vì sao chọn và thuộc họ nào
*Thống kê có suy diễn nhưng linh hoạt; giả định; vì sao dùng spline cho ngày và hạng tuyến tính cho trễ.*

### B3. Cài đặt trong repo
*Hàm, đầu vào, mục tiêu anomaly, lưới 6 cấu hình, cấu hình được chọn ở mỗi fold.*

### B4. Kết quả
*RMSE 8/16/32 ngày; so với AnomAR1 (tỉ lệ, số mùa thắng, CI, p Holm, p TOST); theo giai đoạn lũ; theo cỡ lũ; với cửa sổ 7 mùa. Ghi nguồn từng bảng.*

### B5. Chẩn đoán
*Đồ thị hàm trơn, hệ số trễ, phần dư, 2 mùa tiêu biểu.*

### B6. Điểm mạnh, điểm yếu, hạn chế và cách nói an toàn

### B7. Câu hỏi vấn đáp tự chuẩn bị
*Ít nhất 5 câu về GAM và các câu 1–8 ở mục 18 của file tổng hợp.*

---

## Phần C. Hồi quy đỉnh lũ theo cơ cấu đất (Level 3)

### C1. Mô hình và biến
### C2. Kết quả (hệ số, CI, p, R²)
### C3. Kiểm tra 5 giả định
### C4. Diễn giải và hạn chế (n = 21, đa cộng tuyến, không nhân quả)

---

## Phần D. Nháp chương 3: Data & Methodology

### D1. Dữ liệu và cách tạo bản đồ lũ
### D2. Từ ảnh đến chuỗi số (cắt mặt nạ, quy lưới, cờ chất lượng)
### D3. Bài toán dự báo và biến mục tiêu
### D4. Giao thức so sánh công bằng
### D5. Thước đo và kiểm định

---

## Phần E. Câu hỏi đã gửi thầy và câu trả lời
*Ngày gửi, câu hỏi, câu trả lời; chép kết luận sang `docs/decisions.md`.*
