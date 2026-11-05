# Thiện – LightGBM, giới thiệu, tổng quan tài liệu và tài liệu tham khảo

> File này vừa là **danh sách việc** của Thiện, vừa là **nơi viết report** cho phần LightGBM và nháp chương 1–2 (Introduction, Literature Review) của báo cáo. Phần A là việc cần làm; phần B là khung report, viết thẳng vào các mục bên dưới (xóa dòng hướng dẫn in nghiêng khi viết xong).
> Đọc trước: [docs/tong_hop_du_an.md](../../docs/tong_hop_du_an.md) mục 0, 1, 4, 5, 7.

| Hạng mục | Nội dung |
|---|---|
| Mô hình | LightGBM, 3 biến thể khai báo trước: `raw` (y_t, y_t−1, y_t−2, c_t, c_target, ngày đích), `anomaly` (dự báo anomaly từ a_t, a_t−1, ngày đích), `noclim` (như raw nhưng bỏ climatology) |
| Code | `src/floodbench/models/ml.py`, hàm `_make`, lưới `LGBM_GRID`, `SEED`; chọn cấu hình ở `src/floodbench/models/tuning.py` |
| Tên trong kết quả | `lgbm_raw`, `lgbm_anomaly`, `lgbm_noclim` và bản `_w7` |
| Nhánh git | `feature/lightgbm-thien` (tách từ `dev`, PR vào `dev`) |
| Review chéo | Quỳnh |
| Việc chung | Chương 1 Introduction và chương 2 Literature Review (bảng tài liệu, gap); tra DOI toàn bộ tài liệu tham khảo |
| Hạn gợi ý | Hiểu code và chạy lại: 11-08 · Phân tích LightGBM, tra DOI: 11-12 · Report và nháp chương 1–2: 11-14 |

---

## Phần A. Việc cần làm

### A1. Chuẩn bị
- [ ] Cài môi trường theo mục 6.1 của file tổng hợp, chạy `pytest` thấy 45 test pass.
- [ ] Chạy lại: `python -m floodbench.benchmark mekong lgbm_raw,lgbm_anomaly,lgbm_noclim`, rồi `python -m floodbench.report_tables`. Kiểm tra số không đổi (`git diff results/tables`).
- [ ] Đọc hiểu từng dòng: `_raw`, `_anom`, `_noclim`, `_make` (ml.py), `pick_config` (tuning.py), `build_rows` (features.py). Tự trả lời: vì sao có 3 biến thể và vì sao phải khai báo trước? Ba biến thể khác nhau ở đầu vào nào?

### A2. Phân tích riêng
- [ ] In cấu hình được chọn ở từng fold và từng tầm.
- [ ] **Độ quan trọng đặc trưng** (gain hoặc permutation importance) và nếu kịp thì SHAP cho `lgbm_raw`. Đây là phần Explainable AI của outline.
- [ ] **Kiểm tra lợi thế khi nước rút** (16 ngày: 1.321 so với AnomAR1 1.578): lợi thế có đều ở các mùa test hay chỉ do vài mùa? Random Forest cùng đặc trưng lại không có lợi thế này, thử giải thích.
- [ ] **Chạy nhiều seed** (ví dụ 5 seed) để xem kết quả phụ thuộc ngẫu nhiên bao nhiêu (đợt thử cũ: độ lệch chuẩn RMSE chỉ 0,3–0,7%). Báo trung bình ± độ lệch chuẩn.
- [ ] Giải thích vì sao lợi thế ở mùa lũ nhỏ biến mất khi chỉ học 7 mùa (cỡ lũ lẫn với lượng dữ liệu học).
- [ ] Giải thích vì sao cây không ngoại suy được và hệ quả với năm lũ cực trị.

Code phân tích để ở `notebooks/thien_lightgbm.ipynb`, hình lưu `results/figures/lightgbm/`. Hàm dùng lại được thì đưa vào `src/floodbench/`, kèm test.

### A3. Lấy số cho report (dán vào notebook)
```python
import pandas as pd
M, d = ["lgbm_raw", "lgbm_anomaly", "lgbm_noclim"], "results/tables/"
for f in ["probe_rmse", "probe_by_phase_h2", "probe_by_magnitude_h2", "probe_window7", "probe_window7_by_magnitude_h2"]:
    t = pd.read_csv(d + f + ".csv")
    print(f); print(t[t.model.str.removesuffix("_w7").isin(M + ["anom_ar1"])].to_string(index=False))
t = pd.read_csv(d + "probe_vs_anom_ar1.csv")
print(t[t.model.isin(M)].to_string(index=False))   # tỉ lệ RMSE, CI, p Holm, p TOST
```

### A4. Việc chung
- [ ] **Tra DOI** mọi tài liệu ở mục 14 file tổng hợp, ưu tiên các dòng ⚠️: mở DOI hoặc trang tạp chí, kiểm tác giả, năm, tập, trang. Tìm tác giả và năm của bài ARIMA + ML sông Hồng. Đọc bản đầy đủ Asif 2025 và Kratzert 2018 trước khi trích. Không ghi Q1/Q2 khi chưa tra Scimago. Kết quả: danh mục chuẩn IEEE ở `reports/references/references.md` (hoặc `.bib`).
- [ ] Tìm thêm 5–10 bài so sánh thống kê và ML trong thủy văn/viễn thám nếu cần cho bảng tổng quan.
- [ ] **Nháp chương 1 Introduction:** bối cảnh ĐBSCL, động lực, gap, câu hỏi nghiên cứu, đóng góp; insight chính ngay đầu (lấy từ Phu Hung khi có bảng cuối). Nguồn: mục 14–15 file tổng hợp.
- [ ] **Nháp chương 2 Literature Review:** bảng so sánh tài liệu (bài toán, dữ liệu, mô hình, cách đánh giá, kết luận, thiếu gì) và bảng gap. Viết bằng lời của nhóm (Turnitin ≤ 10%).
- [ ] Viết ở Phần C bên dưới.

### A5. Nộp
- [ ] Report viết xong ở Phần B, mọi số ghi rõ file nguồn trong `results/tables/`.
- [ ] Nếu đổi lưới: vẫn 6 cấu hình mỗi biến thể, ghi `docs/tuning_log.md`, chạy lại và báo cả nhóm. Chạy nhiều seed là phân tích độ ổn định, không dùng để chọn seed tốt nhất.
- [ ] PR vào `dev`, Quỳnh review.

---

## Phần B. Report: LightGBM

### B1. Mô hình là gì
*Giải thích dễ hiểu (nhiều cây nhỏ, mỗi cây sửa lỗi của các cây trước), công thức, ví dụ.*

### B2. Vì sao chọn và thuộc họ nào
*ML; vì sao chọn họ cây/boosting cho dữ liệu bảng cỡ vừa (Grinsztajn 2022); điểm mạnh/yếu lý thuyết; vì sao không ngoại suy.*

### B3. Cài đặt trong repo
*3 biến thể và đầu vào; lưới 6 cấu hình; tham số cố định (min_child_samples, subsample, seed); cấu hình được chọn ở mỗi fold.*

### B4. Kết quả
*RMSE 8/16/32 ngày cho 3 biến thể; so với AnomAR1 (tỉ lệ, số mùa thắng, CI, p Holm, p TOST); theo giai đoạn lũ; theo cỡ lũ; với cửa sổ 7 mùa. Ghi nguồn từng bảng.*

### B5. Chẩn đoán
*Độ quan trọng đặc trưng / SHAP; lợi thế khi nước rút theo từng mùa; độ ổn định theo seed; 2 mùa tiêu biểu.*

### B6. Điểm mạnh, điểm yếu, hạn chế và cách nói an toàn

### B7. Câu hỏi vấn đáp tự chuẩn bị
*Ít nhất 5 câu về LightGBM (ý nghĩa từng tham số, bỏ đi thì sao) và các câu 1–8 ở mục 18 của file tổng hợp.*

---

## Phần C. Nháp chương 1–2 và tài liệu tham khảo

### C1. Introduction
*Bối cảnh, động lực, gap, câu hỏi nghiên cứu, đóng góp, insight chính.*

### C2. Literature Review
*Bảng so sánh tài liệu; tài liệu nói gì; gap.*

### C3. Trạng thái tra DOI
*Tài liệu nào đã kiểm, tài liệu nào bỏ vì không xác nhận được.*
