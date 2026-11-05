# Phu Hung – Baseline, ARX, Ridge, Random Forest, Chronos-2, khung đánh giá và tổng hợp cuối

> File này vừa là **danh sách việc** của Phu Hung, vừa là **nơi viết report** cho các mô hình phụ trách. Phần A là việc cần làm; phần B là khung report, viết thẳng vào các mục bên dưới (xóa dòng hướng dẫn in nghiêng khi viết xong). Bảng so sánh cuối và báo cáo insight viết ở `reports/full_report/`.
> Đọc trước: [docs/tong_hop_du_an.md](../../docs/tong_hop_du_an.md) toàn bộ.

| Hạng mục | Nội dung |
|---|---|
| Mô hình | Baseline (persistence, clim_all, clim5, seasonal_naive); ARX (OLS); Ridge có trễ; Random Forest (raw, anomaly, noclim); Chronos-2 zero-shot |
| Code | `models/baselines.py`, `models/stat.py` (`arx`, `ridge`), `models/forest.py`, `models/ml.py` (`Chronos2Zero`); khung chung: `features.py`, `cv.py`, `tuning.py`, `evaluation.py`, `inference.py`, `benchmark.py`, `report_tables.py`, `report_md.py` |
| Tên trong kết quả | `persistence`, `clim_all`, `clim5`, `seasonal_naive`, `arx`, `ridge`, `rf_raw`, `rf_anomaly`, `rf_noclim`, `chronos2` và bản `_w7` |
| Nhánh git | `feature/eval-phuhung` (tách từ `dev`, PR vào `dev`) |
| Review chéo | Bình |
| Hạn gợi ý | Report mô hình: 11-14 · Bảng so sánh cuối và insight: 11-17 · Báo cáo đầy đủ: 11-22 |

---

## Phần A. Việc cần làm

### A1. Giữ ổn định khung chung
- [ ] Không đổi `features.py`, `cv.py`, `tuning.py`, `evaluation.py`, `inference.py` khi các thành viên đang chạy. Nếu bắt buộc đổi: ghi `docs/decisions.md`, báo cả nhóm, chạy lại **mọi** mô hình.
- [ ] Review PR của Bình; gộp PR các thành viên vào `dev`.

### A2. Phân tích các mô hình phụ trách
- [ ] ARX, Ridge: cấu hình chọn ở từng fold; hệ số; chẩn đoán phần dư (ACF, Ljung–Box, Q–Q); diễn giải kết quả tương đương AnomAR1 ở 8 ngày.
- [ ] Random Forest: cấu hình chọn; độ quan trọng đặc trưng; vì sao RF raw kém LightGBM raw cùng đặc trưng; vì sao RF raw tốt nhất khi nước lên nhưng tệ nhất khi nước rút; chạy vài seed.
- [ ] Chronos-2: vì sao kém ở 32 ngày; thử độ nhạy độ dài ngữ cảnh nếu có thời gian (chỉ báo như độ nhạy, không chọn lại trên test).
- [ ] Baseline: giải thích vì sao clim_all kém nặng (xu hướng giảm của đỉnh lũ).

### A3. Bổ sung kiểm định và độ nhạy (cho bảng cuối)
- [ ] Block bootstrap theo mùa cho CI của chênh lệch RMSE (viết test với đáp án biết trước trước khi dùng).
- [ ] Friedman + Nemenyi trên các mùa.
- [ ] Độ nhạy: bỏ các cặp dự báo chứa ảnh nội suy hoặc nghi bản sao; bản 13 mùa test (2008–2020).
- [ ] Learning curve: số mùa huấn luyện 5, 7, 9, 11, … cho AnomAR1, GAM, LightGBM, RF.
- [ ] Chạy lại kiểm tra biến giả lưới 500 m trên dữ liệu đã quy lưới bằng mean.

### A4. Tổng hợp cuối
- [ ] Gộp kết quả và report của Quỳnh, Bình, Thiện (`reports/members/`).
- [ ] Lập bảng so sánh cuối (sinh bằng code từ `results/tables/`), theo điều kiện: tầm, giai đoạn, cỡ lũ, lượng dữ liệu.
- [ ] Viết insight cho đầu và cuối báo cáo; đối chiếu tiêu chí "đáng làm" (mục 8 file tổng hợp).
- [ ] Viết chương 5–7 (Models & Experiments, Results & Discussion, Conclusion) và ghép chương 1–4 của Thiện, Bình, Quỳnh vào `reports/full_report/`; bài IEEE và slide cùng cả nhóm.

### A5. Lấy số cho report (dán vào notebook)
```python
import pandas as pd
M = ["arx", "ridge", "rf_raw", "rf_anomaly", "rf_noclim", "chronos2", "persistence", "clim_all", "clim5", "seasonal_naive"]
d = "results/tables/"
for f in ["probe_rmse", "probe_by_phase_h2", "probe_by_magnitude_h2", "probe_window7", "probe_window7_by_magnitude_h2"]:
    t = pd.read_csv(d + f + ".csv")
    print(f); print(t[t.model.str.removesuffix("_w7").isin(M + ["anom_ar1"])].to_string(index=False))
t = pd.read_csv(d + "probe_vs_anom_ar1.csv")
print(t[t.model.isin(M)].to_string(index=False))
```
Code phân tích để ở `notebooks/phuhung_*.ipynb`, hình lưu `results/figures/<mô_hình>/`.

---

## Phần B. Report các mô hình phụ trách

Mỗi mô hình viết theo cùng thứ tự: (1) mô hình là gì, (2) vì sao chọn và thuộc họ nào, (3) cài đặt trong repo, (4) kết quả, (5) chẩn đoán, (6) hạn chế và cách nói an toàn, (7) câu hỏi vấn đáp.

### B1. Baseline (persistence, climatology toàn bộ, climatology 5 mùa, seasonal naive)

### B2. ARX (OLS)

### B3. Ridge có trễ

### B4. Random Forest (raw, anomaly, noclim)

### B5. Chronos-2 zero-shot

### B6. Ghi chú cho bảng so sánh cuối
*Những điểm rút ra khi đặt các mô hình trên cạnh AnomAR1, GAM, LightGBM; chuyển sang `reports/full_report/` khi viết báo cáo cuối.*
