# Quỳnh – AnomAR1, Level 3, EDA và thống kê suy diễn

> File này vừa là **danh sách việc** của Quỳnh, vừa là **nơi viết report** cho phần AnomAR1 và nháp chương 4 (Descriptive + Inferential) của báo cáo. Phần A là việc cần làm; phần B là khung report, viết thẳng vào các mục bên dưới (xóa dòng hướng dẫn in nghiêng khi viết xong).
> Đọc trước: [docs/tong_hop_du_an.md](../../docs/tong_hop_du_an.md) mục 0, 1, 4, 7 và 12.1.

| Hạng mục | Nội dung |
|---|---|
| Mô hình | AnomAR1: ŷ(t+h) = c(t+h) + φ^h × [y(t) − c(t)] |
| Code | `src/floodbench/models/baselines.py`, hàm `anom_ar1`; climatology ở `src/floodbench/features.py` |
| Tên trong kết quả | `anom_ar1`, `anom_ar1_w7` |
| Nhánh git | `feature/anomar1-quynh` (tách từ `dev`, PR vào `dev`) |
| Review chéo | Thiện |
| Việc chung | Dựng bảng Level 3 (Bình dùng cho hồi quy); EDA và thống kê suy diễn = chương 4 của báo cáo |
| Hạn gợi ý | Hiểu code và chạy lại: 11-08 · **Level 3: 11-10** · Phân tích AnomAR1 và EDA: 11-12 · Report xong: 11-14 |

---

## Phần A. Việc cần làm

### A1. Chuẩn bị
- [ ] Cài môi trường theo mục 6.1 của file tổng hợp, chạy `pytest` thấy 45 test pass.
- [ ] Chạy lại mô hình của mình: `python -m floodbench.benchmark mekong anom_ar1`, rồi `python -m floodbench.report_tables`. Kiểm tra số không đổi so với trước (`git diff results/tables`).
- [ ] Đọc hiểu từng dòng: `anom_ar1` (baselines.py), `climatology` và `build_rows` (features.py), `rolling_origin` (cv.py). Tự trả lời: vì sao climatology của mùa i chỉ dùng các mùa trước i? φ được ước lượng từ những hàng nào?

### A2. Phân tích riêng
- [ ] In φ ước lượng ở từng fold (11 năm test). φ có ổn định theo thời gian không?
- [ ] Chẩn đoán phần dư: ACF của sai số trong mùa, Ljung–Box, Q–Q. Ghi rõ vi phạm nếu có (sai số h bước vốn có dạng MA(h − 1)).
- [ ] So sánh φ^h với φ ước lượng riêng cho từng tầm (đã có số sơ bộ: 1.088 / 1.356 / 1.675 so với 1.090 / 1.375 / 1.687). Không đổi định nghĩa chính, chỉ báo như một độ nhạy.
- [ ] **Truy nguyên chênh lệch 1.602 so với 1.687 km² ở 32 ngày** (file tổng hợp mục 7.4 và 12.1). Manh mối: persistence và Clim5 cũng lệch, nên kiểm tra tập cặp dự báo (chuỗi đầu vào, khoảng bước gốc t). Script cũ nằm trong vault Obsidian `03_Resources/Code_Snippets/STAT3013_Flood_pilot_benchmark.py`. Ghi kết luận vào `docs/decisions.md`.
- [ ] Vẽ 2 mùa tiêu biểu (một mùa lũ lớn, một mùa lũ thấp như 2015): quan sát và dự báo ở 8/16/32 ngày (có sẵn ở tab "Dự báo từng mùa" của `streamlit run app/streamlit_app.py`).

Code phân tích để ở `notebooks/quynh_anomar1.ipynb`, hình lưu `results/figures/anomar1/`. Hàm dùng lại được thì đưa vào `src/floodbench/`, kèm test.

### A3. Lấy số cho report (dán vào notebook)
```python
import pandas as pd
M, d = ["anom_ar1"], "results/tables/"
for f in ["probe_rmse", "probe_by_phase_h2", "probe_by_magnitude_h2", "probe_window7", "probe_window7_by_magnitude_h2"]:
    t = pd.read_csv(d + f + ".csv")
    print(f); print(t[t.model.str.removesuffix("_w7").isin(M)].to_string(index=False))
```
AnomAR1 là mốc so sánh nên không có dòng riêng trong `probe_vs_anom_ar1.csv`; các mô hình khác so với nó.

### A4. Việc chung: Level 3, EDA và thống kê suy diễn (chương 4)
- [ ] **Dựng bảng Level 3** `data/processed/mekong/level3_yearly.csv` (21 dòng, một dòng một mùa): đỉnh diện tích ngập, ngày đạt đỉnh, diện tích trung bình mùa, và diện tích từng lớp đất (thủy sản, lúa 3 vụ, lúa 2 vụ…). Đỉnh và trung bình lấy từ `level1_series.csv`; cơ cấu đất cần đọc 21 ảnh sử dụng đất trong `data/raw/mekong/` (cắt cùng mặt nạ ĐBSCL). Code đặt ở `src/floodbench/data/level3.py`, đọc cấu hình vùng, có test số dòng. **Bình cần bảng này trước 11-10.**
- [ ] **Thống kê mô tả:** phân phối diện tích ngập, độ lệch (skewness), boxplot theo năm và theo giai đoạn lũ, heatmap mùa × bước, 21 đường cong mùa chồng nhau, đỉnh lũ theo năm.
- [ ] **Suy diễn:** xu hướng đỉnh lũ (Mann–Kendall, OLS với sai số HC3); so sánh trước/sau 2010 (Mann–Whitney); Levene cho phương sai; Kruskal–Wallis theo nhóm năm lũ lớn/vừa/nhỏ. Diễn giải "phù hợp với", không nhân quả.
- [ ] Notebook `notebooks/02_eda_descriptive_inferential.ipynb`, hình lưu `results/figures/eda/`, bảng kết quả lưu `results/tables/eda_*.csv` (sinh bằng code).
- [ ] Viết ở Phần C bên dưới.

### A5. Nộp
- [ ] Report viết xong ở Phần B, mọi số ghi rõ file nguồn trong `results/tables/`.
- [ ] PR vào `dev`, Thiện review.

---

## Phần B. Report: AnomAR1

### B1. Mô hình là gì
*Giải thích dễ hiểu, công thức, một ví dụ bằng số (ví dụ hôm nay lệch +1.000 km², φ = 0,9, 16 ngày sau còn bao nhiêu).*

### B2. Vì sao chọn và thuộc họ nào
*Họ thống kê (AR trên anomaly). Giả định. Vì sao là mốc khó thắng (tự tương quan anomaly ≈ 0,89). Vì sao dùng climatology 5 mùa thay vì mọi mùa.*

### B3. Cài đặt trong repo
*Hàm, đầu vào, cách ước lượng φ (hồi quy qua gốc trên các hàng huấn luyện), không có siêu tham số nên không tune.*

### B4. Kết quả
*RMSE 8/16/32 ngày; so với baseline; theo giai đoạn lũ; theo cỡ lũ; với cửa sổ 7 mùa. Ghi nguồn từng bảng.*

### B5. Chẩn đoán
*φ theo fold; ACF, Ljung–Box, Q–Q của phần dư; φ^h so với φ trực tiếp; 2 mùa tiêu biểu.*

### B6. Chênh lệch 1.602 so với 1.687
*Đã tìm ra nguyên nhân chưa; nếu có, sửa ở đâu và kết quả mới.*

### B7. Hạn chế và cách nói an toàn

### B8. Câu hỏi vấn đáp tự chuẩn bị
*Ít nhất 5 câu về AnomAR1 và các câu 1–8 ở mục 18 của file tổng hợp.*

---

## Phần C. Nháp chương 4: Thống kê mô tả và suy diễn

### C1. Bảng Level 3
*Cách dựng, các cột, nguồn từng cột.*

### C2. Thống kê mô tả
*Phân phối, độ lệch, boxplot, heatmap, đường cong mùa; nhận xét ngắn dưới mỗi hình.*

### C3. Xu hướng đỉnh lũ
*Mann–Kendall, OLS (HC3): hệ số, CI, p; diễn giải "phù hợp với".*

### C4. So sánh nhóm
*Trước/sau 2010, Levene, Kruskal theo nhóm năm; giả thuyết, kiểm định, kết luận.*

### C5. Hạn chế
*n = 21 mùa, dữ liệu là sản phẩm suy diễn, phân loại đất "nhảy" giữa các năm.*
