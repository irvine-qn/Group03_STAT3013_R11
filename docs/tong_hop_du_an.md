# Tổng hợp dự án và phân công – STAT3013, Nhóm 03, R11

> Tài liệu chung cho cả nhóm: dự án là gì, dữ liệu ra sao, đã làm gì, kết quả thăm dò đến đâu, ai làm gì tiếp theo, và chỗ nào chưa chắc.
> **Ảnh chụp ngày 2026-11-05.** Số liệu ở đây là bản chép lại từ `results/tables/` tại ngày này. Khi code hoặc dữ liệu đổi, **`results/` mới là nguồn đúng**; chạy lại theo mục 6 rồi đối chiếu. Báo cáo cuối không trích số từ file này mà trích từ `results/`.
> Quy ước: "đã làm" nghĩa là có mã, có test hoặc có bảng sinh từ mã; "dự kiến" nghĩa là mới nằm trong kế hoạch. Mọi điểm chưa chắc gom ở mục 12.
> Thầy có thể đọc repo, nên mọi ghi chú ở đây viết như tài liệu làm việc nghiêm túc.

---

## 0. Đọc nhanh trong 1 phút

- **Đề tài:** so sánh công bằng mô hình thống kê và Machine Learning khi dự báo diện tích ngập (km²) ở Đồng bằng sông Cửu Long sau 8, 16 và 32 ngày, và báo cáo **theo điều kiện** (tầm dự báo, giai đoạn lũ, cỡ lũ, lượng dữ liệu học), không tuyên bố "mô hình X tốt nhất".
- **Dữ liệu:** bản đồ lũ MODIS của Vu et al. (2022), đã gộp thành một chuỗi số 21 mùa × 27 bước (8 ngày/bước). Dùng được nhưng là sản phẩm suy diễn, và chỉ đối chiếu được với mực nước ở cấp năm (r ≈ 0,67 sau khi bỏ xu hướng).
- **Đã có:** pipeline dữ liệu, khung đánh giá không rò rỉ (rolling-origin theo năm, 11 mùa test 2010–2020), kiểm định cấp mùa, 5 baseline và 8 mô hình (AnomAR1, GAM, ARX, Ridge, LightGBM, Random Forest, Chronos-2 và các biến thể), 45 test, báo cáo tự sinh và app Streamlit.
- **Kết quả thăm dò:** chưa mô hình nào thắng mốc thống kê AnomAR1 một cách có ý nghĩa; ARX, Ridge và GAM tương đương AnomAR1 ở 8 ngày; LightGBM nhỉnh về số học ở 8–16 ngày và tốt hơn khi nước rút nhưng kém ở đỉnh lũ; lợi thế của ML biến mất khi chỉ có 7 mùa huấn luyện. **Hướng đi: báo cáo theo điều kiện.**
- **Phân công (mục 1):** AnomAR1 – Quỳnh; GAM – Bình; LightGBM – Thiện; các mô hình còn lại, khung đánh giá, bảng so sánh cuối và báo cáo insight – Phu Hung.
- **Tra cứu:** dữ liệu (3), giao thức và thước đo (4), mô hình và tham số (5), lệnh chạy (6), kết quả và các lần thử trước (7), rủi ro (12), tài liệu tham khảo và gap (14), khung báo cáo (15), mở rộng vùng (16), từ điển (17), câu hỏi vấn đáp (18), kiến thức theo người (19).

---

## 1. Phân công và cách làm việc

### 1.1 Phạm vi đã chốt (tạm thời)
Nhóm chốt so sánh **bộ mô hình đã chạy ở đợt thăm dò** (mục 4): 5 baseline, AnomAR1, GAM, ARX, Ridge, LightGBM (3 biến thể), Random Forest (3 biến thể), Chronos-2. Các mô hình khác (SARIMAX, ETS, SVR, MLP, LSTM, TabPFN…) để sau, chỉ làm nếu dư thời gian.

### 1.2 Ai làm gì

| Người | Mô hình | Việc riêng nên làm (ngoài khung chung ở 1.3) |
|---|---|---|
| **Quỳnh** | AnomAR1 (mốc thống kê chính) | Giải thích công thức và φ; in φ ước lượng ở từng fold; chẩn đoán phần dư (ACF, Ljung–Box); so sánh φ^h với φ ước lượng riêng từng tầm; truy nguyên chênh lệch 1.602 so với 1.687 km² ở 32 ngày (mục 12.1) |
| **Bình** | GAM | Vẽ hàm trơn s(doy) và hệ số hai trễ; cấu hình (n_splines, lam) được chọn ở từng fold; chẩn đoán phần dư; giải thích vì sao GAM gần ranh giới thống kê/ML; diễn giải kết quả tương đương TOST ở 8 ngày |
| **Thiện** | LightGBM (raw, anomaly, noclim) | Giải thích 3 biến thể và vì sao khai báo trước; cấu hình được chọn ở từng fold; độ quan trọng đặc trưng hoặc SHAP (đáp ứng mục Explainable AI của outline); kiểm tra lợi thế khi nước rút có ổn định theo từng mùa không; chạy vài seed để xem kết quả có phụ thuộc ngẫu nhiên không |
| **Phu Hung** | Baseline, ARX, Ridge, Random Forest, Chronos-2; khung đánh giá | Giữ ổn định `features.py`, `cv.py`, `tuning.py`, `evaluation.py`, `inference.py`; gộp kết quả; lập bảng so sánh cuối và viết report cuối với insight |

### 1.3 Khung chung cho mọi người (để so sánh còn công bằng)

Mô hình của mỗi người **đã có sẵn trong repo** và đã chạy trên cùng giao thức. Việc của từng người không phải viết lại từ đầu theo cách riêng, mà là **chạy lại, hiểu từng dòng, phân tích sâu và viết report** cho mô hình của mình. Nếu mỗi người tự chia dữ liệu, tự chọn đặc trưng hay tự tune theo kiểu khác thì bảng so sánh cuối không còn công bằng.

1. **Không đổi giao thức** khi chưa thống nhất cả nhóm: cách chia (rolling-origin, test 2010–2020), đặc trưng (`features.py`), cách chọn cấu hình (`tuning.pick_config`, 3 mùa cuối của tập học làm validation), thước đo và kiểm định. Muốn đổi thì mở issue/PR, ghi vào `docs/decisions.md`, và chạy lại **mọi** mô hình.
2. **Ngân sách tinh chỉnh bằng nhau:** 6 cấu hình mỗi mô hình (mỗi biến thể). Đổi lưới thì vẫn 6 cấu hình và ghi vào `docs/tuning_log.md`. Không bao giờ chọn tham số bằng điểm trên mùa test.
3. **Kết quả sinh bằng code, không làm bảng tay:**
   `python -m floodbench.benchmark mekong <tên_mô_hình>` chỉ tính lại mô hình đó và ghi vào `results/tables/probe_predictions.csv`; sau đó `python -m floodbench.report_tables` sinh lại mọi bảng. Bảng output cuối của từng mô hình là các dòng của mô hình đó trong `results/tables/probe_*.csv`.
4. **Report từng mô hình** đặt ở `reports/models/<mô_hình>.md`, theo khung chung ở 1.4. Mọi con số phải tra lại được trong `results/tables/` (ghi rõ tên file nguồn).
5. **Git:** tạo nhánh `feature/<mô_hình>-<tên>` từ `dev`, mở Pull Request vào `dev`, có ít nhất một người khác review. Không commit `data/raw/`. Cặp review chéo gợi ý: Quỳnh ↔ Thiện, Bình ↔ Phu Hung.
6. **Vấn đáp:** thầy hay hỏi code từng dòng, nên ngoài mô hình của mình, mỗi người phải giải thích được toàn bộ pipeline (dữ liệu → đặc trưng → chia fold → tune → dự báo → kiểm định). Mã của đợt thăm dò được viết với sự hỗ trợ của AI agent, nên càng cần từng người đọc kỹ chứ không chỉ tin kết quả.

### 1.4 Khung report cho từng mô hình (`reports/models/<mô_hình>.md`)
1. **Mô hình là gì:** giải thích dễ hiểu, công thức, ví dụ bằng số.
2. **Vì sao chọn, thuộc họ nào:** thống kê hay ML, giả định của mô hình, điểm mạnh/yếu về lý thuyết.
3. **Cài đặt trong repo:** file và hàm, đầu vào (đặc trưng), mục tiêu (y hay anomaly), lưới 6 cấu hình, cấu hình được chọn ở mỗi fold.
4. **Kết quả:** RMSE ở 8/16/32 ngày; so với AnomAR1 (tỉ lệ RMSE, số mùa thắng, CI của chênh MSE, p sau Holm, p TOST); theo giai đoạn lũ; theo cỡ lũ; với cửa sổ 7 mùa.
5. **Chẩn đoán riêng:** phần dư (mô hình thống kê) hoặc độ quan trọng đặc trưng (mô hình ML); vẽ 1–2 mùa tiêu biểu (ví dụ 2011 lũ lớn, 2015 lũ thấp nhất).
6. **Hạn chế và cách nói an toàn** (mục 7.5).
7. **Câu hỏi vấn đáp tự chuẩn bị** và câu trả lời.

### 1.5 Lịch gợi ý (nhóm chốt lại)

| Mốc | Việc |
|---|---|
| đến 11-08 | Mỗi người cài môi trường, chạy lại được `pytest` và mô hình của mình, đọc hiểu code |
| đến 11-12 | Phân tích riêng (mục 1.2), PR code/biểu đồ nếu có |
| đến 11-14 | PR report từng mô hình theo khung 1.4 |
| 11-15 → 11-17 | Phu Hung gộp, lập bảng so sánh cuối và insight; cả nhóm review |
| 11-18 → 11-22 | Viết báo cáo đầy đủ, bài IEEE, slide, video; tập vấn đáp |
| 11-23 | Báo cáo |

### 1.6 Việc chưa có người nhận
- **EDA, thống kê mô tả và suy diễn, hồi quy Level 3** (outline môn yêu cầu, xem mục 10).
- Các chương chung của báo cáo: giới thiệu, tổng quan tài liệu và gap, dữ liệu và phương pháp.
- Tra DOI tài liệu tham khảo (mục 12.5).
- Gửi thầy các câu hỏi mở (mục 11).
- Điền bảng thành viên (họ tên, MSSV) trong README.

---

## 2. Dự án là gì

| Hạng mục | Nội dung |
|---|---|
| Môn | STAT3013 – Phân tích thống kê, UIT, lớp R11, Nhóm 03, giảng viên thầy Triều |
| Đề tài | So sánh công bằng mô hình thống kê và Machine Learning trong dự báo diện tích ngập (flood extent) |
| Câu hỏi | Mô hình thống kê cạnh tranh được với ML trong điều kiện nào? Mô hình nào hợp với dữ liệu lũ của từng vùng? |
| Cách trả lời | Một giao thức duy nhất cho mọi mô hình, báo cáo **theo điều kiện** |
| Sản phẩm | Báo cáo đầy đủ ~30 trang, bài IEEE 7–10 trang, slide, video demo, link GitHub, Turnitin ≤ 10% |
| Mốc | **Báo cáo dự kiến: 2026-11-23** |

**Yêu cầu của thầy (theo ghi chú nhóm):** đọc bài báo, tìm gap, giải thích vì sao chọn mô hình này mà không phải mô hình kia; làm sâu vì lũ đã có nhiều người làm; dữ liệu từ bài báo uy tín, dữ liệu Việt Nam, không dùng Kaggle; insight ở đầu và cuối báo cáo; hay hỏi vấn đáp code từng dòng nên cả nhóm phải hiểu toàn bộ pipeline; muốn mở rộng sang các vùng khác của Việt Nam.

**Thầy chấm quy trình công bằng và khả năng giải thích, không chấm việc ai thắng.** Kết quả ngược giả thuyết vẫn hợp lệ.

**Hướng đi đã chọn:** dùng bộ Vu et al. (2022) làm dữ liệu chính, tạm dừng tìm dataset khác; chạy thử nhanh một bộ mô hình đại diện để xem có insight đáng làm không (kết quả ở mục 7 và 8).

**Giả thuyết (để kiểm tra, không phải để chứng minh):**
- H1: ở 8–16 ngày, AR đơn giản không kém ML (tự tương quan anomaly ≈ 0,89).
- H2: ở 32 ngày, không mô hình nào vượt climatology có cập nhật xu hướng.
- H3: ML kém ở năm cực trị vì cây không ngoại suy được.
- H4: càng ít dữ liệu, thống kê càng có lợi.
- H5: ở cấp ô lưới, ML có cơ hội thắng rõ hơn (chưa làm).
- H6: lợi thế của họ mô hình phụ thuộc vùng (chưa làm).

---

## 3. Dữ liệu

### 3.1 Ba bộ trong `data/raw/` (không commit; xem `data/raw/README.md` để tải)
1. `mekong/`: ảnh bản đồ lũ MODIS của ĐBSCL, Vu et al. 2022, *Data in Brief* 42:108268, Mendeley DOI 10.17632/kpftzmsyyz.2 (≈ 2,7 GB sau giải nén).
2. `mekong_hydrology/`: mực nước tại các trạm Tân Châu, Mỹ Thuận, dùng để đối chiếu.
3. `haiphong_tide/`: thủy triều Hải Phòng. **Chưa dùng.**

Không cần tải ảnh gốc để chạy mô hình: bảng `data/processed/mekong/level1_series.csv` đã có trong repo.

### 3.2 Bộ ảnh ĐBSCL
- 565 ảnh trên 567 kỳ vọng. Thiếu `2013 ngày 233` và `2019 ngày 257`.
- 27 ảnh mỗi năm, ngày 153 → 361, cách 8 ngày. 21 mùa (2000–2020). Giữa hai mùa trống khoảng 150 ngày.
- Giá trị pixel: 0 không nước/không dữ liệu, 1 nước, 2 hỗn hợp đất ẩm.
- 21 ảnh sử dụng đất (2000–2020).
- Không có mực nước, mưa, nhiệt độ, thiệt hại, DEM, ranh giới tỉnh.

### 3.3 Năm điều cần cẩn thận
1. **Phạm vi quá rộng:** raster phủ ≈ 380.000 km² gồm Campuchia và Tonle Sap; ĐBSCL chỉ ≈ 39.900 km². Phải cắt bằng mặt nạ (≈ 39.871 km²).
2. **Lưới khác nhau:** 232 m và 500 m. Quy về km² và lưới chung bằng **mean**, không dùng nearest-neighbour.
3. **Hai định dạng:** 485 file `.tif` là GeoTIFF; 80 file `.tiff` (2018–2020) là ENVI thô, đọc bằng `.hdr`. Một file tên có dấu cách (`Flood_2017_ 289.tif`).
4. **Chuỗi đứt:** mỗi mùa là một đoạn 27 điểm. Không nối thành chuỗi liên tục để chạy SARIMA.
5. **Mục tiêu là sản phẩm suy diễn:** cây quyết định trên EVI/LSWI của MODIS. Lớp "nước" gồm cả sông, ao nuôi và ruộng ngập nước canh tác. Phải nêu trong mọi bài viết.

### 3.4 Hình dạng dữ liệu (số từ pipeline cũ nearest-neighbour; số đã kiểm lại ở 3.6)
- Đỉnh diện tích nước mỗi năm: 8.620 km² (2015) đến 16.326 km² (2002), tức 22%–41% đồng bằng.
- Ngày đạt đỉnh: 265–313.
- Đỉnh giảm theo năm: −326 km²/năm (OLS, HC3, p < 0,001); Spearman ρ = −0,86.
- AR(1) của anomaly (bước 8 ngày) ≈ 0,89.
- Vì đỉnh giảm mạnh, climatology lấy trung bình mọi năm bị chệch. Nguyên nhân có thể là đê bao, đập, đổi canh tác nhưng **chỉ là giả thuyết** (tài liệu có bàn, ví dụ Triet et al. 2017 về đê bao; nhóm không kiểm chứng nhân quả); chỉ được nói "phù hợp với".
- Ba ảnh đầu mùa đã có khoảng 4.700 km² "nước" (sông, ao, ruộng), tức nền nước không phải lũ.
- Cơ cấu đất (km²): thủy sản 2.938 (2000) → 8.032 (2020); lúa 3 vụ 4.202 → 8.590; lúa 2 vụ mùa khô 8.078 → 3.472. Phân loại đất "nhảy" giữa các năm (ví dụ lúa 2 vụ mùa khô giảm gần một nửa từ 2012 sang 2013), nên không diễn giải từng năm. Mã lớp đất: 1 thủy sản, 2 tôm–lúa, 5 khác, 6 lúa 1 vụ, 7 lúa 3 vụ, 8 lúa 2 vụ (mùa khô), 9 lúa 2 vụ (mùa mưa).
- Độ chính xác của dữ liệu theo bài báo gốc: R² > 0,96 cho *diện tích lúa/thủy sản* so với thống kê GSO; bản đồ lũ được nói là "kiểm chứng với radar" nhưng chưa thấy số liệu kiểm chứng riêng cho bản đồ lũ (trang 1–8 của bài Data in Brief). Cần đọc bài Vu et al. 2022 trên *Science of the Total Environment* để tìm số này.
- Bản đồ lũ được tạo bằng thuật toán của Sakamoto et al. (2007): lọc wavelet trên ảnh MODIS MOD09A1 8 ngày 500 m, ngưỡng trên EVI/LSWI.

### 3.5 Các cấp dữ liệu
- **Level 1 (đã dựng, dùng cho mọi mô hình):** một chuỗi diện tích ngập toàn vùng, schema `region, season, step, doy, target_km2` + cờ chất lượng. Dự báo `y[s, t+h]` với h ∈ {1, 2, 4} bước (8/16/32 ngày).
- **Level 3 (chưa dựng):** bảng theo năm (n = 21): đỉnh lũ, ngày đỉnh, diện tích trung bình mùa, cơ cấu đất. Dùng cho xu hướng và hồi quy.
- **Level 2 (tùy chọn, chưa làm):** ô lưới 5 km × 5 km.
- **Vùng proxy Z0/Z1/Z2 (từng pilot, cần dựng lại):** chia mặt nạ theo tam phân vị dọc trục Tây Bắc → Đông Nam; là vùng theo tọa độ, không phải ranh giới hành chính.

### 3.6 Đã làm trên dữ liệu
- Dựng lại chuỗi từ 565 ảnh gốc bằng quy lưới **mean**, khớp `data/interim/mekong/series_mean.csv` trong phạm vi 0,01 km² (`floodbench.data.build_series`).
- Bảng Level 1 `data/processed/mekong/level1_series.csv`: 567 dòng (21 × 27). Gắn cờ, **không xóa**:
  - `imputed`: 2 ảnh thiếu, nội suy tuyến tính trong mùa.
  - `duplicate`: 15 ảnh giống hệt từng byte một ảnh sớm hơn (nghi bản sao). Mùa 2016 có 5/27 ảnh như vậy, trong đó 3 ảnh cuối (ngày 345, 353, 361) giống hệt cùng ngày của 2015.
  - `grid_500m`: ảnh gốc ở lưới 500 m.
- **Đối chiếu mực nước** (`results/tables/waterlevel_validation.csv`): đỉnh diện tích với mực nước đỉnh Tân Châu, 2000–2016, n = 17: r = 0,865 khi để nguyên, **0,669 sau khi bỏ xu hướng** (CI 95% 0,277–0,870), 0,681 với chênh lệch bậc nhất. Mỹ Thuận thấp hơn (r bỏ xu hướng 0,48–0,62, một số CI chứa 0). Chỉ kiểm được ở cấp năm.
- **Độ nhạy dữ liệu** (`results/tables/data_sensitivity.csv`): đỉnh giảm −314 km²/năm (21 mùa) và −306 (12 mùa chỉ gồm ảnh lưới 232 m); r bỏ xu hướng không đổi (0,67); bỏ ảnh nội suy hoặc ảnh nghi bản sao không đổi các thống kê dựa trên đỉnh. Các thống kê này không dùng những hàng bị gắn cờ, nên **chưa nói gì về sai số dự báo**.

**Đánh giá chung:** dữ liệu dùng được nhưng bằng chứng yếu hơn vẻ ngoài: sản phẩm suy diễn, đối chiếu chỉ ở cấp năm, r bỏ xu hướng khoảng 0,67 với CI rộng. Kết luận phải dè dặt.

---

## 4. Giao thức so sánh công bằng

Quy tắc không thương lượng (chi tiết ở `AGENTS.md`):
1. Chia theo thời gian, không xáo trộn: rolling origin theo năm.
2. Mọi thống kê tiền xử lý (climatology, scaler, trung bình, hệ số φ) tính chỉ từ tập huấn luyện của từng fold.
3. Baseline luôn nằm cùng bảng với mô hình: persistence, climatology, climatology 5 mùa, seasonal naive.
4. Cùng đặc trưng cho mô hình thống kê và ML, trừ khi ghi rõ.
5. Ngân sách tinh chỉnh ngang nhau, ghi `docs/tuning_log.md`, không tune trên tập test.
6. So sánh ở **cấp mùa** (đơn vị là mùa, không phải 27 bước trong mùa), vì sai số các bước tự tương quan mạnh nên kiểm định gộp mọi bước sẽ đánh giá quá cao ý nghĩa.
7. "Không bác bỏ" không phải "tương đương". Dùng khoảng tin cậy và TOST.
8. Không nói nhân quả (đê, đập). Chỉ nói "phù hợp với".
9. Không dùng Prophet mặc định.

### 4.1 Thiết lập đã cài
- **Chia:** huấn luyện các mùa trước năm Y, kiểm tra Y, Y = 2010..2020 (11 mùa test). Mùa 2000–2002 chỉ dùng để tính climatology, nên số mùa huấn luyện thực tế đi từ 7 (Y = 2010) đến 17 (Y = 2020), tức khoảng 170 đến 400 hàng. Bản độ nhạy 2008..2020 (13 mùa test) **chưa chạy**.
- **Chống rò rỉ:** climatology của mùa i chỉ dùng tối đa 5 mùa liền trước (cần ≥ 3). Có test cố ý tạo bộ dựng đặc trưng bị rò rỉ để chắc bộ kiểm tra bắt được (`tests/test_features.py`).
- **Anomaly và AnomAR1:** anomaly = y − Clim5. AnomAR1: ŷ[t+h] = c[t+h] + φ^h × (y[t] − c[t]), φ ước lượng chỉ từ các hàng huấn luyện.
- **Đặc trưng:** giá trị tại t, t−1, t−2, ngày đích, climatology tại t và t+h, anomaly.
- **Kiểm định cấp mùa (n = 11):** t ghép cặp, Wilcoxon, CI của chênh MSE tương đối, TOST với biên ±5% RMSE, hiệu chỉnh Holm trên 42 phép so sánh (14 mô hình/baseline × 3 tầm).
- **Cửa sổ cố định 7 mùa (`_w7`):** để tách cỡ lũ khỏi lượng dữ liệu học.

### 4.2 Thước đo: đã cài và trong kế hoạch
Ký hiệu e = y − ŷ, n là số cặp dự báo.

| Thước đo | Định nghĩa | Ý nghĩa | Trạng thái |
|---|---|---|---|
| RMSE | √(trung bình e²) | Sai số điển hình (km²), phạt nặng lỗi lớn | Đã cài |
| MAE | trung bình \|e\| | Sai số điển hình, ít nhạy với lỗi lớn | Đã cài |
| MSE theo mùa | trung bình e² trong từng mùa test | Đơn vị cho kiểm định cấp mùa | Đã cài |
| MASE | MAE / MAE của persistence trên tập huấn luyện | < 1 nghĩa là hơn persistence | Chưa |
| NSE | 1 − Σe² / Σ(y − ȳ)² | Thước đo chuẩn thủy văn; dễ cao giả khi chuỗi có mùa vụ mạnh | Chưa |
| Persistence Index (PI) | 1 − Σe² / Σ(y_t+h − y_t)² | PI > 0 nghĩa là hơn persistence; hợp với dự báo đa bước | Chưa |
| Sai số đỉnh | đỉnh dự báo − đỉnh thật mỗi mùa (km²) và sai ngày đỉnh | Điều người ra quyết định quan tâm nhất | Chưa |
| POD / FAR / CSI | "Sự kiện" = y vượt phân vị 75 của tập huấn luyện | Khả năng phát hiện ngập cao | Chưa |
| Khoảng dự báo 80/95% | Bootstrap phần dư hoặc quantile regression; đo độ phủ thật | Điểm cộng | Chưa |
| Chi phí tính toán | Thời gian huấn luyện, số tham số | Một trục khác của "công bằng" | Chưa |

### 4.3 Biến thể mục tiêu (cho phân tích độ nhạy)

| Mục tiêu | Định nghĩa | Khi nào dùng |
|---|---|---|
| T1 (chính, đang dùng) | Diện tích nước (lớp 1) trong mặt nạ ĐBSCL | Mặc định |
| T2 | Nước + hỗn hợp đất ẩm (lớp 1 + lớp 2) | Độ nhạy: gộp đất ẩm |
| T3 | T1 trừ trung bình 3 ảnh đầu mùa | Loại nền nước (sông, ao) |
| T4 | Nước theo từng lớp sử dụng đất | Level 2, so sánh theo vùng canh tác |

### 4.4 Kiểm định: vì sao làm ở cấp mùa
Kiểm định Diebold–Mariano (DM) chuẩn chỉ xử lý tự tương quan sai số đến độ trễ h − 1. Ở đây sai số các bước liền kề trong cùng mùa tương quan mạnh hơn thế nhiều, nên DM trên 27 × 11 điểm gộp sẽ đánh giá quá cao ý nghĩa. Ba cách an toàn, nên báo song song:
1. Lấy MSE trung bình của từng mùa (n = 11), rồi t ghép cặp hoặc Wilcoxon (**đã cài**).
2. Block bootstrap theo mùa: lấy lại nguyên mùa có hoàn lại để ước lượng CI cho chênh lệch RMSE (**chưa cài**).
3. Friedman + Nemenyi khi so nhiều mô hình, với mùa là đơn vị (**chưa cài**).

Với n = 11, sức mạnh kiểm định thấp: mọi chênh dưới khoảng 5% gần như không phân biệt được với nhiễu. Kết quả đáng tin nhất là khoảng tin cậy của chênh lệch.

### 4.5 Checklist chống sai sót
- Mọi thống kê tiền xử lý tính trong tập huấn luyện của từng fold.
- Liệt kê mô hình nào dùng biến nào (ngang đặc trưng).
- Ghi số cấu hình đã thử của mỗi mô hình.
- Mô hình có yếu tố ngẫu nhiên (Random Forest, LightGBM, mạng nơ-ron) chạy nhiều seed, báo trung bình ± độ lệch chuẩn.
- Baseline và mô hình cùng bảng.
- Kiểm tra kết quả khi bỏ các cặp dự báo chứa ảnh nội suy.
- Kiểm tra độ nhạy theo lưới ảnh (232 m so với 500 m) và theo mục tiêu T1/T2/T3.
- Cố định phiên bản thư viện trong `requirements.txt`.

---

## 5. Danh mục mô hình

| Họ | Mô hình | Người phụ trách | Trạng thái |
|---|---|---|---|
| Baseline | Persistence, Climatology toàn bộ, Climatology 5 mùa, Seasonal naive | Phu Hung | Đã chạy |
| Thống kê | **AnomAR1** (mốc thống kê chính) | Quỳnh | Đã chạy |
| Thống kê | **GAM** (pyGAM: s(doy) + l(a_t) + l(a_t−1), mục tiêu anomaly) | Bình | Đã chạy |
| Thống kê | **ARX (OLS)**: anomaly đích ~ 1–3 trễ anomaly + 1–2 cặp sin/cos của ngày đích | Phu Hung | Đã chạy |
| Thống kê | **Ridge có trễ**: cùng đặc trưng (3 trễ, 2 cặp Fourier), chuẩn hóa, phạt L2 | Phu Hung | Đã chạy |
| ML | **LightGBM**, 3 biến thể khai báo trước: `raw`, `anomaly`, `noclim` | Thiện | Đã chạy |
| ML | **Random Forest**, 3 biến thể `raw`, `anomaly`, `noclim` (đúng đặc trưng của LightGBM) | Phu Hung | Đã chạy |
| Foundation | **Chronos-2** (`amazon/chronos-2`, zero-shot, ngữ cảnh anomaly, phân vị 0,5) | Phu Hung | Đã chạy |
| Thống kê | SARIMAX trong mùa, ETS, hồi quy điều hòa + sai số ARMA, MixedLM | – | Để sau |
| ML | SVR, MLP nhiều seed, LSTM/GRU, hybrid ARX + GBM | – | Để sau |
| Khác | MARS; TabPFN (cần API key Prior Labs và chấp nhận giấy phép) | – | Để sau |
| Loại | Kronos (huấn luyện trên dữ liệu nến tài chính, khó biện minh cho chuỗi lũ); Prophet (không dùng mặc định) | – | Không dùng |

**Biến thể của LightGBM và Random Forest:**
- `raw`: y_t, y_t−1, y_t−2, c_t, c_target, ngày đích.
- `anomaly`: dự báo anomaly từ a_t, a_t−1, ngày đích, rồi cộng lại climatology.
- `noclim`: như `raw` nhưng bỏ climatology, để xem lợi thế đến từ đâu.

**Ngân sách tinh chỉnh** (`docs/tuning_log.md`), mỗi mô hình 6 cấu hình, chọn lại ở mỗi fold và mỗi tầm bằng validation trong tập học (3 mùa cuối, fit trên các mùa trước, tối thiểu MSE):
- GAM: (n_splines, lam).
- ARX: p ∈ {1, 2, 3} × K ∈ {1, 2}.
- Ridge: alpha ∈ {0,01; 0,1; 1; 10; 100; 1000}.
- LightGBM: (num_leaves, n_estimators, learning_rate); cố định min_child_samples = 10, subsample = 0,8, seed = 0.
- Random Forest: (max_depth, min_samples_leaf, max_features); cố định 300 cây, seed = 0.
- Chronos-2, AnomAR1 và các baseline không có siêu tham số.

**Lý do chọn (để trả lời vấn đáp):** AnomAR1 là mốc thống kê mạnh vì tự tương quan anomaly rất cao; ARX và Ridge là mở rộng tuyến tính của nó; GAM là thống kê có suy diễn nhưng đủ linh hoạt cho chu kỳ mùa phi tuyến; LightGBM và Random Forest đại diện cây/boosting cho dữ liệu bảng cỡ vừa (Grinsztajn 2022); Chronos-2 đại diện foundation model không cần huấn luyện. Ranh giới cần nói rõ: *thống kê = mô hình tham số có suy diễn; ML = mô hình học từ dữ liệu không giả định dạng hàm*. Thầy sẽ hỏi, và GAM nằm gần ranh giới.

### 5.1 Giải thích tham số (thầy hay hỏi "tham số này làm gì, bỏ đi thì sao")

| Mô hình | Tham số | Ý nghĩa |
|---|---|---|
| AnomAR1 | φ (ước lượng), c = climatology 5 mùa | φ cao thì lũ "nhớ" mức lệch lâu; lệch phai theo φ^h. c lấy từ các mùa trước |
| ARX | p (số trễ), K (số cặp sin/cos) | p lớn dễ quá khớp; K nhỏ cho hình dạng mùa trơn, K lớn bắt chi tiết nhưng nhiễu |
| Ridge | alpha | alpha lớn co hệ số về 0, giảm phương sai đổi lấy chút chệch |
| GAM | n_splines, lam | n_splines là độ "uốn" tối đa của hàm theo ngày; lam lớn làm hàm trơn hơn |
| LightGBM | num_leaves, n_estimators, learning_rate, min_child_samples, subsample | num_leaves là độ phức tạp mỗi cây (bậc tương tác); learning_rate nhỏ cần nhiều cây hơn; min_child_samples lớn chống quá khớp; subsample < 1 thêm ngẫu nhiên nên kết quả phụ thuộc seed |
| Random Forest | max_depth, min_samples_leaf, max_features, số cây | Lá nhỏ dễ quá khớp; max_features < 1 làm các cây ít giống nhau. Cây không ngoại suy được ra ngoài khoảng giá trị đã thấy |
| Chronos-2 | phân vị (0,5), độ dài ngữ cảnh | Không huấn luyện; chỉ chọn cách đưa ngữ cảnh (anomaly của mùa đang dự báo) |
| SVR (chưa dùng) | C, epsilon, gamma | C phạt lỗi, gamma điều khiển độ cong của kernel; phải chuẩn hóa đặc trưng trong vòng huấn luyện |
| MLP, LSTM/GRU (chưa dùng) | số lớp, số nút, alpha L2, dropout, cửa sổ | Dữ liệu chỉ 21 mùa nên rất dễ quá khớp; phải chạy nhiều seed |

---

## 6. Mã nguồn, cài đặt và lệnh chạy

```
.
├── AGENTS.md, README.md, pyproject.toml, requirements.txt
├── app/streamlit_app.py            báo cáo tương tác (7 tab)
├── configs/regions/mekong.yaml     cấu hình vùng
├── data/{raw,interim,processed}/   raw không commit; interim/legacy giữ bản nearest-neighbour để so sánh
├── docs/                           decisions.md, tuning_log.md, tong_hop_du_an.md (file này)
├── notebooks/                      còn trống
├── reports/                        full_report, ieee_paper, slides, references; models/ (report từng mô hình, sắp có)
├── results/                        tables/*.csv và probe_report.md, đều do code sinh
├── src/floodbench/
│   ├── data/        regions, schema, level1, build_series
│   ├── validation/  waterlevel, sensitivity
│   ├── models/      baselines (gồm AnomAR1), stat (GAM, ARX, Ridge), ml (LightGBM, Chronos-2), forest (Random Forest), tuning
│   ├── features.py, cv.py, evaluation.py, inference.py
│   └── benchmark.py, report_tables.py, report_md.py
└── tests/                          45 test
```

Quy ước: logic nằm trong `src/`, notebook chỉ gọi hàm và vẽ; vùng cấu hình bằng YAML, không hard-code đường dẫn hay hằng số của ĐBSCL; mọi vùng dùng một schema chung; `results/` do code sinh, `reports/` trích số từ `results/`.

### 6.1 Lệnh
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell; macOS/Linux: source venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest                                # 45 test

# Chỉ cần khi dựng lại dữ liệu từ ảnh gốc
python -m floodbench.data.build_series data/raw/mekong/kpftzmsyyz-2 data/interim/mekong/series_mean.csv
python -m floodbench.data.level1
python -m floodbench.validation.waterlevel
python -m floodbench.validation.sensitivity

# Mô hình và bảng
python -m floodbench.benchmark                  # chạy tất cả (Random Forest khá lâu, hơn 10 phút)
python -m floodbench.benchmark mekong gam       # chỉ tính lại mô hình nêu tên (cách nhau bằng dấu phẩy)
python -m floodbench.report_tables              # -> results/tables/probe_*.csv
python -m floodbench.report_md                  # -> results/probe_report.md
streamlit run app/streamlit_app.py
```
Tên mô hình dùng trong lệnh: `anom_ar1`, `gam`, `arx`, `ridge`, `lgbm_raw`, `lgbm_anomaly`, `lgbm_noclim`, `rf_raw`, `rf_anomaly`, `rf_noclim`, `chronos2`. Mỗi mô hình (trừ Chronos-2 và các baseline) có thêm bản `_w7` được tính cùng lúc.

Chronos-2 tự tải trọng số từ Hugging Face ở lần chạy đầu và chạy trên CPU.

### 6.2 Thư viện (Python 3.10, đã pin trong `requirements.txt`)
numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, statsmodels 0.15.0, scikit-learn 1.7.2, matplotlib 3.10.9, PyYAML 6.0.3, openpyxl 3.1.5, rasterio 1.4.4, tifffile, lightgbm 4.7.0, pygam 0.12.0, chronos-forecasting 2.3.2, torch 2.14.1, transformers 5.20.2, streamlit 1.46.1, pytest 9.2.0.

### 6.3 Vì sao chạy nhanh
Mỗi lần fit chỉ học từ vài trăm hàng, mỗi hàng 3–10 đặc trưng, và không mô hình nào nhận ảnh làm đầu vào. Thời gian chủ yếu nằm ở số lần fit (6 cấu hình × 11 fold × 3 tầm × 2 cửa sổ). Nhanh là do bài toán nhỏ, không phải dấu hiệu lỗi.

---

## 7. Kết quả thăm dò (11 mùa test 2010–2020, một lần chạy, chưa phải kết quả cuối)

### 7.1 RMSE gộp (km², thấp hơn là tốt), sắp theo tầm 16 ngày

| Mô hình | 8 ngày | 16 ngày | 32 ngày |
|---|--:|--:|--:|
| LightGBM (raw) | 1.051 | 1.284 | 1.688 |
| LightGBM (noclim) | 1.059 | 1.301 | 1.778 |
| Random Forest (anomaly) | 1.087 | 1.338 | 1.673 |
| Ridge có trễ | 1.087 | 1.346 | 1.693 |
| ARX (OLS) | 1.082 | 1.354 | 1.680 |
| LightGBM (anomaly) | 1.101 | 1.364 | 1.699 |
| GAM | 1.096 | 1.372 | 1.688 |
| Random Forest (noclim) | 1.110 | 1.373 | 1.904 |
| **AnomAR1 (mốc thống kê)** | **1.090** | **1.375** | **1.687** |
| Random Forest (raw) | 1.118 | 1.391 | 1.793 |
| Chronos-2 zero-shot | 1.116 | 1.412 | 1.788 |
| Climatology 5 mùa | 1.812 | 1.845 | 1.916 |
| Persistence | 1.312 | 2.068 | 3.456 |
| Seasonal naive | 2.215 | 2.249 | 2.336 |
| Climatology toàn bộ | 2.605 | 2.657 | 2.771 |

Phần lớn mô hình nằm trong dải hẹp quanh AnomAR1; chênh lệch lớn chỉ có giữa nhóm này và các baseline ngốc.

### 7.2 Kiểm định cấp mùa so với AnomAR1 (n = 11, họ Holm = 42 phép)
- **Không mô hình nào thắng AnomAR1 có ý nghĩa sau Holm.** Chỉ persistence kém hơn rõ (Holm 0,011 ở 16 ngày và 0,0008 ở 32 ngày); climatology toàn bộ và seasonal naive kém hơn sát ngưỡng ở 8 ngày (Holm khoảng 0,09).
- **Tương đương trong ±5% RMSE (TOST, p chưa hiệu chỉnh đa kiểm định):** ARX ở 8 ngày (p = 0,0001) và 16 ngày (0,043); GAM ở 8 ngày (0,0017); Ridge ở 8 ngày (0,0036); Random Forest anomaly ở 8 ngày (0,044). Họ thống kê tuyến tính bám sát AnomAR1 ở 8 ngày, đúng như kỳ vọng vì cùng dự báo anomaly từ vài trễ. Các kết luận có p quanh 0,04 là yếu.
- **LightGBM raw:** nhỉnh về số học ở 8 và 16 ngày (tỉ lệ RMSE 0,965 và 0,933; p t thô 0,41 và 0,14; CI chứa 0), hòa ở 32 ngày (1,001). Chưa kết luận được.
- **Random Forest:** raw kém LightGBM raw dù cùng đặc trưng và ngân sách (16 ngày: 1.391 so với 1.284). Biến thể anomaly bám sát AnomAR1 và nhỉnh nhất ở 32 ngày (1.673) nhưng CI chứa 0. Biến thể noclim kém rõ ở 32 ngày (tỉ lệ 1,129, p t thô 0,046, Holm không còn ý nghĩa).
- **Chronos-2:** kém hơn ở 32 ngày (tỉ lệ 1,060, CI chênh MSE [+4,0; +24,5] %, p t thô 0,005, sau Holm 0,18). Ở 8 và 16 ngày không khác có ý nghĩa.

### 7.3 Theo điều kiện (tầm 16 ngày, RMSE km²)
- **Giai đoạn lũ:** LightGBM raw tốt hơn AnomAR1 khi nước rút (1.321 so với 1.578) nhưng kém ở đỉnh lũ (1.531 so với 1.391). Bản noclim cũng tốt khi nước rút (1.235), nên lợi thế này không đến từ climatology. **Lợi thế nước rút không lặp lại ở họ cây nói chung:** Random Forest raw tốt nhất khi nước lên (979) nhưng tệ nhất khi nước rút (1.662); ARX bằng AnomAR1 (1.578). Không được nói chung chung "ML tốt hơn khi nước rút".
- **Cỡ lũ:** với toàn bộ dữ liệu học, LightGBM raw tốt hơn ở mùa lũ nhỏ (1.261 so với 1.488) và kém hơn ở lũ lớn (1.318 so với 1.260).
- **Cỡ lũ lẫn với lượng dữ liệu học:** các mùa lũ nhỏ đều là những năm cuối (có nhiều mùa học hơn). Cố định cửa sổ 7 mùa thì ưu thế ở lũ nhỏ biến mất (LightGBM raw 1.523 so với AnomAR1 cùng cửa sổ 1.466) và kém hơn ở lũ lớn (1.468 so với 1.245).
- **Lượng dữ liệu học:** với 7 mùa, LightGBM raw mất lợi thế ở mọi tầm (16 ngày: 1.446 so với AnomAR1 1.355). GAM ổn định hơn. ARX với 7 mùa tốt hơn bản đầy đủ ở 8–16 ngày nhưng kém hơn ở 32 ngày (1.756 so với 1.680). Random Forest với 7 mùa kém hơn bản đầy đủ ở mọi tầm.

### 7.4 Các thử nghiệm trước đó (mã cũ, không có trong repo, chỉ để tham khảo)
Trước khi có pipeline hiện tại, nhóm đã chạy hai đợt thử bằng script riêng. Các số này **không tái lập được bằng mã trong repo** và không được trích như kết quả đồ án; giữ lại để biết lịch sử và vì chúng gợi ý vài hướng phân tích.

**(a) Pilot chưa tune** (lưới nearest-neighbour, cấu hình cố định: RF 300 cây, GBM 200 cây độ sâu 2, Ridge alpha = 1), RMSE km²:

| Tầm | Persist. | Clim | Clim5 | SeasNaive | AnomAR1 | RF | GBM | Ridge |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 ngày | 1.312 | 2.604 | 1.811 | 2.212 | 1.083 | 1.099 | 1.079 | 1.090 |
| 16 ngày | 2.069 | 2.656 | 1.844 | 2.246 | 1.430 | 1.436 | 1.411 | 1.436 |
| 32 ngày | 3.460 | 2.770 | 1.915 | 2.333 | 1.950 | 2.007 | 1.995 | 1.930 |

**(b) Pilot theo 3 vùng proxy Z0/Z1/Z2** (cùng điều kiện như (a)): ở cả ba vùng và toàn đồng bằng, mọi chênh lệch thống kê–ML đều có p từ 0,12 đến 0,96 (t ghép cặp theo mùa, chưa Holm). Ở Z0 và Z2, GBM nhỉnh hơn AnomAR1 khoảng 4–6% (ví dụ Z0 16 ngày: 828 so với 876 km²; Z2 8 ngày: 327 so với 343); ở Z1 và toàn đồng bằng thì ngang. Ở Z0, 32 ngày, mọi mô hình ML/Ridge hơn cả Clim5 lẫn AnomAR1, khác với toàn đồng bằng, nên "tầm dài thì climatology thắng" không phải quy luật chung. Các vùng cùng chịu một năm lũ nên không độc lập.

**(c) Đợt "2 thống kê + 2 ML" đã tune** (6 cấu hình mỗi mô hình, 11 mùa test), RMSE km²:

| Mô hình | 8 ngày | 16 ngày | 32 ngày |
|---|--:|--:|--:|
| Persistence | 1.299 | 2.003 | 3.326 |
| Clim5 | 1.767 | 1.787 | 1.844 |
| AnomAR1 | 1.089 | 1.312 | 1.602 |
| GAM | 1.082 | 1.314 | 1.606 |
| MixedLM (BLUP, hiệu ứng ngẫu nhiên theo mùa) | 1.082 | 1.304 | 1.566 |
| State-space (local level + AR, chọn bằng AIC) | 1.082 | 1.298 | 1.573 |
| GPR (6 kernel) | 1.093 | 1.299 | 1.592 |
| GRU nhỏ (trung bình 5 seed) | 1.059 | 1.359 | 1.590 |
| LightGBM thô | 1.033 | 1.247 | 1.625 |
| LightGBM anomaly | 1.081 | 1.336 | 1.623 |
| Chronos-2 anomaly (zero-shot) | 1.122 | 1.376 | 1.730 |
| Chronos-2 giá trị thô (zero-shot) | 1.287 | 1.845 | 2.875 |

Điều rút ra từ đợt (c), khớp với kết quả hiện tại:
- Mọi mô hình thống kê và LightGBM nằm trong ±5% RMSE của AnomAR1, CI của chênh lệch đều chứa 0. Ngoại lệ duy nhất vượt Holm (18 phép) là Chronos-2 kém hơn ở 32 ngày (p Holm 0,016, MSE +17%, chỉ thắng 1/11 mùa).
- Khi tune công bằng, lợi thế của LightGBM co lại (cấu hình cố định đầu tiên cho tỉ lệ 0,93/0,92, sau tune còn 0,948/0,950).
- Theo giai đoạn (16 ngày): LightGBM thô tốt nhất khi nước rút (1.335 so với 1.550), kém ở đỉnh (1.457 so với 1.361); GRU ngược lại, khá khi nước lên, kém khi nước rút (1.640).
- **Độ ổn định theo seed của LightGBM** (5 seed): độ lệch chuẩn RMSE chỉ 0,3–0,7% (8 ngày 1.030 ± 6,9; 16 ngày 1.253 ± 6,1; 32 ngày 1.628 ± 4,4 km²). GRU dao động 10–25 km² theo seed và số epoch chọn rất không ổn định.
- GPR: khoảng dự báo 90% thực tế chỉ phủ 85,5–87,4%, tức hơi hẹp.
- **State-space bị loại** dù kết quả tốt: AIC chọn "hằng số + AR(1)" ở 33/33 lần, tức rơi về cùng họ với AnomAR1, không thêm giá trị diễn giải. Nếu thầy chấp nhận họ AR/ARIMA thì đây là đối chứng thống kê đáng giữ.
- **MixedLM** từng nằm trong "bộ 4" đề xuất (GAM, MixedLM, LightGBM, Chronos-2); hiện không nằm trong phạm vi đã chốt.
- Đợt (c) có vài lựa chọn quyết định sau khi xem test (chọn biến thể Chronos-2 và LightGBM đại diện, thiết kế lưới, thêm MixedLM). Pipeline hiện tại tránh điều này bằng cách khai báo trước các biến thể và báo tất cả.

**Manh mối cho chênh lệch AnomAR1 1.602 so với 1.687 km² (mục 12.1):** con số 1.602 đến từ đợt (c). Trong đợt đó cả persistence (1.299 / 2.003 / 3.326) và Clim5 (1.767 / 1.787 / 1.844) cũng khác bản hiện tại (1.312 / 2.068 / 3.456 và 1.812 / 1.845 / 1.916), trong khi hai baseline này không có tham số nào để ước lượng. Vậy rất có thể chênh lệch nằm ở **tập cặp dự báo được đánh giá** (khác chuỗi đầu vào, hoặc khác khoảng bước gốc t trong mỗi mùa), chứ không phải ở công thức AnomAR1. Đáng chú ý: persistence của bản hiện tại khớp gần hết pilot (a) (1.312 / 2.069 / 3.460).

### 7.5 Cách nói an toàn
"Trong thử nghiệm sơ bộ trên 11 mùa, chưa thấy bằng chứng ML vượt thống kê một cách có ý nghĩa; các mô hình thống kê tuyến tính tương đương AnomAR1 ở 8 ngày; thứ hạng thay đổi theo giai đoạn lũ và theo lượng dữ liệu học." Không nói "ML kém hơn" hay "thống kê thắng".

---

## 8. Có đáng làm tiếp không? (tiêu chí đặt trước khi chạy)

Tiêu chí: đạt **ít nhất 2 trong 3**: (a) thứ hạng đổi theo điều kiện và CI ủng hộ; (b) có kết luận tương đương hợp lệ hoặc "không đủ dữ liệu"; (c) có phát hiện về phương pháp hoặc dữ liệu.

Hiện tại:
- (a) nước rút so với đỉnh lũ, và lượng dữ liệu học, **nhưng CI còn rộng, mức ủng hộ yếu**;
- (b) nhóm thống kê tuyến tính tương đương AnomAR1 ở 8 ngày;
- (c) cỡ lũ lẫn với lượng dữ liệu học; mô hình zero-shot không hơn thống kê đơn giản.

Kết luận: **đáng làm, theo hướng "so sánh theo điều kiện", không phải "chứng minh ML thắng"**. Giá trị nằm ở quy trình công bằng và cách báo cáo, đúng điều thầy chấm.

---

## 9. Kế hoạch còn lại

Tính đến 2026-11-05, còn 18 ngày đến buổi báo cáo. Phân công và lịch ở mục 1.

| Hạng mục | Trạng thái |
|---|---|
| Chốt phạm vi với thầy | Chưa gửi các câu hỏi mở (mục 11); chưa điền bảng thành viên |
| Pipeline dữ liệu | Xong phần cần cho mô hình. Còn: Level 3 (bảng theo năm), notebook kiểm kê, vùng proxy Z0–Z2 |
| EDA, suy diễn, hồi quy | Chưa: thống kê mô tả, Mann–Kendall, trước/sau 2010, hồi quy đỉnh lũ theo cơ cấu đất (n = 21, kiểm tra 5 giả định) |
| Mô hình thống kê | Xong: baseline, AnomAR1, GAM, ARX, Ridge. Còn: **chẩn đoán phần dư** |
| Mô hình ML | Xong: LightGBM, Random Forest, Chronos-2. Còn: **độ quan trọng đặc trưng / SHAP** |
| Kiểm định theo điều kiện | Xong một phần: kiểm định cấp mùa, giai đoạn lũ, cỡ lũ, cửa sổ 7 mùa. Còn: learning curve đầy đủ (5, 8, 11, 14 mùa), block bootstrap theo mùa, Friedman + Nemenyi, độ nhạy mô hình (bỏ ảnh gắn cờ, 13 mùa test) |
| Mở rộng vùng khác | Chưa bắt đầu, chờ thầy; nếu không kịp thì chuyển thành hướng phát triển |
| Viết báo cáo | Chưa; có `results/probe_report.md` và app Streamlit làm nguyên liệu |

---

## 10. Ánh xạ vào outline môn

| Yêu cầu outline | Cách ghép | Trạng thái |
|---|---|:-:|
| Descriptive | Phân phối, boxplot theo năm/giai đoạn, heatmap mùa × năm | Chưa làm |
| Inferential | Trước/sau 2010 (Mann–Whitney), Levene, Kruskal theo nhóm năm, kiểm định cấp mùa | Một phần |
| Regression | Đỉnh lũ ~ năm + thủy sản + lúa 3 vụ (n = 21, 5 giả định) | Chưa làm |
| ARIMA/Forecasting | AnomAR1, ARX (họ AR); SARIMAX để sau | Một phần |
| ML/DL | LightGBM, Random Forest, Chronos-2 | Đã có |
| Logistic | Chỉ có nếu làm Level 2 (ô có ngập hay không) | Chưa, hỏi thầy |
| SEM | SEM cần biến tiềm ẩn đo bằng nhiều câu hỏi khảo sát; dữ liệu viễn thám không có biến tiềm ẩn nào, ép vào chỉ thành path analysis gượng. Đề xuất bỏ; phần "yếu tố ảnh hưởng" đáp ứng bằng xu hướng đỉnh lũ, hồi quy theo cơ cấu đất và so sánh theo vùng | Hỏi thầy |
| Explainable AI | Độ quan trọng đặc trưng / SHAP cho LightGBM | Chưa làm (Thiện) |
| Theme Economics/Business/Healthcare | Gắn vào kinh tế nông nghiệp – thủy sản vùng | Hỏi thầy |

---

## 11. Câu hỏi mở cho thầy (chưa gửi)

1. Chuyển ảnh thành chuỗi số có được coi là "dùng ảnh" không?
2. Bắt buộc đủ các phương pháp trong outline hay chọn 2–3? Bỏ SEM được không?
3. Đề tài lũ có hợp theme kinh tế/kinh doanh/y tế không?
4. Biến ngoài dataset (mưa, mực nước, shapefile) có được làm phần mở rộng không?
5. Vùng proxy Z0/Z1/Z2 có chấp nhận không?
6. Mở rộng vùng nào, dữ liệu nào? Nếu khác loại biến (mực nước và diện tích ngập), có so sánh chéo được không?
7. 30 trang có gồm tài liệu tham khảo và phụ lục không?
8. Tên repo/mã lớp dùng tên nào?
9. Nếu kết luận là "chưa thấy ML vượt thống kê", có coi là kết quả hợp lệ không?
10. GAM và MixedLM có tính là "mô hình truyền thống" không? Foundation model zero-shot như Chronos-2 (không huấn luyện lại trên dữ liệu của nhóm) có tính là một mô hình ML không?

---

## 12. Điểm chưa chắc và rủi ro

### 12.1 Chênh lệch AnomAR1 chưa giải thích được (Quỳnh nhận)
Báo cáo thăm dò cũ cho AnomAR1 = 1.602 km² ở 32 ngày; bản hiện tại ra **1.687** (biến thể φ ước lượng riêng cho từng tầm cho 1.675). Chưa rõ nguyên nhân; mã cũ không có trong repo (nằm trong vault Obsidian của nhóm: `03_Resources/Code_Snippets/STAT3013_Flood_pilot_benchmark.py` và các script cùng thư mục). Manh mối ở cuối mục 7.4: persistence và Clim5 cũng lệch, nên nhiều khả năng do khác tập cặp dự báo, không do công thức. Chưa trích con số 32 ngày như kết quả cuối cho đến khi giải thích được. Đã ghi trong `docs/decisions.md`.

### 12.2 Các lựa chọn đã cố định trước khi xem kết quả
- AnomAR1 giữ φ^h, không đổi sang biến thể φ riêng từng tầm sau khi thấy biến thể này tốt hơn nhẹ (ghi như một độ nhạy).
- LightGBM và Random Forest có 3 biến thể khai báo trước; cả 3 đều nằm trong bảng và trong họ Holm.
- Biên ±5% cho TOST được chọn trước, nhưng chưa có cơ sở thực chất (ví dụ ý nghĩa nông nghiệp). Cần bàn.

### 12.3 Thêm mô hình sau đợt đầu
ARX, Ridge và Random Forest được thêm sau khi đã thấy kết quả của GAM, LightGBM, Chronos-2. Việc thêm nằm trong kế hoạch từ đầu, và dự báo của các mô hình cũ không đổi, nhưng họ Holm tăng từ 30 lên 42 phép nên p Holm của các mô hình cũ tăng (Chronos-2 ở 32 ngày từ 0,10 lên 0,18). Báo cáo phải nói rõ. p TOST cũng chưa hiệu chỉnh đa kiểm định.

### 12.4 Hạn chế thống kê
- Chỉ 11 mùa test, một lần chạy, một chuỗi, CI rộng; khác biệt vài phần trăm không đáng tin.
- Chronos-2 chỉ chạy zero-shot với ngữ cảnh anomaly.
- Mỗi nhóm cỡ lũ chỉ khoảng 3–4 mùa; phân tích theo giai đoạn và cỡ lũ hiện chỉ là mô tả, chưa kiểm định riêng.
- Chưa chẩn đoán phần dư cho các mô hình thống kê.
- Các hàng trong cùng mùa tương quan mạnh, nên vài trăm hàng huấn luyện chứa ít thông tin độc lập hơn con số cho thấy.

### 12.5 Hạn chế dữ liệu và trích dẫn
- Mục tiêu là sản phẩm suy diễn; "nước" gồm cả ao nuôi và ruộng ngập canh tác.
- Đối chiếu mực nước chỉ ở cấp năm.
- Pilot cũ có dấu hiệu lệch do lưới 500 m (hệ số +851 km², p ≈ 0,049, n = 21); **chưa kiểm lại** trên dữ liệu đã quy lưới bằng mean.
- 15 ảnh nghi bản sao chưa rõ nguyên nhân.
- Nhiều tài liệu tham khảo chưa tra DOI (Breiman 2001, Shmueli 2010, Hewamalage 2023, Diebold–Mariano 1995, HLN 1997, Hyndman–Koehler 2006, Demšar 2006, Hansen–Lunde–Nason 2011, Nash–Sutcliffe 1970). Asif 2025 và Kratzert 2018 phải đọc bản đầy đủ trước khi trích. Không ghi Q1/Q2 khi chưa tra Scimago.

### 12.6 Mở rộng vùng
Chưa có vùng, chưa có nguồn. Global Flood Database chỉ có bản đồ theo sự kiện, không phải chuỗi liên tục. MRC Data Portal (mực nước) cần tài khoản. Vùng Campuchia/Tonle Sap có sẵn trong raster nhưng không phải Việt Nam.

### 12.7 Quy trình
- Các dòng "Phu Hung (đề xuất)" trong `docs/decisions.md` cần cả nhóm xác nhận, đặc biệt: dùng Chronos-2 thay Kronos, hoãn TabPFN, gắn cờ thay vì xóa ảnh, định nghĩa AnomAR1.
- Nếu ai đổi giao thức chung mà không chạy lại mọi mô hình, bảng so sánh cuối sẽ không còn công bằng (mục 1.3).

### 12.8 Bảng rủi ro

| Rủi ro | Mức | Cách xử lý |
|---|:-:|---|
| Lệch do lưới 232 m / 500 m | Cao | Đã quy lưới bằng mean; còn phải chạy lại chỉ với năm cùng lưới và kiểm tra biến giả lưới |
| Rò rỉ dữ liệu | Cao | Mọi thống kê chỉ từ fold huấn luyện; test cố ý rò rỉ; review chéo |
| Vấn đáp code từng dòng | Cao | Mỗi người giải thích được toàn pipeline; luân phiên trình bày module của người khác |
| Mục tiêu là sản phẩm suy diễn | Trung bình | Nêu định nghĩa; thử T2/T3 |
| Chỉ 21 mùa | Trung bình | Kiểm định cấp mùa, CI, TOST, bootstrap theo mùa |
| Tune lệch ngân sách | Trung bình | `docs/tuning_log.md`, cùng cách chọn cấu hình |
| Dữ liệu vùng mới không có hoặc không tương thích | Trung bình | Chốt sớm với thầy; có phương án chỉ ĐBSCL |
| Diễn giải nhân quả (đê bao, đập) | Trung bình | Chỉ nói "phù hợp với" |
| Mạng nơ-ron quá khớp (nếu làm) | Thấp | Thử nghiệm phụ, nhiều seed, early stopping |

---

## 13. Tài liệu trong repo

- `README.md`: giới thiệu, cài đặt, lộ trình.
- `AGENTS.md`: quy tắc cho AI coding agent, cũng là tóm tắt tốt các quy tắc so sánh công bằng.
- `docs/decisions.md`: nhật ký quyết định kỹ thuật.
- `docs/tuning_log.md`: nhật ký tinh chỉnh.
- `results/probe_report.md`: báo cáo thăm dò tự sinh từ `results/tables/`.
- `app/streamlit_app.py`: bản tương tác (tóm tắt, dữ liệu, RMSE, kiểm định, theo điều kiện, dự báo từng mùa, báo cáo).
- Ghi chú nội bộ cũ của nhóm (brainstorm, chọn mô hình, lộ trình M0–M8, kịch bản báo cáo hướng đi) không nằm trong repo; nội dung cần thiết đã được gộp vào file này (mục 7.4 và 14–19).

---

## 14. Tài liệu tham khảo và gap

Ký hiệu: ✅ đã kiểm ít nhất metadata/abstract; ⚠️ chưa tra DOI. **Mọi dòng ⚠️ phải mở DOI hoặc trang tạp chí, kiểm tác giả, năm, tập, trang trước khi đưa vào References.** Nhiều dòng ✅ mới chỉ đọc abstract.

### 14.1 Lõi: so sánh thống kê và ML
| Tài liệu | Dùng để | Kiểm |
|---|---|:-:|
| Makridakis, Spiliotis & Assimakopoulos (2018), *Statistical and Machine Learning forecasting methods: Concerns and ways forward*, PLOS ONE 13(3):e0194889 | Trên 1.045 chuỗi tháng (M3), ML kém 8 phương pháp thống kê và tốn tính toán hơn; mẫu lập luận "so sánh công bằng" | ✅ |
| Makridakis et al. (2018), *The M4 Competition: Results, findings, conclusion and way forward*, Int. J. Forecasting 34(4):802–808 | Các phương pháp tốt nhất chủ yếu là kết hợp; người thắng là hybrid ES–RNN | ✅ |
| Smyl (2020), *A hybrid method of exponential smoothing and RNNs*, IJF 36(1):75–85 | Thiết kế hybrid | ✅ (metadata) |
| Grinsztajn, Oyallon & Varoquaux (2022), *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS 2022, arXiv:2207.08815 | Biện minh dùng cây/boosting thay vì mạng sâu cho dữ liệu bảng cỡ vừa | ✅ (abstract) |
| Breiman (2001), *Statistical Modeling: The Two Cultures*, Statistical Science 16(3):199–231 | Định nghĩa hai họ mô hình | ⚠️ |
| Shmueli (2010), *To Explain or to Predict?*, Statistical Science 25(3):289–310 | Phân biệt giải thích và dự báo | ⚠️ |

### 14.2 Lũ và ML
| Tài liệu | Dùng để | Kiểm |
|---|---|:-:|
| Asif, Kuglitsch, Pelivan & Albano (2025), *Review and Intercomparison of ML Applications for Short-term Flood Forecasting*, Water Resources Management, DOI 10.1007/s11269-025-04093-x | Xếp hạng mô hình thay đổi theo tầm dự báo, động lực "phụ thuộc điều kiện" | ✅ (abstract; phải đọc toàn văn trước khi trích) |
| Mosavi, Ozturk & Chau (2018), *Flood Prediction Using ML Models: Literature Review*, Water 10(11):1536, DOI 10.3390/w10111536 | Tổng quan; hybrid, ensemble là chiến lược nổi bật | ✅ |
| Kratzert et al. (2018), *Rainfall–runoff modelling using LSTM networks*, HESS 22:6005–6022, DOI 10.5194/hess-22-6005-2018 | LSTM thắng khi huấn luyện trên 241 lưu vực; nhóm chỉ có 1 chuỗi × 21 mùa | ✅ (phải đọc toàn văn trước khi trích) |
| *Combining statistical ML models with ARIMA for water level forecasting: The case of the Red river*, ScienceDirect S0309170819311546 | Nghiên cứu Việt Nam ARIMA + ML | ✅ tiêu đề; **chưa xác nhận tác giả, năm** |
| *Short-Term Streamflow Forecasting … ARIMA Models and RNN*, Hydrology 2026, 13(3):82, DOI 10.3390/hydrology13030082 | Độ phức tạp không bảo đảm tốt hơn | ✅ (snippet) |
| Nevo et al. (2022), *Flood forecasting with ML models in an operational framework*, arXiv:2111.02780 | Hệ thống vận hành thực tế | ✅ (URL) |
| Hanh et al. (2024), *Multi-step-ahead prediction of water levels using ML … Vietnamese Mekong Delta*, VJES 46(4):468–488, DOI 10.15625/2615-9783/21067 | Cùng khu vực, đa bước; SVR tốt nhất | ✅ |
| Le, Ho, Lee & Jung (2019), *Application of LSTM Neural Network for Flood Forecasting*, Water 11(7):1387, DOI 10.3390/w11071387 | LSTM cho dòng chảy Việt Nam | ✅ |

### 14.3 Dữ liệu, MODIS và bối cảnh ĐBSCL
| Tài liệu | Dùng để | Kiểm |
|---|---|:-:|
| Vu et al. (2022), *Datasets of land use change and flood dynamics in the Vietnamese Mekong Delta*, Data in Brief 42:108268; dữ liệu Mendeley DOI 10.17632/kpftzmsyyz.2 | Dataset chính (trang 2: đặc tả; trang 4–5: bảng lúa/thủy sản so với GSO; trang 6–7: đỉnh lũ) | ✅ đã đọc |
| Vu et al. (2022), *Land use change in the Vietnamese Mekong Delta: New evidence from remote sensing*, Sci. Total Environ. 813:151918, DOI 10.1016/j.scitotenv.2021.151918 | Nguồn kiểm chứng độ chính xác bản đồ; cần tải qua thư viện | ✅ (metadata) |
| Sakamoto et al. (2007), *Detecting temporal changes in the extent of annual flooding … from MODIS time-series imagery*, Remote Sensing of Environment | Thuật toán gốc tạo bản đồ lũ; cần đọc bản gốc để kiểm ngưỡng EVI | ✅ (abstract) |
| Triet et al. (2017), *Has dyke development in the Vietnamese Mekong Delta shifted flood hazard downstream?*, HESS 21:3991–4010, DOI 10.5194/hess-21-3991-2017 | Bối cảnh cho xu hướng giảm đỉnh lũ; **không** dùng để khẳng định nguyên nhân trong dữ liệu của nhóm | ✅ |
| Remote Sensing 2023, DOI 10.3390/rs15082001 (Sentinel-1 + CNN/MLP/RF, An Giang); Environments, DOI 10.3390/environments6050057 (MODIS Campuchia 2002–2013) | Cho thấy nghiên cứu lân cận *lập bản đồ* chứ không *dự báo* diện tích ngập | ✅ (snippet) |

### 14.4 Phương pháp đánh giá
| Tài liệu | Dùng để | Kiểm |
|---|---|:-:|
| Hyndman & Athanasopoulos (2021), *Forecasting: Principles and Practice*, 3rd ed., otexts.com/fpp3 | Cross-validation chuỗi thời gian, thước đo, ARIMA/ETS | ✅ |
| Cerqueira, Torgo & Mozetič (2020), *Evaluating time series forecasting models*, Machine Learning 109:1997–2028, DOI 10.1007/s10994-020-05910-7 | Biện minh rolling-origin cho chuỗi không dừng | ✅ |
| Hewamalage, Ackermann & Bergmeir (2023), *Forecast evaluation for data scientists*, DMKD 37:788–832 | Checklist lỗi đánh giá | ⚠️ |
| Diebold & Mariano (1995), JBES 13(3):253–263; Harvey, Leybourne & Newbold (1997), IJF 13(2):281–291 | Kiểm định DM và hiệu chỉnh HLN | ⚠️ số trang |
| Hyndman & Koehler (2006), IJF 22(4):679–688 | MASE | ⚠️ |
| Demšar (2006), JMLR 7:1–30 | Friedman + Nemenyi | ⚠️ |
| Hansen, Lunde & Nason (2011), *The Model Confidence Set*, Econometrica 79(2):453–497 | Tùy chọn khi nhiều mô hình xấp xỉ nhau | ⚠️ |
| Gneiting & Raftery (2007), JASA 102:359–378 | Nếu làm dự báo xác suất | ⚠️ |
| Nash & Sutcliffe (1970), J. Hydrology 10(3):282–290; Kitanidis & Bras (1980) | NSE và Persistence Index | ⚠️ |
| Hollmann et al. (2025), *Nature* 637:319–326, DOI 10.1038/s41586-024-08328-6; Hoo et al. (2025), arXiv:2501.02945 | TabPFN và TabPFN-TS (nếu dùng) | ✅ |

Không ghi Q1/Q2 khi chưa tra Scimago. Turnitin phải ≤ 10%, nên viết lại bằng lời của nhóm.

### 14.5 Tài liệu nói gì
| Chiều | Phát hiện | Nguồn |
|---|---|---|
| Dữ liệu ít, chuỗi đơn | Thống kê thường ngang/hơn ML | Makridakis 2018, M4 |
| Dữ liệu bảng cỡ vừa | Cây/boosting vượt mạng sâu | Grinsztajn 2022 |
| Nhiều chuỗi huấn luyện | LSTM vượt mô hình khái niệm | Kratzert 2018 |
| Tầm dự báo | Xếp hạng mô hình thay đổi theo lead time | Asif 2025 |
| Hybrid | Chiến lược cải thiện được nhắc nhiều nhất | M4, Smyl 2020, Mosavi 2018 |

### 14.6 Gap (viết như đề xuất, không khẳng định "chưa ai làm")
1. Ít bài đặt ML cạnh baseline thống kê mạnh dưới cùng một giao thức.
2. Hiếm báo cáo theo điều kiện (giai đoạn lũ, năm cực trị, lượng dữ liệu, vùng).
3. Tính không dừng bị xem nhẹ: đỉnh lũ giảm khoảng 300 km²/năm, climatology cổ điển bị chệch nặng (RMSE 2.605 so với 1.812 km² của Clim5 ở 8 ngày), ít bài đưa baseline thích nghi xu hướng vào so sánh.
4. Theo tìm kiếm của nhóm, chưa thấy so sánh dự báo trên chuỗi diện tích ngập MODIS của ĐBSCL; các bài lân cận lập bản đồ, không dự báo.

Câu gap gợi ý (viết lại bằng lời của nhóm): *Nhiều nghiên cứu so sánh các thuật toán ML cho dự báo lũ, nhưng kết quả phụ thuộc vào tầm dự báo, biến đầu vào và loại lũ (Asif et al., 2025), và hiếm khi được đặt cạnh các baseline thống kê mạnh dưới cùng một giao thức đánh giá. Chúng tôi so sánh … trên chuỗi diện tích ngập MODIS 2000–2020 của ĐBSCL và báo cáo kết quả theo điều kiện.*

---

## 15. Câu chuyện của báo cáo và kế hoạch viết

### 15.1 Storyline (insight ở đầu và cuối)
1. **Vấn đề:** ĐBSCL khoảng 40.000 km², lũ hằng năm vừa là rủi ro vừa là nguồn phù sa/thủy sản; đỉnh lũ đang giảm trong khi thủy sản và lúa 3 vụ tăng. Dự báo ngập 1–4 tuần phục vụ lịch xuống giống, điều tiết cống.
2. **Gap:** mục 14.6.
3. **Giải pháp:** pipeline biến ảnh MODIS thành chuỗi; benchmark thống nhất (rolling-origin theo năm, kiểm định cấp mùa, theo điều kiện).
4. **Kết quả:** "bản đồ điều kiện": mô hình nào tốt ở đâu. Dù kết luận là "thống kê ngang ML" hay "ML thắng khi nước rút", đều dịch được thành khuyến nghị.
5. **Khuyến nghị (dè dặt):** ví dụ mô hình đơn giản, giải thích được, chạy hằng tuần ở cấp tỉnh không cần GPU; khi nào mới đáng đầu tư mô hình phức tạp. Không khẳng định nhân quả.

### 15.2 Khung báo cáo ~30 trang

| Chương | Nội dung | Trang |
|---|---|:-:|
| 1. Introduction | Bối cảnh, động lực, gap, đóng góp; insight chính ngay đầu | 3 |
| 2. Literature Review | Bảng so sánh tài liệu, bảng gap | 4 |
| 3. Data & Methodology | Dataset, cách tạo bản đồ MODIS, pipeline cắt và quy lưới, định nghĩa mục tiêu, giao thức | 6 |
| 4. Descriptive + Inferential | Phân phối, xu hướng, kiểm định nhóm | 4 |
| 5. Models & Experiments | Mô hình thống kê (kèm chẩn đoán phần dư), ML (kèm tune) | 5 |
| 6. Results & Discussion | Bảng chính, theo điều kiện, kiểm định, insight | 5 |
| 7. Conclusion | Tóm tắt, khuyến nghị, hạn chế, hướng mở; insight nhắc lại | 2 |

Hình/bảng nên có: bản đồ ĐBSCL với mặt nạ; 21 mùa chồng nhau; đỉnh lũ theo năm kèm xu hướng; bảng RMSE mô hình × tầm; heatmap giai đoạn × mô hình; learning curve; lỗi theo năm; kết quả kiểm định với CI.

**Bài IEEE 7–10 trang:** Abstract (vấn đề, phương pháp, dữ liệu, kết quả chính theo điều kiện, đóng góp), Related Work gọn với bảng gap, Methodology, Results (1 bảng chính + 2 hình), Discussion, Conclusion.

**Streamlit demo** (`app/streamlit_app.py`) có thể dùng làm điểm cộng khi vấn đáp, nếu cả nhóm giải thích được từng dòng.

---

## 16. Mở rộng vùng, dữ liệu bổ sung và Level 2 (chưa chốt, chờ thầy)

### 16.1 Ứng viên dữ liệu đa vùng (đã tìm, chưa tải thử)
| Nguồn | Vùng | Thời gian | Loại | Dùng làm gì |
|---|---|---|---|---|
| ENACTS daily rainfall (IRI Data Library) | Toàn Việt Nam, lưới ~0,05° | 1981–2021 | Mưa ngày | Biến phụ cho mọi vùng; nguồn đa vùng liên tục duy nhất đã xác nhận |
| VnGP 0,25° (DIAS, DOI 10.20783/DIAS.271) | Toàn Việt Nam | 1980–2010 | Mưa lưới | Kiểm tra chéo |
| 172 trạm mưa Việt Nam (World Bank, CC BY 4.0) | Toàn quốc | 1975–2006 | Mưa trạm | Kiểm chứng ngoài; không nối tới 2020 |
| Red River sediment & discharge (Mendeley, DOI 10.17632/gmtw3zxhzc.1) | Hệ sông Hồng | Chưa kiểm | Lưu lượng, bùn cát | Mục tiêu thủy văn cho miền Bắc nếu đủ dài |
| Sentinel-1 Quảng Nam 2020, Thừa Thiên Huế 2022, đồng bằng sông Hồng 2019; UNOSAT/HDX | Bắc, Trung | Sự kiện | Diện tích ngập | Chỉ kiểm chứng định tính, không phải chuỗi |
| Global Flood Database (Earth Engine) | Toàn cầu | Sự kiện | Bản đồ ngập | Không dùng để dự báo |
| Kaggle, dữ liệu quốc tế | – | – | – | Loại |

**Kết luận:** không tìm được chuỗi diện tích ngập liên tục nhiều năm cho miền Bắc hoặc miền Trung. Phương án đề xuất: giữ mục tiêu ĐBSCL; thêm mưa ENACTS của ba vùng làm biến phụ (nếu thầy cho); dùng các sự kiện Sentinel-1 làm kiểm chứng định tính; nếu dữ liệu sông Hồng đủ dài thì thêm một mục mở rộng. Vùng Campuchia/Tonle Sap có trong raster nhưng không phải Việt Nam, chỉ làm phụ lục nếu thầy đồng ý.

### 16.2 Biến phụ khác (chỉ ở mục mở rộng, kết quả bản chính phải tách riêng)
| Nguồn | Biến | Ghi chú |
|---|---|---|
| MRC Data Portal | Mực nước Tân Châu, Châu Đốc, Kratie theo ngày | Cần tài khoản; cho ML "nguyên liệu" thật vì lũ ĐBSCL phụ thuộc nước thượng nguồn |
| ERA5-Land, CHIRPS | Mưa theo ô | Dữ liệu toàn cầu, cần xử lý lưới |
| DEM (SRTM, MERIT) | Cao độ | Cho Level 2 |
| NOAA ONI | Chỉ số ENSO theo tháng | Biến khí hậu quy mô lớn |
| GADM, DesInventar | Ranh giới tỉnh | Biến vùng proxy thành vùng quản lý thật |

### 16.3 Mở rộng ngay trong dataset hiện có
- Vùng theo ranh giới thật (Đồng Tháp Mười, tứ giác Long Xuyên, ven biển) khi có shapefile.
- Chuỗi ngập riêng theo loại đất (lúa 3 vụ, lúa 2 vụ, thủy sản): nhanh, nên làm nếu có thời gian.
- Vùng proxy Z0/Z1/Z2 (mục 7.4b): cần dựng lại với lưới mean.

### 16.4 Level 2 (ô lưới, tùy chọn)
Ô 5 km × 5 km (≈ 1.600 ô, ≈ 840.000 dòng). Mục tiêu: tỉ lệ pixel nước (hồi quy) hoặc "ô ngập > 50%" (phân loại, dùng được logistic, OR, ROC-AUC theo outline). Đặc trưng: tỉ lệ nước tại t, t−1, trung bình 8 ô lân cận, ngày trong năm, cơ cấu đất, tọa độ. Thống kê: fractional logit hoặc GAM; ML: Random Forest, LightGBM. Đánh giá: rolling-origin theo năm và spatial block CV. Cảnh báo: không có DEM nên ML có thể học "vị trí" thay vì cơ chế. Chỉ làm khi Level 1 đã xong chắc.

---

## 17. Từ điển nhanh

| Thuật ngữ | Nghĩa trong dự án |
|---|---|
| Mùa (season) | Một năm lũ, 27 ảnh cách nhau 8 ngày (ngày 153 → 361); có 21 mùa |
| Bước (step, t) | Chỉ số ảnh trong mùa, 1..27 |
| Tầm dự báo (h) | Số bước dự báo trước: h = 1, 2, 4 là 8, 16, 32 ngày |
| Climatology | Trung bình theo bước tính từ các năm trước; Clim5 chỉ dùng 5 năm gần nhất |
| Anomaly | Giá trị thực trừ climatology |
| AR(1) trên anomaly | Lũ "nhớ" mức lệch hiện tại, phai dần theo φ^h |
| Persistence | Dự báo bằng giá trị hiện tại |
| Seasonal naive | Dự báo bằng giá trị cùng bước của mùa trước |
| Rolling origin | Huấn luyện các năm trước Y, kiểm tra năm Y, lặp qua nhiều Y |
| Rò rỉ (leakage) | Thông tin tương lai lọt vào huấn luyện; lỗi nguy hiểm nhất |
| Fold | Một lần chia huấn luyện/kiểm tra (ở đây: một năm test) |
| Ngân sách tinh chỉnh | Số cấu hình được thử cho mỗi mô hình (ở đây 6) |
| RMSE, MAE | Sai số căn trung bình bình phương, sai số tuyệt đối trung bình (km²) |
| DM-test | Kiểm định hai mô hình có sai số dự báo khác nhau không |
| Holm | Hiệu chỉnh p khi làm nhiều kiểm định cùng lúc |
| TOST | Kiểm định *tương đương* (ngang nhau trong biên ±Δ, ở đây ±5% RMSE) |
| Block bootstrap theo mùa | Lấy lại nguyên mùa để ước lượng CI |
| MixedLM / BLUP | Mô hình hiệu ứng ngẫu nhiên; BLUP là dự báo hiệu ứng ngẫu nhiên của từng mùa |
| Zero-shot | Dùng mô hình huấn luyện sẵn, không học thêm trên dữ liệu của nhóm |
| Z0/Z1/Z2 | 3 vùng con theo tọa độ của ĐBSCL (proxy) |
| Level 1/2/3 | Chuỗi toàn vùng / ô lưới 5 km / bảng theo năm |
| ENVI raw | Định dạng ảnh của 80 file 2018–2020, đọc bằng `.hdr` |

---

## 18. Bộ câu hỏi vấn đáp

Mỗi người trả lời được ít nhất các câu liên quan mô hình của mình và các câu 1–8. Học ý, không học thuộc.

| # | Thầy có thể hỏi | Ý trả lời |
|:-:|---|---|
| 1 | Vì sao không dùng Kaggle? | Theo yêu cầu: dataset từ bài báo uy tín, dữ liệu Việt Nam |
| 2 | Ảnh vệ tinh thì "dữ liệu" ở đâu? | Cắt vùng, quy lưới, tính diện tích km², tạo chuỗi 21 mùa × 27 bước. Mô hình không nhận ảnh |
| 3 | Bản đồ lũ chính xác đến đâu? | Sản phẩm suy diễn từ EVI/LSWI; lớp nước gồm cả sông, ao, ruộng ngập. Đối chiếu với mực nước Tân Châu: r ≈ 0,67 sau khi bỏ xu hướng, chỉ ở cấp năm. Nêu trong Limitations |
| 4 | Sao chia theo năm mà không ngẫu nhiên? | Chia ngẫu nhiên làm lộ tương lai, kết quả lạc quan giả. Dự báo thật chỉ có quá khứ |
| 5 | Baseline để làm gì? | Không thắng baseline đơn giản thì mô hình vô nghĩa. Clim mọi năm kém nặng vì xu hướng giảm, nên dùng Clim5 và AnomAR1 |
| 6 | Sao chọn cây/boosting mà không LSTM? | 21 mùa là ít; cây/boosting mạnh với dữ liệu bảng cỡ vừa (Grinsztajn 2022); mạng nơ-ron dễ quá khớp. Thử GRU ở đợt trước cũng không hơn |
| 7 | 21 mùa thì kiểm định ý nghĩa thế nào? | Kiểm định cấp mùa, CI, TOST, Holm. "Không bác bỏ" khác "ngang nhau" |
| 8 | Nếu ML không hơn thống kê thì sao? | Vẫn hợp lệ: cho ra bản đồ điều kiện "ai tốt ở đâu" và khuyến nghị mô hình đơn giản, giải thích được |
| 9 | "Thống kê" và "ML" khác nhau thế nào? | Thống kê = mô hình tham số có suy diễn (AR, ARX, GAM); ML = học từ dữ liệu không giả định dạng hàm (cây, boosting). Ridge và GAM ở ranh giới, nói rõ |
| 10 | Sao không gộp 27 bước để kiểm định? | Lỗi các bước liền kề tự tương quan mạnh (AR(1) ≈ 0,89), gộp sẽ thổi phồng ý nghĩa; đơn vị đúng là mùa |
| 11 | Vì sao đỉnh lũ giảm? | Dữ liệu chỉ cho thấy xu hướng. Đê bao, đập là giả thuyết trong tài liệu (Triet 2017); nhóm không kiểm chứng nhân quả |
| 12 | Sao ML có thể kém ở năm cực trị? | Cây không ngoại suy ra ngoài miền đã thấy. Nhưng ở đây cỡ lũ lẫn với lượng dữ liệu học, nên chưa kết luận được |
| 13 | Ba biến thể LightGBM để làm gì? | Khai báo trước để tránh chọn sau khi xem test; `noclim` là thí nghiệm bỏ climatology để xem lợi thế đến từ đâu |
| 14 | Sao dùng Chronos-2 dù nó không huấn luyện? | Đại diện foundation model mới; zero-shot cho thấy mô hình huấn luyện sẵn trên dữ liệu khác có áp dụng được không. Kết quả: không hơn AnomAR1, kém ở 32 ngày |
| 15 | Chọn tham số thế nào, có nhìn test không? | 6 cấu hình mỗi mô hình, chọn trên 3 mùa cuối của tập huấn luyện, chọn lại ở từng fold; không bao giờ dùng mùa test |
| 16 | Giải thích một tham số bất kỳ | Xem mục 5.1 |
| 17 | Sao thêm vùng khác? | Một vùng, 21 mùa thì kết luận dễ vỡ; thêm vùng để xem kết luận có lặp lại không |

---

## 19. Kiến thức cần nắm theo người

| Người | Cần nắm chắc |
|---|---|
| Quỳnh (AnomAR1) | Tính dừng, ACF/PACF, AR(1), ước lượng φ bằng hồi quy qua gốc, φ^h so với ước lượng trực tiếp, kiểm tra phần dư (Ljung–Box, Q–Q), vì sao không ghép các mùa thành một chuỗi liên tục |
| Bình (GAM) | Spline, tham số làm trơn, hàm trơn s(·) so với hạng tuyến tính l(·), cách đọc đồ thị hàm trơn, kiểm tra phần dư, vì sao GAM là thống kê nhưng linh hoạt |
| Thiện (LightGBM) | Gradient boosting, ý nghĩa từng tham số, vì sao cây không ngoại suy, ngang đặc trưng, chống quá khớp, seed, độ quan trọng đặc trưng / SHAP |
| Phu Hung (còn lại, đánh giá) | Vì sao kiểm định gộp bước là sai, kiểm định cấp mùa, TOST, Holm, block bootstrap, cách đọc bảng theo điều kiện; ARX/Ridge, Random Forest, Chronos-2 |

Mọi người đọc thêm phần của người khác ít nhất một lượt. Nhịp gợi ý: họp ngắn 2 lần/tuần; mỗi tuần mỗi người trình bày 5 phút một module của người khác.

**Tài liệu nền nên đọc:** Vu et al. (2022) Data in Brief, trang 2, 7, 8; Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (otexts.com/fpp3), mục time series cross-validation và ARIMA.

**Khi bí:** tìm trong file này và `docs/decisions.md`; hỏi cả nhóm. Không tự quyết những thứ ảnh hưởng kết quả (lưới, mục tiêu, cách chia) mà không ghi lại.
