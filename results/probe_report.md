# Báo cáo thăm dò: dữ liệu và mô hình dự báo diện tích ngập (ĐBSCL)

> Bản thăm dò của một thành viên để cả nhóm cùng xem, **chưa phải kết quả cuối**. Mọi số được sinh tự động từ
> `results/tables/` bằng `python -m floodbench.report_md`. Xem bản tương tác: `streamlit run app/streamlit_app.py`.

## 1. Tóm tắt

- **Dữ liệu dùng được, nhưng bằng chứng còn yếu hơn vẻ ngoài.** Đỉnh diện tích ngập khớp với mực nước đỉnh ở Tân Châu: r = 0.865 khi để nguyên, còn 0.67 sau khi bỏ xu hướng (CI 95% 0.28–0.87, n = 17). Mới kiểm được ở cấp năm.
- **Chưa mô hình nào thắng AnomAR1 một cách có ý nghĩa** sau hiệu chỉnh Holm (không có trường hợp ngoại lệ), với chỉ 11 mùa test.
- **Tương đương AnomAR1 trong biên ±5% RMSE** (TOST, p chưa hiệu chỉnh đa kiểm định): ARX (OLS) ở 8 ngày (p = 0.0001); GAM ở 8 ngày (p = 0.0017); Ridge có trễ ở 8 ngày (p = 0.0036); Random Forest (anomaly) ở 8 ngày (p = 0.0439); ARX (OLS) ở 16 ngày (p = 0.0429). Nhóm ARX/Ridge/GAM bám sát AnomAR1, đúng với việc chúng cùng dự báo anomaly từ vài trễ.
- **Random Forest thô kém LightGBM thô** ở cùng đặc trưng và ngân sách (RMSE 16 ngày: 1,391 so với 1,284 km²); dự báo anomaly (biến thể `anomaly`) giúp RF bám sát AnomAR1. Ở 8 ngày tỉ lệ RF thô/AnomAR1 là 1.026, ở 32 ngày là 1.063.
- **LightGBM (thô)** nhỉnh hơn về số học ở 8 và 16 ngày (tỉ lệ RMSE 0.965 và 0.933) nhưng khoảng tin cậy chứa 0, nên **chưa kết luận được**.
- **Chronos-2 zero-shot kém hơn** ở 32 ngày (tỉ lệ 1.060, p thô 0.005, sau Holm 0.176).
- **Điều đáng bàn nhất là kết quả theo điều kiện**, không phải "ai thắng" (mục 4).

## 2. Dữ liệu

- Nguồn: Vu et al. (2022), bản đồ lũ MODIS suy diễn từ EVI/LSWI (**không phải quan sát trực tiếp**), gộp thành một chuỗi diện tích ngập (km²) toàn ĐBSCL. 21 mùa × 27 bước (8 ngày/bước).
- Cờ chất lượng: 2 ảnh thiếu (nội suy), 15 ảnh nghi là bản sao, 12 mùa chỉ gồm ảnh lưới 232 m, các mùa còn lại có một phần hoặc toàn bộ ảnh lưới 500 m.
- Độ nhạy ở cấp dữ liệu (chỉ các thống kê dựa trên đỉnh lũ, **chưa** nói gì về sai số dự báo):

| Biến thể | Số mùa | Xu hướng đỉnh (km²/năm) | r (bỏ xu hướng) |
|---|---|---|---|
| default | 21 | -314 | 0.67 |
| drop_imputed | 21 | -314 | 0.67 |
| drop_suspected_copies | 21 | -314 | 0.67 |
| only_232m_seasons | 12 | -306 | 0.67 |

- Đối chiếu mực nước: Tân Châu, đỉnh diện tích với mực nước lớn nhất, 2000–2016. Chỉ ở cấp năm, n = 17.

## 3. Thiết lập thử nghiệm

- Dự báo diện tích ngập sau 8, 16, 32 ngày. Chia theo thời gian: huấn luyện các mùa trước năm Y, kiểm tra năm Y, Y = 2010–2020 (11 mùa test).
- Mọi thống kê tiền xử lý (climatology 5 mùa trước, hệ số φ) chỉ tính từ dữ liệu huấn luyện. Có test cố ý tạo rò rỉ để chắc bộ kiểm tra bắt được.
- AnomAR1 = mức bình thường tại bước đích + φ^h × (lệch hôm nay so với mức bình thường). Đây là mốc thống kê để so sánh.
- Ngân sách tinh chỉnh như nhau (6 cấu hình/mô hình), chọn trên 3 mùa cuối của tập huấn luyện, không đụng tập test. Chi tiết ở `docs/tuning_log.md`.
- So sánh ở **cấp mùa** (n = 11), vì sai số các bước trong cùng mùa tự tương quan mạnh. Dùng t ghép cặp, Wilcoxon, CI của chênh MSE, TOST ±5% RMSE và Holm trên 42 phép so sánh.

## 4. Kết quả

### 4.1 RMSE gộp (km², thấp hơn là tốt hơn), baseline cùng bảng

| Mô hình | 8 ngày | 16 ngày | 32 ngày |
|---|---|---|---|
| LightGBM (thô) | 1,051 | 1,284 | 1,688 |
| LightGBM (không climatology) | 1,059 | 1,301 | 1,778 |
| Random Forest (anomaly) | 1,087 | 1,338 | 1,673 |
| Ridge có trễ | 1,087 | 1,346 | 1,693 |
| ARX (OLS) | 1,082 | 1,354 | 1,680 |
| LightGBM (anomaly) | 1,101 | 1,364 | 1,699 |
| GAM | 1,096 | 1,372 | 1,688 |
| Random Forest (không climatology) | 1,110 | 1,373 | 1,904 |
| AnomAR1 (thống kê, mốc so sánh) | 1,090 | 1,375 | 1,687 |
| Random Forest (thô) | 1,118 | 1,391 | 1,793 |
| Chronos-2 (zero-shot) | 1,116 | 1,412 | 1,788 |
| Climatology 5 mùa | 1,812 | 1,845 | 1,916 |
| Persistence | 1,312 | 2,068 | 3,456 |
| Seasonal naive | 2,215 | 2,249 | 2,336 |
| Climatology toàn bộ | 2,605 | 2,657 | 2,771 |

### 4.2 So với AnomAR1 ở cấp mùa

| Mô hình | Tầm | RMSE / AnomAR1 | Số mùa thắng | Chênh MSE (%) [CI 95%] | p (t, sau Holm) | p TOST ±5% |
|---|---|---|---|---|---|---|
| LightGBM (thô) | 8 ngày | 0.965 | 5/11 | -4.9 [-18.7; +8.9] | 1.000 | 0.363 |
| LightGBM (không climatology) | 8 ngày | 0.972 | 5/11 | -5.3 [-21.0; +10.4] | 1.000 | 0.438 |
| ARX (OLS) | 8 ngày | 0.993 | 8/11 | -0.5 [-4.3; +3.3] | 1.000 | 0.000 |
| Ridge có trễ | 8 ngày | 0.998 | 6/11 | +0.5 [-6.1; +7.2] | 1.000 | 0.004 |
| Random Forest (anomaly) | 8 ngày | 0.998 | 4/11 | +0.2 [-9.9; +10.3] | 1.000 | 0.044 |
| GAM | 8 ngày | 1.006 | 5/11 | +2.2 [-2.8; +7.2] | 1.000 | 0.002 |
| LightGBM (anomaly) | 8 ngày | 1.010 | 6/11 | +3.7 [-7.0; +14.4] | 1.000 | 0.071 |
| Random Forest (không climatology) | 8 ngày | 1.018 | 5/11 | +6.6 [-14.3; +27.6] | 1.000 | 0.221 |
| Chronos-2 (zero-shot) | 8 ngày | 1.024 | 4/11 | +7.7 [-1.0; +16.5] | 1.000 | 0.216 |
| Random Forest (thô) | 8 ngày | 1.026 | 4/11 | +9.5 [-10.6; +29.5] | 1.000 | 0.315 |
| LightGBM (thô) | 16 ngày | 0.933 | 9/11 | -5.3 [-27.0; +16.4] | 1.000 | 0.511 |
| LightGBM (không climatology) | 16 ngày | 0.946 | 7/11 | -8.4 [-24.3; +7.5] | 1.000 | 0.602 |
| Random Forest (anomaly) | 16 ngày | 0.973 | 6/11 | -1.5 [-14.4; +11.4] | 1.000 | 0.187 |
| Ridge có trễ | 16 ngày | 0.979 | 7/11 | -2.6 [-14.0; +8.9] | 1.000 | 0.139 |
| ARX (OLS) | 16 ngày | 0.984 | 7/11 | -1.5 [-10.3; +7.3] | 1.000 | 0.043 |
| LightGBM (anomaly) | 16 ngày | 0.992 | 5/11 | +0.4 [-10.5; +11.2] | 1.000 | 0.074 |
| GAM | 16 ngày | 0.997 | 7/11 | +2.0 [-10.5; +14.5] | 1.000 | 0.053 |
| Random Forest (không climatology) | 16 ngày | 0.999 | 5/11 | +5.4 [-16.7; +27.6] | 1.000 | 0.185 |
| Random Forest (thô) | 16 ngày | 1.011 | 5/11 | +7.2 [-15.1; +29.5] | 1.000 | 0.229 |
| Chronos-2 (zero-shot) | 16 ngày | 1.027 | 2/11 | +7.2 [-2.4; +16.8] | 1.000 | 0.211 |
| Random Forest (anomaly) | 32 ngày | 0.992 | 7/11 | +11.8 [-17.4; +41.0] | 1.000 | 0.342 |
| ARX (OLS) | 32 ngày | 0.996 | 7/11 | +6.6 [-21.9; +35.0] | 1.000 | 0.194 |
| GAM | 32 ngày | 1.001 | 7/11 | +9.4 [-23.0; +41.8] | 1.000 | 0.253 |
| LightGBM (thô) | 32 ngày | 1.001 | 5/11 | +11.8 [-23.9; +47.5] | 1.000 | 0.293 |
| Ridge có trễ | 32 ngày | 1.004 | 6/11 | +6.1 [-23.4; +35.7] | 1.000 | 0.218 |
| LightGBM (anomaly) | 32 ngày | 1.007 | 5/11 | +10.4 [-14.8; +35.6] | 1.000 | 0.325 |
| LightGBM (không climatology) | 32 ngày | 1.054 | 4/11 | +11.9 [-4.3; +28.2] | 1.000 | 0.451 |
| Chronos-2 (zero-shot) | 32 ngày | 1.060 | 2/11 | +14.3 [+4.0; +24.5] | 0.176 | 0.748 |
| Random Forest (thô) | 32 ngày | 1.063 | 3/11 | +26.2 [-20.3; +72.7] | 1.000 | 0.641 |
| Random Forest (không climatology) | 32 ngày | 1.129 | 2/11 | +26.2 [+8.9; +43.4] | 1.000 | 0.950 |

Đọc bảng: tỉ lệ < 1 nghĩa là tốt hơn AnomAR1. CI chứa 0 nghĩa là chưa phân biệt được. p TOST nhỏ nghĩa là có bằng chứng tương đương trong ±5%. p TOST **chưa** hiệu chỉnh đa kiểm định (cột Holm chỉ áp dụng cho p của t ghép cặp).

### 4.3 Theo giai đoạn lũ (tầm 16 ngày, RMSE km²)

| Mô hình | Nước lên | Đỉnh | Nước rút |
|---|---|---|---|
| AnomAR1 (thống kê, mốc so sánh) | 1,154 | 1,391 | 1,578 |
| ARX (OLS) | 1,106 | 1,368 | 1,578 |
| Chronos-2 (zero-shot) | 1,134 | 1,468 | 1,633 |
| Climatology 5 mùa | 1,258 | 1,838 | 2,340 |
| Climatology toàn bộ | 1,797 | 3,081 | 3,089 |
| GAM | 1,121 | 1,408 | 1,584 |
| LightGBM (anomaly) | 1,115 | 1,413 | 1,567 |
| LightGBM (không climatology) | 1,166 | 1,554 | 1,235 |
| LightGBM (thô) | 1,048 | 1,531 | 1,321 |
| Persistence | 1,099 | 2,323 | 2,624 |
| Random Forest (anomaly) | 1,101 | 1,383 | 1,533 |
| Random Forest (không climatology) | 1,187 | 1,458 | 1,497 |
| Random Forest (thô) | 979 | 1,516 | 1,662 |
| Ridge có trễ | 1,067 | 1,359 | 1,596 |
| Seasonal naive | 1,541 | 2,129 | 2,911 |

LightGBM thô tốt hơn AnomAR1 khi nước rút (1,321 so với 1,578) nhưng kém hơn ở đỉnh lũ (1,531 so với 1,391). Bản không dùng climatology cũng tốt khi nước rút (1,235), nên lợi thế này không đến từ climatology.

### 4.4 Theo cỡ lũ của mùa test (tầm 16 ngày, RMSE km²)

| Mô hình | Lũ nhỏ | Lũ vừa | Lũ lớn |
|---|---|---|---|
| AnomAR1 (thống kê, mốc so sánh) | 1,488 | 1,367 | 1,260 |
| ARX (OLS) | 1,440 | 1,348 | 1,267 |
| Chronos-2 (zero-shot) | 1,579 | 1,323 | 1,295 |
| Climatology 5 mùa | 2,105 | 1,474 | 1,818 |
| Climatology toàn bộ | 3,282 | 2,380 | 2,095 |
| GAM | 1,460 | 1,334 | 1,307 |
| LightGBM (anomaly) | 1,457 | 1,366 | 1,263 |
| LightGBM (không climatology) | 1,437 | 1,252 | 1,189 |
| LightGBM (thô) | 1,261 | 1,269 | 1,318 |
| Persistence | 1,830 | 2,154 | 2,220 |
| Random Forest (anomaly) | 1,418 | 1,331 | 1,259 |
| Random Forest (không climatology) | 1,411 | 1,412 | 1,304 |
| Random Forest (thô) | 1,425 | 1,260 | 1,449 |
| Ridge có trễ | 1,432 | 1,344 | 1,257 |
| Seasonal naive | 2,133 | 1,905 | 2,577 |

Với toàn bộ dữ liệu huấn luyện, LightGBM thô tốt hơn ở mùa lũ nhỏ (1,261 so với 1,488). **Nhưng cỡ lũ và lượng dữ liệu học bị lẫn nhau**: các mùa lũ nhỏ đều là những năm cuối, khi mô hình có nhiều mùa học hơn. Khi cố định cửa sổ huấn luyện 7 mùa:

| Mô hình | Lũ nhỏ | Lũ vừa | Lũ lớn |
|---|---|---|---|
| AnomAR1 (thống kê, mốc so sánh) | 1,488 | 1,367 | 1,260 |
| AnomAR1 (thống kê, mốc so sánh) [7 mùa gần nhất] | 1,466 | 1,342 | 1,245 |
| ARX (OLS) [7 mùa gần nhất] | 1,439 | 1,321 | 1,243 |
| GAM [7 mùa gần nhất] | 1,457 | 1,334 | 1,329 |
| LightGBM (anomaly) [7 mùa gần nhất] | 1,498 | 1,432 | 1,383 |
| LightGBM (không climatology) [7 mùa gần nhất] | 1,368 | 1,300 | 1,305 |
| LightGBM (thô) [7 mùa gần nhất] | 1,523 | 1,303 | 1,468 |
| Random Forest (anomaly) [7 mùa gần nhất] | 1,507 | 1,329 | 1,263 |
| Random Forest (không climatology) [7 mùa gần nhất] | 1,322 | 1,368 | 1,428 |
| Random Forest (thô) [7 mùa gần nhất] | 1,679 | 1,338 | 1,625 |
| Ridge có trễ [7 mùa gần nhất] | 1,473 | 1,346 | 1,370 |

ưu thế ở lũ nhỏ biến mất (LightGBM thô 1,523 so với AnomAR1 cùng cửa sổ 1,466) và nó kém hơn ở lũ lớn (1,468 so với 1,245).

### 4.5 Lượng dữ liệu huấn luyện (RMSE km², "[7 mùa gần nhất]" = chỉ dùng 7 mùa liền trước)

| Mô hình | 8 ngày | 16 ngày | 32 ngày |
|---|---|---|---|
| AnomAR1 (thống kê, mốc so sánh) | 1,090 | 1,375 | 1,687 |
| AnomAR1 (thống kê, mốc so sánh) [7 mùa gần nhất] | 1,084 | 1,355 | 1,682 |
| ARX (OLS) [7 mùa gần nhất] | 1,083 | 1,338 | 1,756 |
| GAM | 1,096 | 1,372 | 1,688 |
| GAM [7 mùa gần nhất] | 1,090 | 1,378 | 1,711 |
| LightGBM (anomaly) [7 mùa gần nhất] | 1,164 | 1,439 | 1,811 |
| LightGBM (không climatology) [7 mùa gần nhất] | 1,052 | 1,327 | 1,782 |
| LightGBM (thô) | 1,051 | 1,284 | 1,688 |
| LightGBM (thô) [7 mùa gần nhất] | 1,100 | 1,446 | 1,829 |
| Random Forest (anomaly) [7 mùa gần nhất] | 1,133 | 1,374 | 1,762 |
| Random Forest (không climatology) [7 mùa gần nhất] | 1,107 | 1,374 | 1,895 |
| Random Forest (thô) [7 mùa gần nhất] | 1,202 | 1,573 | 1,869 |
| Ridge có trễ [7 mùa gần nhất] | 1,095 | 1,402 | 1,764 |

## 5. Có đáng làm tiếp không?

Theo tiêu chí tự đặt (đạt ít nhất 2 trong 3): (a) thứ hạng đổi theo điều kiện, (b) có kết luận tương đương hợp lệ, (c) có phát hiện về phương pháp hoặc dữ liệu. Hiện đạt cả ba: nước rút so với đỉnh lũ và lượng dữ liệu học; các mô hình thống kê tuyến tính (ARX, Ridge, GAM) tương đương AnomAR1 ở 8 ngày; cỡ lũ lẫn với lượng dữ liệu học. **Hướng nên theo là "so sánh theo điều kiện", không phải "chứng minh ML thắng".**

## 6. Hạn chế (đọc kỹ trước khi trích số)

- Chỉ 11 mùa test, một lần chạy, một chuỗi (toàn ĐBSCL), CI rộng. Các khác biệt vài phần trăm không đáng tin.
- Không tái lập được con số AnomAR1 = 1.602 km² ở 32 ngày của lần thăm dò trước (bản này ra 1,687). Nguyên nhân chưa rõ, ghi ở `docs/decisions.md`.
- Dữ liệu đích là sản phẩm suy diễn; đối chiếu mực nước mới ở cấp năm và r bỏ xu hướng chỉ khoảng 0,67.
- Chronos-2 chỉ chạy zero-shot với ngữ cảnh anomaly, chưa tinh chỉnh.
- Không kết luận nhân quả (đê, đập); chỉ nói "phù hợp với".

## 7. Việc tiếp theo và câu hỏi cho thầy

1. Truy nguyên chênh lệch 1.602 so với 1.687 km².
2. Chạy lại benchmark sau khi bỏ ảnh nội suy/nghi bản sao; thêm bản 13 mùa test (2008–2020).
3. Thêm ARIMA/SARIMA trong từng mùa (không nối mùa) và MARS nếu cài được; TabPFN khi có API key Prior Labs.
4. Hỏi thầy: GAM/MixedLM có tính là "mô hình truyền thống" không; có được thêm biến phụ (mưa, mực nước) không; foundation model zero-shot có tính là một thuật toán không.
