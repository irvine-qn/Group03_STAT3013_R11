# Chronos-2 zero-shot

> **Bắt đầu từ [`notebooks/Chronus-2/chronos2_overview.ipynb`](../../notebooks/Chronus-2/chronos2_overview.ipynb)**:
> notebook đi qua từng bước bằng hàm thật và chỉ sang file chi tiết bên dưới.

## 1. Chronos-2 là gì

Chronos-2 (`amazon/chronos-2`) là một **mô hình nền (foundation model)** cho chuỗi thời gian:
một mạng transformer đã được huấn luyện trước trên rất nhiều chuỗi thời gian khác nhau.
Đưa vào một đoạn chuỗi (ngữ cảnh), model trả về phân phối dự báo cho các bước tiếp theo.

Ở đây dùng kiểu **zero-shot**: không huấn luyện lại, không tinh chỉnh siêu tham số trên dữ liệu lũ.
Lấy **trung vị** (phân vị 0,5) làm dự báo điểm.

## 2. Vì sao chọn, thuộc họ nào

- Đại diện cho hướng "mô hình lớn dùng ngay" đang được quảng bá nhiều, để trả lời câu hỏi: không
  cần học gì từ dữ liệu ĐBSCL thì có thắng được mốc thống kê đơn giản không?
- Không phải mô hình thống kê (không có giả định tham số, không có suy diễn trên hệ số), cũng không
  phải ML truyền thống (không học từ dữ liệu của nhóm). Xếp riêng thành nhóm **foundation model**.
- Mốc so sánh là **AnomAR1** (mốc thống kê chính của nhóm): mức bình thường + φ^h × độ lệch hôm nay.

## 3. Luồng dữ liệu và thứ tự đọc code

```
data/processed/level1_series.csv
        │  flood_data.py      đọc thành ma trận Y: 21 mùa × 27 bước (km² ngập)
        ▼
   protocol.py               climatology 5 mùa trước, anomaly, gốc dự báo, mùa huấn luyện, AnomAR1
        │
   chronos_model.py          ngữ cảnh anomaly bước 1..t → Chronos-2 → trung vị 4 bước tới
        ▼
   run_forecast.py  (bước 1) 11 mùa test × mọi gốc t × tầm 8/16/32 ngày → predictions.csv
        ▼
   evaluate.py      (bước 2) RMSE, kiểm định cấp mùa, bảng theo điều kiện, hình
```

Ngữ cảnh đưa vào là **anomaly** (giá trị − climatology) của **chính mùa đang dự báo**, bước 1 đến t.
Lý do: chu kỳ lên–đỉnh–rút đã biết qua climatology, nên model chỉ cần dự báo phần lệch; các mùa
cách nhau khoảng 150 ngày khô nên không nối thành một chuỗi liền. AnomAR1 dùng đúng thông tin này.

## 4. Chạy lại

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

.venv\Scripts\python.exe src/Chronus-2/run_forecast.py   # bước 1, lần đầu tải trọng số model từ Hugging Face
.venv\Scripts\python.exe src/Chronus-2/evaluate.py       # bước 2
.venv\Scripts\python.exe -m pytest tests/Chronus-2       # 26 test, dùng pipeline giả, không cần model
```

Dự báo chạy lại khớp bản thăm dò ban đầu của nhóm tới sai số làm tròn máy (≤ 1e-12 km²).

## 5. Kết quả (11 mùa test 2010–2020)

RMSE gộp, km² (`results/Chronus-2/tables/rmse.csv`):

| Mô hình | 8 ngày | 16 ngày | 32 ngày |
|---|--:|--:|--:|
| AnomAR1 | 1090 | 1375 | 1687 |
| Chronos-2 | 1116 | 1412 | 1788 |

So với AnomAR1 ở cấp mùa, n = 11 (`results/Chronus-2/tables/vs_anom_ar1.csv`):

| Tầm | Tỉ lệ RMSE | Mùa thắng | Chênh MSE (CI 95%) | p t | p Wilcoxon | p TOST ±5% | p Holm |
|---|--:|--:|--:|--:|--:|--:|--:|
| 8 ngày | 1,024 | 4/11 | +7,7% (−1,0; +16,5) | 0,303 | 0,175 | 0,216 | 0,605 |
| 16 ngày | 1,027 | 2/11 | +7,2% (−2,4; +16,8) | 0,346 | 0,175 | 0,211 | 0,605 |
| 32 ngày | 1,060 | 2/11 | +14,3% (+4,0; +24,5) | 0,005 | 0,010 | 0,748 | 0,015 |

Theo điều kiện, tầm 16 ngày (`rmse_by_phase_h2.csv`, `rmse_by_size_h2.csv`):

| | Nước lên | Đỉnh | Nước rút | Lũ nhỏ | Lũ vừa | Lũ lớn |
|---|--:|--:|--:|--:|--:|--:|
| AnomAR1 | 1154 | 1391 | 1578 | 1488 | 1367 | 1260 |
| Chronos-2 | 1134 | 1468 | 1633 | 1579 | 1323 | 1295 |

Theo vị trí gốc dự báo trong mùa (`rmse_by_season_part.csv`):

| Tầm | Gốc t ≤ 20: AnomAR1 / Chronos-2 | Gốc t ≥ 21: AnomAR1 / Chronos-2 |
|---|--:|--:|
| 8 ngày | 974 / 1005 | 1380 / 1396 |
| 16 ngày | 1285 / 1311 | 1661 / 1726 |
| 32 ngày | 1687 / 1703 | 1689 / **2236** |

Hình: `results/Chronus-2/figures/rmse_by_horizon.png`, `rmse_by_origin.png`, `example_seasons.png`.

## 6. Đọc kết quả

- Chronos-2 **không tốt hơn** AnomAR1 ở tầm nào. Ở 8 và 16 ngày khác biệt không có ý nghĩa;
  ở 32 ngày **kém hơn có ý nghĩa** (RMSE cao hơn 6%, CI của chênh MSE nằm hẳn trên 0).
- Phần thua ở 32 ngày tập trung ở **cuối mùa**: với gốc t ≤ 20 hai mô hình gần như ngang nhau
  (1687 so với 1703), với gốc t ≥ 21 (đích rơi vào cuối giai đoạn nước rút) Chronos-2 sai nhiều hơn
  hẳn (2236 so với 1689). Vậy điểm yếu **không** đến từ ngữ cảnh ngắn đầu mùa.
- Chronos-2 nhỉnh hơn một chút khi nước lên (1134 so với 1154 ở 16 ngày) nhưng kém ở đỉnh và khi
  nước rút. Các lát cắt này chỉ để mô tả, chưa kiểm định riêng.

## 7. Hạn chế và cách nói an toàn

- Nói "trên 11 mùa test, Chronos-2 zero-shot không tốt hơn AnomAR1, và kém hơn ở 32 ngày".
  Không khái quát thành "foundation model kém": chỉ một model, một cách đưa ngữ cảnh, không tinh chỉnh.
- "Không khác có ý nghĩa" ở 8 và 16 ngày **không** có nghĩa là tương đương: p TOST > 0,05.
- **Holm ở đây chỉ hiệu chỉnh trong 3 phép so sánh** (Chronos-2 × 3 tầm). Bảng tổng hợp của nhóm
  so sánh nhiều mô hình hơn nên p Holm sẽ lớn hơn (bản thăm dò với 42 phép cho 0,18 ở 32 ngày).
- Chỉ 11 mùa test và một chuỗi toàn vùng, nên khoảng tin cậy rộng.
- Cả hai mô hình đôi khi dự báo **diện tích âm** ở cuối mùa (AnomAR1 8 lần, Chronos-2 15 lần, các mùa
  2012, 2014, 2015, 2019). Không cắt về 0 để giữ đúng giao thức chung của nhóm.
- Mục tiêu là sản phẩm suy diễn từ ảnh MODIS (Vu et al. 2022), không phải đo trực tiếp.
- Không giải thích nhân quả (đê bao, đập); chỉ nói "phù hợp với".

## 8. Câu hỏi vấn đáp dự kiến

**Zero-shot nghĩa là gì, vậy Chronos-2 có "học" gì từ dữ liệu không?**
Không. Trọng số giữ nguyên; dữ liệu ĐBSCL chỉ dùng để tính climatology (trừ chu kỳ mùa) và làm ngữ cảnh.

**Vì sao đưa anomaly mà không đưa km² thô?**
Để Chronos-2 và AnomAR1 dùng cùng thông tin: cả hai dự báo phần lệch so với mức bình thường rồi cộng lại climatology.

**Có rò rỉ dữ liệu tương lai không?**
Không: climatology chỉ dùng 5 mùa trước, ngữ cảnh chỉ có bước 1..t. Có test cố ý đổi dữ liệu tương lai
thành 1e9 và kiểm tra kết quả không đổi (`tests/Chronus-2/test_chronos2_protocol.py`, `test_chronos2_model.py`).

**Vì sao kiểm định theo mùa (n = 11) mà không dùng hơn 700 dự báo?**
Sai số các bước trong một mùa tương quan rất mạnh, coi chúng độc lập sẽ thổi phồng ý nghĩa. Mùa là đơn vị độc lập hợp lý.

**TOST là gì, sao không kết luận "tương đương" ở 8 ngày?**
TOST kiểm định giả thuyết "chênh ít nhất ±5%"; chỉ khi bác bỏ được (p < 0,05) mới nói tương đương.
Ở đây p TOST ≈ 0,21 nên chưa đủ bằng chứng.

**Vì sao không tinh chỉnh (fine-tune) Chronos-2?**
Mục tiêu là đánh giá cách dùng zero-shot; tinh chỉnh trên chỉ 7–17 mùa huấn luyện dễ quá khớp, và sẽ phải dành
ngân sách tinh chỉnh ngang với các mô hình khác.

**φ của AnomAR1 sao khác nhau giữa các tầm?**
φ được ước lượng trên đúng các dòng huấn luyện của từng tầm (gốc t = 3..27 − h), giống cách các mô hình khác
được huấn luyện ở tầm đó, nên tập dòng hơi khác nhau.

**Sao chạy Chronos-2 một lần cho mỗi gốc mà có đủ 3 tầm?**
Mỗi lần gọi xin luôn 4 bước tới; tầm h lấy bước thứ h. Đây đúng là cách bản thăm dò ban đầu của nhóm làm, nên số khớp.
