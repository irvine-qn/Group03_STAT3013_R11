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
        │
   chronos2_data.py       đọc thành ma trận Y: 21 mùa × 27 bước (km² ngập)
        │
   chronos2_protocol.py   climatology 5 mùa trước, anomaly, gốc dự báo, mùa huấn luyện, AnomAR1
        │
   chronos2_model.py      ngữ cảnh anomaly bước 1..t → Chronos-2 → trung vị 4 bước tới
        ▼
   chronos2_forecast.py   (bước 1) 11 mùa test × mọi gốc t × tầm 8/16/32 ngày → predictions.csv
        ▼
   chronos2_evaluate.py   (bước 2) RMSE, kiểm định cấp mùa, bảng theo điều kiện, hình
```

Ngữ cảnh đưa vào là **anomaly** (giá trị − climatology) của **chính mùa đang dự báo**, bước 1 đến t.
Lý do: chu kỳ lên–đỉnh–rút đã biết qua climatology, nên model chỉ cần dự báo phần lệch; các mùa
cách nhau khoảng 150 ngày khô nên không nối thành một chuỗi liền. AnomAR1 dùng đúng thông tin này.

## 4. Chạy lại

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

.venv\Scripts\python.exe src/Chronus-2/chronos2_forecast.py        # bước 1, lần đầu tải trọng số model từ Hugging Face
.venv\Scripts\python.exe src/Chronus-2/chronos2_evaluate.py        # bước 2
.venv\Scripts\python.exe src/Chronus-2/chronos2_finetune_check.py  # kiểm tra phụ, cần GPU (mục 7)
.venv\Scripts\python.exe -m pytest tests/Chronus-2                 # dùng pipeline giả, không cần model
```

Kiểm tra lại kết quả:
- Dự báo chạy lại trên CPU khớp bản thăm dò ban đầu của nhóm tới sai số làm tròn máy (≤ 1e-12 km²).
- Chạy lại zero-shot trên GPU (RTX 4050, torch CUDA cài riêng trên máy, `requirements.txt` vẫn là bản CPU):
  1496 dự báo lệch bản CPU tối đa 0,002 km²; RMSE, tỉ lệ, số mùa thắng và mọi p-value trùng tới 4 chữ số.
  Kết quả chính thức vẫn là bản CPU.

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

Theo vị trí gốc dự báo trong mùa (`rmse_by_season_part.csv`). **Bảng mô tả, lập sau khi xem hình
`rmse_by_origin.png`**: mốc chia t = 21 chọn sau khi đã thấy dữ liệu, không khai báo trước, không kiểm định.

| Tầm | Gốc t ≤ 20: AnomAR1 / Chronos-2 | Gốc t ≥ 21: AnomAR1 / Chronos-2 |
|---|--:|--:|
| 8 ngày | 974 / 1005 | 1380 / 1396 |
| 16 ngày | 1285 / 1311 | 1661 / 1726 |
| 32 ngày | 1687 / 1703 | 1689 / **2236** |

Hình: `results/Chronus-2/figures/rmse_by_horizon.png`, `rmse_by_origin.png`, `example_seasons.png`.

## 6. Đọc kết quả

- Chronos-2 **không tốt hơn** AnomAR1 ở tầm nào. Ở 8 và 16 ngày khác biệt không có ý nghĩa;
  ở 32 ngày **kém hơn có ý nghĩa** (RMSE cao hơn 6%, CI của chênh MSE nằm hẳn trên 0).
- Ở 32 ngày, **phần chênh lớn nhất tính theo km²** nằm ở cuối mùa: gốc t ≥ 21 (đích rơi vào cuối
  giai đoạn nước rút) cho 2236 so với 1689; gộp các gốc t ≤ 20 thì gần ngang nhau (1703 so với 1687).
  Nhưng ở vài gốc đầu tiên Chronos-2 **cũng kém hơn** khoảng 10–18% (t = 3: 705 so với 597;
  t = 5: 1247 so với 1124, `rmse_by_origin.csv`), chỉ là sai số đầu mùa nhỏ về km² nên bị che khi
  gộp. Phần chênh cuối mùa do một vài mùa chi phối (2012, 2014, 2015, 2019, cũng là các mùa có dự
  báo âm), chỉ có 3 gốc × 11 mùa. Đây là mô tả, chưa đủ để khẳng định nguyên nhân.
- Chronos-2 nhỉnh hơn một chút khi nước lên (1134 so với 1154 ở 16 ngày) nhưng kém ở đỉnh và khi
  nước rút. Các lát cắt này chỉ để mô tả, chưa kiểm định riêng.

## 7. Kiểm tra phụ: fine-tune có đổi kết luận không? (thăm dò)

`chronos2_finetune_check.py` fine-tune Chronos-2 trên GPU cho **từng** mùa test, chỉ học từ anomaly
của các mùa trước nó (2003..Y−1), rồi dự báo như bản zero-shot. Hai cách: `full` (sửa mọi trọng số,
50 bước, learning rate 1e-6) và `lora` (chỉ học thêm vài ma trận nhỏ, 50 bước, learning rate 1e-5).

RMSE gộp, km² (`results/Chronus-2/tables/finetune_check.csv`):

| Mô hình | 8 ngày | 16 ngày | 32 ngày | 32 ngày, gốc t ≤ 20 | 32 ngày, gốc t ≥ 21 |
|---|--:|--:|--:|--:|--:|
| AnomAR1 | 1090 | 1375 | 1687 | 1687 | 1689 |
| Chronos-2 zero-shot | 1116 | 1412 | 1788 | 1703 | 2236 |
| Fine-tune full | 1121 | 1403 | 1758 | 1671 | 2213 |
| Fine-tune LoRA | 1185 | 1501 | 1939 | 1846 | 2425 |

- Fine-tune nhẹ (`full`) gần như **không đổi** so với zero-shot (khoảng ±2%) và vẫn kém AnomAR1 ở 32 ngày.
- Fine-tune mạnh hơn (`lora`) **tệ hơn** zero-shot ở cả ba tầm: với 7–17 mùa × 27 điểm, model dễ học thuộc.
- Phần chênh ở cuối mùa (gốc t ≥ 21) **không** giảm khi fine-tune.
- Khi thử trên riêng mùa 2020, bản LoRA lại trông tốt nhất ở 8 ngày; trên cả 11 mùa thì tệ nhất. Đây là
  ví dụ thật cho lý do giao thức cấm chọn cấu hình bằng mùa test.

**Vì sao chỉ là thăm dò:** hai cấu hình được chọn sau khi đã thử trên mùa 2020 (có nhìn mùa test),
không theo quy trình chọn 6 cấu hình bằng 3 mùa cuối của tập học như các mô hình khác; mỗi cấu hình
chạy một lần trên GPU. Vì vậy bảng này không vào bảng so sánh chính và không kiểm định; nó chỉ cho thấy
fine-tune **không làm đổi kết luận**.

## 8. Hạn chế và cách nói an toàn

- Nói "trên 11 mùa test, Chronos-2 zero-shot không tốt hơn AnomAR1, và kém hơn ở 32 ngày; fine-tune thử
  nghiệm cũng không đổi kết luận". Không khái quát thành "foundation model kém": chỉ một model, một cách
  đưa ngữ cảnh.
- "Không khác có ý nghĩa" ở 8 và 16 ngày **không** có nghĩa là tương đương: p TOST > 0,05.
- **Holm ở đây chỉ hiệu chỉnh trong 3 phép so sánh** (Chronos-2 × 3 tầm). Bảng tổng hợp của nhóm
  so sánh nhiều mô hình hơn nên p Holm sẽ lớn hơn (bản thăm dò với 42 phép cho 0,18 ở 32 ngày).
- Chỉ 11 mùa test và một chuỗi toàn vùng, nên khoảng tin cậy rộng.
- Cả hai mô hình đôi khi dự báo **diện tích âm** ở cuối mùa (AnomAR1 8 lần, Chronos-2 15 lần, các mùa
  2012, 2014, 2015, 2019). Không cắt về 0 để giữ đúng giao thức chung của nhóm.
- Mục tiêu là sản phẩm suy diễn từ ảnh MODIS (Vu et al. 2022), không phải đo trực tiếp.
- Không giải thích nhân quả (đê bao, đập); chỉ nói "phù hợp với".

## 9. Câu hỏi vấn đáp dự kiến

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
Bản chính đánh giá cách dùng zero-shot. Nhóm có thử fine-tune (mục 7), học lại riêng cho từng mùa test:
bản nhẹ gần như không đổi, bản mạnh hơn tệ đi vì chỉ có 7–17 mùa để học, và vẫn không thắng AnomAR1.
Muốn đưa fine-tune vào bảng chính thì phải chọn cấu hình đúng giao thức (6 cấu hình, validation 3 mùa cuối).

**Chạy trên GPU thì kết quả có khác không?**
Bản zero-shot chạy lại trên GPU lệch tối đa 0,002 km² so với CPU, mọi bảng và p-value giữ nguyên.

**φ của AnomAR1 sao khác nhau giữa các tầm?**
φ được ước lượng trên đúng các dòng huấn luyện của từng tầm (gốc t = 3..27 − h), giống cách các mô hình khác
được huấn luyện ở tầm đó, nên tập dòng hơi khác nhau.

**Sao chạy Chronos-2 một lần cho mỗi gốc mà có đủ 3 tầm?**
Mỗi lần gọi xin luôn 4 bước tới; tầm h lấy bước thứ h. Đây đúng là cách bản thăm dò ban đầu của nhóm làm, nên số khớp.
