# Fair Comparison of Statistical and Machine Learning Models for Flood Extent Forecasting in Vietnam

Đồ án môn **STAT3013 – Phân tích thống kê** (lớp R11, Nhóm 03), Trường Đại học Công nghệ Thông tin (UIT).

> **Trạng thái:** đang thực hiện. Báo cáo dự kiến ngày 2026-11-23.

## Câu hỏi nghiên cứu

> Mô hình thống kê có thể cạnh tranh với Machine Learning trong những điều kiện nào? Mô hình nào phù hợp hơn với dữ liệu lũ của từng khu vực?

Thay vì kết luận "mô hình X tốt nhất", nhóm so sánh các mô hình dưới **cùng một giao thức** và báo cáo kết quả **theo điều kiện**:

| Điều kiện | Câu hỏi |
|---|---|
| Tầm dự báo | Lợi thế của ML (nếu có) xuất hiện ở 8, 16 hay 32 ngày? |
| Giai đoạn lũ | Nước lên, đỉnh lũ hay nước rút? |
| Cường độ lũ | Năm lũ lớn/nhỏ bất thường, mô hình nào "gãy" trước? |
| Lượng dữ liệu huấn luyện | Cần bao nhiêu mùa lũ để ML bắt kịp thống kê? |
| Vùng và quy mô không gian | Kết luận có lặp lại giữa các vùng và giữa các cấp độ không? |

## Dữ liệu

- **Vùng chính (Đồng bằng sông Cửu Long):** Vu et al. (2022), *Data in Brief* 42:108268. 565 bản đồ lũ MODIS (8 ngày/ảnh, mùa lũ 2000–2020) và 21 bản đồ sử dụng đất. Dữ liệu: [Mendeley Data, DOI 10.17632/kpftzmsyyz.2](https://data.mendeley.com/datasets/kpftzmsyyz/2).
- **Các vùng bổ sung của Việt Nam:** dự kiến mở rộng để tăng độ tin cậy của kết luận. Nguồn dữ liệu đang được chốt.
- Ảnh vệ tinh được **tổng hợp thành chuỗi số** (diện tích ngập, km²) rồi mới mô hình hóa. Mô hình không nhận ảnh làm đầu vào.
- Dữ liệu thô không nằm trong repo. Xem [data/raw/README.md](data/raw/README.md).

## Phương pháp

- **Biến mục tiêu:** diện tích ngập (km²) sau 8, 16 và 32 ngày.
- **Chia dữ liệu:** rolling-origin theo năm (huấn luyện các năm trước, kiểm tra năm kế tiếp), không xáo trộn, mọi thống kê tiền xử lý chỉ tính từ tập huấn luyện.
- **Baseline:** persistence, climatology, climatology 5 năm, seasonal naive.
- **Thống kê:** AR trên anomaly, ARX/SARIMAX, ETS, hồi quy điều hòa + sai số ARMA, Ridge.
- **Machine Learning:** Random Forest, Gradient Boosting, SVR, MLP (LSTM/GRU là thử nghiệm phụ).
- **Kiểm định:** Diebold–Mariano (hiệu chỉnh HLN) và block bootstrap theo mùa, kiểm định tương đương (TOST), hiệu chỉnh đa kiểm định (Holm).

## Cấu trúc thư mục (mục tiêu)

Mục đánh dấu ✅ đã có, 🔜 sẽ được thêm theo [lộ trình](#lộ-trình).

```
.
├── README.md                     ✅
├── AGENTS.md                     ✅  hướng dẫn cho AI coding agent
├── requirements.txt              ✅
├── pyproject.toml                ✅  để `pip install -e .`
├── data/
│   ├── raw/<dataset>/            ✅  dữ liệu gốc theo bộ (mekong, mekong_hydrology, haiphong_tide), không commit
│   ├── interim/<region>/         ✅  chuỗi/bảng trung gian (CSV nhỏ)
│   └── processed/<region>/       ✅  bảng Level 1 (🔜 Level 2 / 3)
├── configs/
│   └── regions/<region>.yaml     ✅  mỗi vùng một file: lịch ảnh, cột mục tiêu, đường dẫn (mekong.yaml)
├── src/floodbench/               ✅  gói code tái sử dụng
│   ├── data/                     ✅  build_series, level1, schema, regions (🔜 build_zones, build_grid)
│   ├── validation/               ✅  waterlevel (đối chiếu mực nước), sensitivity (độ nhạy dữ liệu)
│   ├── models/                   🔜  baselines, stat, ml, hybrid
│   ├── cv.py                     🔜  rolling-origin theo năm
│   ├── features.py               🔜  lag, anomaly, climatology (chỉ từ tập train)
│   ├── evaluation.py             🔜  RMSE, MAE, MASE, PI, sai số đỉnh
│   └── inference.py              🔜  DM-HLN, bootstrap theo mùa, TOST, MixedLM, Holm
├── tests/                        ✅  pytest (kiểm tra rò rỉ, kiểm định có đáp án biết trước)
├── notebooks/                    ✅  01 audit → 02 EDA → 03 thống kê → 04 ML → 05 benchmark
│                                     → 06 đa vùng → 07 lưới (tùy chọn)
├── results/{tables,figures}/     ✅  đầu ra do code sinh ra
├── reports/                      ✅  full_report, ieee_paper, slides, references
└── docs/                         ✅  decisions.md, tuning_log.md
```

Quy ước chung: logic nằm trong `src/`, notebook chỉ gọi hàm và vẽ hình. Báo cáo chỉ trích số từ `results/`.

## Cài đặt

```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Đã kiểm tra với Python 3.10.11. Sau đó cài gói ở chế độ phát triển và chạy kiểm thử:

```bash
pip install -e .
pytest
```

Dựng lại bảng dữ liệu (sau khi giải nén ảnh gốc theo [data/raw/README.md](data/raw/README.md)):

```bash
python -m floodbench.data.build_series data/raw/mekong/kpftzmsyyz-2 data/interim/mekong/series_mean.csv
python -m floodbench.data.level1            # -> data/processed/mekong/level1_series.csv
python -m floodbench.validation.waterlevel  # -> results/tables/waterlevel_validation.csv
python -m floodbench.validation.sensitivity # -> results/tables/data_sensitivity.csv
python -m floodbench.benchmark             # -> results/tables/probe_predictions.csv
python -m floodbench.report_tables         # -> results/tables/probe_*.csv
python -m floodbench.report_md             # -> results/probe_report.md
streamlit run app/streamlit_app.py         # báo cáo tương tác
```

## Lộ trình

| Giai đoạn | Nội dung | Mốc |
|---|---|---|
| 0 | Chốt phạm vi và cấu trúc repo | 2026-10-23 |
| 1 | Pipeline dữ liệu ĐBSCL (cắt mặt nạ, chuẩn hóa lưới) | 2026-10-24 |
| 2 | EDA, thống kê mô tả và suy diễn | 2026-10-28 |
| 3 | Baseline và mô hình thống kê | 2026-11-01 |
| 4 | Mô hình ML và benchmark | 2026-11-08 |
| 5 | Kiểm định và phân tích theo điều kiện | 2026-11-13 |
| 6 | Mở rộng sang các vùng khác của Việt Nam | đang chốt dữ liệu |
| 7 | Báo cáo, bài IEEE, slide | 2026-11-22 |

## Giới hạn đã biết

- Bản đồ lũ MODIS là sản phẩm suy diễn từ chỉ số EVI/LSWI, không phải quan sát trực tiếp. Lớp "nước" gồm cả sông, ao nuôi và ruộng ngập nước canh tác.
- Chỉ có 21 mùa lũ mỗi vùng, nên kiểm định có sức mạnh thấp. Nhóm báo cáo khoảng tin cậy thay vì chỉ p-value.
- Dữ liệu không có mưa, mực nước hay địa hình. Mô hình ở giai đoạn đầu là đơn biến.

## Thành viên

Nhóm 03, lớp R11. Giảng viên hướng dẫn: thầy Triều.

| Họ tên | MSSV | Vai trò chính |
|---|---|---|
| | | Dữ liệu và pipeline |
| | | Mô hình thống kê |
| | | Machine Learning |
| | | Đánh giá và viết báo cáo |

## Tài liệu tham khảo chính

- Vu, D. T. et al. (2022). Datasets of land use change and flood dynamics in the Vietnamese Mekong Delta. *Data in Brief*, 42, 108268.
- Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2018). Statistical and Machine Learning forecasting methods: Concerns and ways forward. *PLOS ONE*, 13(3), e0194889.
- Hyndman, R. J., Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed.

Danh mục đầy đủ nằm trong [reports/references/](reports/references/).
