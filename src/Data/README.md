# Data – dựng chuỗi diện tích ngập từ ảnh gốc (code khôi phục)

Thư mục này chứa code đã dựng `data/processed/level1_series.csv`, đầu vào chung của cả 4 mô hình. Code bị gỡ
khỏi repo ở commit `db99215` và được **khôi phục từ commit `db99215^`** (bản ngay trước khi gỡ), từ thư mục
`src/floodbench/data/`. Logic giữ nguyên.

> **Chưa chạy lại được từ ảnh gốc** vì repo không chứa ảnh Mendeley. Đã kiểm tra một phần: chạy `build_level1`
> trên bảng trung gian `series_mean.csv` lấy từ lịch sử git (`git show d35c8aa^:data/interim/mekong/series_mean.csv`)
> cho ra đúng `data/processed/level1_series.csv` (567 dòng, lệch `target_km2` = 0, mọi cờ trùng khớp).

## 1. Các file

| File | Nguồn | Việc làm |
|---|---|---|
| `build_series.py` | `src/floodbench/data/build_series.py` @ `db99215^` (giữ nguyên từng dòng) | Ảnh lũ MODIS → km² ngập mỗi ảnh. Lưới chung 500 m của `Landuse_2000`; mặt nạ ĐBSCL = hợp các pixel ≠ 0 của 21 ảnh sử dụng đất; quy lưới bằng `Resampling.average` trên chỉ thị lớp 1 (nước) và lớp 2 (hỗn hợp); ghi MD5 từng file |
| `level1.py` | `src/floodbench/data/level1.py` @ `db99215^` | Bảng trung gian → bảng Level 1: đủ 21 × 27 (mùa, bước); ảnh thiếu nội suy tuyến tính trong mùa và gắn cờ `imputed`; ảnh trùng MD5 với ảnh sớm hơn gắn cờ `duplicate` (giữ lại); `grid_500m` theo độ phân giải |
| `regions.py` | `src/floodbench/data/regions.py` @ `db99215^` | Đọc cấu hình vùng (`RegionConfig`). Cần cho `level1.py` |
| `schema.py` | `src/floodbench/data/schema.py` @ `db99215^` | Tên cột chung và `validate_long`. Cần cho `level1.py` |

## 2. Dòng đã đổi so với bản gốc (chỉ để bỏ phụ thuộc gói `floodbench`)

| File | Dòng | Gốc | Mới |
|---|---|---|---|
| `level1.py` | 12 | `from floodbench.data.regions import RegionConfig` | `from regions import RegionConfig` |
| `level1.py` | 13 | `from floodbench.data.schema import FLAG_COLUMNS, REQUIRED_COLUMNS` | `from schema import FLAG_COLUMNS, REQUIRED_COLUMNS` |
| `level1.py` | 52 | `from floodbench.data.regions import load_region` | `from regions import load_region` |
| `level1.py` | 53 | `from floodbench.data.schema import validate_long` | `from schema import validate_long` |
| `schema.py` | 6 | `from floodbench.data.regions import RegionConfig` | `from regions import RegionConfig` |
| `regions.py` | 15 | `return Path(__file__).resolve().parents[3]` | `return Path(__file__).resolve().parents[2]` (file nằm sâu ít hơn 1 cấp, `parents[2]` mới là gốc repo) |

`build_series.py` không đổi. Docstring của hai file vẫn ghi lệnh cũ `python -m floodbench.data...`; dùng lệnh ở mục 4.

## 3. Cần gì

- Ảnh gốc: Vu et al. (2022), Mendeley Data, DOI `10.17632/kpftzmsyyz.2`. Giải nén sao cho có hai thư mục con
  `Land use/` (file `Landuse_<năm>.tif`) và `Flooding/` (file `Flood_<năm>_<ngày>.tif` hoặc `.tiff`), ví dụ
  `data/raw/kpftzmsyyz-2/` (đường dẫn `raw_dir` trong `configs/regions/mekong.yaml`). `data/raw/` không được commit.
- Thư viện: `rasterio` (cho `build_series.py`), `pandas`, `numpy`, `PyYAML` (đã có trong `requirements.txt`).
- `configs/regions/mekong.yaml` hiện **thiếu khóa `interim_series`** mà `RegionConfig` bắt buộc. Muốn chạy `level1.py`
  bằng lệnh ở mục 4 thì phải thêm dòng `interim_series: data/interim/mekong/series_mean.csv` vào YAML
  (chưa thêm, để không đổi file đang có). Test trong `tests/Data/` tự thêm khóa này khi dựng cấu hình.

## 4. Chạy (PowerShell, từ gốc repo)

```powershell
# Bước 1: ảnh gốc -> bảng trung gian (một dòng mỗi ảnh có thật, 565 dòng)
.venv\Scripts\python.exe src/Data/build_series.py data/raw/kpftzmsyyz-2 data/interim/mekong/series_mean.csv
# Bước 2: bảng trung gian -> bảng Level 1 (567 dòng), ghi vào processed_level1 của YAML
.venv\Scripts\python.exe src/Data/level1.py mekong
.venv\Scripts\python.exe -m pytest tests/Data
```

Đầu ra cuối: `data/processed/level1_series.csv` (đường dẫn `processed_level1` trong YAML). **Lệnh bước 2 ghi đè bảng
chung**: chỉ chạy khi cố ý dựng lại dữ liệu, và kiểm tra bằng `git diff data/processed/level1_series.csv`.

## 5. Kiểm tra

- `tests/Data/test_level1.py`: các test có đáp án biết trước từ `tests/test_level1.py` cũ (nội suy tuyến tính, cờ
  bản sao chỉ ở bản sau, cờ lưới 500 m, lỗi khi thiếu ảnh đầu mùa), cộng kiểm tra bảng chung đúng schema. Test dựng
  lại từ `series_mean.csv` tự bỏ qua vì bảng trung gian không có trong repo.
- `build_series.py` không có test (cần ảnh gốc).
