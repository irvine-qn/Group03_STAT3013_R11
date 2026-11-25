"""Hằng số, đường dẫn và vài hàm nhỏ dùng chung cho mọi bước của GAM (src/GAM/).

Không có logic mô hình ở đây; mô hình nằm ở `gam_model.py`.
Bố cục theo quy ước chung của nhóm: code ở src/GAM/, kết quả ở results/GAM/,
dữ liệu sinh ra ở data/processed/GAM/, test ở tests/GAM/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from floodbench.data.regions import load_region, repo_root  # cấu hình vùng (lịch ảnh, đường dẫn)
from floodbench.features import to_wide  # bảng dài (mùa, bước) -> ma trận Y[mùa, bước]

# ---- Đường dẫn -----------------------------------------------------------------------------
ROOT = repo_root()                               # thư mục gốc repo
DATASET = ROOT / "data" / "processed" / "GAM"     # bảng đầu vào để xem (bước 1)
TABLES = ROOT / "results" / "GAM" / "tables"      # bảng kết quả (bước 2, 3)
FIGURES = ROOT / "results" / "GAM" / "figures"    # hình (bước 4)
# Bảng dự báo chung của cả nhóm (do `python -m floodbench.benchmark` sinh). Chỉ ĐỌC, không ghi.
TEAM_PREDICTIONS = repo_root() / "results" / "tables" / "probe_predictions.csv"

# ---- Thiết lập thử nghiệm (giống hệt benchmark của nhóm) -------------------------------------
REGION = "mekong"
HORIZONS = (1, 2, 4)            # tầm dự báo tính bằng SỐ BƯỚC; 1 bước = 8 ngày
DAYS = {1: 8, 2: 16, 4: 32}     # đổi tầm (bước) sang ngày để ghi lên bảng/hình
WINDOW = 7                      # biến thể "_w7": chỉ học 7 mùa gần nhất
BASELINE = "anom_ar1"           # mốc thống kê chính để so sánh

# Ranh giới giai đoạn lũ theo ngày trong năm của bước đích (giống floodbench.report_tables.PHASES)
PHASE_EDGES = [0, 257, 298, 999]
PHASE_NAMES = ["rising", "peak", "falling"]
PHASE_VI = {"rising": "Nước lên", "peak": "Đỉnh lũ", "falling": "Nước rút"}
SIZE_NAMES = ["small", "medium", "large"]
SIZE_VI = {"small": "Lũ nhỏ", "medium": "Lũ vừa", "large": "Lũ lớn"}


def load_data():
    """Trả về (cfg, Y). Y có 21 hàng (mùa 2000..2020) x 27 cột (bước 1..27), đơn vị km²."""
    cfg = load_region(REGION)
    Y = to_wide(pd.read_csv(cfg.path("processed_level1")), cfg)
    return cfg, Y


def season_index(cfg, seasons) -> list[int]:
    """Đổi năm (2010) sang chỉ số hàng của Y (10). `build_rows` nhận chỉ số hàng."""
    return [s - cfg.first_year for s in seasons]


def rmse(err) -> float:
    """Căn bậc hai của sai số bình phương trung bình."""
    return float(np.sqrt(np.mean(np.square(err))))


def ensure_dirs() -> None:
    for d in (DATASET, TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
