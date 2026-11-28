"""Hằng số, đường dẫn và vài hàm nhỏ dùng chung cho mọi bước của GAM (src/GAM/).

Không có logic mô hình ở đây; mô hình nằm ở `gam_model.py`, luật chơi chung ở `gam_protocol.py`.
Bố cục theo quy ước chung của nhóm: code ở src/GAM/, kết quả ở results/GAM/,
dữ liệu sinh ra ở data/processed/GAM/, test ở tests/GAM/.
"""
from __future__ import annotations

import numpy as np

from gam_data import REPO_ROOT, load_flood_matrix, load_region
from gam_protocol import HORIZON_DAYS, HORIZONS, WINDOW  # noqa: F401  (dùng lại ở các bước sau)

# ---- Đường dẫn -----------------------------------------------------------------------------
DATASET = REPO_ROOT / "data" / "processed" / "GAM"     # bảng đầu vào để xem (bước 1)
TABLES = REPO_ROOT / "results" / "GAM" / "tables"      # bảng kết quả (bước 2, 3)
FIGURES = REPO_ROOT / "results" / "GAM" / "figures"    # hình (bước 4)

# ---- Thiết lập thử nghiệm ----------------------------------------------------------------------
REGION = "mekong"
DAYS = HORIZON_DAYS             # đổi tầm (bước) sang ngày để ghi lên bảng/hình
BASELINE = "anom_ar1"           # mốc thống kê chính để so sánh

# Ranh giới giai đoạn lũ theo ngày trong năm của bước đích (cùng ranh giới với bảng thăm dò của nhóm)
PHASE_EDGES = [0, 257, 298, 999]
PHASE_NAMES = ["rising", "peak", "falling"]
PHASE_LABEL = {"rising": "Rising", "peak": "Peak", "falling": "Falling"}
SIZE_NAMES = ["small", "medium", "large"]
SIZE_LABEL = {"small": "Small flood", "medium": "Medium flood", "large": "Large flood"}


def load_data():
    """Trả về (cfg, Y). Y có 21 hàng (mùa 2000..2020) x 27 cột (bước 1..27), đơn vị km²."""
    cfg = load_region(REGION)
    return cfg, load_flood_matrix(cfg)


def season_index(cfg, seasons) -> list[int]:
    """Đổi năm (2010) sang chỉ số hàng của Y (10). `build_rows` nhận chỉ số hàng."""
    return [s - cfg.first_year for s in seasons]


def rmse(err) -> float:
    """Căn bậc hai của sai số bình phương trung bình."""
    return float(np.sqrt(np.mean(np.square(err))))


def ensure_dirs() -> None:
    for d in (DATASET, TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
