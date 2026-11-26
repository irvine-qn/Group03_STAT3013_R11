"""Luật chơi chung của bài so sánh và công thức AnomAR1.

Ba luật chống rò rỉ thông tin từ tương lai:
1. Climatology (mức "bình thường") của một mùa chỉ lấy từ các mùa TRƯỚC nó, không gồm chính nó.
2. Tại gốc dự báo t, chỉ được biết các bước 1..t của mùa đang dự báo.
3. φ chỉ ước lượng từ các mùa huấn luyện, trước mùa test.

Một "dòng" = một gốc dự báo (mùa, bước t) ở tầm h: dự báo y ở bước t + h.
Bước t đánh số từ 1 như trong báo cáo; mảng numpy đánh số từ 0, nên bước t nằm ở cột t − 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 2, 4)                   # tầm dự báo, tính bằng số bước 8 ngày
HORIZON_DAYS = {1: 8, 2: 16, 4: 32}
TEST_SEASONS = range(2010, 2021)       # 11 mùa test
N_PRIOR = 5                            # climatology = trung bình 5 mùa liền trước
MIN_PRIOR = 3                          # cần ít nhất 3 mùa trước mới tính climatology
FIRST_ORIGIN = 3                       # gốc đầu tiên là bước 3 (bảng chung cần 2 trễ)
WINDOW = 7                             # biến thể "_w7": chỉ học 7 mùa gần nhất


def climatology(Y: np.ndarray, season_idx: int) -> np.ndarray:
    """Mức bình thường theo từng bước: trung bình tối đa 5 mùa liền trước mùa season_idx."""
    if season_idx < MIN_PRIOR:
        raise ValueError(f"mùa thứ {season_idx} có ít hơn {MIN_PRIOR} mùa trước")
    return Y[max(0, season_idx - N_PRIOR):season_idx].mean(axis=0)


def forecast_origins(n_steps: int, h: int) -> range:
    """Các gốc dự báo t sao cho đích t + h vẫn nằm trong mùa (t ≤ n_steps − h)."""
    return range(FIRST_ORIGIN, n_steps - h + 1)


def training_seasons(test_season: int, first_year: int, window: int | None = None) -> list[int]:
    """Mùa huấn luyện cho mùa test: từ mùa thứ 4 (đủ 3 mùa trước) đến ngay trước mùa test.

    window: chỉ giữ `window` mùa gần nhất, để lượng dữ liệu học không đổi giữa các mùa test.
    """
    seasons = list(range(first_year + MIN_PRIOR, test_season))
    return seasons[-window:] if window else seasons


def build_rows(Y: np.ndarray, config: dict, seasons, h: int) -> pd.DataFrame:
    """Bảng dòng dự báo của các mùa `seasons` (năm) ở tầm h.

    y_t, y_t1: giá trị ở bước t và t − 1; c_*: climatology ở cùng bước; c_target: ở bước đích.
    """
    first_year, n = config["first_year"], Y.shape[1]
    records = []
    for season in seasons:
        i = season - first_year
        c = climatology(Y, i)
        for t in forecast_origins(n, h):
            o = t - 1
            records.append({
                "season": season, "t": t,
                "y_t": Y[i, o], "y_t1": Y[i, o - 1],
                "c_t": c[o], "c_t1": c[o - 1], "c_target": c[o + h],
                "doy_target": config["first_doy"] + config["step_days"] * (t + h - 1),
                "y_target": Y[i, o + h],
            })
    return pd.DataFrame.from_records(records)


def regression_xy(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Cặp hồi quy của φ: x = anomaly bước t − 1, y = anomaly bước t."""
    x = (rows["y_t1"] - rows["c_t1"]).to_numpy()
    y = (rows["y_t"] - rows["c_t"]).to_numpy()
    return x, y


def estimate_phi(rows: pd.DataFrame) -> float:
    """Hồi quy qua gốc: φ = Σ x·y / Σ x², chỉ trên các dòng huấn luyện."""
    x, y = regression_xy(rows)
    return float((x * y).sum() / (x**2).sum())


def anom_ar1(train: pd.DataFrame, test: pd.DataFrame, h: int) -> np.ndarray:
    """AnomAR1: mức bình thường tại đích + φ^h × độ lệch hôm nay.

    Ý nghĩa: độ lệch so với bình thường tắt dần theo cấp số nhân khi nhìn xa hơn.
    """
    phi = estimate_phi(train)
    return test["c_target"].to_numpy() + phi**h * (test["y_t"] - test["c_t"]).to_numpy()


def persistence(test: pd.DataFrame) -> np.ndarray:
    """Mốc ngây thơ: giá trị hôm nay giữ nguyên tới bước đích."""
    return test["y_t"].to_numpy()


def clim5(test: pd.DataFrame) -> np.ndarray:
    """Mốc chỉ dùng mức bình thường tại bước đích (bỏ qua độ lệch hôm nay)."""
    return test["c_target"].to_numpy()
