"""Đặc trưng dự báo. Mọi thống kê của một mùa chỉ dùng các mùa TRƯỚC nó (chống rò rỉ).

Một dòng = một gốc dự báo (mùa i, bước t) với tầm h: dự báo y ở bước t+h.
Climatology của mùa i là trung bình tối đa `n_prior` mùa liền trước (mùa i không tự dùng
chính nó), nên dòng của mùa huấn luyện cũng không nhìn thấy mùa test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from floodbench.data.regions import RegionConfig

# Các cột mô hình được phép dùng làm đầu vào (không gồm nhãn y_target)
FEATURE_COLUMNS = ["y_t", "y_t1", "y_t2", "c_t", "c_t1", "c_t2", "c_target", "doy_target",
                   "y_prev_season", "clim_all_target"]


def to_wide(level1: pd.DataFrame, cfg: RegionConfig) -> np.ndarray:
    """Ma trận (mùa, bước) của diện tích ngập; hàng i ứng với mùa first_year + i."""
    wide = level1.pivot(index="season", columns="step", values="target_km2")
    wide = wide.reindex(index=list(cfg.seasons), columns=range(1, cfg.n_steps + 1))
    return wide.to_numpy(dtype=float)


def climatology(Y: np.ndarray, i: int, n_prior: int = 5) -> np.ndarray:
    """Trung bình theo bước của tối đa n_prior mùa liền trước mùa i (NaN nếu i = 0)."""
    if i == 0:
        return np.full(Y.shape[1], np.nan)
    return Y[max(0, i - n_prior):i].mean(axis=0)


def build_rows(Y: np.ndarray, cfg: RegionConfig, season_idx, h: int,
               n_prior: int = 5, min_prior: int = 3) -> pd.DataFrame:
    n = Y.shape[1]
    records = []
    for i in season_idx:
        if i < min_prior:
            raise ValueError(f"season index {i} has fewer than {min_prior} prior seasons")
        c = climatology(Y, i, n_prior)
        for t in range(3, n - h + 1):  # t = bước gốc (1-based), cần 2 trễ
            o = t - 1
            records.append({
                "season": cfg.first_year + i, "t": t,
                "y_t": Y[i, o], "y_t1": Y[i, o - 1], "y_t2": Y[i, o - 2],
                "c_t": c[o], "c_t1": c[o - 1], "c_t2": c[o - 2], "c_target": c[o + h],
                "doy_target": cfg.doy_of(t + h),
                "y_prev_season": Y[i - 1, o + h],
                "clim_all_target": Y[:i, o + h].mean(),
                "y_target": Y[i, o + h],
            })
    return pd.DataFrame.from_records(records)
