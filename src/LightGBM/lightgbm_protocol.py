"""Luật chơi chung của bài so sánh: fold rolling origin, bảng đặc trưng, chọn cấu hình, mốc AnomAR1.

Ba luật chống rò rỉ thông tin từ tương lai:
1. Climatology (mức "bình thường") của một mùa chỉ lấy từ các mùa TRƯỚC nó, không gồm chính nó.
2. Tại gốc dự báo t, chỉ được biết các bước 1..t của mùa đang dự báo.
3. Siêu tham số chọn bằng 3 mùa cuối của tập huấn luyện; mùa test không bao giờ được nhìn.

Một "dòng" = một gốc dự báo (mùa, bước t) ở tầm h: dự báo y ở bước t + h.
Bước t đánh số từ 1 như trong báo cáo; mảng numpy đánh số từ 0, nên bước t nằm ở cột t − 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZONS = (1, 2, 4)                   # tầm dự báo, tính bằng số bước 8 ngày
HORIZON_DAYS = {1: 8, 2: 16, 4: 32}
FIRST_TEST, LAST_TEST = 2010, 2020     # 11 mùa test
N_PRIOR = 5                            # climatology = trung bình 5 mùa liền trước
MIN_PRIOR = 3                          # cần ít nhất 3 mùa trước mới tính climatology
FIRST_ORIGIN = 3                       # gốc đầu tiên là bước 3 (cần 2 trễ y_t1, y_t2)
N_VALIDATION_SEASONS = 3               # chọn cấu hình bằng 3 mùa cuối của tập học
# Giai đoạn lũ theo ngày đích: [từ ngày, đến trước ngày, tên]
PHASES = [(0, 257, "rising"), (257, 298, "peak"), (298, 999, "falling")]


@dataclass(frozen=True)
class Fold:
    train_seasons: list[int]
    test_season: int


def rolling_origin(config: dict, first_test: int = FIRST_TEST, last_test: int = LAST_TEST,
                   window: int | None = None):
    """Mỗi mùa test Y học trên các mùa (first_year + 3)..Y−1, hoặc `window` mùa gần nhất."""
    start = config["first_year"] + MIN_PRIOR
    for test in range(first_test, last_test + 1):
        train = list(range(start, test))
        yield Fold(train_seasons=train[-window:] if window else train, test_season=test)


def climatology(Y: np.ndarray, season_idx: int) -> np.ndarray:
    """Mức bình thường theo từng bước: trung bình tối đa 5 mùa liền trước mùa season_idx."""
    if season_idx < MIN_PRIOR:
        raise ValueError(f"season {season_idx} has fewer than {MIN_PRIOR} prior seasons")
    return Y[max(0, season_idx - N_PRIOR):season_idx].mean(axis=0)


def build_rows(Y: np.ndarray, config: dict, seasons, h: int) -> pd.DataFrame:
    """Bảng đặc trưng của các mùa `seasons` (năm) ở tầm h.

    y_t, y_t1, y_t2: diện tích ở bước t, t − 1, t − 2; c_*: climatology ở cùng bước;
    c_target: climatology ở bước đích; doy_target: ngày trong năm của bước đích.
    """
    first_year, n = config["first_year"], Y.shape[1]
    records = []
    for season in seasons:
        i = season - first_year
        c = climatology(Y, i)
        for t in range(FIRST_ORIGIN, n - h + 1):
            o = t - 1
            records.append({
                "season": season, "t": t,
                "y_t": Y[i, o], "y_t1": Y[i, o - 1], "y_t2": Y[i, o - 2],
                "c_t": c[o], "c_t1": c[o - 1], "c_t2": c[o - 2], "c_target": c[o + h],
                "doy_target": config["first_doy"] + config["step_days"] * (t + h - 1),
                "y_target": Y[i, o + h],
            })
    return pd.DataFrame.from_records(records)


def pick_config(train: pd.DataFrame, configs: list, fit_predict, h: int):
    """Chọn cấu hình bằng validation BÊN TRONG tập huấn luyện (3 mùa cuối làm validation).

    fit_predict(train_part, valid_part, config, h) -> dự báo. Trả về (cấu hình tốt nhất, MSE từng cấu hình).
    """
    seasons = sorted(train["season"].unique())
    cut = seasons[-N_VALIDATION_SEASONS]
    fit, valid = train[train["season"] < cut], train[train["season"] >= cut]
    scores = []
    for cfg in configs:
        pred = fit_predict(fit, valid, cfg, h)
        scores.append(float(np.mean((valid["y_target"].to_numpy() - pred) ** 2)))
    return configs[int(np.argmin(scores))], scores


def anom_ar1(train: pd.DataFrame, test: pd.DataFrame, h: int) -> np.ndarray:
    """Mốc AnomAR1: c_{t+h} + φ^h (y_t − c_t); φ hồi quy qua gốc chỉ trên các dòng huấn luyện."""
    a_now = (train["y_t"] - train["c_t"]).to_numpy()
    a_prev = (train["y_t1"] - train["c_t1"]).to_numpy()
    phi = float((a_now * a_prev).sum() / (a_prev**2).sum())
    return test["c_target"].to_numpy() + phi**h * (test["y_t"] - test["c_t"]).to_numpy()


def clim5(train: pd.DataFrame, test: pd.DataFrame, h: int) -> np.ndarray:
    """Mốc chỉ dùng mức bình thường tại bước đích."""
    return test["c_target"].to_numpy()
