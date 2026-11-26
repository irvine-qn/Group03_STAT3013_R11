"""Luật chơi chung của bài so sánh, viết lại tối giản cho Chronos-2 và mốc AnomAR1.

Ba luật chống rò rỉ thông tin từ tương lai:
1. Climatology (mức "bình thường") của một mùa chỉ lấy từ các mùa TRƯỚC nó, không gồm chính nó.
2. Tại gốc dự báo t, chỉ được biết các bước 1..t của mùa đang dự báo.
3. Mọi tham số (ở đây là φ của AnomAR1) chỉ ước lượng từ các mùa huấn luyện, trước mùa test.

Quy ước chỉ số: bước t và gốc dự báo đánh số từ 1 như trong báo cáo; mảng numpy đánh số từ 0,
nên bước t nằm ở cột t − 1.
"""
from __future__ import annotations

import numpy as np

HORIZONS = (1, 2, 4)                   # tầm dự báo, tính bằng số bước 8 ngày
HORIZON_DAYS = {1: 8, 2: 16, 4: 32}
TEST_SEASONS = range(2010, 2021)       # 11 mùa test
N_PRIOR = 5                            # climatology = trung bình 5 mùa liền trước
MIN_PRIOR = 3                          # cần ít nhất 3 mùa trước mới tính climatology
FIRST_ORIGIN = 3                       # gốc đầu tiên là bước 3, giống bảng chung (các mô hình khác cần 2 trễ)


def climatology(Y: np.ndarray, season_idx: int) -> np.ndarray:
    """Mức bình thường theo từng bước: trung bình tối đa 5 mùa liền trước mùa season_idx."""
    if season_idx < MIN_PRIOR:
        raise ValueError(f"mùa thứ {season_idx} có ít hơn {MIN_PRIOR} mùa trước")
    first = max(0, season_idx - N_PRIOR)
    return Y[first:season_idx].mean(axis=0)


def anomaly(Y: np.ndarray, season_idx: int) -> np.ndarray:
    """Độ lệch của mùa so với mức bình thường của nó (dương = ngập nhiều hơn bình thường)."""
    return Y[season_idx] - climatology(Y, season_idx)


def forecast_origins(n_steps: int, h: int) -> range:
    """Các gốc dự báo t sao cho đích t + h vẫn nằm trong mùa (t ≤ n_steps − h)."""
    return range(FIRST_ORIGIN, n_steps - h + 1)


def training_seasons(test_season: int, first_year: int) -> list[int]:
    """Mùa huấn luyện cho mùa test: từ mùa thứ 4 (đủ 3 mùa trước) đến ngay trước mùa test."""
    return list(range(first_year + MIN_PRIOR, test_season))


def estimate_phi(Y: np.ndarray, train_idx: list[int], h: int) -> float:
    """Hệ số tự tương quan φ của anomaly: hồi quy qua gốc a[t] theo a[t − 1].

    φ = Σ a[t]·a[t−1] / Σ a[t−1]², tính trên các gốc dự báo của tầm h ở các mùa huấn luyện.
    Dùng đúng các dòng mà mô hình khác được huấn luyện ở tầm h, nên φ hơi khác nhau giữa các tầm.
    """
    a_now, a_prev = [], []
    for season_idx in train_idx:
        a = anomaly(Y, season_idx)
        for origin in forecast_origins(Y.shape[1], h):
            a_now.append(a[origin - 1])    # bước t
            a_prev.append(a[origin - 2])   # bước t − 1
    a_now, a_prev = np.array(a_now), np.array(a_prev)
    return float((a_now * a_prev).sum() / (a_prev**2).sum())


def anom_ar1_forecast(Y: np.ndarray, season_idx: int, origin: int, h: int, phi: float) -> float:
    """AnomAR1: mức bình thường tại đích + φ^h × độ lệch hôm nay.

    Ý nghĩa: độ lệch so với bình thường tắt dần theo cấp số nhân khi nhìn xa hơn.
    """
    c = climatology(Y, season_idx)
    a_today = Y[season_idx, origin - 1] - c[origin - 1]
    return float(c[origin + h - 1] + phi**h * a_today)
