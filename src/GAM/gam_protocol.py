"""Luật chơi chung của bài so sánh, viết lại cho GAM (không cần gói chung nào khác).

Logic chép đúng giao thức nhóm đã dùng cho benchmark thăm dò (rolling origin theo năm, climatology
5 mùa trước, chọn cấu hình bằng 3 mùa cuối của tập học, kiểm định ở cấp mùa), nên dự báo GAM
không đổi so với lúc còn chạy bằng gói chung.

Ba luật chống rò rỉ thông tin từ tương lai:
1. Climatology (mức "bình thường") của một mùa chỉ lấy từ các mùa TRƯỚC nó.
2. Tại gốc dự báo t, chỉ dùng các bước ≤ t của mùa đang dự báo.
3. Mọi tham số và mọi lựa chọn cấu hình chỉ dùng các mùa huấn luyện, trước mùa test.

Quy ước chỉ số: bước t đánh số từ 1 như trong báo cáo; mảng numpy đánh số từ 0,
nên bước t nằm ở cột t − 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from gam_data import Region

HORIZONS = (1, 2, 4)                # tầm dự báo, tính bằng số bước 8 ngày
HORIZON_DAYS = {1: 8, 2: 16, 4: 32}
FIRST_TEST, LAST_TEST = 2010, 2020  # 11 mùa test
N_PRIOR = 5                         # climatology = trung bình tối đa 5 mùa liền trước
MIN_PRIOR = 3                       # mùa cần ít nhất 3 mùa trước mới được dùng
FIRST_ORIGIN = 3                    # gốc đầu tiên là bước 3 (bảng dòng chung có 2 trễ)
WINDOW = 7                          # biến thể "_w7": chỉ học 7 mùa gần nhất
N_VALIDATION_SEASONS = 3            # chọn cấu hình bằng 3 mùa cuối của tập học


# ---- 1. Mức bình thường -------------------------------------------------------------------------
def climatology(Y: np.ndarray, i: int, n_prior: int = N_PRIOR) -> np.ndarray:
    """Trung bình theo bước của tối đa n_prior mùa liền trước mùa thứ i (NaN nếu i = 0)."""
    if i == 0:
        return np.full(Y.shape[1], np.nan)
    return Y[max(0, i - n_prior):i].mean(axis=0)


# ---- 2. Bảng dòng dự báo -------------------------------------------------------------------------
def build_rows(Y: np.ndarray, region: Region, season_idx, h: int) -> pd.DataFrame:
    """Một dòng = một gốc dự báo (mùa i, bước t) với tầm h: dự báo y ở bước t + h.

    Gốc chạy từ bước 3 đến bước 27 − h (đích phải còn trong mùa), nên mỗi mùa có 27 − 2 − h
    dòng: 24 dòng ở 8 ngày, 23 ở 16 ngày, 21 ở 32 ngày.
    """
    n = Y.shape[1]
    records = []
    for i in season_idx:
        if i < MIN_PRIOR:
            raise ValueError(f"season {i} has fewer than {MIN_PRIOR} prior seasons")
        c = climatology(Y, i)
        for t in range(FIRST_ORIGIN, n - h + 1):
            o = t - 1  # cột của bước gốc
            records.append({
                "season": region.first_year + i, "t": t,
                "y_t": Y[i, o], "y_t1": Y[i, o - 1],         # hôm nay và 8 ngày trước
                "c_t": c[o], "c_t1": c[o - 1], "c_target": c[o + h],
                "doy_target": region.doy_of(t + h),
                "y_target": Y[i, o + h],                      # đáp án (chỉ dùng để học / chấm)
            })
    return pd.DataFrame.from_records(records)


# ---- 3. Chia fold theo năm -----------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    train_seasons: list[int]
    test_season: int


def rolling_origin(region: Region, window: int | None = None):
    """Mỗi mùa test Y (2010..2020) học trên các mùa 2003..Y−1 (hoặc `window` mùa gần nhất)."""
    start = region.first_year + MIN_PRIOR
    for test in range(FIRST_TEST, LAST_TEST + 1):
        train = list(range(start, test))
        if window is not None:
            train = train[-window:]
        yield Fold(train_seasons=train, test_season=test)


# ---- 4. Chọn cấu hình ----------------------------------------------------------------------------
def pick_config(train: pd.DataFrame, configs, fit_predict, h: int):
    """Học trên các mùa huấn luyện trừ 3 mùa cuối, chấm MSE trên 3 mùa cuối, lấy cấu hình tốt nhất.

    fit_predict(train_part, valid_part, config, h) -> dự báo. Trả về (cấu hình tốt nhất, MSE của từng cấu hình).
    """
    seasons = sorted(train["season"].unique())
    cut = seasons[-N_VALIDATION_SEASONS]
    fit, valid = train[train["season"] < cut], train[train["season"] >= cut]
    scores = [float(np.mean((valid["y_target"].to_numpy() - fit_predict(fit, valid, cfg, h)) ** 2))
              for cfg in configs]
    return configs[int(np.argmin(scores))], scores


# ---- 5. Mốc so sánh ------------------------------------------------------------------------------
def anom_ar1(train: pd.DataFrame, test: pd.DataFrame, h: int) -> np.ndarray:
    """AnomAR1: c_{t+h} + φ^h (y_t − c_t); φ ước lượng (hồi quy qua gốc) chỉ từ dòng huấn luyện."""
    a_now = (train["y_t"] - train["c_t"]).to_numpy()
    a_prev = (train["y_t1"] - train["c_t1"]).to_numpy()
    phi = float((a_now * a_prev).sum() / (a_prev ** 2).sum())
    return test["c_target"].to_numpy() + phi ** h * (test["y_t"] - test["c_t"]).to_numpy()


def persistence(test: pd.DataFrame) -> np.ndarray:
    """Baseline: diện tích sau h bước bằng diện tích hôm nay."""
    return test["y_t"].to_numpy()


def clim5(test: pd.DataFrame) -> np.ndarray:
    """Baseline: diện tích sau h bước bằng mức bình thường ở bước đích."""
    return test["c_target"].to_numpy()


# ---- 6. Kiểm định ở cấp mùa ----------------------------------------------------------------------
def season_mse(pred: pd.DataFrame) -> pd.DataFrame:
    """MSE của từng (mô hình, tầm, mùa test). Đơn vị so sánh là mùa, không phải từng bước."""
    p = pred.assign(sq=(pred["y_target"] - pred["pred"]) ** 2)
    return p.groupby(["model", "h", "season"], as_index=False)["sq"].mean().rename(columns={"sq": "mse"})


def holm(p) -> np.ndarray:
    """Hiệu chỉnh Holm–Bonferroni (giữ nguyên thứ tự đầu vào)."""
    p = np.asarray(p, dtype=float)
    adj, running = np.empty(len(p)), 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def compare_to_baseline(smse: pd.DataFrame, baseline: str, margin: float = 0.05) -> pd.DataFrame:
    """Mỗi (mô hình, tầm) so với baseline trên MSE từng mùa.

    rmse_ratio: căn của tỉ số MSE trung bình (< 1 là tốt hơn baseline)
    mse_diff_pct: trung bình theo mùa của (MSE_m − MSE_b)/MSE_b, kèm CI 95% t
    p_t / p_wilcoxon: kiểm định cặp hai phía
    p_tost: tương đương, H0: |log tỉ số RMSE| ≥ log(1 + margin); p nhỏ = tương đương
    p_t_holm: p_t hiệu chỉnh Holm trong mọi dòng của bảng này
    """
    wide = smse.pivot_table(index=["h", "season"], columns="model", values="mse")
    rows = []
    for h, grp in wide.groupby(level=0):
        b = grp[baseline].to_numpy()
        for model in grp.columns:
            if model == baseline:
                continue
            m = grp[model].to_numpy()
            n = len(m)
            rel = (m - b) / b
            se_rel = rel.std(ddof=1) / np.sqrt(n)
            tcrit = stats.t.ppf(0.975, n - 1)
            d = 0.5 * np.log(m / b)  # log tỉ số RMSE theo mùa
            mean_d, se_d = d.mean(), d.std(ddof=1) / np.sqrt(n)
            lim = np.log1p(margin)
            p_low = stats.t.sf((mean_d + lim) / se_d, n - 1)
            p_high = stats.t.cdf((mean_d - lim) / se_d, n - 1)
            try:
                p_w = stats.wilcoxon(m, b).pvalue
            except ValueError:
                p_w = float("nan")
            rows.append({
                "model": model, "h": h, "n_seasons": n,
                "rmse_ratio": float(np.sqrt(m.mean() / b.mean())),
                "wins": int((m < b).sum()),
                "mse_diff_pct": 100 * rel.mean(),
                "mse_diff_pct_ci_low": 100 * (rel.mean() - tcrit * se_rel),
                "mse_diff_pct_ci_high": 100 * (rel.mean() + tcrit * se_rel),
                "p_t": float(stats.ttest_rel(m, b).pvalue),
                "p_wilcoxon": float(p_w),
                "p_tost": float(max(p_low, p_high)),
            })
    out = pd.DataFrame(rows)
    out["p_t_holm"] = holm(out["p_t"].to_numpy())
    return out
