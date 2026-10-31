"""So sánh mô hình với baseline ở CẤP MÙA (n = số mùa test), không gộp các bước trong mùa.

Các bước liền nhau tự tương quan mạnh nên gộp chúng sẽ thổi phồng ý nghĩa thống kê.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def holm(p: np.ndarray) -> np.ndarray:
    """Hiệu chỉnh Holm–Bonferroni (p đã hiệu chỉnh, giữ nguyên thứ tự đầu vào)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def compare_to_baseline(season_mse: pd.DataFrame, baseline: str, margin: float = 0.05) -> pd.DataFrame:
    """Mỗi (model, h) so với baseline trên MSE từng mùa.

    rmse_ratio: căn của tỷ số MSE trung bình (<1 là tốt hơn baseline)
    mse_diff_pct: trung bình theo mùa của (MSE_m - MSE_b)/MSE_b, kèm CI 95% t
    p_t / p_wilcoxon: kiểm định cặp hai phía (p_t trên chênh lệch MSE theo mùa)
    p_tost: kiểm định tương đương, H0: |log tỷ số RMSE| >= log(1+margin); p nhỏ = tương đương
    """
    wide = season_mse.pivot_table(index=["h", "season"], columns="model", values="mse")
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
            d = 0.5 * np.log(m / b)  # log tỷ số RMSE theo mùa
            mean_d, se_d = d.mean(), d.std(ddof=1) / np.sqrt(n)
            lim = np.log1p(margin)
            p_low = stats.t.sf((mean_d + lim) / se_d, n - 1)   # H0: d <= -lim
            p_high = stats.t.cdf((mean_d - lim) / se_d, n - 1)  # H0: d >= +lim
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
