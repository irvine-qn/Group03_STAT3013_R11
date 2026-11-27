"""Bước 3 – Kiểm định phần dư thống nhất cho 4 mô hình chính (chỉ đọc kết quả đã có, không học lại mô hình).

Chạy (sau khi cả 4 mô hình đã ghi predictions.csv):  python src/Comparison/comparison_residuals.py
Đọc:  results/<Mô hình>/tables/predictions.csv  (anom_ar1 lấy từ thư mục AnomAR1)
Ghi:  results/Comparison/tables/
        residual_tests_all.csv      mỗi (mô hình, tầm, mùa test): n, bias, ACF trễ 1–5, p Ljung–Box trễ 10
        residual_summary_all.csv    mỗi (mô hình, tầm): bias, SD, xiên, nhọn dư, Shapiro–Wilk trên sai số gộp,
                                    ACF1–3 trung bình qua mùa, số và tỉ lệ mùa có Ljung–Box p < 0,05
        paired_diff_normality.csv   giả định chuẩn của các kiểm định so sánh cấp mùa (n = 11)
      results/Comparison/figures/residual_diagnostics_all.png   Q–Q và ACF trung bình, mô hình × tầm

Quy ước: sai số e = y_target − pred (> 0 nghĩa là dự báo THẤP hơn thực tế), sắp theo (h, season, t).
ACF và Ljung–Box dùng đúng công thức của src/GAM/gam_analysis.py (test so với statsmodels).

Lưu ý khi đọc Shapiro–Wilk trên sai số gộp: các sai số trong cùng một mùa tự tương quan mạnh, nên các quan sát
không độc lập; p chỉ mang tính tham khảo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from comparison_tables import BASELINE, FIGURES, HORIZON_DAYS, HORIZONS, SOURCES, TABLES, read_predictions

MAIN_MODELS = ["anom_ar1", "gam", "lgbm_raw", "chronos2"]  # mốc + 3 bản chính, khai báo trước
CONTENDERS = MAIN_MODELS[1:]
N_ACF = 5          # ACF trễ 1..5 cho từng mùa
LB_LAGS = 10       # Ljung–Box đến trễ 10 (giống AnomAR1 và GAM)
ALPHA = 0.05       # ngưỡng đếm số mùa có Ljung–Box p < 0,05
SHAPIRO_NOTE = "pooled errors are autocorrelated within seasons; p is indicative only"


# ---------- Đọc dữ liệu ----------


def owner(model: str) -> str:
    """Thư mục sở hữu một mô hình (anom_ar1 thuộc thư mục AnomAR1)."""
    for source, models in SOURCES.items():
        if model in models:
            return source
    raise KeyError(model)


def load_main(results: Path | None = None) -> pd.DataFrame:
    """Dự báo của 4 mô hình chính, mỗi mô hình lấy từ thư mục sở hữu nó, kèm cột sai số `err`."""
    parts = []
    for model in MAIN_MODELS:
        source = owner(model)
        pred = read_predictions(source) if results is None else read_predictions(source, results)
        parts.append(pred[pred["model"] == model])
    pred = pd.concat(parts, ignore_index=True)
    pred = pred.assign(err=pred["y_target"] - pred["pred"])
    return pred.sort_values(["model", "h", "season", "t"], ignore_index=True)


# ---------- Công thức ----------


def acf(x, nlags: int) -> np.ndarray:
    """Tự tương quan mẫu r_1..r_nlags (cùng công thức với gam_analysis.acf và statsmodels acf)."""
    x = np.asarray(x, float) - np.mean(x)
    denom = np.sum(x * x)
    return np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, nlags + 1)])


def ljung_box_p(x, lags: int) -> float:
    """Ljung–Box: H0 = không tự tương quan đến trễ `lags`. Q = n(n+2) Σ r_k²/(n−k) ~ χ²(lags)."""
    n = len(x)
    r = acf(x, lags)
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1)))
    return float(stats.chi2.sf(q, lags))


def season_tests(err) -> dict:
    """Chẩn đoán sai số của một (mô hình, tầm, mùa): n, bias, ACF trễ 1..5, p Ljung–Box trễ 10."""
    err = np.asarray(err, float)
    r = acf(err, N_ACF)
    return {"n": len(err), "bias_km2": float(err.mean()),
            **{f"acf_{k}": float(r[k - 1]) for k in range(1, N_ACF + 1)},
            "ljung_box_p_lag10": ljung_box_p(err, LB_LAGS)}


def pooled_summary(err) -> dict:
    """Tóm tắt sai số gộp mọi mùa: bias, SD (ddof = 1), xiên, nhọn dư, Shapiro–Wilk (W, p)."""
    err = np.asarray(err, float)
    sw = stats.shapiro(err)
    return {"n": len(err), "bias_km2": float(err.mean()), "sd_km2": float(err.std(ddof=1)),
            "skew": float(stats.skew(err)), "excess_kurtosis": float(stats.kurtosis(err)),
            "shapiro_w": float(sw.statistic), "shapiro_p": float(sw.pvalue)}


def paired_quantities(m, b) -> dict:
    """Ba đại lượng theo mùa mà các kiểm định so sánh dùng (m, b = MSE từng mùa của mô hình và mốc).

    mse_diff        MSE_m − MSE_b                (t ghép cặp p_t giả định đại lượng này chuẩn)
    rel_mse_diff    (MSE_m − MSE_b) / MSE_b      (CI của mse_diff_pct giả định đại lượng này chuẩn)
    log_rmse_ratio  ½·ln(MSE_m / MSE_b)          (TOST giả định đại lượng này chuẩn)
    """
    m, b = np.asarray(m, float), np.asarray(b, float)
    return {"mse_diff": m - b, "rel_mse_diff": (m - b) / b, "log_rmse_ratio": 0.5 * np.log(m / b)}


def normality_row(values) -> dict:
    """n, trung bình, SD, Shapiro–Wilk (W, p) của một dãy giá trị."""
    v = np.asarray(values, float)
    sw = stats.shapiro(v)
    return {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "shapiro_w": float(sw.statistic), "shapiro_p": float(sw.pvalue)}


# ---------- Bảng ----------


def residual_tests_all(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, h, season), grp in pred.groupby(["model", "h", "season"], sort=False):
        rows.append({"model": model, "h": int(h), "h_days": HORIZON_DAYS[int(h)], "season": int(season),
                     **season_tests(grp.sort_values("t")["err"])})
    out = pd.DataFrame(rows)
    return _order(out, ["h", "season"])


def residual_summary_all(pred: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, h), grp in pred.groupby(["model", "h"], sort=False):
        ps = tests[(tests["model"] == model) & (tests["h"] == h)]
        below = int((ps["ljung_box_p_lag10"] < ALPHA).sum())
        rows.append({"model": model, "h": int(h), "h_days": HORIZON_DAYS[int(h)],
                     **pooled_summary(grp["err"]),
                     **{f"mean_acf_{k}": float(ps[f"acf_{k}"].mean()) for k in (1, 2, 3)},
                     "n_seasons": len(ps), "seasons_lb_p_lt_0.05": below,
                     "share_seasons_lb_p_lt_0.05": below / len(ps),
                     "note": SHAPIRO_NOTE})
    return _order(pd.DataFrame(rows), ["h"])


def season_mse(pred: pd.DataFrame) -> pd.DataFrame:
    """MSE từng (mô hình, tầm, mùa test): đơn vị của các kiểm định so sánh."""
    return (pred.assign(sq=pred["err"] ** 2).groupby(["model", "h", "season"], as_index=False)["sq"]
                .mean().rename(columns={"sq": "mse"}))


def paired_diff_normality(pred: pd.DataFrame) -> pd.DataFrame:
    wide = season_mse(pred).pivot_table(index=["h", "season"], columns="model", values="mse")
    rows = []
    for model in CONTENDERS:
        for h in HORIZONS:
            m, b = wide.loc[h, model].to_numpy(), wide.loc[h, BASELINE].to_numpy()
            for quantity, values in paired_quantities(m, b).items():
                rows.append({"model": model, "h": h, "h_days": HORIZON_DAYS[h], "quantity": quantity,
                             **normality_row(values)})
    return pd.DataFrame(rows)


def _order(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rank = frame["model"].map({m: k for k, m in enumerate(MAIN_MODELS)})
    return (frame.assign(_rank=rank).sort_values(["_rank"] + by).drop(columns="_rank")
                 .reset_index(drop=True))


# ---------- Hình ----------


def plot_diagnostics(pred: pd.DataFrame, tests: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(MAIN_MODELS), 2 * len(HORIZONS), figsize=(18, 11))
    for i, model in enumerate(MAIN_MODELS):
        for j, h in enumerate(HORIZONS):
            err = pred[(pred["model"] == model) & (pred["h"] == h)]["err"].to_numpy()
            (osm, osr), (slope, intercept, _) = stats.probplot(err)
            ax = axes[i, 2 * j]
            ax.plot(osm, osr, ".", markersize=3)
            ax.plot(osm, slope * np.asarray(osm) + intercept, color="black", linewidth=0.8)
            ax.set_title(f"{model}, {HORIZON_DAYS[h]} d: Q-Q", fontsize=8)
            mean_acf = tests[(tests["model"] == model) & (tests["h"] == h)][[f"acf_{k}" for k in range(1, 6)]].mean()
            ax = axes[i, 2 * j + 1]
            ax.bar(range(1, 6), mean_acf)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set(ylim=(-0.5, 1))
            ax.set_title(f"{model}, {HORIZON_DAYS[h]} d: mean ACF", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pred = load_main()
    tests = residual_tests_all(pred)
    summary = residual_summary_all(pred, tests)
    normality = paired_diff_normality(pred)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    tests.to_csv(TABLES / "residual_tests_all.csv", index=False)
    summary.to_csv(TABLES / "residual_summary_all.csv", index=False)
    normality.to_csv(TABLES / "paired_diff_normality.csv", index=False)
    plot_diagnostics(pred, tests, FIGURES / "residual_diagnostics_all.png")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summary.drop(columns="note").round(4).to_string(index=False), "\n")
        print(normality.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
