"""Bước 3 – Chẩn đoán AnomAR1: φ^h so với φ ước lượng trực tiếp, kiểm định phần dư, hình.

Chạy (sau anomar1_forecast.py):  python src/AnomAR1/anomar1_diagnostics.py
Đọc:  results/AnomAR1/tables/predictions.csv, phi_by_fold.csv
Ghi:  results/AnomAR1/tables/phi_direct_sensitivity.csv, residual_tests.csv, residual_summary.csv
      results/AnomAR1/figures/residual_diagnostics.png, phi_by_fold.png, typical_seasons.png
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")  # vẽ ra file, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from anomar1_data import FIGURES, TABLES, load_config, load_flood_matrix
from anomar1_evaluate import BASELINE, load_predictions, rmse
from anomar1_protocol import HORIZON_DAYS, HORIZONS, TEST_SEASONS, build_rows, training_seasons


def direct_phi(train: pd.DataFrame) -> float:
    """Hồi quy qua gốc của anomaly ĐÍCH theo anomaly hôm nay, riêng cho một tầm."""
    x = (train["y_t"] - train["c_t"]).to_numpy()
    y = (train["y_target"] - train["c_target"]).to_numpy()
    return float((x * y).sum() / (x**2).sum())


def compare_direct_phi(Y: np.ndarray, config: dict, pred: pd.DataFrame) -> pd.DataFrame:
    """RMSE khi dùng φ^h (định nghĩa chính) so với φ ước lượng thẳng cho từng tầm (kiểm tra phụ)."""
    rows = []
    for test_season in TEST_SEASONS:
        seasons = training_seasons(test_season, config["first_year"])
        for h in HORIZONS:
            train = build_rows(Y, config, seasons, h)
            test = build_rows(Y, config, [test_season], h)
            phi = direct_phi(train)
            rows.append(test.assign(h=h, pred=test["c_target"] + phi * (test["y_t"] - test["c_t"])))
    direct = pd.concat(rows)
    return pd.DataFrame([
        {"h": h, "days": HORIZON_DAYS[h],
         "rmse_phi_pow_h": rmse(pred[pred["h"] == h]),
         "rmse_direct_phi": rmse(direct[direct["h"] == h])}
        for h in HORIZONS
    ])


def residual_tests(pred: pd.DataFrame) -> pd.DataFrame:
    """Mỗi (tầm, mùa): sai lệch trung bình, ACF trễ 1..5 của sai số, Ljung–Box trễ 10."""
    rows = []
    for (h, season), grp in pred.groupby(["h", "season"]):
        grp = grp.sort_values("t")
        err = (grp["y_target"] - grp["pred"]).to_numpy()
        r = acf(err, nlags=5, fft=False)
        rows.append({
            "h": h, "season": season, "bias_km2": err.mean(),
            **{f"acf_{k}": r[k] for k in range(1, 6)},
            "ljung_box_p": float(acorr_ljungbox(err, lags=[10])["lb_pvalue"].iloc[0]),
        })
    return pd.DataFrame(rows)


def summarize_residuals(tests: pd.DataFrame) -> pd.DataFrame:
    return tests.groupby("h").agg(
        mean_bias_km2=("bias_km2", "mean"),
        mean_acf_1=("acf_1", "mean"),
        seasons_lb_p_below_05=("ljung_box_p", lambda p: int((p < 0.05).sum())),
        n_seasons=("season", "size"),
    )


def plot_diagnostics(pred: pd.DataFrame, tests: pd.DataFrame, phi: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, h in enumerate(HORIZONS):
        p = pred[pred["h"] == h]
        sm.qqplot(p["y_target"] - p["pred"], line="s", ax=axes[0, col], markersize=3)
        axes[0, col].set_title(f"Q-Q plot of errors, {HORIZON_DAYS[h]} days")
        mean_acf = tests[tests["h"] == h][[f"acf_{k}" for k in range(1, 6)]].mean()
        axes[1, col].bar(range(1, 6), mean_acf)
        axes[1, col].axhline(0, color="black", linewidth=0.8)
        axes[1, col].set(title=f"Mean within-season ACF, {HORIZON_DAYS[h]} days",
                         xlabel="Lag (steps)", ylim=(-0.5, 1))
    fig.tight_layout()
    fig.savefig(FIGURES / "residual_diagnostics.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for h in HORIZONS:
        p = phi[(phi["model"] == BASELINE) & (phi["h"] == h)]
        ax.plot(p["test_season"], p["phi"], marker="o", label=f"Rows of {HORIZON_DAYS[h]}-day horizon")
    ax.set(xlabel="Test season", ylabel="phi", title="Estimated phi by fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "phi_by_fold.png", dpi=150)
    plt.close(fig)


def plot_seasons(pred: pd.DataFrame, seasons: list[int]) -> None:
    fig, axes = plt.subplots(len(seasons), 3, figsize=(13, 6), sharey="row")
    for row, season in enumerate(seasons):
        for col, h in enumerate(HORIZONS):
            p = pred[(pred["season"] == season) & (pred["h"] == h)].sort_values("doy_target")
            axes[row, col].plot(p["doy_target"], p["y_target"], color="black", label="Observed")
            axes[row, col].plot(p["doy_target"], p["pred"], color="tab:blue", label="Forecast")
            axes[row, col].set(title=f"{season}, {HORIZON_DAYS[h]} days", xlabel="Day of year", ylabel="km²")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "typical_seasons.png", dpi=150)
    plt.close(fig)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    FIGURES.mkdir(parents=True, exist_ok=True)
    config = load_config("mekong")
    Y = load_flood_matrix(config)
    all_pred = load_predictions()
    pred = all_pred[all_pred["model"] == BASELINE]
    phi = pd.read_csv(TABLES / "phi_by_fold.csv")

    direct = compare_direct_phi(Y, config, pred)
    tests = residual_tests(pred)
    summary = summarize_residuals(tests)
    direct.to_csv(TABLES / "phi_direct_sensitivity.csv", index=False)
    tests.to_csv(TABLES / "residual_tests.csv", index=False)
    summary.to_csv(TABLES / "residual_summary.csv")

    plot_diagnostics(pred, tests, phi)
    peaks = pred.groupby("season")["y_target"].max()
    plot_seasons(pred, [int(peaks.idxmax()), int(peaks.idxmin())])

    print("φ^h so với φ trực tiếp:\n", direct.round(3).to_string(index=False), "\n")
    print("Phần dư:\n", summary.round(3).to_string())
    print("Mùa test lũ lớn nhất / nhỏ nhất:", int(peaks.idxmax()), int(peaks.idxmin()))


if __name__ == "__main__":
    main()
