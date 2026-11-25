"""Bước 2 – Đánh giá Chronos-2 so với mốc AnomAR1, sinh bảng và hình.

Chạy (sau run_forecast.py):  python src/Chronus-2/evaluate.py
Đọc:  results/Chronus-2/tables/predictions.csv
Ghi:  results/Chronus-2/tables/
        rmse.csv               RMSE gộp theo mô hình × tầm
        vs_anom_ar1.csv        kiểm định cấp mùa so với AnomAR1
        rmse_by_phase_h2.csv   RMSE theo giai đoạn lũ (tầm 16 ngày)
        rmse_by_size_h2.csv    RMSE theo cỡ lũ của mùa test (tầm 16 ngày)
        rmse_by_origin.csv     RMSE theo gốc t, tức theo độ dài ngữ cảnh
      results/Chronus-2/figures/
        rmse_by_horizon.png, rmse_by_origin.png, example_seasons.png

Vì sao kiểm định ở CẤP MÙA: sai số các bước liền nhau trong một mùa tương quan rất mạnh
(AR(1) của anomaly ≈ 0,89), nên coi 27 bước là 27 quan sát độc lập sẽ thổi phồng ý nghĩa.
Đơn vị so sánh là mùa: n = 11.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # vẽ ra file, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from flood_data import REPO_ROOT, load_config, load_flood_matrix, step_to_doy
from protocol import HORIZON_DAYS
from run_forecast import PREDICTIONS_PATH

TABLES = REPO_ROOT / "results" / "Chronus-2" / "tables"
FIGURES = REPO_ROOT / "results" / "Chronus-2" / "figures"
BASELINE = "anom_ar1"
TOST_MARGIN = 0.05  # "tương đương" nghĩa là RMSE chênh không quá ±5%
# Giai đoạn lũ theo ngày đích: [từ ngày, đến trước ngày, tên]
PHASES = [(0, 257, "rising"), (257, 298, "peak"), (298, 999, "falling")]
LABELS = {"chronos2": "Chronos-2", "anom_ar1": "AnomAR1"}


# ---------- Đọc dữ liệu ----------

def load_predictions(path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"không thấy {path}; hãy chạy run_forecast.py trước")
    return pd.read_csv(path)


# ---------- Thước đo sai số ----------

def pooled_rmse(pred: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """RMSE gộp mọi dòng trong từng nhóm `by`."""
    squared = pred.assign(sq_error=(pred["y_target"] - pred["pred"]) ** 2)
    rmse = squared.groupby(by)["sq_error"].mean() ** 0.5
    return rmse.rename("rmse").reset_index()


def season_mse(pred: pd.DataFrame) -> pd.DataFrame:
    """MSE của từng (mô hình, tầm, mùa test): mỗi mùa là một quan sát cho kiểm định."""
    squared = pred.assign(mse=(pred["y_target"] - pred["pred"]) ** 2)
    return squared.groupby(["model", "h", "season"], as_index=False)["mse"].mean()


# ---------- Kiểm định cấp mùa ----------

def holm(p_values: np.ndarray) -> np.ndarray:
    """Hiệu chỉnh Holm–Bonferroni: p nhỏ nhất nhân m, p kế tiếp nhân m − 1, ..., giữ đơn điệu."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(np.argsort(p_values)):
        running_max = max(running_max, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def _relative_mse_diff(model_mse: np.ndarray, base_mse: np.ndarray) -> dict:
    """Chênh MSE tương đối theo mùa, (M − B)/B, trung bình và khoảng tin cậy 95% (phân phối t)."""
    rel = (model_mse - base_mse) / base_mse
    n = len(rel)
    half_width = stats.t.ppf(0.975, n - 1) * rel.std(ddof=1) / np.sqrt(n)
    return {
        "mse_diff_pct": 100 * rel.mean(),
        "mse_diff_pct_ci_low": 100 * (rel.mean() - half_width),
        "mse_diff_pct_ci_high": 100 * (rel.mean() + half_width),
    }


def _tost_p(model_mse: np.ndarray, base_mse: np.ndarray, margin: float) -> float:
    """Kiểm định tương đương TOST trên log tỉ số RMSE theo mùa.

    H0: chênh ít nhất ±margin. p nhỏ ⇒ bác bỏ H0 ⇒ kết luận "tương đương trong ±margin".
    Lưu ý: "không khác có ý nghĩa" KHÔNG đồng nghĩa "tương đương"; phải dùng TOST.
    """
    d = 0.5 * np.log(model_mse / base_mse)  # log(RMSE_m / RMSE_b) của từng mùa
    n, limit = len(d), np.log1p(margin)
    se = d.std(ddof=1) / np.sqrt(n)
    p_lower = stats.t.sf((d.mean() + limit) / se, n - 1)   # H0: d ≤ −limit
    p_upper = stats.t.cdf((d.mean() - limit) / se, n - 1)  # H0: d ≥ +limit
    return float(max(p_lower, p_upper))


def _wilcoxon_p(model_mse: np.ndarray, base_mse: np.ndarray) -> float:
    try:
        return float(stats.wilcoxon(model_mse, base_mse).pvalue)
    except ValueError:  # mọi chênh lệch bằng 0
        return float("nan")


def compare_with_anom_ar1(season_errors: pd.DataFrame, margin: float = TOST_MARGIN) -> pd.DataFrame:
    """So mỗi mô hình với AnomAR1 ở từng tầm, trên MSE của 11 mùa test.

    rmse_ratio < 1 nghĩa là tốt hơn AnomAR1; wins = số mùa mô hình có MSE thấp hơn.
    Holm hiệu chỉnh trong họ các phép so sánh của bảng này (Chronos-2 × 3 tầm).
    """
    wide = season_errors.pivot_table(index=["h", "season"], columns="model", values="mse")
    rows = []
    models = [name for name in wide.columns if name != BASELINE]
    horizons = wide.index.get_level_values("h").unique()
    for model in models:
        for h in horizons:
            m = wide.loc[h, model].to_numpy()       # MSE của mô hình ở từng mùa
            b = wide.loc[h, BASELINE].to_numpy()    # MSE của AnomAR1 ở cùng các mùa
            rows.append(_one_comparison(model, h, m, b, margin))
    result = pd.DataFrame(rows)
    result["p_t_holm"] = holm(result["p_t"].to_numpy())
    return result


def _one_comparison(model: str, h: int, m: np.ndarray, b: np.ndarray, margin: float) -> dict:
    """Mọi con số so sánh của một mô hình ở một tầm."""
    return {
        "model": model, "h": int(h), "n_seasons": len(m),
        "rmse_ratio": float(np.sqrt(m.mean() / b.mean())),
        "wins": int((m < b).sum()),
        **_relative_mse_diff(m, b),
        "p_t": float(stats.ttest_rel(m, b).pvalue),
        "p_wilcoxon": _wilcoxon_p(m, b),
        "p_tost": _tost_p(m, b, margin),
    }


# ---------- Chia theo điều kiện ----------

def add_flood_phase(pred: pd.DataFrame) -> pd.DataFrame:
    """Gắn giai đoạn lũ (nước lên / đỉnh / nước rút) theo ngày của bước đích."""
    edges = [start for start, _, _ in PHASES] + [PHASES[-1][1]]
    names = [name for _, _, name in PHASES]
    phase = pd.cut(pred["doy_target"], edges, right=False, labels=names)
    return pred.assign(phase=phase.astype(str))


def add_flood_size(pred: pd.DataFrame) -> pd.DataFrame:
    """Gắn cỡ lũ của mùa test: chia 11 mùa thành 3 nhóm theo đỉnh diện tích ngập."""
    peak = pred.groupby("season")["y_target"].max()
    size = pd.qcut(peak, 3, labels=["small", "medium", "large"])
    return pred.assign(size=pred["season"].map(size).astype(str))


# ---------- Hình ----------

def plot_rmse_by_horizon(rmse: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.38
    for k, model in enumerate(["anom_ar1", "chronos2"]):
        values = rmse.loc[model]
        x = np.arange(len(values)) + (k - 0.5) * width
        ax.bar(x, values, width, label=LABELS[model])
    ax.set_xticks(np.arange(len(rmse.columns)), [f"{HORIZON_DAYS[h]} days" for h in rmse.columns])
    ax.set_ylabel("RMSE (km²)")
    ax.set_title("Pooled RMSE, 11 test seasons (2010–2020)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rmse_by_origin(by_origin: pd.DataFrame, path: Path) -> None:
    horizons = sorted(by_origin["h"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(12, 3.6), sharey=True)
    for ax, h in zip(axes, horizons):
        for model in ["anom_ar1", "chronos2"]:
            part = by_origin[(by_origin["h"] == h) & (by_origin["model"] == model)]
            ax.plot(part["t"], part["rmse"], marker=".", label=LABELS[model])
        ax.set_title(f"Horizon {HORIZON_DAYS[h]} days")
        ax.set_xlabel("Origin step t (= context length)")
    axes[0].set_ylabel("RMSE (km²)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_example_seasons(pred: pd.DataFrame, Y: np.ndarray, config: dict, path: Path,
                         seasons=(2011, 2015), h: int = 4) -> None:
    doys = [step_to_doy(config, s) for s in range(1, config["n_steps"] + 1)]
    fig, axes = plt.subplots(1, len(seasons), figsize=(11, 3.8), sharey=True)
    for ax, season in zip(axes, seasons):
        ax.plot(doys, Y[season - config["first_year"]], color="black", label="Observed")
        for model in ["anom_ar1", "chronos2"]:
            part = pred[(pred["season"] == season) & (pred["h"] == h) & (pred["model"] == model)]
            ax.plot(part["doy_target"], part["pred"], marker=".", label=f"{LABELS[model]} ({HORIZON_DAYS[h]} d)")
        ax.set_title(f"Season {season}")
        ax.set_xlabel("Day of year")
    axes[0].set_ylabel("Flooded area (km²)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- Chạy ----------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pred = load_predictions()
    config = load_config("mekong")
    Y = load_flood_matrix(config)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    rmse = pooled_rmse(pred, ["model", "h"]).pivot(index="model", columns="h", values="rmse")
    tests = compare_with_anom_ar1(season_mse(pred))
    h2 = pred[pred["h"] == 2]
    by_phase = pooled_rmse(add_flood_phase(h2), ["model", "phase"]).pivot(
        index="model", columns="phase", values="rmse")[["rising", "peak", "falling"]]
    by_size = pooled_rmse(add_flood_size(h2), ["model", "size"]).pivot(
        index="model", columns="size", values="rmse")[["small", "medium", "large"]]
    by_origin = pooled_rmse(pred, ["model", "h", "t"])

    rmse.round(0).to_csv(TABLES / "rmse.csv")
    tests.round(4).to_csv(TABLES / "vs_anom_ar1.csv", index=False)
    by_phase.round(0).to_csv(TABLES / "rmse_by_phase_h2.csv")
    by_size.round(0).to_csv(TABLES / "rmse_by_size_h2.csv")
    by_origin.round(1).to_csv(TABLES / "rmse_by_origin.csv", index=False)

    plot_rmse_by_horizon(rmse, FIGURES / "rmse_by_horizon.png")
    plot_rmse_by_origin(by_origin, FIGURES / "rmse_by_origin.png")
    plot_example_seasons(pred, Y, config, FIGURES / "example_seasons.png")

    print("RMSE (km², cột = tầm h):\n", rmse.round(0).to_string(), "\n")
    print("So với AnomAR1 (cấp mùa):\n", tests.round(4).to_string(index=False), "\n")
    print("Theo giai đoạn lũ, h = 2:\n", by_phase.round(0).to_string(), "\n")
    print("Theo cỡ lũ, h = 2:\n", by_size.round(0).to_string())


if __name__ == "__main__":
    main()
