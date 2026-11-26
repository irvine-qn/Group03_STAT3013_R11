"""Bước 2 – Đánh giá LightGBM so với mốc AnomAR1: bảng chuẩn của nhóm và bảng theo điều kiện.

Chạy (sau lightgbm_forecast.py):  python src/LightGBM/lightgbm_evaluate.py
Đọc:  results/LightGBM/tables/predictions.csv
Ghi:  results/LightGBM/tables/
        rmse.csv               RMSE gộp theo mô hình × tầm (cùng dạng với mọi mô hình của nhóm)
        vs_anom_ar1.csv        kiểm định cấp mùa so với AnomAR1
        rmse_by_phase_h2.csv   RMSE theo giai đoạn lũ (tầm 16 ngày)
        rmse_by_size_h2.csv    RMSE theo cỡ lũ của mùa test (tầm 16 ngày)

Vì sao kiểm định ở CẤP MÙA: sai số các bước liền nhau trong một mùa tương quan rất mạnh,
nên coi 27 bước là 27 quan sát độc lập sẽ thổi phồng ý nghĩa. Đơn vị so sánh là mùa: n = 11.
Holm hiệu chỉnh trên cả 3 biến thể × 3 tầm = 9 phép so sánh.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from lightgbm_data import TABLES
from lightgbm_protocol import PHASES

BASELINE = "anom_ar1"
TOST_MARGIN = 0.05  # "tương đương" nghĩa là RMSE chênh không quá ±5%
MODEL_ORDER = ["anom_ar1", "lgbm_raw", "lgbm_anomaly", "lgbm_noclim"]


def load_predictions(path: Path = TABLES / "predictions.csv") -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"không thấy {path}; hãy chạy lightgbm_forecast.py trước")
    return pd.read_csv(path)


# ---------- Thước đo sai số ----------

def pooled_rmse(pred: pd.DataFrame, by) -> pd.DataFrame:
    """RMSE gộp mọi dòng trong từng nhóm `by`."""
    squared = pred.assign(sq_error=(pred["y_target"] - pred["pred"]) ** 2)
    out = squared.groupby(list(by))["sq_error"].mean() ** 0.5
    return out.rename("rmse").reset_index()


def season_mse(pred: pd.DataFrame) -> pd.DataFrame:
    """MSE của từng (mô hình, tầm, mùa test): mỗi mùa là một quan sát cho kiểm định."""
    squared = pred.assign(mse=(pred["y_target"] - pred["pred"]) ** 2)
    return squared.groupby(["model", "h", "season"], as_index=False)["mse"].mean()


def _ordered(table: pd.DataFrame) -> pd.DataFrame:
    order = [m for m in MODEL_ORDER if m in table.index] + [m for m in table.index if m not in MODEL_ORDER]
    return table.loc[order]


def rmse_table(pred: pd.DataFrame) -> pd.DataFrame:
    """Bảng chuẩn rmse.csv: dòng = mô hình, cột = tầm h (1, 2, 4)."""
    return _ordered(pooled_rmse(pred, ["model", "h"]).pivot(index="model", columns="h", values="rmse"))


# ---------- Kiểm định cấp mùa ----------

def holm(p_values) -> np.ndarray:
    """Hiệu chỉnh Holm–Bonferroni: p nhỏ nhất nhân m, p kế tiếp nhân m − 1, ..., giữ đơn điệu."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(np.argsort(p_values)):
        running_max = max(running_max, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def _relative_mse_diff(m: np.ndarray, b: np.ndarray) -> dict:
    """Chênh MSE tương đối theo mùa, (M − B)/B, trung bình và khoảng tin cậy 95% (phân phối t)."""
    rel = (m - b) / b
    n = len(rel)
    half_width = stats.t.ppf(0.975, n - 1) * rel.std(ddof=1) / np.sqrt(n)
    return {
        "mse_diff_pct": 100 * rel.mean(),
        "mse_diff_pct_ci_low": 100 * (rel.mean() - half_width),
        "mse_diff_pct_ci_high": 100 * (rel.mean() + half_width),
    }


def _tost_p(m: np.ndarray, b: np.ndarray, margin: float) -> float:
    """TOST trên log tỉ số RMSE theo mùa. p nhỏ ⇒ kết luận "tương đương trong ±margin"."""
    d = 0.5 * np.log(m / b)
    n, limit = len(d), np.log1p(margin)
    se = d.std(ddof=1) / np.sqrt(n)
    p_lower = stats.t.sf((d.mean() + limit) / se, n - 1)
    p_upper = stats.t.cdf((d.mean() - limit) / se, n - 1)
    return float(max(p_lower, p_upper))


def _wilcoxon_p(m: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(stats.wilcoxon(m, b).pvalue)
    except ValueError:  # mọi chênh lệch bằng 0
        return float("nan")


def compare_with_anom_ar1(season_errors: pd.DataFrame, margin: float = TOST_MARGIN) -> pd.DataFrame:
    """Bảng chuẩn vs_anom_ar1.csv: so mỗi mô hình với AnomAR1 ở từng tầm, trên MSE của 11 mùa.

    rmse_ratio < 1 nghĩa là tốt hơn AnomAR1; wins = số mùa có MSE thấp hơn.
    """
    wide = season_errors.pivot_table(index=["h", "season"], columns="model", values="mse")
    models = [m for m in MODEL_ORDER if m in wide.columns and m != BASELINE]
    models += [m for m in wide.columns if m not in MODEL_ORDER]
    rows = []
    for model in models:
        for h in wide.index.get_level_values("h").unique():
            m = wide.loc[h, model].to_numpy()
            b = wide.loc[h, BASELINE].to_numpy()
            rows.append({
                "model": model, "h": int(h), "n_seasons": len(m),
                "rmse_ratio": float(np.sqrt(m.mean() / b.mean())),
                "wins": int((m < b).sum()),
                **_relative_mse_diff(m, b),
                "p_t": float(stats.ttest_rel(m, b).pvalue),
                "p_wilcoxon": _wilcoxon_p(m, b),
                "p_tost": _tost_p(m, b, margin),
            })
    result = pd.DataFrame(rows)
    result["p_t_holm"] = holm(result["p_t"].to_numpy())
    return result


# ---------- Theo điều kiện ----------

def add_flood_phase(pred: pd.DataFrame) -> pd.DataFrame:
    """Gắn giai đoạn lũ (nước lên / đỉnh / nước rút) theo ngày của bước đích."""
    edges = [start for start, _, _ in PHASES] + [PHASES[-1][1]]
    phase = pd.cut(pred["doy_target"], edges, right=False, labels=[n for _, _, n in PHASES])
    return pred.assign(phase=phase.astype(str))


def add_flood_size(pred: pd.DataFrame) -> pd.DataFrame:
    """Gắn cỡ lũ của mùa test: chia 11 mùa thành 3 nhóm theo đỉnh diện tích ngập."""
    peak = pred.groupby("season")["y_target"].max()
    size = pd.qcut(peak, 3, labels=["small", "medium", "large"])
    return pred.assign(size=pred["season"].map(size).astype(str))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pred = load_predictions()

    rmse = rmse_table(pred)
    tests = compare_with_anom_ar1(season_mse(pred))
    h2 = pred[pred["h"] == 2]
    by_phase = _ordered(pooled_rmse(add_flood_phase(h2), ["model", "phase"]).pivot(
        index="model", columns="phase", values="rmse")[["rising", "peak", "falling"]])
    by_size = _ordered(pooled_rmse(add_flood_size(h2), ["model", "size"]).pivot(
        index="model", columns="size", values="rmse")[["small", "medium", "large"]])

    rmse.round(0).to_csv(TABLES / "rmse.csv")
    tests.round(4).to_csv(TABLES / "vs_anom_ar1.csv", index=False)
    by_phase.round(0).to_csv(TABLES / "rmse_by_phase_h2.csv")
    by_size.round(0).to_csv(TABLES / "rmse_by_size_h2.csv")

    print("RMSE (km², cột = tầm h):\n", rmse.round(0).to_string(), "\n")
    print("So với AnomAR1 (cấp mùa):\n", tests.round(4).to_string(index=False), "\n")
    print("Theo giai đoạn lũ, h = 2:\n", by_phase.round(0).to_string(), "\n")
    print("Theo cỡ lũ, h = 2:\n", by_size.round(0).to_string())


if __name__ == "__main__":
    main()
