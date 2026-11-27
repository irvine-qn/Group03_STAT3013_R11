"""Bước 2 – Bảng kết quả chuẩn của nhóm và bảng theo điều kiện.

Chạy (sau anomar1_forecast.py):  python src/AnomAR1/anomar1_evaluate.py
Đọc:  results/AnomAR1/tables/predictions.csv
Ghi:  results/AnomAR1/tables/
        rmse.csv               RMSE gộp theo mô hình × tầm (cùng dạng với mọi mô hình của nhóm)
        vs_anom_ar1.csv        kiểm định cấp mùa của các mốc khác so với AnomAR1
        rmse_by_phase.csv      RMSE theo giai đoạn lũ, mỗi tầm × mô hình (cùng dạng với mọi mô hình)
        rmse_by_size.csv       RMSE theo cỡ lũ của mùa test, mỗi tầm × mô hình (cùng dạng với mọi mô hình)

Vì sao kiểm định ở CẤP MÙA: sai số các bước liền nhau trong một mùa tương quan rất mạnh,
nên coi 27 bước là 27 quan sát độc lập sẽ thổi phồng ý nghĩa. Đơn vị so sánh là mùa: n = 11.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from anomar1_data import TABLES

BASELINE = "anom_ar1"
TOST_MARGIN = 0.05  # "tương đương" nghĩa là RMSE chênh không quá ±5%
MODEL_ORDER = ["anom_ar1", "anom_ar1_w7", "clim5", "persistence"]
# Giai đoạn lũ theo ngày đích: [0, 257) nước lên, [257, 298) đỉnh, [298, ...) nước rút
PHASE_EDGES = [0, 257, 298, 999]
PHASES = ["rising", "peak", "falling"]
SIZES = ["small", "medium", "large"]


def load_predictions(path: Path = TABLES / "predictions.csv") -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"không thấy {path}; hãy chạy anomar1_forecast.py trước")
    return pd.read_csv(path)


# ---------- Thước đo sai số ----------

def rmse(frame: pd.DataFrame) -> float:
    """RMSE gộp mọi dòng của bảng."""
    return float(np.sqrt(((frame["y_target"] - frame["pred"]) ** 2).mean()))


def pooled_rmse(pred: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """RMSE gộp mọi dòng trong từng nhóm `by`."""
    squared = pred.assign(sq_error=(pred["y_target"] - pred["pred"]) ** 2)
    out = squared.groupby(by)["sq_error"].mean() ** 0.5
    return out.rename("rmse").reset_index()


def season_mse(pred: pd.DataFrame) -> pd.DataFrame:
    """MSE của từng (mô hình, tầm, mùa test): mỗi mùa là một quan sát cho kiểm định."""
    squared = pred.assign(mse=(pred["y_target"] - pred["pred"]) ** 2)
    return squared.groupby(["model", "h", "season"], as_index=False)["mse"].mean()


def rmse_table(pred: pd.DataFrame) -> pd.DataFrame:
    """Bảng chuẩn rmse.csv: dòng = mô hình, cột = tầm h (1, 2, 4)."""
    table = pooled_rmse(pred, ["model", "h"]).pivot(index="model", columns="h", values="rmse")
    order = [m for m in MODEL_ORDER if m in table.index] + [m for m in table.index if m not in MODEL_ORDER]
    return table.loc[order]


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
    Holm hiệu chỉnh trong họ mọi phép so sánh của bảng.
    """
    wide = season_errors.pivot_table(index=["h", "season"], columns="model", values="mse")
    rows = []
    for model in [m for m in wide.columns if m != BASELINE]:
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

def label_conditions(pred: pd.DataFrame) -> pd.DataFrame:
    """Gắn giai đoạn lũ (theo ngày đích) và cỡ lũ (tam phân vị đỉnh của mùa test) cho mỗi dòng."""
    size = pd.qcut(pred.groupby("season")["y_target"].max(), 3, labels=SIZES)
    return pred.assign(
        phase=pd.cut(pred["doy_target"], PHASE_EDGES, right=False, labels=PHASES).astype(str),
        size=pred["season"].map(size).astype(str),
    )


def condition_table(pred: pd.DataFrame, by: str) -> pd.DataFrame:
    """Bảng chuẩn rmse_by_phase.csv / rmse_by_size.csv: dòng = (tầm, mô hình), cột = nhóm.

    Cùng dạng ở mọi mô hình của nhóm: cột h, model rồi rising/peak/falling hoặc small/medium/large.
    """
    groups = PHASES if by == "phase" else SIZES
    table = (pooled_rmse(label_conditions(pred), ["h", "model", by])
             .pivot(index=["h", "model"], columns=by, values="rmse")
             .reindex(columns=groups))
    order = {m: k for k, m in enumerate(MODEL_ORDER)}
    rank = [order.get(m, len(order)) for m in table.index.get_level_values("model")]
    table = table.assign(_rank=rank).reset_index()
    return table.sort_values(["h", "_rank", "model"]).drop(columns="_rank").reset_index(drop=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pred = load_predictions()

    table = rmse_table(pred)
    tests = compare_with_anom_ar1(season_mse(pred))
    phase = condition_table(pred, "phase").round(0)
    size = condition_table(pred, "size").round(0)

    table.round(0).to_csv(TABLES / "rmse.csv")
    tests.round(4).to_csv(TABLES / "vs_anom_ar1.csv", index=False)
    phase.to_csv(TABLES / "rmse_by_phase.csv", index=False)
    size.to_csv(TABLES / "rmse_by_size.csv", index=False)

    print("RMSE (km², cột = tầm h):\n", table.round(0).to_string(), "\n")
    print("So với AnomAR1 (cấp mùa):\n", tests.round(4).to_string(index=False), "\n")
    print("Theo giai đoạn lũ:\n", phase.to_string(index=False), "\n")
    print("Theo cỡ lũ:\n", size.to_string(index=False))


if __name__ == "__main__":
    main()
