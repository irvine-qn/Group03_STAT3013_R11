"""Bước 1 – Bảng so sánh chung của 4 mô hình (chỉ đọc kết quả đã có, không học lại mô hình).

Chạy (sau khi cả 4 mô hình đã ghi predictions.csv):  python src/Comparison/comparison_tables.py
Đọc:  results/<Mô hình>/tables/predictions.csv  với <Mô hình> = AnomAR1, GAM, LightGBM, Chronos-2
Ghi:  results/Comparison/tables/
        baseline_check.csv     cột anom_ar1 của mỗi thư mục lệch bao nhiêu so với thư mục AnomAR1
        rmse_all.csv           RMSE gộp của mọi mô hình × tầm, một bảng
        r2_all.csv             R² gộp (1 − SSE/SST trên cùng các dòng) của mọi mô hình × tầm
        vs_anom_ar1_all.csv    kiểm định cấp mùa so với AnomAR1, Holm trên TOÀN BỘ 9 phép so chính
        vs_anom_ar1_main.csv   chỉ họ chính (9 dòng), không làm tròn, có cả p Holm theo từng mô hình và theo họ chính
        rmse_by_phase.csv      RMSE theo giai đoạn lũ, 4 bản chính × 3 tầm
        rmse_by_size.csv       RMSE theo cỡ lũ của mùa test, 4 bản chính × 3 tầm
        season_rmse.csv        RMSE từng mùa test của 4 bản chính và tỉ lệ so với AnomAR1

Mọi số đều tính lại từ predictions.csv, cùng công thức với bảng của từng mô hình, nên khớp với số
trong results/<Mô hình>/tables/ (test kiểm tra điều này).

Vì sao cần Holm toàn cục: mỗi mô hình chỉ hiệu chỉnh trong các phép so của chính nó. Khi đặt 3 bản chính
(GAM, LightGBM thô, Chronos-2) cạnh nhau ở 3 tầm, nhóm thực hiện 9 phép so với AnomAR1, nên p phải hiệu
chỉnh trên cả 9. Các biến thể và mốc đơn giản vẫn có p thô, nhưng không vào họ hiệu chỉnh này.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
TABLES = RESULTS / "Comparison" / "tables"
FIGURES = RESULTS / "Comparison" / "figures"

BASELINE = "anom_ar1"
TOST_MARGIN = 0.05  # "tương đương" nghĩa là RMSE chênh không quá ±5%
HORIZONS = [1, 2, 4]
HORIZON_DAYS = {1: 8, 2: 16, 4: 32}
COLUMNS = ["season", "t", "doy_target", "y_target", "model", "h", "pred"]

# Mỗi thư mục kết quả "sở hữu" các mô hình nào. AnomAR1 và các mốc đơn giản lấy từ thư mục AnomAR1;
# các thư mục khác cũng có cột anom_ar1 (để tự so sánh), dùng để kiểm tra mọi người cùng một mốc.
SOURCES = {
    "AnomAR1": ["anom_ar1", "anom_ar1_w7", "clim5", "persistence"],
    "GAM": ["gam", "gam_w7"],
    "LightGBM": ["lgbm_raw", "lgbm_anomaly", "lgbm_noclim"],
    "Chronos-2": ["chronos2"],
}
MAIN = [
    BASELINE,
    "gam",
    "lgbm_raw",
    "chronos2",
]  # bản chính của mỗi thành viên, khai báo trước
CONTENDERS = MAIN[1:]
ROLE = {
    "anom_ar1": "baseline",
    "gam": "main",
    "lgbm_raw": "main",
    "chronos2": "main",
    "anom_ar1_w7": "variant",
    "gam_w7": "variant",
    "lgbm_anomaly": "variant",
    "lgbm_noclim": "variant",
    "clim5": "simple",
    "persistence": "simple",
}
ORDER = list(ROLE)  # thứ tự dòng: mốc, 3 bản chính, biến thể, mốc đơn giản
PHASE_EDGES, PHASES = [0, 257, 298, 999], ["rising", "peak", "falling"]
SIZES = ["small", "medium", "large"]
BASELINE_TOL = 1e-6  # km²; lệch lớn hơn nghĩa là các thư mục không cùng một mốc


# ---------- Đọc dữ liệu ----------


def read_predictions(source: str, results: Path = RESULTS) -> pd.DataFrame:
    path = results / source / "tables" / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"không thấy {path}; hãy chạy bước dự báo của {source} trước"
        )
    return pd.read_csv(path)[COLUMNS]


def baseline_check(results: Path = RESULTS) -> pd.DataFrame:
    """So cột anom_ar1 của từng thư mục với thư mục AnomAR1: cùng số dòng và cùng giá trị."""
    key = ["season", "t", "h"]
    ref = (
        read_predictions("AnomAR1", results)
        .query("model == @BASELINE")
        .set_index(key)
        .sort_index()
    )
    rows = []
    for source in SOURCES:
        own = (
            read_predictions(source, results)
            .query("model == @BASELINE")
            .set_index(key)
            .sort_index()
        )
        same_rows = own.index.equals(ref.index)
        diff = (own["pred"] - ref["pred"]).abs().max() if same_rows else np.inf
        rows.append(
            {
                "source": source,
                "n_rows": len(own),
                "same_rows": same_rows,
                "max_abs_diff": float(diff),
            }
        )
    return pd.DataFrame(rows)


def load_all(results: Path = RESULTS) -> pd.DataFrame:
    """Dự báo của mọi mô hình, mỗi mô hình lấy đúng một lần từ thư mục sở hữu nó."""
    check = baseline_check(results)
    bad = check[~check["same_rows"] | (check["max_abs_diff"] > BASELINE_TOL)]
    if len(bad):
        raise ValueError(
            f"cột anom_ar1 không khớp giữa các thư mục:\n{bad.to_string(index=False)}"
        )
    parts = []
    for source, models in SOURCES.items():
        pred = read_predictions(source, results)
        parts.append(pred[pred["model"].isin(models)].assign(source=source))
    return pd.concat(parts, ignore_index=True)


# ---------- Thước đo sai số ----------


def pooled_rmse(pred: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """RMSE gộp mọi dòng trong từng nhóm `by`."""
    squared = pred.assign(sq_error=(pred["y_target"] - pred["pred"]) ** 2)
    out = squared.groupby(by, observed=True)["sq_error"].mean() ** 0.5
    return out.rename("rmse").reset_index()


def season_mse(pred: pd.DataFrame) -> pd.DataFrame:
    """MSE của từng (mô hình, tầm, mùa test): mỗi mùa là một quan sát cho kiểm định."""
    squared = pred.assign(mse=(pred["y_target"] - pred["pred"]) ** 2)
    return squared.groupby(["model", "h", "season"], as_index=False)["mse"].mean()


def _sort_models(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rank = frame["model"].map({m: k for k, m in enumerate(ORDER)}).fillna(len(ORDER))
    return (
        frame.assign(_rank=rank)
        .sort_values(by + ["_rank"])
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def rmse_all(pred: pd.DataFrame) -> pd.DataFrame:
    """Một bảng RMSE: dòng = mô hình (kèm thư mục nguồn và vai trò), cột = tầm h."""
    wide = pooled_rmse(pred, ["model", "h"]).pivot(
        index="model", columns="h", values="rmse"
    )
    wide.columns = [str(h) for h in wide.columns]
    source = pred.drop_duplicates("model").set_index("model")["source"]
    out = wide.assign(source=source, role=wide.index.map(ROLE)).reset_index()
    out = _sort_models(out, [])
    return out[["model", "source", "role"] + [str(h) for h in HORIZONS]]


def r2_all(pred: pd.DataFrame) -> pd.DataFrame:
    """R² gộp = 1 − SSE/SST, cùng dạng rmse_all. SST tính quanh trung bình y_target của mỗi tầm,
    nên mọi mô hình ở cùng tầm có chung mẫu số (cùng 748/3 dòng) và so được với nhau."""
    sst = (
        pred.drop_duplicates(["season", "t", "h"])
        .groupby("h")["y_target"]
        .agg(lambda y: ((y - y.mean()) ** 2).sum())
    )
    sse = (
        pred.assign(sq=(pred["y_target"] - pred["pred"]) ** 2)
        .groupby(["model", "h"])["sq"]
        .sum()
    )
    r2 = (
        (1 - sse / sse.index.get_level_values("h").map(sst).to_numpy())
        .rename("r2")
        .reset_index()
    )
    wide = r2.pivot(index="model", columns="h", values="r2")
    wide.columns = [str(h) for h in wide.columns]
    source = pred.drop_duplicates("model").set_index("model")["source"]
    out = _sort_models(
        wide.assign(source=source, role=wide.index.map(ROLE)).reset_index(), []
    )
    return out[["model", "source", "role"] + [str(h) for h in HORIZONS]]


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


def compare_with_anom_ar1(
    pred: pd.DataFrame, margin: float = TOST_MARGIN
) -> pd.DataFrame:
    """So mọi mô hình với AnomAR1 ở từng tầm, trên MSE của 11 mùa.

    Cột giống vs_anom_ar1.csv của từng mô hình, trừ phần Holm:
      p_t_holm_main  Holm trên 9 phép so của 3 bản chính × 3 tầm (để trống cho biến thể và mốc đơn giản).
    """
    wide = season_mse(pred).pivot_table(
        index=["h", "season"], columns="model", values="mse"
    )
    source = pred.drop_duplicates("model").set_index("model")["source"]
    rows = []
    for model in [m for m in ORDER if m in wide.columns and m != BASELINE]:
        for h in HORIZONS:
            m, b = wide.loc[h, model].to_numpy(), wide.loc[h, BASELINE].to_numpy()
            rows.append(
                {
                    "model": model,
                    "source": source[model],
                    "role": ROLE[model],
                    "h": h,
                    "n_seasons": len(m),
                    "rmse_ratio": float(np.sqrt(m.mean() / b.mean())),
                    "wins": int((m < b).sum()),
                    **_relative_mse_diff(m, b),
                    "p_t": float(stats.ttest_rel(m, b).pvalue),
                    "p_wilcoxon": _wilcoxon_p(m, b),
                    "p_tost": _tost_p(m, b, margin),
                }
            )
    out = pd.DataFrame(rows)
    main = out["role"] == "main"
    out["p_t_holm_main"] = np.nan
    out.loc[main, "p_t_holm_main"] = holm(out.loc[main, "p_t"].to_numpy())
    return out


def model_families(results: Path = RESULTS) -> dict[str, list[str]]:
    """Họ Holm của từng thư mục = các mô hình có trong vs_anom_ar1.csv của thư mục đó."""
    return {
        s: list(
            pd.read_csv(results / s / "tables" / "vs_anom_ar1.csv")["model"].unique()
        )
        for s in SOURCES
    }


def main_family(pred: pd.DataFrame, results: Path = RESULTS) -> pd.DataFrame:
    """Bảng vs_anom_ar1_main.csv: họ chính 3 bản chính × 3 tầm = 9 phép so, KHÔNG làm tròn.

    p_t tính lại từ predictions.csv (MSE từng mùa) nên Holm nhân trên p chưa làm tròn.
      p_t_holm_model  Holm trong họ của thư mục mô hình đó (tái tạo đúng vs_anom_ar1.csv của thư mục)
      p_t_holm_main   Holm trên 9 phép so của họ chính
    """
    full = compare_with_anom_ar1(pred)
    full["p_t_holm_model"] = np.nan
    for source, family in model_families(results).items():
        rows = full["model"].isin(family)
        full.loc[rows, "p_t_holm_model"] = holm(full.loc[rows, "p_t"].to_numpy())
    main = full[full["role"] == "main"].copy()
    main["model"] = pd.Categorical(main["model"], CONTENDERS, ordered=True)
    main = (
        main.sort_values(["model", "h"]).astype({"model": str}).reset_index(drop=True)
    )
    return main[
        [
            "model",
            "h",
            "n_seasons",
            "rmse_ratio",
            "wins",
            "mse_diff_pct",
            "mse_diff_pct_ci_low",
            "mse_diff_pct_ci_high",
            "p_t",
            "p_wilcoxon",
            "p_tost",
            "p_t_holm_model",
            "p_t_holm_main",
        ]
    ]


# ---------- Theo điều kiện và theo mùa ----------


def label_conditions(pred: pd.DataFrame) -> pd.DataFrame:
    """Gắn giai đoạn lũ (theo ngày đích) và cỡ lũ (tam phân vị đỉnh của mùa test) cho mỗi dòng."""
    size = pd.qcut(pred.groupby("season")["y_target"].max(), 3, labels=SIZES)
    return pred.assign(
        phase=pd.cut(
            pred["doy_target"], PHASE_EDGES, right=False, labels=PHASES
        ).astype(str),
        size=pred["season"].map(size).astype(str),
    )


def condition_table(pred: pd.DataFrame, by: str) -> pd.DataFrame:
    """Cùng dạng rmse_by_phase.csv / rmse_by_size.csv của từng mô hình: cột h, model rồi các nhóm."""
    groups = PHASES if by == "phase" else SIZES
    table = (
        pooled_rmse(label_conditions(pred), ["h", "model", by])
        .pivot(index=["h", "model"], columns=by, values="rmse")
        .reindex(columns=groups)
        .reset_index()
    )
    table.columns.name = None
    return _sort_models(table, ["h"])


def season_table(pred: pd.DataFrame) -> pd.DataFrame:
    """RMSE từng (tầm, mùa, mô hình), đỉnh và cỡ lũ của mùa, tỉ lệ so với AnomAR1 cùng mùa."""
    per = pooled_rmse(label_conditions(pred), ["h", "season", "size", "model"])
    base = per[per["model"] == BASELINE].set_index(["h", "season"])["rmse"]
    per["ratio_vs_anom_ar1"] = per["rmse"] / per.set_index(["h", "season"]).index.map(
        base
    )
    peak = pred.groupby("season")["y_target"].max().rename("peak_km2")
    per = per.join(peak, on="season")
    per = _sort_models(per, ["h", "season"])
    return per[
        ["h", "season", "peak_km2", "size", "model", "rmse", "ratio_vs_anom_ar1"]
    ]


def build_tables(pred: pd.DataFrame, check: pd.DataFrame) -> dict[str, pd.DataFrame]:
    main = pred[pred["model"].isin(MAIN)]
    return {
        "baseline_check": check,
        "rmse_all": rmse_all(pred).round(0),
        "r2_all": r2_all(pred).round(4),
        "vs_anom_ar1_all": compare_with_anom_ar1(pred).round(4),
        "rmse_by_phase": condition_table(main, "phase").round(0),
        "rmse_by_size": condition_table(main, "size").round(0),
        "season_rmse": season_table(main).round(
            {"peak_km2": 0, "rmse": 0, "ratio_vs_anom_ar1": 4}
        ),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pred = load_all()
    tables = build_tables(pred, baseline_check())
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(TABLES / f"{name}.csv", index=False)
    family = main_family(pred)
    family.to_csv(
        TABLES / "vs_anom_ar1_main.csv", index=False
    )  # không làm tròn: video và báo cáo trích từ đây
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for name in [
            "baseline_check",
            "rmse_all",
            "r2_all",
            "vs_anom_ar1_all",
            "rmse_by_phase",
            "rmse_by_size",
        ]:
            print(f"\n== {name}\n{tables[name].to_string(index=False)}")


if __name__ == "__main__":
    main()
