"""Bước 4 – Độ vững của TOST và t ghép cặp ở 9 phép so của họ chính (phân tích phụ, không thay kết quả chính).

Chạy (sau comparison_tables.py):  python src/Comparison/comparison_robustness.py
Đọc:  results/<Mô hình>/tables/predictions.csv  (như comparison_residuals.py)
Ghi:  results/Comparison/tables/
        robustness_main.csv           9 dòng: gam, lgbm_raw, chronos2 × h = 1, 2, 4
        robustness_season_values.csv  99 dòng: giá trị theo mùa, cờ ngoại lai, p khi bỏ mùa đó

Lý do: Shapiro–Wilk bác bỏ giả định chuẩn của các đại lượng theo mùa ở một số phép so
(paired_diff_normality.csv). Ba cách kiểm tra không cần (hoặc ít cần) giả định chuẩn:
  1. ngoại lai theo mùa: |z| > 2,5 hoặc ngoài [Q1 − 1,5·IQR; Q3 + 1,5·IQR];
  2. bootstrap theo mùa (lấy lại nguyên mùa, có hoàn lại, B = 10000, seed 0):
     - CI 90% percentile của trung bình d = ½·ln(MSE_m/MSE_b) → "tương đương ±5%" nếu nằm trọn trong
       (−ln 1,05; +ln 1,05) (dạng khoảng tin cậy của TOST mức 5%);
     - CI 95% percentile của trung bình MSE_m − MSE_b → "khác có ý nghĩa" nếu không chứa 0;
     - p bootstrap hai phía của chênh MSE: tỉ lệ trung bình bootstrap (sau khi dịch tâm về 0) có
       |trung bình| ≥ |trung bình quan sát|;
  3. bỏ lần lượt từng mùa (leave-one-season-out): tính lại p_tost, p_t và p Holm của họ chính.
Kiểm chứng: CI t-thông thường 90% của trung bình d cho đúng kết luận của p_tost (p_tost < 0,05 ⇔ CI trong biên).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from comparison_residuals import load_main, season_mse
from comparison_tables import BASELINE, HORIZONS, TABLES, TOST_MARGIN, _tost_p, holm

CONTENDERS = ["gam", "lgbm_raw", "chronos2"]  # họ chính, khai báo trước
N_BOOT = 10_000
SEED = 0
ALPHA = 0.05
Z_LIMIT = 2.5
IQR_K = 1.5


# ---------- Công thức (mỗi hàm có test đáp án biết trước) ----------


def season_quantities(m, b) -> dict:
    """d = ½·ln(MSE_m/MSE_b) (log tỉ số RMSE) và MSE_m − MSE_b, theo mùa."""
    m, b = np.asarray(m, float), np.asarray(b, float)
    return {"d": 0.5 * np.log(m / b), "mse_diff": m - b}


def outlier_flags(x) -> dict:
    """z = (x − trung bình)/SD (ddof = 1) và cờ ngoài 1,5·IQR (phân vị nội suy tuyến tính của numpy)."""
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    z = (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    out_iqr = (x < q1 - IQR_K * iqr) | (x > q3 + IQR_K * iqr)
    return {"z": z, "z_out": np.abs(z) > Z_LIMIT, "iqr_out": out_iqr}


def t_ci(x, level: float) -> tuple[float, float]:
    """Khoảng tin cậy t hai phía mức `level` cho trung bình."""
    x = np.asarray(x, float)
    n = len(x)
    half = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean() - half), float(x.mean() + half)


def bootstrap_indices(n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """Chỉ số mùa lấy lại có hoàn lại, kích thước (n_boot, n). Dùng chung cho d và chênh MSE (cặp mùa)."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_means(x, idx: np.ndarray) -> np.ndarray:
    return np.asarray(x, float)[idx].mean(axis=1)


def percentile_ci(boot_means, level: float) -> tuple[float, float]:
    lo, hi = np.percentile(boot_means, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def bootstrap_p_two_sided(x, idx: np.ndarray) -> float:
    """p hai phía: dịch x về trung bình 0, đếm tỉ lệ |trung bình bootstrap| ≥ |trung bình quan sát|."""
    x = np.asarray(x, float)
    centred = bootstrap_means(x - x.mean(), idx)
    return float(np.mean(np.abs(centred) >= abs(x.mean())))


def within_margin(ci: tuple[float, float], margin: float = TOST_MARGIN) -> bool:
    lim = np.log1p(margin)
    return bool(-lim < ci[0] and ci[1] < lim)


def excludes_zero(ci: tuple[float, float]) -> bool:
    return bool(ci[0] > 0 or ci[1] < 0)


def p_t(m, b) -> float:
    return float(stats.ttest_rel(m, b).pvalue)


# ---------- Bảng ----------


def season_mse_wide(pred: pd.DataFrame) -> pd.DataFrame:
    return season_mse(pred).pivot_table(index=["h", "season"], columns="model", values="mse")


def holm_main(wide: pd.DataFrame, drop_season: int | None = None) -> dict:
    """p_t và p Holm của 9 phép so họ chính; tùy chọn bỏ một mùa ở mọi phép so."""
    keys, ps = [], []
    for model in CONTENDERS:
        for h in HORIZONS:
            w = wide.loc[h]
            if drop_season is not None:
                w = w.drop(index=drop_season)
            keys.append((model, h))
            ps.append(p_t(w[model].to_numpy(), w[BASELINE].to_numpy()))
    adj = holm(np.array(ps))
    return {k: (p, a) for k, p, a in zip(keys, ps, adj)}


def robustness_tables(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = season_mse_wide(pred)
    seasons = sorted(wide.index.get_level_values("season").unique())
    full = holm_main(wide)
    loso_holm = {s: holm_main(wide, s) for s in seasons}
    main_rows, season_rows = [], []
    for model in CONTENDERS:
        for h in HORIZONS:
            w = wide.loc[h].loc[seasons]
            m, b = w[model].to_numpy(), w[BASELINE].to_numpy()
            q = season_quantities(m, b)
            flags = {name: outlier_flags(v) for name, v in q.items()}
            idx = bootstrap_indices(len(m))
            ci_d_t = t_ci(q["d"], 0.90)
            ci_d_boot = percentile_ci(bootstrap_means(q["d"], idx), 0.90)
            ci_diff_boot = percentile_ci(bootstrap_means(q["mse_diff"], idx), 0.95)
            p_tost_full = _tost_p(m, b, TOST_MARGIN)
            p_t_full, p_holm_full = full[(model, h)]
            equiv_full, sig_full = p_tost_full < ALPHA, p_holm_full < ALPHA

            loso = []
            for k, s in enumerate(seasons):
                keep = np.arange(len(m)) != k
                loso.append({"season": s, "p_tost": _tost_p(m[keep], b[keep], TOST_MARGIN),
                             "p_t": p_t(m[keep], b[keep]), "p_t_holm_main": loso_holm[s][(model, h)][1]})
            loso = pd.DataFrame(loso)
            flip_tost = loso.loc[(loso["p_tost"] < ALPHA) != equiv_full, "season"].tolist()
            flip_holm = loso.loc[(loso["p_t_holm_main"] < ALPHA) != sig_full, "season"].tolist()

            outliers = []
            for name, f in flags.items():
                hit = [str(s) for s, zo, io in zip(seasons, f["z_out"], f["iqr_out"]) if zo or io]
                if hit:
                    outliers.append(f"{name}: " + ", ".join(hit))
            main_rows.append({
                "model": model, "h": h, "n_seasons": len(m),
                "p_t": p_t_full, "p_tost": p_tost_full,
                "mean_d": float(q["d"].mean()),
                "ci90_d_t_low": ci_d_t[0], "ci90_d_t_high": ci_d_t[1],
                "ci90_d_boot_low": ci_d_boot[0], "ci90_d_boot_high": ci_d_boot[1],
                "equiv_tost": bool(equiv_full), "equiv_t": within_margin(ci_d_t),
                "equiv_boot": within_margin(ci_d_boot),
                "mean_msediff": float(q["mse_diff"].mean()),
                "ci95_msediff_boot_low": ci_diff_boot[0], "ci95_msediff_boot_high": ci_diff_boot[1],
                "p_boot_msediff": bootstrap_p_two_sided(q["mse_diff"], idx),
                "p_t_holm_main": p_holm_full,
                "sig_t_holm_main": bool(sig_full), "sig_boot": excludes_zero(ci_diff_boot),
                "loso_p_tost_min": loso["p_tost"].min(), "loso_p_tost_max": loso["p_tost"].max(),
                "loso_p_t_min": loso["p_t"].min(), "loso_p_t_max": loso["p_t"].max(),
                "loso_p_t_holm_main_min": loso["p_t_holm_main"].min(),
                "loso_p_t_holm_main_max": loso["p_t_holm_main"].max(),
                "loso_equiv_flip_seasons": ", ".join(map(str, flip_tost)),
                "loso_sig_holm_flip_seasons": ", ".join(map(str, flip_holm)),
                "outlier_seasons": "; ".join(outliers),
            })
            for k, s in enumerate(seasons):
                season_rows.append({
                    "model": model, "h": h, "season": s, "mse_model": m[k], "mse_anom_ar1": b[k],
                    "d": q["d"][k], "mse_diff": q["mse_diff"][k],
                    "z_d": flags["d"]["z"][k], "iqr_out_d": bool(flags["d"]["iqr_out"][k]),
                    "z_mse_diff": flags["mse_diff"]["z"][k],
                    "iqr_out_mse_diff": bool(flags["mse_diff"]["iqr_out"][k]),
                    "outlier_d": bool(flags["d"]["z_out"][k] or flags["d"]["iqr_out"][k]),
                    "outlier_mse_diff": bool(flags["mse_diff"]["z_out"][k] or flags["mse_diff"]["iqr_out"][k]),
                    "loso_p_tost": loso.loc[k, "p_tost"], "loso_p_t": loso.loc[k, "p_t"],
                    "loso_p_t_holm_main": loso.loc[k, "p_t_holm_main"],
                })
    return pd.DataFrame(main_rows), pd.DataFrame(season_rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    main_table, season_table = robustness_tables(load_main())
    TABLES.mkdir(parents=True, exist_ok=True)
    main_table.to_csv(TABLES / "robustness_main.csv", index=False)
    season_table.to_csv(TABLES / "robustness_season_values.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(main_table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
