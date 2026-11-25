"""BƯỚC 3 – Phân tích kết quả GAM (chỉ đọc file của bước 2, không học lại mô hình).

Chạy từ gốc repo:  python src/GAM/analysis.py

Ghi vào results/GAM/tables/:
  gam_rmse.csv              RMSE gộp 8/16/32 ngày: GAM, AnomAR1, bản 7 mùa và vài mô hình tham chiếu
  gam_vs_anom_ar1.csv       so với AnomAR1 ở CẤP MÙA: tỉ lệ RMSE, số mùa thắng, CI, p, Holm, TOST
  gam_season_rmse.csv       RMSE và thiên lệch của từng mùa test
  gam_by_phase.csv          RMSE theo giai đoạn lũ (nước lên / đỉnh / nước rút)
  gam_by_magnitude.csv      RMSE theo cỡ lũ của mùa test, đủ mùa và 7 mùa
  gam_residual_tests.csv    chẩn đoán phần dư từng mùa (thiên lệch, ACF, Ljung–Box)
  gam_residual_summary.csv  tóm tắt phần dư theo tầm, so với ACF lý thuyết
  climatology_bias.csv      anomaly trung bình theo ngày: climatology 5 mùa lệch bao nhiêu
  gam_ablation_rmse.csv     PHẦN PHỤ: bóc tách GAM (xem run_model.ablation_variants)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import (BASELINE, DAYS, HORIZONS, PHASE_EDGES, PHASE_NAMES, SIZE_NAMES, TABLES,
                    TEAM_PREDICTIONS, DATASET, load_data, rmse)
from floodbench.evaluation import season_mse      # MSE của từng (mô hình, tầm, mùa)
from floodbench.inference import compare_to_baseline  # kiểm định cấp mùa dùng chung của nhóm

# Mô hình tham chiếu lấy từ bảng chung, chỉ để đặt GAM vào bối cảnh
REFERENCE = ["arx", "ridge", "lgbm_raw", "clim5", "persistence"]


def rmse_table(pred: pd.DataFrame) -> pd.DataFrame:
    """RMSE gộp mọi dòng của 11 mùa test, theo (mô hình, tầm)."""
    team = pd.read_csv(TEAM_PREDICTIONS)
    extra = team[team["model"].isin(REFERENCE + ["anom_ar1_w7"])]
    both = pd.concat([pred, extra], ignore_index=True)
    out = (both.assign(sq=(both["y_target"] - both["pred"]) ** 2)
               .groupby(["model", "h"])["sq"].mean().pow(0.5).unstack("h"))
    out.columns = [f"rmse_{DAYS[h]}d" for h in out.columns]
    order = ["gam", BASELINE, "gam_w7", "anom_ar1_w7"] + REFERENCE
    return out.loc[order].reset_index()


def versus_baseline(pred: pd.DataFrame) -> pd.DataFrame:
    """Kiểm định cấp mùa, dùng ĐÚNG hàm và ĐÚNG họ so sánh của bảng chung.

    Hiệu chỉnh Holm phụ thuộc số phép so sánh trong họ. Để p Holm khớp `results/tables/
    probe_vs_anom_ar1.csv`, ta đưa vào mọi mô hình chính của nhóm (thay dự báo GAM bằng bản tự
    chạy, vốn trùng khớp) rồi chỉ giữ dòng GAM.
    """
    team = pd.read_csv(TEAM_PREDICTIONS)
    team = team[~team["model"].str.endswith("_w7") & (team["model"] != "gam")]
    ours = pred[pred["model"] == "gam"][team.columns]
    tests = compare_to_baseline(season_mse(pd.concat([team, ours])), BASELINE)
    main = tests[tests["model"] == "gam"].assign(comparison="gam vs anom_ar1 (đủ mùa)")

    # Bản 7 mùa: so GAM_w7 với AnomAR1_w7 (cùng lượng dữ liệu). Họ so sánh chỉ gồm 3 tầm.
    w7 = pd.concat([pred[pred["model"] == "gam_w7"],
                    pd.read_csv(TEAM_PREDICTIONS).query("model == 'anom_ar1_w7'")])
    w7 = w7.assign(model=w7["model"].str.removesuffix("_w7"))
    small = compare_to_baseline(season_mse(w7[team.columns]), BASELINE)
    small = small.assign(comparison="gam_w7 vs anom_ar1_w7 (Holm trong 3 tầm)")
    out = pd.concat([main, small], ignore_index=True)
    out.insert(2, "h_days", out["h"].map(DAYS))
    return out


def season_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, season), g in pred[pred["model"].isin(["gam", BASELINE])].groupby(["h", "season"]):
        gam, base = g[g["model"] == "gam"], g[g["model"] == BASELINE]
        r_g, r_b = rmse(gam["y_target"] - gam["pred"]), rmse(base["y_target"] - base["pred"])
        rows.append({"h": h, "h_days": DAYS[h], "season": season,
                     "peak_km2": g["y_target"].max(),
                     "rmse_gam": r_g, "rmse_anom_ar1": r_b, "ratio": r_g / r_b,
                     # thiên lệch = trung bình (thật − dự báo): > 0 nghĩa là dự báo THẤP hơn thực tế
                     "bias_gam": (gam["y_target"] - gam["pred"]).mean(),
                     "bias_anom_ar1": (base["y_target"] - base["pred"]).mean(),
                     "gam_better": r_g < r_b})
    return pd.DataFrame(rows)


def label_conditions(pred: pd.DataFrame) -> pd.DataFrame:
    """Gắn giai đoạn lũ (theo ngày đích) và cỡ lũ (tam phân vị đỉnh của mùa test) cho mỗi dòng."""
    peak = pred.groupby("season")["y_target"].max()
    size = pd.qcut(peak, 3, labels=SIZE_NAMES)
    return pred.assign(
        phase=pd.cut(pred["doy_target"], PHASE_EDGES, right=False, labels=PHASE_NAMES),
        size=pred["season"].map(size))


def condition_table(pred: pd.DataFrame, by: str, models) -> pd.DataFrame:
    p = label_conditions(pred)
    p = p[p["model"].isin(models)]
    out = (p.assign(sq=(p["y_target"] - p["pred"]) ** 2)
            .groupby(["h", "model", by], observed=True)["sq"].mean().pow(0.5)
            .unstack(by).reset_index())
    out.insert(1, "h_days", out["h"].map(DAYS))
    return out


# ---- Chẩn đoán phần dư ---------------------------------------------------------------------
def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Tự tương quan mẫu r_1..r_nlags (cùng công thức với statsmodels.tsa.stattools.acf)."""
    x = np.asarray(x, float) - np.mean(x)
    denom = np.sum(x * x)
    return np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, nlags + 1)])


def ljung_box_p(x: np.ndarray, lags: int) -> float:
    """Kiểm định Ljung–Box: H0 = không có tự tương quan đến độ trễ `lags`.

    Q = n(n+2) Σ_{k=1..m} r_k² / (n−k)  ~  χ²(m) dưới H0.
    """
    n = len(x)
    r = acf(x, lags)
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1)))
    return float(stats.chi2.sf(q, lags))


def theoretical_acf(phi: float, h: int, k: int) -> float:
    """ACF của sai số dự báo h bước nếu anomaly đúng là AR(1) với hệ số phi.

    Sai số dự báo trực tiếp h bước là tổng h cú sốc gần nhất: e = Σ_{j<h} phi^j ε_{t+h−j}.
    Hai sai số cách nhau k bước (k < h) dùng chung h−k cú sốc, nên tương quan khác 0:
        ρ(k) = Σ_{j=0..h−1−k} phi^j·phi^{j+k} / Σ_{j=0..h−1} phi^{2j}  ,  ρ(k) = 0 nếu k ≥ h.
    Nghĩa là: phần dư tự tương quan ở tầm > 8 ngày là ĐIỀU TẤT YẾU, không phải lỗi mô hình.
    """
    if k >= h:
        return 0.0
    num = sum(phi ** j * phi ** (j + k) for j in range(h - k))
    den = sum(phi ** (2 * j) for j in range(h))
    return num / den


def residual_tables(pred: pd.DataFrame, phi: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = pred[pred["model"] == "gam"].sort_values(["h", "season", "t"])
    per_season = []
    for (h, season), grp in g.groupby(["h", "season"]):
        err = (grp["y_target"] - grp["pred"]).to_numpy()
        r = acf(err, 5)
        per_season.append({"h": h, "h_days": DAYS[h], "season": season, "n": len(err),
                           "bias_km2": err.mean(), **{f"acf_{k}": r[k - 1] for k in range(1, 6)},
                           "ljung_box_p_lag10": ljung_box_p(err, 10)})
    per_season = pd.DataFrame(per_season)

    summary = []
    for h in HORIZONS:
        err = (g[g["h"] == h]["y_target"] - g[g["h"] == h]["pred"]).to_numpy()
        ps = per_season[per_season["h"] == h]
        summary.append({
            "h": h, "h_days": DAYS[h], "n": len(err),
            "bias_km2": err.mean(), "sd_km2": err.std(ddof=1),
            "skew": stats.skew(err), "excess_kurtosis": stats.kurtosis(err),
            "shapiro_p": stats.shapiro(err).pvalue,
            **{f"mean_acf_{k}": ps[f"acf_{k}"].mean() for k in (1, 2, 3)},
            **{f"theory_acf_{k}": theoretical_acf(phi, h, k) for k in (1, 2, 3)},
            "share_seasons_ljung_box_p_lt_0.05": (ps["ljung_box_p_lag10"] < 0.05).mean(),
        })
    return per_season, pd.DataFrame(summary)


def pooled_phi() -> float:
    """φ của AnomAR1 ước lượng trên mọi dòng 8 ngày (mùa 2003–2020), chỉ để vẽ ACF lý thuyết."""
    rows = pd.read_csv(DATASET / "gam_rows.csv").query("h_steps == 1")
    x, y = rows["a_t1"].to_numpy(), rows["a_t"].to_numpy()
    return float((x * y).sum() / (x ** 2).sum())


def climatology_bias(Y: np.ndarray, cfg) -> pd.DataFrame:
    """Anomaly (y − climatology 5 mùa trước) trung bình theo khoảng ngày, mùa 2003–2020.

    Nếu climatology không lệch, cột mean_anomaly phải quanh 0 ở mọi khoảng ngày.
    """
    rows = pd.read_csv(DATASET / "gam_rows.csv").query("h_steps == 1")
    bins = pd.cut(rows["doy_target"], [170, 200, 230, 260, 290, 320, 362])
    out = rows.groupby(bins, observed=True)["anomaly_target"].agg(["mean", "std", "count"])
    out = out.rename(columns={"mean": "mean_anomaly_km2", "std": "sd_anomaly_km2", "count": "n"})
    peaks = Y.max(axis=1)
    slope = np.polyfit(np.arange(cfg.first_year, cfg.last_year + 1), peaks, 1)[0]
    out["peak_trend_km2_per_year_2000_2020"] = slope
    return out.reset_index().rename(columns={"doy_target": "doy_bin"})


def ablation_table(pred: pd.DataFrame) -> pd.DataFrame:
    abl = pd.read_csv(TABLES / "gam_ablation_predictions.csv")
    both = pd.concat([pred[pred["model"].isin(["gam", BASELINE])], abl], ignore_index=True)
    sm = season_mse(both).pivot_table(index=["h", "season"], columns="model", values="mse")
    rows = []
    for (model, h), g in both.groupby(["model", "h"]):
        s = sm.loc[h]
        rows.append({"model": model, "h": h, "h_days": DAYS[h], "rmse": rmse(g["y_target"] - g["pred"]),
                     # số mùa test (trên 11) mà biến thể có MSE thấp hơn GAM gốc
                     "wins_vs_gam": int((s[model] < s["gam"]).sum()) if model != "gam" else np.nan,
                     "n_seasons": len(s)})
    order = {"gam": 0, BASELINE: 1, "gam_linear_doy": 2, "ar1_direct_intercept": 3}
    out = pd.DataFrame(rows)
    return out.sort_values(["h", "model"], key=lambda c: c.map(order) if c.name == "model" else c)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg, Y = load_data()
    pred = pd.read_csv(TABLES / "gam_predictions.csv")
    main_pred = pred[pred["model"].isin(["gam", BASELINE])]

    tables = {
        "gam_rmse": rmse_table(pred),
        "gam_vs_anom_ar1": versus_baseline(pred),
        "gam_season_rmse": season_table(pred),
        "gam_by_phase": condition_table(main_pred, "phase", ["gam", BASELINE]),
        "gam_by_magnitude": condition_table(
            pd.concat([pred, pd.read_csv(TEAM_PREDICTIONS).query("model == 'anom_ar1_w7'")]),
            "size", ["gam", BASELINE, "gam_w7", "anom_ar1_w7"]),
        "climatology_bias": climatology_bias(Y, cfg),
        "gam_ablation_rmse": ablation_table(pred),
    }
    phi = pooled_phi()
    tables["gam_residual_tests"], tables["gam_residual_summary"] = residual_tables(pred, phi)
    for name, t in tables.items():
        t.to_csv(TABLES / f"{name}.csv", index=False)
        print(f"\n== {name}\n{t.round(3).to_string(index=False)}")
    print(f"\nφ gộp (dùng cho ACF lý thuyết) = {phi:.3f}")


if __name__ == "__main__":
    main()
