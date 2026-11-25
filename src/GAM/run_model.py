"""BƯỚC 2 – Chạy GAM theo đúng giao thức của nhóm và lưu mọi thứ cần cho phân tích.

Chạy từ gốc repo:  python src/GAM/run_model.py      (khoảng 2–4 phút)

Giao thức (giống `floodbench.benchmark`):
  - Rolling origin theo năm: mỗi mùa test Y = 2010..2020 học trên các mùa 2003..Y-1.
  - Biến thể "_w7": chỉ học 7 mùa gần nhất (để xem mô hình cần bao nhiêu dữ liệu).
  - 3 tầm: 1, 2, 4 bước = 8, 16, 32 ngày. Mỗi (fold, tầm) chọn cấu hình riêng.

Ghi vào results/GAM/tables/:
  gam_predictions.csv     mỗi dòng một dự báo (gam, gam_w7) + AnomAR1 cùng fold để so sánh
  gam_chosen_configs.csv  cấu hình được chọn + MSE validation của cả 6 cấu hình
  gam_coefficients.csv    β0, β1, β2, bậc tự do hiệu dụng của hàm trơn, ở từng fold/tầm
  gam_seasonal_curve.csv  đường β0 + s(ngày) của từng fold/tầm (để vẽ)
  reproduction_check.csv  độ lệch so với dự báo chung của nhóm (phải ~0)
  gam_ablation_predictions.csv  PHẦN PHỤ, xem `ablation_variants`
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from pygam import LinearGAM, l

from common import (BASELINE, DAYS, HORIZONS, TABLES, TEAM_PREDICTIONS, WINDOW, ensure_dirs,
                    load_data, season_index)
from floodbench.cv import rolling_origin           # sinh các fold theo năm
from floodbench.features import build_rows          # dòng dự báo, climatology chỉ từ mùa trước
from floodbench.models.baselines import anom_ar1    # mốc so sánh chính của nhóm
from gam_model import GAM_GRID, anomaly_target, coefficients, design_matrix, gam_forecast, seasonal_curve

DOY_GRID = np.arange(177, 362, 4)  # ngày đích có thể có: bước 4 (ngày 177) .. bước 27 (ngày 361)


def run(Y: np.ndarray, cfg, windows=((None, ""), (WINDOW, "_w7"))) -> dict[str, pd.DataFrame]:
    preds, configs, coefs, curves = [], [], [], []
    for window, suffix in windows:
        for fold in rolling_origin(cfg, window=window):
            t0 = time.time()
            for h in HORIZONS:
                # Dựng dòng BÊN TRONG fold: dòng test chỉ dùng mùa test và các mùa trước nó.
                train = build_rows(Y, cfg, season_index(cfg, fold.train_seasons), h)
                test = build_rows(Y, cfg, season_index(cfg, [fold.test_season]), h)
                fit = gam_forecast(train, test, h)
                key = {"model": "gam" + suffix, "test_season": fold.test_season, "h": h,
                       "h_days": DAYS[h], "n_train_seasons": len(fold.train_seasons),
                       "n_train_rows": len(train)}

                cols = test[["season", "t", "doy_target", "y_target", "c_target"]]
                preds.append(cols.assign(model="gam" + suffix, h=h, pred=fit.pred))
                if window is None:  # AnomAR1 cùng fold, để so sánh ngay trong thư mục này
                    preds.append(cols.assign(model=BASELINE, h=h, pred=anom_ar1(train, test, h)))

                configs.append({**key, "n_splines": fit.config[0], "lam": fit.config[1],
                                **{f"val_mse_{n}_{lam}": m for (n, lam), m in zip(GAM_GRID, fit.val_mse)},
                                "val_spread_pct": 100 * (max(fit.val_mse) / min(fit.val_mse) - 1)})
                coefs.append({**key, **coefficients(fit.model)})
                curves.append(pd.DataFrame({**key, "doy": DOY_GRID,
                                            "beta0_plus_s": seasonal_curve(fit.model, DOY_GRID)}))
            print(f"  fold {fold.test_season}{suffix or ' (đủ mùa)'}: {time.time() - t0:.1f}s", flush=True)
    return {"predictions": pd.concat(preds, ignore_index=True), "configs": pd.DataFrame(configs),
            "coefficients": pd.DataFrame(coefs), "curves": pd.concat(curves, ignore_index=True)}


def reproduction_check(pred: pd.DataFrame) -> pd.DataFrame:
    """So từng dự báo với file chung `results/tables/probe_predictions.csv`. Lệch phải ~0."""
    team = pd.read_csv(TEAM_PREDICTIONS)
    out = []
    for model in ("gam", "gam_w7", BASELINE):
        a = pred[pred["model"] == model].set_index(["season", "t", "h"])["pred"]
        b = team[team["model"] == model].set_index(["season", "t", "h"])["pred"]
        both = a.to_frame("ours").join(b.rename("team"), how="inner")
        out.append({"model": model, "n_ours": len(a), "n_team": len(b), "n_matched": len(both),
                    "max_abs_diff_km2": float((both["ours"] - both["team"]).abs().max())})
    return pd.DataFrame(out)


# ---- PHẦN PHỤ: thí nghiệm bóc tách -------------------------------------------------------------
# Mục đích: hiểu PHẦN NÀO của GAM tạo ra kết quả, không phải để chọn mô hình tốt hơn.
# Các biến thể này được nghĩ ra SAU KHI đã thấy kết quả test, nên KHÔNG được đưa vào bảng
# so sánh chính. Muốn dùng biến thể nào thì phải khai báo trước và chạy lại cho cả nhóm.
def ablation_variants(Y: np.ndarray, cfg) -> pd.DataFrame:
    out = []
    for fold in rolling_origin(cfg):
        for h in HORIZONS:
            train = build_rows(Y, cfg, season_index(cfg, fold.train_seasons), h)
            test = build_rows(Y, cfg, season_index(cfg, [fold.test_season]), h)
            X, Xt, y = design_matrix(train), design_matrix(test), anomaly_target(train)
            c = test["c_target"].to_numpy()
            variants = {
                # thay hàm trơn s(ngày) bằng một đường thẳng theo ngày
                "gam_linear_doy": LinearGAM(l(0) + l(1) + l(2)).fit(X, y).predict(Xt),
                # bỏ hẳn ngày và a_{t-1}: chỉ còn β0 + β1·a_t (AR(1) trực tiếp có hệ số chặn)
                "ar1_direct_intercept": LinearGAM(l(0)).fit(X[:, [1]], y).predict(Xt[:, [1]]),
            }
            for name, anomaly in variants.items():
                out.append(test[["season", "t", "doy_target", "y_target"]].assign(
                    model=name, h=h, pred=c + anomaly))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_dirs()
    cfg, Y = load_data()
    print("Chạy GAM trên 11 fold x 3 tầm, đủ mùa và 7 mùa ...")
    res = run(Y, cfg)
    res["predictions"].to_csv(TABLES / "gam_predictions.csv", index=False)
    res["configs"].to_csv(TABLES / "gam_chosen_configs.csv", index=False)
    res["coefficients"].to_csv(TABLES / "gam_coefficients.csv", index=False)
    res["curves"].to_csv(TABLES / "gam_seasonal_curve.csv", index=False)

    check = reproduction_check(res["predictions"])
    check.to_csv(TABLES / "reproduction_check.csv", index=False)
    print("\nĐối chiếu với dự báo chung của nhóm:\n", check.to_string(index=False))

    print("\nPhần phụ: thí nghiệm bóc tách ...")
    ablation_variants(Y, cfg).to_csv(TABLES / "gam_ablation_predictions.csv", index=False)
    print(f"Xong -> {TABLES}")


if __name__ == "__main__":
    main()
