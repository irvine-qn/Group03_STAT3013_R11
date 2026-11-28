"""BƯỚC 2 – Chạy GAM theo đúng giao thức của nhóm và lưu mọi thứ cần cho phân tích.

Chạy từ gốc repo:  python src/GAM/gam_run_model.py      (khoảng 2–4 phút)

Giao thức (gam_protocol.py, giống benchmark thăm dò của nhóm):
  - Rolling origin theo năm: mỗi mùa test Y = 2010..2020 học trên các mùa 2003..Y-1.
  - Biến thể "_w7": chỉ học 7 mùa gần nhất (để xem mô hình cần bao nhiêu dữ liệu).
  - 3 tầm: 1, 2, 4 bước = 8, 16, 32 ngày. Mỗi (fold, tầm) chọn cấu hình riêng.

Ghi vào results/GAM/tables/:
  predictions.csv         mỗi dòng một dự báo: gam, gam_w7 và các mốc so sánh trên CÙNG fold
                          (anom_ar1, anom_ar1_w7, clim5, persistence); cùng 7 cột với mọi mô hình
                          của nhóm. Dự báo clim5 chính là mức bình thường c_target.
  gam_chosen_configs.csv  cấu hình được chọn + MSE validation của cả 6 cấu hình
  gam_coefficients.csv    β0, β1, β2, bậc tự do hiệu dụng của hàm trơn, ở từng fold/tầm
  gam_seasonal_curve.csv  đường β0 + s(ngày) của từng fold/tầm (để vẽ)
  gam_ablation_predictions.csv  PHẦN PHỤ, xem `ablation_variants`
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from pygam import LinearGAM, l

from gam_common import BASELINE, DAYS, HORIZONS, TABLES, WINDOW, ensure_dirs, load_data, season_index
from gam_protocol import anom_ar1, build_rows, clim5, persistence, rolling_origin
from gam_model import GAM_GRID, anomaly_target, coefficients, design_matrix, gam_forecast, seasonal_curve

DOY_GRID = np.arange(177, 362, 4)  # ngày đích có thể có: bước 4 (ngày 177) .. bước 27 (ngày 361)
COLUMNS = ["season", "t", "doy_target", "y_target", "model", "h", "pred"]  # bảng dự báo chung của nhóm


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
                # Mốc so sánh trên CÙNG fold, cùng dòng test: AnomAR1 (cả bản 7 mùa) và 2 baseline
                preds.append(cols.assign(model=BASELINE + suffix, h=h, pred=anom_ar1(train, test, h)))
                if window is None:
                    preds.append(cols.assign(model="clim5", h=h, pred=clim5(test)))
                    preds.append(cols.assign(model="persistence", h=h, pred=persistence(test)))

                configs.append({**key, "n_splines": fit.config[0], "lam": fit.config[1],
                                **{f"val_mse_{n}_{lam}": m for (n, lam), m in zip(GAM_GRID, fit.val_mse)},
                                "val_spread_pct": 100 * (max(fit.val_mse) / min(fit.val_mse) - 1)})
                coefs.append({**key, **coefficients(fit.model)})
                curves.append(pd.DataFrame({**key, "doy": DOY_GRID,
                                            "beta0_plus_s": seasonal_curve(fit.model, DOY_GRID)}))
            print(f"  fold {fold.test_season}{suffix or ' (all seasons)'}: {time.time() - t0:.1f}s", flush=True)
    return {"predictions": pd.concat(preds, ignore_index=True), "configs": pd.DataFrame(configs),
            "coefficients": pd.DataFrame(coefs), "curves": pd.concat(curves, ignore_index=True)}


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
    print("Running GAM on 11 folds x 3 horizons, all seasons and 7 seasons ...")
    res = run(Y, cfg)
    res["predictions"][COLUMNS].to_csv(TABLES / "predictions.csv", index=False)
    res["configs"].to_csv(TABLES / "gam_chosen_configs.csv", index=False)
    res["coefficients"].to_csv(TABLES / "gam_coefficients.csv", index=False)
    res["curves"].to_csv(TABLES / "gam_seasonal_curve.csv", index=False)

    print("\nSupplementary: ablation experiment ...")
    ablation_variants(Y, cfg).to_csv(TABLES / "gam_ablation_predictions.csv", index=False)
    print(f"Xong -> {TABLES}")


if __name__ == "__main__":
    main()
