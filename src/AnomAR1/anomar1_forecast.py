"""Bước 1 – Sinh dự báo AnomAR1 (và các mốc đơn giản) cho 11 mùa test, rolling origin theo năm.

Chạy:  python src/AnomAR1/anomar1_forecast.py
Ghi:   results/AnomAR1/tables/predictions.csv   season, t, doy_target, y_target, model, h, pred
       results/AnomAR1/tables/phi_by_fold.csv   φ của từng (mô hình, mùa test, tầm) kèm sai số chuẩn
       results/AnomAR1/tables/regression_rows.csv  bảng hồi quy của φ (mọi mùa 2003–2020)
Bước 2 là anomar1_evaluate.py.

Mô hình trong bảng:
  anom_ar1      AnomAR1 học trên mọi mùa 2003..Y−1 (mốc chính của nhóm)
  anom_ar1_w7   AnomAR1 chỉ học 7 mùa gần nhất (lượng dữ liệu học không đổi giữa các mùa test)
  persistence   giá trị hôm nay giữ nguyên
  clim5         chỉ mức bình thường (trung bình 5 mùa trước)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from anomar1_data import REPO_ROOT, TABLES, load_config, load_flood_matrix
from anomar1_protocol import (
    HORIZONS,
    MIN_PRIOR,
    TEST_SEASONS,
    WINDOW,
    anom_ar1,
    build_rows,
    clim5,
    estimate_phi,
    persistence,
    regression_xy,
    training_seasons,
)

COLUMNS = ["season", "t", "doy_target", "y_target", "model", "h", "pred"]
AR_VARIANTS = ((None, "anom_ar1"), (WINDOW, "anom_ar1_w7"))
SIMPLE_BASELINES = {"persistence": persistence, "clim5": clim5}


def regression_table(Y: np.ndarray, config: dict) -> pd.DataFrame:
    """Bảng hồi quy của φ cho mọi mùa có đủ 3 mùa trước, mọi tầm (để xem và kiểm tra tay)."""
    seasons = range(config["first_year"] + MIN_PRIOR, config["last_year"] + 1)
    frames = []
    for h in HORIZONS:
        rows = build_rows(Y, config, seasons, h)
        x, y = regression_xy(rows)
        frames.append(rows.assign(h=h, x_anomaly_prev=x, y_anomaly_now=y))
    return pd.concat(frames, ignore_index=True)


def phi_record(name: str, test_season: int, h: int, seasons: list[int], train: pd.DataFrame) -> dict:
    """φ của một fold, kèm φ và sai số chuẩn tính lại bằng statsmodels để đối chiếu."""
    x, y = regression_xy(train)
    ols = sm.OLS(y, x).fit()
    phi = estimate_phi(train)
    return {
        "model": name, "test_season": test_season, "h": h,
        "first_train": seasons[0], "last_train": seasons[-1],
        "n_train_seasons": len(seasons), "n_train_rows": len(train),
        "phi": phi, "phi_statsmodels": float(ols.params[0]), "phi_se": float(ols.bse[0]),
        "phi_pow_h": phi**h,
    }


def run(Y: np.ndarray, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dự báo của mọi mô hình trên 11 mùa test; φ ước lượng lại ở mỗi mùa test và mỗi tầm."""
    preds, phis = [], []
    for test_season in TEST_SEASONS:
        for h in HORIZONS:
            test = build_rows(Y, config, [test_season], h)
            keep = test[["season", "t", "doy_target", "y_target"]]
            for window, name in AR_VARIANTS:
                seasons = training_seasons(test_season, config["first_year"], window)
                train = build_rows(Y, config, seasons, h)
                preds.append(keep.assign(model=name, h=h, pred=anom_ar1(train, test, h)))
                phis.append(phi_record(name, test_season, h, seasons, train))
            for name, baseline in SIMPLE_BASELINES.items():
                preds.append(keep.assign(model=name, h=h, pred=baseline(test)))
    pred = pd.concat(preds, ignore_index=True)[COLUMNS]
    return pred.sort_values(["model", "h", "season", "t"], ignore_index=True), pd.DataFrame(phis)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_config("mekong")
    Y = load_flood_matrix(config)
    TABLES.mkdir(parents=True, exist_ok=True)

    pred, phi = run(Y, config)
    pred.to_csv(TABLES / "predictions.csv", index=False)
    phi.to_csv(TABLES / "phi_by_fold.csv", index=False)
    regression_table(Y, config).to_csv(TABLES / "regression_rows.csv", index=False)

    print(f"{len(pred)} forecasts -> {(TABLES / 'predictions.csv').relative_to(REPO_ROOT)}")
    print("max |φ − statsmodels|:", float((phi["phi"] - phi["phi_statsmodels"]).abs().max()))
    print(phi[phi["model"] == "anom_ar1"].pivot(index="test_season", columns="h", values="phi")
          .round(4).to_string())


if __name__ == "__main__":
    main()
