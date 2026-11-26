"""Bước 1 – Sinh dự báo LightGBM (3 biến thể, seed 0) và mốc AnomAR1 cho 11 mùa test.

Chạy:  python src/LightGBM/lightgbm_forecast.py
Ghi:   results/LightGBM/tables/predictions.csv   season, t, doy_target, y_target, model, h, pred
(cùng cột với bảng của mọi mô hình trong nhóm). Bước 2 là lightgbm_evaluate.py.
Độ ổn định theo seed, độ quan trọng đặc trưng... nằm ở lightgbm_diag.py.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from lightgbm_data import REPO_ROOT, TABLES, load_config, load_flood_matrix
from lightgbm_model import lgbm_anomaly, lgbm_noclim, lgbm_raw
from lightgbm_protocol import HORIZONS, anom_ar1, build_rows, rolling_origin

COLUMNS = ["season", "t", "doy_target", "y_target", "model", "h", "pred"]
MODELS = {"lgbm_raw": lgbm_raw, "lgbm_anomaly": lgbm_anomaly, "lgbm_noclim": lgbm_noclim,
          "anom_ar1": anom_ar1}


def run(Y: np.ndarray, config: dict, models: dict = MODELS) -> pd.DataFrame:
    """Dự báo của mọi mô hình trong `models` (tên -> hàm (train, test, h)) trên 11 mùa test."""
    preds = []
    for fold in rolling_origin(config):
        for h in HORIZONS:
            train = build_rows(Y, config, fold.train_seasons, h)
            test = build_rows(Y, config, [fold.test_season], h)
            keep = test[["season", "t", "doy_target", "y_target"]]
            for name, model in models.items():
                preds.append(keep.assign(model=name, h=h, pred=model(train, test, h)))
        print(f"  mùa {fold.test_season}: xong", flush=True)
    return pd.concat(preds, ignore_index=True)[COLUMNS]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_config("mekong")
    Y = load_flood_matrix(config)
    pred = run(Y, config)
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / "predictions.csv"
    pred.to_csv(path, index=False)
    print(f"{len(pred)} dự báo -> {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
