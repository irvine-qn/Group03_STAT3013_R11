"""Bước 1 – Sinh dự báo Chronos-2 và mốc AnomAR1 cho 11 mùa test (rolling origin theo năm).

Chạy:  python src/Chronos-2/chronos2_forecast.py
Kết quả: results/Chronos-2/tables/predictions.csv, mỗi dòng là một dự báo:
    season, t, doy_target, y_target, model, h, pred
(cùng cột với bảng chung của nhóm để ghép được). Bước 2 là chronos2_evaluate.py.

Rolling origin: với mùa test Y, AnomAR1 ước lượng φ từ các mùa 2003..Y−1 rồi dự báo mùa Y.
Chronos-2 không học gì từ các mùa trước; nó chỉ cần climatology (5 mùa trước) để tính anomaly.
"""
from __future__ import annotations

import sys

import pandas as pd

from chronos2_model import Chronos2ZeroShot, build_anomaly_context, load_pipeline
from chronos2_data import REPO_ROOT, load_config, load_flood_matrix, step_to_doy
from chronos2_protocol import (
    HORIZONS,
    TEST_SEASONS,
    anom_ar1_forecast,
    climatology,
    estimate_phi,
    forecast_origins,
    training_seasons,
)

PREDICTIONS_PATH = REPO_ROOT / "results" / "Chronos-2" / "tables" / "predictions.csv"
COLUMNS = ["season", "t", "doy_target", "y_target", "model", "h", "pred"]


def chronos_season(Y, config, chronos: Chronos2ZeroShot, test_season: int) -> list[dict]:
    """Dự báo Chronos-2 cho một mùa test, mọi gốc t và mọi tầm h.

    Mỗi gốc chỉ gọi model MỘT lần, xin luôn 4 bước tới, rồi lấy bước thứ h cho từng tầm.
    """
    season_idx = test_season - config["first_year"]
    c = climatology(Y, season_idx)
    n_steps, max_h = Y.shape[1], max(HORIZONS)
    rows = []
    for origin in forecast_origins(n_steps, h=1):  # gốc t = 3..26
        context = build_anomaly_context(Y, season_idx, origin)
        anomaly_forecast = chronos.forecast_anomaly(context, n_ahead=max_h)
        for h in HORIZONS:
            if origin + h > n_steps:  # đích nằm ngoài mùa
                continue
            pred = c[origin + h - 1] + anomaly_forecast[h - 1]  # cộng lại mức bình thường
            rows.append(prediction_row(Y, config, test_season, origin, h, "chronos2", pred))
    return rows


def anom_ar1_season(Y, config, test_season: int) -> list[dict]:
    """Dự báo AnomAR1 cho một mùa test; φ ước lượng lại ở mỗi mùa test và mỗi tầm."""
    season_idx = test_season - config["first_year"]
    train_idx = [s - config["first_year"] for s in training_seasons(test_season, config["first_year"])]
    rows = []
    for h in HORIZONS:
        phi = estimate_phi(Y, train_idx, h)
        for origin in forecast_origins(Y.shape[1], h):
            pred = anom_ar1_forecast(Y, season_idx, origin, h, phi)
            rows.append(prediction_row(Y, config, test_season, origin, h, "anom_ar1", pred))
    return rows


def prediction_row(Y, config, season: int, origin: int, h: int, model: str, pred: float) -> dict:
    """Một dòng kết quả: dự báo cho bước t + h, kèm giá trị thật để tính sai số sau."""
    target_step = origin + h
    return {
        "season": season,
        "t": origin,
        "doy_target": step_to_doy(config, target_step),
        "y_target": Y[season - config["first_year"], target_step - 1],
        "model": model,
        "h": h,
        "pred": float(pred),
    }


def make_predictions(Y, config: dict, chronos: Chronos2ZeroShot) -> pd.DataFrame:
    """Chạy cả hai mô hình trên 11 mùa test và gộp thành một bảng dài."""
    rows = []
    for test_season in TEST_SEASONS:
        rows += chronos_season(Y, config, chronos, test_season)
        rows += anom_ar1_season(Y, config, test_season)
        print(f"  season {test_season}: done", flush=True)
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # để in tiếng Việt trên console Windows
    config = load_config("mekong")
    Y = load_flood_matrix(config)
    print("Loading Chronos-2 ...", flush=True)
    chronos = Chronos2ZeroShot(load_pipeline())
    predictions = make_predictions(Y, config, chronos)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(f"{len(predictions)} forecasts -> {PREDICTIONS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
