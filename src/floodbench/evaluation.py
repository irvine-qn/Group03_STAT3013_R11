"""Thước đo sai số. Đơn vị so sánh là mùa, không phải 27 bước trong mùa."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(err) -> float:
    return float(np.sqrt(np.mean(np.square(err))))


def mae(err) -> float:
    return float(np.mean(np.abs(err)))


def season_mse(pred: pd.DataFrame) -> pd.DataFrame:
    """MSE của từng (model, h, mùa test). pred có cột model, h, season, y_target, pred."""
    p = pred.assign(sq=(pred["y_target"] - pred["pred"]) ** 2)
    return p.groupby(["model", "h", "season"], as_index=False)["sq"].mean().rename(columns={"sq": "mse"})


def pooled_rmse(pred: pd.DataFrame, by=("model", "h")) -> pd.DataFrame:
    p = pred.assign(sq=(pred["y_target"] - pred["pred"]) ** 2)
    out = p.groupby(list(by))["sq"].mean().pow(0.5).rename("rmse").reset_index()
    return out
