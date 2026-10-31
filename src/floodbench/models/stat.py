"""Mô hình thống kê: GAM (pyGAM) dự báo anomaly từ ngày trong năm (spline) và hai trễ anomaly."""
from __future__ import annotations

import numpy as np
from pygam import LinearGAM, l, s

from floodbench.models.tuning import pick_config

# 6 cấu hình (n_splines, lam), cùng ngân sách với LightGBM
GAM_GRID = [(6, 0.1), (6, 1.0), (8, 0.6), (8, 5.0), (12, 1.0), (12, 10.0)]


def _xy(rows):
    a_t = (rows["y_t"] - rows["c_t"]).to_numpy()
    a_t1 = (rows["y_t1"] - rows["c_t1"]).to_numpy()
    X = np.column_stack([rows["doy_target"].to_numpy(), a_t, a_t1])
    return X


def _fit_predict(train, test, cfg, h):
    n_splines, lam = cfg
    y = (train["y_target"] - train["c_target"]).to_numpy()
    gam = LinearGAM(s(0, n_splines=n_splines, lam=lam) + l(1) + l(2)).fit(_xy(train), y)
    return test["c_target"].to_numpy() + gam.predict(_xy(test))


def gam(train, test, h):
    cfg, _ = pick_config(train, GAM_GRID, _fit_predict, h)
    return _fit_predict(train, test, cfg, h)
