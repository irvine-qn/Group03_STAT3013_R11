"""Mô hình LightGBM: 3 biến thể KHAI BÁO TRƯỚC khi chạy test, không chọn lại sau khi xem điểm test.

  lgbm_raw      đặc trưng thô y_t, y_t-1, y_t-2, c_t, c_target, ngày đích (bản chính)
  lgbm_anomaly  dự báo anomaly (y − climatology) từ a_t, a_t-1, ngày đích
  lgbm_noclim   như lgbm_raw nhưng KHÔNG có climatology (ablation: lợi thế đến từ đâu)

Mỗi (mùa test, tầm): chọn 1 trong 6 cấu hình bằng 3 mùa cuối của tập học (pick_config),
rồi học lại trên toàn bộ tập học với cấu hình đó và dự báo mùa test.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from lightgbm_protocol import pick_config

# 6 cấu hình (num_leaves, n_estimators, learning_rate)
LGBM_GRID = [(4, 200, 0.05), (4, 100, 0.05), (8, 200, 0.05), (4, 400, 0.03), (2, 300, 0.05), (8, 100, 0.1)]
SEED = 0  # seed của bagging (subsample = 0.8); bản chính thức dùng seed 0
VARIANTS = ("raw", "anomaly", "noclim")

# Lưới Random Forest của nhóm (max_depth, min_samples_leaf, max_features), 300 cây cố định.
# Chỉ dùng trong thí nghiệm đối chứng của lightgbm_diag.py.
RF_GRID = [(4, 5, 1.0), (8, 5, 1.0), (None, 5, 1.0), (None, 10, 1.0), (8, 5, 0.5), (None, 5, 0.5)]
N_TREES = 300


def _raw(rows: pd.DataFrame) -> np.ndarray:
    return rows[["y_t", "y_t1", "y_t2", "c_t", "c_target", "doy_target"]].to_numpy()


def _noclim(rows: pd.DataFrame) -> np.ndarray:
    return rows[["y_t", "y_t1", "y_t2", "doy_target"]].to_numpy()


def _anom(rows: pd.DataFrame) -> np.ndarray:
    return np.column_stack([rows["y_t"] - rows["c_t"], rows["y_t1"] - rows["c_t1"], rows["doy_target"]])


DESIGN = {"raw": _raw, "anomaly": _anom, "noclim": _noclim}


def make_model(variant: str):
    """Trả về hàm (train, test, h) -> dự báo km² của biến thể `variant`."""
    feats = DESIGN[variant]
    on_anomaly = variant == "anomaly"

    def fit_predict(train, test, cfg, h):
        leaves, n_est, lr = cfg
        y = train["y_target"] - (train["c_target"] if on_anomaly else 0)
        model = lgb.LGBMRegressor(num_leaves=leaves, n_estimators=n_est, learning_rate=lr,
                                  min_child_samples=10, subsample=0.8, subsample_freq=1,
                                  random_state=SEED, verbose=-1)
        model.fit(feats(train), y.to_numpy())
        out = model.predict(feats(test))
        return out + (test["c_target"].to_numpy() if on_anomaly else 0)

    def run(train, test, h):
        cfg, _ = pick_config(train, LGBM_GRID, fit_predict, h)
        return fit_predict(train, test, cfg, h)

    return run


lgbm_raw, lgbm_anomaly, lgbm_noclim = make_model("raw"), make_model("anomaly"), make_model("noclim")
