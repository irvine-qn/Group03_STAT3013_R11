"""Mô hình ML: LightGBM (3 biến thể KHAI BÁO TRƯỚC khi chạy test) và Chronos-2 zero-shot.

Biến thể LightGBM, không được chọn lại sau khi xem điểm test:
  lgbm_raw      đặc trưng thô y_t, y_t-1, y_t-2, c_t, c_target, ngày đích (bản chính)
  lgbm_anomaly  dự báo anomaly từ a_t, a_t-1, ngày đích
  lgbm_noclim   như lgbm_raw nhưng KHÔNG có climatology (ablation: lợi thế đến từ đâu)
Chronos-2: ngữ cảnh là anomaly của mùa test từ bước 1 đến t, lấy phân vị 0,5, cộng lại climatology.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

from floodbench.features import climatology
from floodbench.models.tuning import pick_config

# 6 cấu hình (num_leaves, n_estimators, learning_rate)
LGBM_GRID = [(4, 200, 0.05), (4, 100, 0.05), (8, 200, 0.05), (4, 400, 0.03), (2, 300, 0.05), (8, 100, 0.1)]
SEED = 0


def _raw(rows):
    return rows[["y_t", "y_t1", "y_t2", "c_t", "c_target", "doy_target"]].to_numpy()


def _noclim(rows):
    return rows[["y_t", "y_t1", "y_t2", "doy_target"]].to_numpy()


def _anom(rows):
    return np.column_stack([rows["y_t"] - rows["c_t"], rows["y_t1"] - rows["c_t1"], rows["doy_target"]])


def _make(variant):
    feats = {"raw": _raw, "noclim": _noclim, "anomaly": _anom}[variant]
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


lgbm_raw, lgbm_anomaly, lgbm_noclim = _make("raw"), _make("anomaly"), _make("noclim")


class Chronos2Zero:
    """Zero-shot: không huấn luyện, không tune. Tải trọng số amazon/chronos-2 một lần."""

    def __init__(self, Y, cfg, max_horizon=4):
        import torch
        from chronos import Chronos2Pipeline

        self._torch = torch
        self.pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
        self.Y, self.cfg, self.max_h = Y, cfg, max_horizon
        self._cache = {}

    def _forecast(self, season, t):
        key = (season, t)
        if key not in self._cache:
            i = season - self.cfg.first_year
            c = climatology(self.Y, i)
            ctx = self._torch.tensor(self.Y[i, :t] - c[:t], dtype=self._torch.float32)
            q, _ = self.pipe.predict_quantiles([ctx], prediction_length=self.max_h, quantile_levels=[0.5])
            self._cache[key] = q[0][0, :, 0].numpy()
        return self._cache[key]

    def __call__(self, train, test, h):
        anomaly = np.array([self._forecast(s, t)[h - 1] for s, t in zip(test["season"], test["t"])])
        return test["c_target"].to_numpy() + anomaly
