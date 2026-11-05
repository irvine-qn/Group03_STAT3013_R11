"""Random Forest với 3 biến thể khai báo trước, đúng đặc trưng và đúng ngân sách của LightGBM.

  rf_raw      y_t, y_t-1, y_t-2, c_t, c_target, ngày đích
  rf_anomaly  dự báo anomaly từ a_t, a_t-1, ngày đích
  rf_noclim   như rf_raw nhưng không có climatology
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from floodbench.models.ml import SEED, _anom, _noclim, _raw
from floodbench.models.tuning import pick_config

# 6 cấu hình (max_depth, min_samples_leaf, max_features); 300 cây cố định
RF_GRID = [(4, 5, 1.0), (8, 5, 1.0), (None, 5, 1.0), (None, 10, 1.0), (8, 5, 0.5), (None, 5, 0.5)]
N_TREES = 300


def _make(variant):
    feats = {"raw": _raw, "noclim": _noclim, "anomaly": _anom}[variant]
    on_anomaly = variant == "anomaly"

    def fit_predict(train, test, cfg, h):
        depth, leaf, max_features = cfg
        y = train["y_target"] - (train["c_target"] if on_anomaly else 0)
        model = RandomForestRegressor(n_estimators=N_TREES, max_depth=depth, min_samples_leaf=leaf,
                                      max_features=max_features, random_state=SEED, n_jobs=-1)
        model.fit(feats(train), y.to_numpy())
        return model.predict(feats(test)) + (test["c_target"].to_numpy() if on_anomaly else 0)

    def run(train, test, h):
        cfg, _ = pick_config(train, RF_GRID, fit_predict, h)
        return fit_predict(train, test, cfg, h)

    return run


rf_raw, rf_anomaly, rf_noclim = _make("raw"), _make("anomaly"), _make("noclim")
