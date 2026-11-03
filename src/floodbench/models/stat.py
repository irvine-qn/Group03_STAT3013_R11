"""Mô hình thống kê: GAM (pyGAM), ARX và Ridge có trễ. Cả ba dự báo anomaly y_{t+h} - c_{t+h}.

ARX = hồi quy tuyến tính (OLS) của anomaly đích lên p trễ anomaly (a_t, a_t-1, a_t-2) và K cặp
sin/cos của ngày đích (điều hòa mùa vụ), dự báo trực tiếp cho từng tầm h. Lưới (p, K) có 6 cấu hình.
Ridge dùng cùng đặc trưng (p = 3, K = 2) đã chuẩn hóa, chỉ khác phạt L2; scaler fit trên dòng huấn luyện.
"""
from __future__ import annotations

import numpy as np
from pygam import LinearGAM, l, s
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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

# 6 cấu hình (p trễ, K cặp Fourier), cùng ngân sách với GAM và LightGBM
ARX_GRID = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
RIDGE_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # hệ số phạt alpha
RIDGE_P, RIDGE_K = 3, 2


def _lag_fourier(rows, p, K):
    lags = [rows["y_t"] - rows["c_t"], rows["y_t1"] - rows["c_t1"], rows["y_t2"] - rows["c_t2"]][:p]
    angle = 2 * np.pi * rows["doy_target"].to_numpy() / 365.25
    harmonics = [f(k * angle) for k in range(1, K + 1) for f in (np.sin, np.cos)]
    return np.column_stack([a.to_numpy() for a in lags] + harmonics)


def _arx_fit_predict(train, test, cfg, h):
    p, K = cfg
    X = np.column_stack([np.ones(len(train)), _lag_fourier(train, p, K)])
    beta, *_ = np.linalg.lstsq(X, (train["y_target"] - train["c_target"]).to_numpy(), rcond=None)
    Xt = np.column_stack([np.ones(len(test)), _lag_fourier(test, p, K)])
    return test["c_target"].to_numpy() + Xt @ beta


def arx(train, test, h):
    cfg, _ = pick_config(train, ARX_GRID, _arx_fit_predict, h)
    return _arx_fit_predict(train, test, cfg, h)


def _ridge_fit_predict(train, test, alpha, h):
    X = _lag_fourier(train, RIDGE_P, RIDGE_K)
    scaler = StandardScaler().fit(X)  # chỉ từ dòng huấn luyện
    model = Ridge(alpha=alpha).fit(scaler.transform(X), (train["y_target"] - train["c_target"]).to_numpy())
    return test["c_target"].to_numpy() + model.predict(scaler.transform(_lag_fourier(test, RIDGE_P, RIDGE_K)))


def ridge(train, test, h):
    alpha, _ = pick_config(train, RIDGE_GRID, _ridge_fit_predict, h)
    return _ridge_fit_predict(train, test, alpha, h)
