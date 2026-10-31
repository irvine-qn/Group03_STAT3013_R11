"""Baseline bắt buộc. Mỗi mô hình là hàm (train_rows, test_rows, h) -> mảng dự báo."""
from __future__ import annotations

import numpy as np


def persistence(train, test, h):
    return test["y_t"].to_numpy()


def clim_all(train, test, h):
    """Trung bình MỌI mùa trước (bị chệch vì đỉnh lũ giảm theo năm)."""
    return test["clim_all_target"].to_numpy()


def clim5(train, test, h):
    return test["c_target"].to_numpy()


def seasonal_naive(train, test, h):
    """Giá trị cùng bước của mùa liền trước."""
    return test["y_prev_season"].to_numpy()


def anom_ar1(train, test, h):
    """c_{t+h} + phi^h (y_t - c_t); phi ước lượng (hồi quy qua gốc) chỉ từ các dòng huấn luyện."""
    a_now = (train["y_t"] - train["c_t"]).to_numpy()
    a_prev = (train["y_t1"] - train["c_t1"]).to_numpy()
    phi = float((a_now * a_prev).sum() / (a_prev ** 2).sum())
    return test["c_target"].to_numpy() + phi ** h * (test["y_t"] - test["c_t"]).to_numpy()


BASELINES = {
    "persistence": persistence,
    "clim_all": clim_all,
    "clim5": clim5,
    "seasonal_naive": seasonal_naive,
    "anom_ar1": anom_ar1,
}
