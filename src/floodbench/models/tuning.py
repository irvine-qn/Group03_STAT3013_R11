"""Chọn cấu hình bằng validation BÊN TRONG tập huấn luyện (3 mùa cuối làm validation).

Mọi mô hình có siêu tham số dùng đúng hàm này và đúng 6 cấu hình, không bao giờ nhìn mùa test.
"""
from __future__ import annotations

import numpy as np

N_VALIDATION_SEASONS = 3


def pick_config(train, configs, fit_predict, h):
    """fit_predict(train_part, valid_part, config, h) -> dự báo. Trả về (config tốt nhất, log)."""
    seasons = sorted(train["season"].unique())
    cut = seasons[-N_VALIDATION_SEASONS]
    fit, valid = train[train["season"] < cut], train[train["season"] >= cut]
    scores = []
    for cfg in configs:
        pred = fit_predict(fit, valid, cfg, h)
        scores.append(float(np.mean((valid["y_target"].to_numpy() - pred) ** 2)))
    best = int(np.argmin(scores))
    return configs[best], scores
