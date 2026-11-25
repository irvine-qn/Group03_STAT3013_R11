"""Test luật chơi chung (protocol.py): climatology, gốc dự báo, AnomAR1."""
import numpy as np
import pytest

from flood_data import load_config, load_flood_matrix
from protocol import (
    anom_ar1_forecast,
    climatology,
    estimate_phi,
    forecast_origins,
    training_seasons,
)


def constant_rows(n_seasons=10, n_steps=27):
    """Ma trận giả: mọi bước của mùa i đều bằng i."""
    return np.repeat(np.arange(n_seasons, dtype=float)[:, None], n_steps, axis=1)


def test_climatology_uses_only_prior_seasons():
    Y = constant_rows()
    expected = np.full(27, 4.0)  # trung bình các mùa 2, 3, 4, 5, 6
    np.testing.assert_allclose(climatology(Y, 7), expected)
    Y[7:] = 1e9  # mùa đang xét và các mùa sau không được ảnh hưởng
    np.testing.assert_allclose(climatology(Y, 7), expected)


def test_climatology_needs_three_prior():
    with pytest.raises(ValueError):
        climatology(constant_rows(), 2)


def test_forecast_origins():
    assert list(forecast_origins(27, 4)) == list(range(3, 24))
    assert list(forecast_origins(27, 1))[-1] == 26


def test_training_seasons():
    assert training_seasons(2010, 2000) == list(range(2003, 2010))


def test_phi_recovers_noise_free_ar1():
    n_seasons, n_steps, phi = 12, 27, 0.8
    Y = np.empty((n_seasons, n_steps))
    for i in range(n_seasons):
        a = np.empty(n_steps)
        a[0] = 100.0 * (i % 3 + 1) * (-1) ** i
        for t in range(1, n_steps):
            a[t] = phi * a[t - 1]
        Y[i] = 5000.0 + a
    # Bỏ 5 mùa đầu để climatology của mùa huấn luyện là trung bình của mùa có anomaly khác dấu.
    # Anomaly so với climatology khi đó vẫn là AR(1) với cùng phi vì phép trừ tuyến tính.
    assert estimate_phi(Y, list(range(5, 12)), h=1) == pytest.approx(phi, abs=1e-9)


def test_phi_uses_rows_of_horizon():
    Y = load_flood_matrix(load_config())
    train_idx = list(range(3, 10))
    assert estimate_phi(Y, train_idx, 1) != estimate_phi(Y, train_idx, 4)


def test_anom_ar1_forecast_formula():
    Y = load_flood_matrix(load_config())
    season_idx, origin, h, phi = 12, 10, 2, 0.9
    c = climatology(Y, season_idx)
    a_now = Y[season_idx, origin - 1] - c[origin - 1]  # bước t (1-based) là chỉ số t − 1
    expected = c[origin + h - 1] + phi**h * a_now
    assert anom_ar1_forecast(Y, season_idx, origin, h, phi) == pytest.approx(expected)
