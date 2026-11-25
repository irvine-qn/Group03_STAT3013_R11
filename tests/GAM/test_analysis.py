"""Test các hàm thống kê tự viết trong analysis.py (đáp án biết trước)."""
import numpy as np
import pytest

from analysis import acf, ljung_box_p, theoretical_acf


def test_acf_of_alternating_series_is_minus_one_like():
    x = np.tile([1.0, -1.0], 50)
    assert acf(x, 1)[0] == pytest.approx(-0.99, abs=0.01)


def test_acf_matches_statsmodels():
    tsa = pytest.importorskip("statsmodels.tsa.stattools")
    x = np.random.default_rng(0).normal(size=40).cumsum()
    np.testing.assert_allclose(acf(x, 5), tsa.acf(x, nlags=5, fft=False)[1:])


def test_ljung_box_matches_statsmodels():
    diag = pytest.importorskip("statsmodels.stats.diagnostic")
    x = np.random.default_rng(1).normal(size=30)
    expected = diag.acorr_ljungbox(x, lags=[10])["lb_pvalue"].iloc[0]
    assert ljung_box_p(x, 10) == pytest.approx(expected)


def test_ljung_box_separates_noise_from_strong_autocorrelation():
    rng = np.random.default_rng(2)
    noise = rng.normal(size=200)
    ar = np.zeros(200)
    for t in range(1, 200):
        ar[t] = 0.9 * ar[t - 1] + rng.normal()
    assert ljung_box_p(noise, 10) > 0.05
    assert ljung_box_p(ar, 10) < 1e-6


def test_theoretical_acf_known_values():
    phi = 0.8
    assert theoretical_acf(phi, 1, 1) == 0.0                           # 1 bước: sai số độc lập
    assert theoretical_acf(phi, 2, 1) == pytest.approx(phi / (1 + phi ** 2))  # MA(1)
    assert theoretical_acf(phi, 2, 2) == 0.0
    assert theoretical_acf(phi, 4, 3) == pytest.approx(phi ** 3 / (1 + phi ** 2 + phi ** 4 + phi ** 6))
