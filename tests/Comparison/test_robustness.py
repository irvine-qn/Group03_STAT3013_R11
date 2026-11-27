"""Độ vững của TOST và t ghép cặp (src/Comparison/comparison_robustness.py).

Nhóm 1: công thức với đáp án tính tay. Nhóm 2: bảng trên dữ liệu thật khớp bảng họ chính.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from comparison_robustness import (
    bootstrap_indices,
    bootstrap_means,
    bootstrap_p_two_sided,
    excludes_zero,
    outlier_flags,
    percentile_ci,
    robustness_tables,
    season_quantities,
    t_ci,
    within_margin,
)
from comparison_residuals import load_main
from comparison_tables import _tost_p

RESULTS = Path(__file__).resolve().parents[2] / "results"


# ---- Nhóm 1: công thức --------------------------------------------------------------------------
def test_season_quantities_known_values():
    q = season_quantities(m=[4.0, 1.0], b=[1.0, 1.0])
    np.testing.assert_allclose(q["d"], [np.log(2.0), 0.0])  # ½·ln 4 = ln 2
    np.testing.assert_allclose(q["mse_diff"], [3.0, 0.0])


def test_outlier_flags_known_values():
    x = np.array([0.0] * 10 + [10.0])
    f = outlier_flags(x)
    # trung bình 10/11, SD = √(10/11 · 10) = √(100/11); z cuối = (10 − 10/11)/√(100/11) = √11 · 0,9091… ≈ 3,015
    assert f["z"][-1] == pytest.approx((10 - 10 / 11) / np.sqrt(100 / 11))
    assert f["z_out"].tolist() == [False] * 10 + [True]
    assert f["iqr_out"].tolist() == [False] * 10 + [True]  # Q1 = Q3 = 0, IQR = 0
    clean = outlier_flags(np.arange(11.0))
    assert not clean["z_out"].any() and not clean["iqr_out"].any()


def test_t_ci_known_values():
    lo, hi = t_ci([-0.02, 0.0, 0.02], 0.90)
    half = stats.t.ppf(0.95, 2) * 0.02 / np.sqrt(3)  # SD = 0,02; t(0,95; 2) = 2,919986
    assert half == pytest.approx(2.919986 * 0.02 / np.sqrt(3), rel=1e-6)
    assert (lo, hi) == pytest.approx((-half, half))


def test_bootstrap_indices_are_reproducible_and_in_range():
    a, b = bootstrap_indices(11, 500, seed=0), bootstrap_indices(11, 500, seed=0)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (500, 11) and a.min() >= 0 and a.max() <= 10


def test_bootstrap_means_known_indices():
    x = np.array([1.0, 2.0, 3.0])
    idx = np.array([[0, 0, 0], [0, 1, 2], [2, 2, 1]])
    np.testing.assert_allclose(bootstrap_means(x, idx), [1.0, 2.0, 8 / 3])


def test_percentile_ci_known_values():
    lo, hi = percentile_ci(np.arange(101.0), 0.90)  # phân vị 5 và 95 của 0..100
    assert (lo, hi) == pytest.approx((5.0, 95.0))


def test_constant_data_bootstrap():
    x = np.full(11, 3.0)
    idx = bootstrap_indices(11, 200)
    assert percentile_ci(bootstrap_means(x, idx), 0.95) == pytest.approx((3.0, 3.0))
    # dịch tâm về 0 thì mọi trung bình bootstrap = 0 < 3, nên p = 0
    assert bootstrap_p_two_sided(x, idx) == 0.0


def test_bootstrap_p_hand_example():
    x = np.array([1.0, -1.0, 3.0])  # trung bình 1; dịch tâm: [0, −2, 2]
    idx = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 2]])  # trung bình: 0, −2, 2, 0
    assert bootstrap_p_two_sided(x, idx) == pytest.approx(0.5)  # |−2| ≥ 1 và |2| ≥ 1: 2/4


def test_margin_and_zero_rules():
    lim = np.log(1.05)
    assert within_margin((-lim + 1e-9, lim - 1e-9)) and not within_margin((-lim - 1e-9, 0.0))
    assert excludes_zero((0.1, 0.2)) and excludes_zero((-0.2, -0.1)) and not excludes_zero((-0.1, 0.1))


def test_t_ci90_inside_margin_iff_tost_p_below_005():
    """Dạng khoảng tin cậy của TOST: p_tost < 0,05 ⇔ CI t 90% của d nằm trong (−ln 1,05; ln 1,05)."""
    rng = np.random.default_rng(3)
    for _ in range(300):
        b = rng.uniform(1e5, 2e6, 11)
        m = b * np.exp(2 * rng.normal(rng.uniform(-0.05, 0.05), rng.uniform(0.005, 0.08), 11))
        d = 0.5 * np.log(m / b)
        assert (_tost_p(m, b, 0.05) < 0.05) == within_margin(t_ci(d, 0.90))


# ---- Nhóm 2: dữ liệu thật -----------------------------------------------------------------------
@pytest.fixture(scope="module")
def real():
    return robustness_tables(load_main())


def test_real_shapes(real):
    main, seasons = real
    assert len(main) == 9 and len(seasons) == 99


def test_real_p_values_match_main_family_table(real):
    main, _ = real
    ref = pd.read_csv(RESULTS / "Comparison" / "tables" / "vs_anom_ar1_main.csv").set_index(["model", "h"])
    mine = main.set_index(["model", "h"]).loc[ref.index]
    for col in ["p_t", "p_tost", "p_t_holm_main"]:
        np.testing.assert_allclose(mine[col], ref[col], rtol=1e-10)


def test_real_t_ci_reproduces_tost_conclusion(real):
    main, _ = real
    assert (main["equiv_t"] == (main["p_tost"] < 0.05)).all()
    assert (main["equiv_t"] == main["equiv_tost"]).all()


def test_real_season_values_rebuild_means(real):
    main, seasons = real
    means = seasons.groupby(["model", "h"])[["d", "mse_diff"]].mean()
    mine = main.set_index(["model", "h"]).loc[means.index]
    np.testing.assert_allclose(mine["mean_d"], means["d"])
    np.testing.assert_allclose(mine["mean_msediff"], means["mse_diff"])
