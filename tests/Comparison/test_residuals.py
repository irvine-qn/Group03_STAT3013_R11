"""Kiểm định phần dư thống nhất (src/Comparison/comparison_residuals.py).

Nhóm 1: công thức khớp đáp án tính tay và khớp statsmodels / scipy.
Nhóm 2: bảng trên dữ liệu thật khớp bảng của từng mô hình (GAM, AnomAR1) và bảng so sánh chung.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf as sm_acf

from comparison_residuals import (
    MAIN_MODELS,
    acf,
    ljung_box_p,
    load_main,
    normality_row,
    paired_diff_normality,
    paired_quantities,
    pooled_summary,
    residual_summary_all,
    residual_tests_all,
    season_tests,
)

RESULTS = Path(__file__).resolve().parents[2] / "results"


# ---- Nhóm 1: công thức --------------------------------------------------------------------------
def test_acf_alternating_series_known_answer():
    # x = [1, −1, 1, −1]: trung bình 0, Σx² = 4, Σ x_k x_{k+1} = −3, Σ x_k x_{k+2} = 2
    np.testing.assert_allclose(acf([1, -1, 1, -1], 2), [-0.75, 0.5])


def test_acf_matches_statsmodels():
    x = np.random.default_rng(1).normal(size=40)
    np.testing.assert_allclose(acf(x, 5), sm_acf(x, nlags=5, fft=False)[1:])


def test_ljung_box_known_answer():
    # x = [1, −1, 1, −1], 2 trễ: Q = 4·6·(0,5625/3 + 0,25/2) = 7,5 ~ χ²(2)
    assert ljung_box_p([1, -1, 1, -1], 2) == pytest.approx(stats.chi2.sf(7.5, 2))


def test_ljung_box_matches_statsmodels():
    x = np.random.default_rng(2).normal(size=24)
    expected = float(acorr_ljungbox(x, lags=[10])["lb_pvalue"].iloc[0])
    assert ljung_box_p(x, 10) == pytest.approx(expected)


def test_season_tests_known_values():
    out = season_tests([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
    assert out["n"] == 12 and out["bias_km2"] == pytest.approx(2 / 12)
    assert set(out) == {"n", "bias_km2", "acf_1", "acf_2", "acf_3", "acf_4", "acf_5", "ljung_box_p_lag10"}


def test_pooled_summary_known_values():
    err = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    out = pooled_summary(err)
    assert out["bias_km2"] == pytest.approx(4.0)
    assert out["sd_km2"] == pytest.approx(np.sqrt(((err - 4) ** 2).sum() / 4))  # = √12,5
    assert out["sd_km2"] == pytest.approx(np.sqrt(12.5))
    assert out["skew"] == pytest.approx(stats.skew(err))
    assert out["excess_kurtosis"] == pytest.approx(stats.kurtosis(err))
    assert out["shapiro_p"] == pytest.approx(stats.shapiro(err).pvalue)


def test_paired_quantities_known_values():
    q = paired_quantities(m=[2.0, 4.0], b=[1.0, 4.0])
    np.testing.assert_allclose(q["mse_diff"], [1.0, 0.0])
    np.testing.assert_allclose(q["rel_mse_diff"], [1.0, 0.0])
    np.testing.assert_allclose(q["log_rmse_ratio"], [0.5 * np.log(2.0), 0.0])


def test_normality_row_matches_scipy():
    v = np.array([0.1, -0.3, 0.25, 0.0, 0.4, -0.2, 0.05, 0.15, -0.1, 0.3, -0.05])
    out = normality_row(v)
    sw = stats.shapiro(v)
    assert out["n"] == 11 and out["mean"] == pytest.approx(v.mean())
    assert out["shapiro_w"] == pytest.approx(sw.statistic) and out["shapiro_p"] == pytest.approx(sw.pvalue)


def test_tables_on_a_small_known_prediction_table():
    """Hai mô hình giả, 1 tầm, 2 mùa: số dòng và bias đúng như tính tay."""
    rows = []
    for model, shift in [("anom_ar1", 1.0), ("gam", -2.0)]:
        for season in (2010, 2011):
            for t in range(3, 15):
                rows.append({"model": model, "h": 1, "season": season, "t": t,
                             "err": shift + (-1) ** t})
    pred = pd.DataFrame(rows)
    tests = residual_tests_all(pred)
    assert len(tests) == 4
    np.testing.assert_allclose(tests.set_index(["model", "season"]).loc[("gam", 2010), "bias_km2"], -2.0)
    summary = residual_summary_all(pred, tests).set_index("model")
    assert summary.loc["anom_ar1", "bias_km2"] == pytest.approx(1.0)
    assert summary.loc["gam", "n_seasons"] == 2


# ---- Nhóm 2: dữ liệu thật -----------------------------------------------------------------------
@pytest.fixture(scope="module")
def real():
    pred = load_main()
    tests = residual_tests_all(pred)
    return pred, tests, residual_summary_all(pred, tests), paired_diff_normality(pred)


def test_real_table_shapes(real):
    pred, tests, summary, normality = real
    assert sorted(pred["model"].unique()) == sorted(MAIN_MODELS)
    assert pred.groupby("model").size().eq(748).all()
    assert len(tests) == 4 * 3 * 11 and len(summary) == 4 * 3 and len(normality) == 3 * 3 * 3


def test_gam_matches_gam_folder(real):
    _, tests, summary, _ = real
    theirs = pd.read_csv(RESULTS / "GAM" / "tables" / "gam_residual_tests.csv").set_index(["h", "season"])
    mine = tests[tests["model"] == "gam"].set_index(["h", "season"]).loc[theirs.index]
    for col, their_col in [("n", "n"), ("bias_km2", "bias_km2"), ("ljung_box_p_lag10", "ljung_box_p_lag10")] + \
            [(f"acf_{k}", f"acf_{k}") for k in range(1, 6)]:
        np.testing.assert_allclose(mine[col], theirs[their_col], rtol=1e-9, atol=1e-9)
    their_sum = pd.read_csv(RESULTS / "GAM" / "tables" / "gam_residual_summary.csv").set_index("h")
    my_sum = summary[summary["model"] == "gam"].set_index("h").loc[their_sum.index]
    for col in ["n", "bias_km2", "sd_km2", "skew", "excess_kurtosis",
                "mean_acf_1", "mean_acf_2", "mean_acf_3"]:
        np.testing.assert_allclose(my_sum[col], their_sum[col], rtol=1e-9, atol=1e-9)
    # scipy.stats.shapiro tính bằng float32 bên trong, nên thứ tự đầu vào làm p lệch ở chữ số thứ 8
    np.testing.assert_allclose(my_sum["shapiro_p"], their_sum["shapiro_p"], rtol=1e-6)
    np.testing.assert_allclose(my_sum["share_seasons_lb_p_lt_0.05"],
                               their_sum["share_seasons_ljung_box_p_lt_0.05"])


def test_anom_ar1_matches_anomar1_folder(real):
    _, tests, _, _ = real
    theirs = pd.read_csv(RESULTS / "AnomAR1" / "tables" / "residual_tests.csv").set_index(["h", "season"])
    mine = tests[tests["model"] == "anom_ar1"].set_index(["h", "season"]).loc[theirs.index]
    for col in ["bias_km2"] + [f"acf_{k}" for k in range(1, 6)]:
        np.testing.assert_allclose(mine[col], theirs[col], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(mine["ljung_box_p_lag10"], theirs["ljung_box_p"], rtol=1e-9, atol=1e-12)


def test_relative_diff_mean_matches_comparison_table(real):
    """Trung bình rel_mse_diff × 100 phải bằng mse_diff_pct của bảng họ chính (cùng 11 mùa)."""
    _, _, _, normality = real
    main = pd.read_csv(RESULTS / "Comparison" / "tables" / "vs_anom_ar1_main.csv").set_index(["model", "h"])
    rel = normality[normality["quantity"] == "rel_mse_diff"].set_index(["model", "h"]).loc[main.index]
    np.testing.assert_allclose(100 * rel["mean"], main["mse_diff_pct"], rtol=1e-9)
