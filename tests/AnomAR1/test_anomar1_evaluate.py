"""Test bảng đánh giá, chẩn đoán và EDA."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anomar1_diagnostics import direct_phi
from anomar1_eda import inference, season_peaks
from anomar1_evaluate import compare_with_anom_ar1, condition_tables, holm, rmse, rmse_table, season_mse

SRC = Path(__file__).resolve().parents[2] / "src" / "AnomAR1"


def test_modules_have_anomar1_prefix():
    names = [p.name for p in SRC.glob("*.py")]
    assert names and all(n.startswith("anomar1_") for n in names)


def test_direct_phi_recovers_known_value():
    x = np.array([100.0, -50.0, 30.0])
    train = pd.DataFrame({"y_t": x + 5, "c_t": 5.0, "y_target": 0.6 * x + 7, "c_target": 7.0})
    assert direct_phi(train) == pytest.approx(0.6)


def test_rmse_known_value():
    assert rmse(pd.DataFrame({"y_target": [3.0, 0.0], "pred": [0.0, 4.0]})) == pytest.approx(np.sqrt(12.5))


def test_condition_tables_use_phase_boundaries():
    pred = pd.DataFrame({
        "h": 1,
        "season": [2010, 2010, 2010, 2011, 2012],
        "doy_target": [256, 257, 298, 200, 200],
        "y_target": [10.0, 10.0, 10.0, 20.0, 30.0],
        "pred": [9.0, 8.0, 7.0, 20.0, 30.0],
    })
    phase, magnitude = condition_tables(pred)
    assert phase.loc[1, "peak"] == pytest.approx(2.0) and phase.loc[1, "falling"] == pytest.approx(3.0)
    assert magnitude.loc[1, "large"] == pytest.approx(0.0)


def test_holm_known_answer():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def _two_model_predictions(scale):
    rng = np.random.default_rng(1)
    rows = []
    for season in range(2010, 2021):
        for t in range(3, 10):
            y = 5000 + rng.normal()
            for model, err in (("anom_ar1", rng.uniform(90, 110)), ("other", rng.uniform(90, 110) * scale)):
                rows.append({"season": season, "t": t, "doy_target": 200, "y_target": y,
                             "model": model, "h": 1, "pred": y + err * rng.choice([-1, 1])})
    return pd.DataFrame(rows)


def test_standard_tables_have_team_columns():
    pred = _two_model_predictions(scale=0.5)
    table = rmse_table(pred)
    assert list(table.index) == ["anom_ar1", "other"] and list(table.columns) == [1]
    tests = compare_with_anom_ar1(season_mse(pred))
    assert list(tests.columns) == ["model", "h", "n_seasons", "rmse_ratio", "wins", "mse_diff_pct",
                                   "mse_diff_pct_ci_low", "mse_diff_pct_ci_high", "p_t", "p_wilcoxon",
                                   "p_tost", "p_t_holm"]
    row = tests.iloc[0]
    assert row["rmse_ratio"] == pytest.approx(0.5, abs=0.05) and row["wins"] == 11


def test_season_peaks_takes_peak_day_and_mean():
    level1 = pd.DataFrame({"season": [2000, 2000, 2001, 2001], "doy": [153, 161, 153, 161],
                           "target_km2": [1.0, 5.0, 8.0, 2.0]})
    out = season_peaks(level1).set_index("season")
    assert out.loc[2000, "peak_km2"] == 5.0 and out.loc[2000, "peak_doy"] == 161
    assert out.loc[2001, "peak_doy"] == 153 and out.loc[2001, "mean_km2"] == 5.0


def test_inference_detects_a_clear_decline():
    seasons = np.arange(2000, 2021)
    peaks = pd.DataFrame({"season": seasons,
                          "peak_km2": 16000.0 - 300 * (seasons - 2000) + (seasons % 3) * 50,
                          "peak_doy": 281 + (seasons % 3) * 8})
    tests = inference(peaks).set_index("test")
    assert tests.loc["Mann-Kendall (peak ~ year)", "statistic"] < -0.9
    assert tests.loc["OLS HC3 (peak ~ year)", "statistic"] == pytest.approx(-300, abs=10)
    assert tests.loc["Mann-Whitney (peak before vs from 2010)", "p"] < 0.001
