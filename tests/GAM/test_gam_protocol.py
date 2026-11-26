"""Test luật chơi chung (gam_protocol.py) bằng đáp án biết trước.  Chạy: pytest tests/GAM"""
import numpy as np
import pandas as pd
import pytest

from gam_data import load_flood_matrix, load_region
from gam_protocol import (anom_ar1, build_rows, climatology, compare_to_baseline, holm, pick_config,
                          rolling_origin, season_mse)

CFG = load_region("mekong")


def synthetic_Y(seed=0):
    return np.random.default_rng(seed).uniform(1000, 9000, size=(21, 27))


# ---- Dữ liệu ------------------------------------------------------------------------------------
def test_real_matrix_shape_and_calendar():
    Y = load_flood_matrix(CFG)
    assert Y.shape == (21, 27) and np.isfinite(Y).all()
    assert CFG.doy_of(1) == 153 and CFG.doy_of(27) == 361


# ---- Mức bình thường ----------------------------------------------------------------------------
def test_climatology_is_mean_of_previous_five_seasons():
    Y = synthetic_Y()
    np.testing.assert_allclose(climatology(Y, 10), Y[5:10].mean(axis=0))
    np.testing.assert_allclose(climatology(Y, 3), Y[0:3].mean(axis=0))  # chỉ có 3 mùa trước


# ---- Bảng dòng ----------------------------------------------------------------------------------
@pytest.mark.parametrize("h, n_rows", [(1, 24), (2, 23), (4, 21)])
def test_rows_per_season(h, n_rows):
    rows = build_rows(synthetic_Y(), CFG, [10], h)
    assert len(rows) == n_rows
    assert rows["t"].min() == 3 and rows["t"].max() == 27 - h


def test_row_values_point_to_the_right_cells():
    Y = synthetic_Y()
    r = build_rows(Y, CFG, [10], 2).query("t == 11").iloc[0]
    c = Y[5:10].mean(axis=0)
    assert r.y_t == Y[10, 10] and r.y_t1 == Y[10, 9] and r.y_target == Y[10, 12]
    assert r.c_t == pytest.approx(c[10]) and r.c_target == pytest.approx(c[12])
    assert r.doy_target == 249 and r.season == 2010


def test_rows_of_a_season_ignore_later_seasons():
    Y = synthetic_Y()
    Y2 = Y.copy()
    Y2[11:] = 0.0
    pd.testing.assert_frame_equal(build_rows(Y, CFG, [10], 2), build_rows(Y2, CFG, [10], 2))


def test_too_few_prior_seasons_is_an_error():
    with pytest.raises(ValueError):
        build_rows(synthetic_Y(), CFG, [2], 1)


# ---- Fold ---------------------------------------------------------------------------------------
def test_rolling_origin_folds():
    folds = list(rolling_origin(CFG))
    assert [f.test_season for f in folds] == list(range(2010, 2021))
    assert folds[0].train_seasons == list(range(2003, 2010))
    assert folds[-1].train_seasons == list(range(2003, 2020))
    assert list(rolling_origin(CFG, window=7))[-1].train_seasons == list(range(2013, 2020))


# ---- Chọn cấu hình ------------------------------------------------------------------------------
def test_pick_config_scores_only_the_last_three_seasons():
    rows = pd.DataFrame({"season": np.repeat(range(2003, 2010), 2), "y_target": 1.0})
    seen = {}

    def fit_predict(fit, valid, cfg, h):
        seen["fit"], seen["valid"] = sorted(fit.season.unique()), sorted(valid.season.unique())
        return np.full(len(valid), cfg)

    best, scores = pick_config(rows, [0.0, 1.0, 3.0], fit_predict, 1)
    assert best == 1.0 and scores == [1.0, 0.0, 4.0]
    assert seen["fit"] == list(range(2003, 2007)) and seen["valid"] == [2007, 2008, 2009]


# ---- AnomAR1 ------------------------------------------------------------------------------------
def test_anom_ar1_recovers_known_phi():
    x = np.array([100.0, -50.0, 30.0, 200.0])
    train = pd.DataFrame({"y_t1": 10 + x, "c_t1": 10.0, "y_t": 10 + 0.8 * x, "c_t": 10.0})
    test = pd.DataFrame({"y_t": [600.0], "c_t": [500.0], "c_target": [700.0]})
    assert anom_ar1(train, test, 2)[0] == pytest.approx(700 + 0.8 ** 2 * 100)


# ---- Kiểm định ----------------------------------------------------------------------------------
def test_holm_known_values():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])


def test_season_mse_and_compare_to_baseline():
    pred = pd.DataFrame({"model": ["m"] * 4 + ["b"] * 4, "h": 1, "season": [1, 1, 2, 2] * 2,
                         "y_target": 0.0, "pred": [1, 1, 2, 2, 2, 2, 2, 2]})
    smse = season_mse(pred)
    assert smse.query("model == 'm'")["mse"].tolist() == [1.0, 4.0]
    out = compare_to_baseline(smse.assign(mse=smse["mse"] + [0, 0, 0, 0.5]), "b").iloc[0]
    assert out.model == "m" and out.wins == 1 and out.n_seasons == 2
