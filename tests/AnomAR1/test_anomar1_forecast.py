"""Test luật chơi, công thức AnomAR1 và bảng dự báo."""
import numpy as np
import pandas as pd
import pytest

from anomar1_data import load_config
from anomar1_forecast import COLUMNS, regression_table, run
from anomar1_protocol import anom_ar1, build_rows, climatology, estimate_phi, training_seasons

CONFIG = load_config("mekong")


def synthetic_Y(seed=0):
    return np.random.default_rng(seed).uniform(1000, 9000, size=(21, 27))


def test_estimate_phi_recovers_known_value():
    x = np.array([100.0, -50.0, 30.0, 200.0])
    rows = pd.DataFrame({"y_t1": 10 + x, "c_t1": 10.0, "y_t": 10 + 0.8 * x, "c_t": 10.0})
    assert estimate_phi(rows) == pytest.approx(0.8)


def test_anom_ar1_matches_hand_formula():
    Y = synthetic_Y()
    train = build_rows(Y, CONFIG, range(2003, 2010), 2)
    test = build_rows(Y, CONFIG, [2010], 2)
    phi = estimate_phi(train)
    hand = test["c_target"] + phi**2 * (test["y_t"] - test["c_t"])
    np.testing.assert_allclose(anom_ar1(train, test, 2), hand)


def test_climatology_uses_only_prior_seasons():
    Y = synthetic_Y()
    np.testing.assert_allclose(climatology(Y, 10), Y[5:10].mean(axis=0))
    with pytest.raises(ValueError):
        climatology(Y, 2)


def test_training_seasons_and_window():
    assert training_seasons(2010, 2000) == list(range(2003, 2010))
    assert training_seasons(2020, 2000, window=7) == list(range(2013, 2020))


def test_regression_table_has_expected_rows_per_horizon():
    table = regression_table(synthetic_Y(), CONFIG)
    assert table.groupby("h").size().to_dict() == {1: 18 * 24, 2: 18 * 23, 4: 18 * 21}
    assert table["season"].min() == 2003


def test_prediction_table_matches_team_columns_and_size():
    pred, phi = run(synthetic_Y(), CONFIG)
    assert list(pred.columns) == COLUMNS
    # mỗi mô hình: 11 mùa × (24 + 23 + 21) gốc
    assert pred.groupby("model").size().to_dict() == {m: 11 * 68 for m in
                                                      ["anom_ar1", "anom_ar1_w7", "clim5", "persistence"]}
    assert len(phi) == 2 * 11 * 3


def test_future_seasons_do_not_change_a_fold_prediction():
    Y = synthetic_Y()
    Y2 = Y.copy()
    Y2[11:] = synthetic_Y(seed=9)[11:]
    a, _ = run(Y, CONFIG)
    b, _ = run(Y2, CONFIG)
    key = (a["season"] == 2010) & (a["model"] == "anom_ar1")
    np.testing.assert_allclose(a.loc[key, "pred"], b.loc[key, "pred"])
