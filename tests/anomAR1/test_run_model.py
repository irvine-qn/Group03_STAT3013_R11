"""Chạy: pytest src/anomAR1/test"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "process"))
from floodbench.data.regions import load_region
from floodbench.features import build_rows
from floodbench.models.baselines import anom_ar1
from run_model import estimate_phi, regression_table, run

CFG = load_region("mekong")


def synthetic_Y(seed=0):
    return np.random.default_rng(seed).uniform(1000, 9000, size=(21, 27))


def test_estimate_phi_recovers_known_value():
    x = np.array([100.0, -50.0, 30.0, 200.0])
    rows = pd.DataFrame(
        {"y_t1": 10 + x, "c_t1": 10.0, "y_t": 10 + 0.8 * x, "c_t": 10.0}
    )
    assert estimate_phi(rows) == pytest.approx(0.8)


def test_hand_formula_matches_team_function():
    Y = synthetic_Y()
    train, test = build_rows(Y, CFG, range(3, 10), 2), build_rows(Y, CFG, [10], 2)
    phi = estimate_phi(train)
    hand = test["c_target"] + phi**2 * (test["y_t"] - test["c_t"])
    np.testing.assert_allclose(anom_ar1(train, test, 2), hand)


def test_regression_table_has_expected_rows_per_horizon():
    table = regression_table(synthetic_Y(), CFG)
    assert table.groupby("h").size().to_dict() == {1: 18 * 24, 2: 18 * 23, 4: 18 * 21}
    assert table["season"].min() == 2003


def test_future_seasons_do_not_change_a_fold_prediction():
    Y = synthetic_Y()
    Y2 = Y.copy()
    Y2[11:] = synthetic_Y(seed=9)[11:]
    a, _ = run(Y, CFG)
    b, _ = run(Y2, CFG)
    key = (a["season"] == 2010) & (a["model"] == "anom_ar1")
    np.testing.assert_allclose(a.loc[key, "pred"], b.loc[key, "pred"])
