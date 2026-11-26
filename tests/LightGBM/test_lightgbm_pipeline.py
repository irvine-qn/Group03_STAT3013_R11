"""Test luật chơi, bảng dự báo và bảng đánh giá chuẩn của LightGBM."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lightgbm_data import load_config
from lightgbm_evaluate import compare_with_anom_ar1, holm, rmse_table, season_mse
from lightgbm_forecast import COLUMNS, run
from lightgbm_protocol import anom_ar1, build_rows, climatology, clim5, pick_config, rolling_origin

CONFIG = load_config("mekong")
SRC = Path(__file__).resolve().parents[2] / "src" / "LightGBM"


def synthetic_Y(seed=0):
    return np.random.default_rng(seed).uniform(1000, 9000, size=(21, 27))


def test_modules_have_lightgbm_prefix():
    names = [p.name for p in SRC.glob("*.py")]
    assert names and all(n.startswith("lightgbm_") for n in names)


def test_rolling_origin_folds():
    folds = list(rolling_origin(CONFIG))
    assert [f.test_season for f in folds] == list(range(2010, 2021))
    assert folds[0].train_seasons == list(range(2003, 2010))
    assert list(rolling_origin(CONFIG, window=7))[-1].train_seasons == list(range(2013, 2020))


def test_rows_use_only_prior_seasons_for_climatology():
    Y = synthetic_Y()
    rows = build_rows(Y, CONFIG, [2010], 2)
    c = Y[5:10].mean(axis=0)
    first = rows.iloc[0]  # t = 3
    assert first["t"] == 3 and first["c_t"] == pytest.approx(c[2]) and first["c_target"] == pytest.approx(c[4])
    assert first["y_t2"] == Y[10, 0] and first["y_target"] == Y[10, 4]
    assert len(rows) == 27 - 2 - 2  # gốc t = 3..25
    with pytest.raises(ValueError):
        climatology(Y, 2)


def test_pick_config_validates_on_last_three_training_seasons():
    train = pd.DataFrame({"season": np.repeat(np.arange(2003, 2010), 2), "y_target": 1.0})
    seen = []

    def fit_predict(fit, valid, cfg, h):
        seen.append((sorted(fit["season"].unique()), sorted(valid["season"].unique())))
        return np.full(len(valid), cfg)

    best, scores = pick_config(train, [5.0, 1.0, 3.0], fit_predict, 1)
    assert best == 1.0 and scores == [16.0, 0.0, 4.0]
    assert seen[0] == (list(range(2003, 2007)), [2007, 2008, 2009])


def test_prediction_table_matches_team_columns():
    pred = run(synthetic_Y(), CONFIG, {"anom_ar1": anom_ar1, "clim5": clim5})
    assert list(pred.columns) == COLUMNS
    assert pred.groupby("model").size().to_dict() == {"anom_ar1": 11 * 68, "clim5": 11 * 68}


def test_future_seasons_do_not_change_a_fold_prediction():
    Y = synthetic_Y()
    Y2 = Y.copy()
    Y2[11:] = synthetic_Y(seed=9)[11:]
    a = run(Y, CONFIG, {"anom_ar1": anom_ar1})
    b = run(Y2, CONFIG, {"anom_ar1": anom_ar1})
    key = a["season"] == 2010
    np.testing.assert_allclose(a.loc[key, "pred"], b.loc[key, "pred"])


def test_holm_known_answer():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_standard_tables_have_team_columns():
    rng = np.random.default_rng(1)
    rows = []
    for season in range(2010, 2021):
        for t in range(3, 10):
            y = 5000.0
            for model, err in (("anom_ar1", rng.uniform(90, 110)), ("lgbm_raw", rng.uniform(45, 55))):
                rows.append({"season": season, "t": t, "doy_target": 200, "y_target": y,
                             "model": model, "h": 1, "pred": y + err * rng.choice([-1, 1])})
    pred = pd.DataFrame(rows)
    assert list(rmse_table(pred).index) == ["anom_ar1", "lgbm_raw"]
    tests = compare_with_anom_ar1(season_mse(pred))
    assert list(tests.columns) == ["model", "h", "n_seasons", "rmse_ratio", "wins", "mse_diff_pct",
                                   "mse_diff_pct_ci_low", "mse_diff_pct_ci_high", "p_t", "p_wilcoxon",
                                   "p_tost", "p_t_holm"]
    assert tests.iloc[0]["rmse_ratio"] == pytest.approx(0.5, abs=0.05) and tests.iloc[0]["wins"] == 11
