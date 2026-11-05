import numpy as np
import pandas as pd
import pytest

from floodbench.data.regions import load_region, repo_root
from floodbench.features import FEATURE_COLUMNS, build_rows, climatology, to_wide

CFG = load_region("mekong")
N_STEPS = CFG.n_steps


def random_Y(n_seasons=12, seed=0):
    return np.random.default_rng(seed).uniform(1000, 9000, size=(n_seasons, N_STEPS))


def test_climatology_is_mean_of_exactly_the_five_previous_seasons():
    Y = np.arange(10 * N_STEPS, dtype=float).reshape(10, N_STEPS)
    np.testing.assert_allclose(climatology(Y, 7), Y[2:7].mean(axis=0))
    np.testing.assert_allclose(climatology(Y, 3), Y[0:3].mean(axis=0))  # fewer than five available
    assert np.isnan(climatology(Y, 0)).all()


def test_row_has_known_values():
    Y = random_Y()
    rows = build_rows(Y, CFG, [8], h=2)
    r = rows[rows["t"] == 10].iloc[0]  # origin step 10 -> index 9, target step 12 -> index 11
    c = Y[3:8].mean(axis=0)
    assert r["y_t"] == Y[8, 9] and r["y_t1"] == Y[8, 8] and r["y_t2"] == Y[8, 7]
    assert r["c_t"] == pytest.approx(c[9]) and r["c_target"] == pytest.approx(c[11])
    assert r["y_target"] == Y[8, 11]
    assert r["y_prev_season"] == Y[7, 11]
    assert r["clim_all_target"] == pytest.approx(Y[:8, 11].mean())
    assert r["doy_target"] == CFG.doy_of(12)


def test_features_of_a_season_do_not_depend_on_later_seasons():
    Y = random_Y()
    i = 6
    base = build_rows(Y, CFG, [i], h=2)
    Y2 = Y.copy()
    Y2[i + 1:] = random_Y(len(Y) - i - 1, seed=99)
    pd.testing.assert_frame_equal(base, build_rows(Y2, CFG, [i], h=2))


def test_features_at_an_origin_do_not_depend_on_later_steps_of_the_same_season():
    Y = random_Y()
    i, origin = 6, 12  # 1-based step
    Y2 = Y.copy()
    Y2[i, origin:] = 123456.0  # everything after the origin, in the same season
    a = build_rows(Y, CFG, [i], h=2).set_index("t").loc[origin, FEATURE_COLUMNS]
    b = build_rows(Y2, CFG, [i], h=2).set_index("t").loc[origin, FEATURE_COLUMNS]
    pd.testing.assert_series_equal(a, b)


def leaky_rows(Y, i, h):
    """A deliberately wrong builder: climatology over ALL seasons, including later ones."""
    rows = build_rows(Y, CFG, [i], h)
    rows["c_target"] = Y[:, rows["t"].to_numpy() + h - 1].mean(axis=0)
    return rows


def test_leak_detection_catches_a_deliberately_leaky_builder():
    Y = random_Y()
    i = 6
    Y2 = Y.copy()
    Y2[i + 1:] = random_Y(len(Y) - i - 1, seed=99)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(leaky_rows(Y, i, 2), leaky_rows(Y2, i, 2))


def test_season_index_below_min_prior_is_rejected():
    with pytest.raises(ValueError):
        build_rows(random_Y(), CFG, [1], h=1)


def test_to_wide_matches_level1_file():
    level1 = pd.read_csv(repo_root() / "data" / "processed" / "mekong" / "level1_series.csv")
    Y = to_wide(level1, CFG)
    assert Y.shape == (21, 27)
    first = level1[(level1.season == 2000) & (level1.step == 1)].target_km2.iloc[0]
    assert Y[0, 0] == first
    assert not np.isnan(Y).any()
