import numpy as np
import pandas as pd
import pytest

from floodbench.validation.waterlevel import compare_to_gauge, season_summary


def test_season_summary_takes_peak_and_mean_per_season():
    level1 = pd.DataFrame(
        {
            "season": [2000, 2000, 2000, 2001, 2001, 2001],
            "target_km2": [1.0, 5.0, 3.0, 2.0, 2.0, 8.0],
        }
    )
    out = season_summary(level1).set_index("season")
    assert out.loc[2000, "peak_km2"] == 5.0
    assert out.loc[2000, "mean_km2"] == pytest.approx(3.0)
    assert out.loc[2001, "peak_km2"] == 8.0


def test_shared_noise_survives_detrending():
    rng = np.random.default_rng(0)
    years = np.arange(2000, 2020)
    noise = rng.normal(size=years.size)
    area = pd.DataFrame({"season": years, "peak_km2": -300 * (years - 2000) + 1000 * noise})
    gauge = pd.DataFrame({"season": years, "level": -0.1 * (years - 2000) + 0.5 * noise})
    res = compare_to_gauge(area, gauge, "peak_km2", "level")
    assert res["n"] == 20
    assert res["r_levels"] > 0.9
    assert res["r_detrended"] > 0.9
    assert res["r_first_diff"] > 0.9


def test_shared_trend_only_is_exposed_by_detrending():
    rng = np.random.default_rng(1)
    years = np.arange(2000, 2020)
    area = pd.DataFrame({"season": years, "peak_km2": -300 * (years - 2000) + 150 * rng.normal(size=20)})
    gauge = pd.DataFrame({"season": years, "level": -0.1 * (years - 2000) + 0.05 * rng.normal(size=20)})
    res = compare_to_gauge(area, gauge, "peak_km2", "level")
    assert res["r_levels"] > 0.9
    assert abs(res["r_detrended"]) < 0.6


def test_only_overlapping_seasons_are_used():
    area = pd.DataFrame({"season": [2000, 2001, 2002, 2003], "peak_km2": [1.0, 2.0, 3.0, 4.5]})
    gauge = pd.DataFrame({"season": [2001, 2002, 2003, 2004], "level": [2.0, 3.0, 4.0, 9.0]})
    res = compare_to_gauge(area, gauge, "peak_km2", "level")
    assert res["n"] == 3
    assert (res["first_season"], res["last_season"]) == (2001, 2003)
