"""test cho level3, analysis và eda."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "process"))
from analysis import condition_tables, direct_phi, rmse
from eda import inference
from level3 import LAND_USE, build_level3


def test_level3_takes_peak_day_mean_and_land_use():
    level1 = pd.DataFrame(
        {
            "season": [2000, 2000, 2001, 2001],
            "doy": [153, 161, 153, 161],
            "target_km2": [1.0, 5.0, 8.0, 2.0],
        }
    )
    land = {f"lu{c}_km2": [float(c)] * 4 for c in LAND_USE}
    interim = pd.DataFrame({"year": [2000, 2000, 2001, 2001], **land})
    out = build_level3(level1, interim).set_index("season")
    assert len(out) == 2
    assert out.loc[2000, "peak_km2"] == 5.0 and out.loc[2000, "peak_doy"] == 161
    assert out.loc[2001, "peak_doy"] == 153 and out.loc[2001, "mean_km2"] == 5.0
    assert (
        out.loc[2000, "aquaculture_km2"] == 1.0
        and out.loc[2000, "rice_triple_km2"] == 7.0
    )


def test_direct_phi_recovers_known_value():
    x = np.array([100.0, -50.0, 30.0])
    train = pd.DataFrame(
        {"y_t": x + 5, "c_t": 5.0, "y_target": 0.6 * x + 7, "c_target": 7.0}
    )
    assert direct_phi(train) == pytest.approx(0.6)


def test_rmse_known_value():
    assert rmse(
        pd.DataFrame({"y_target": [3.0, 0.0], "pred": [0.0, 4.0]})
    ) == pytest.approx(np.sqrt(12.5))


def test_condition_tables_use_phase_boundaries():
    pred = pd.DataFrame(
        {
            "h": 1,
            "season": [2010, 2010, 2010, 2011, 2012],
            "doy_target": [256, 257, 298, 200, 200],
            "y_target": [10.0, 10.0, 10.0, 20.0, 30.0],
            "pred": [9.0, 8.0, 7.0, 20.0, 30.0],
        }
    )
    phase, magnitude = condition_tables(pred)
    assert phase.loc[1, "peak"] == pytest.approx(2.0) and phase.loc[
        1, "falling"
    ] == pytest.approx(3.0)
    assert magnitude.loc[1, "large"] == pytest.approx(0.0)


def test_inference_detects_a_clear_decline():
    seasons = np.arange(2000, 2021)
    level3 = pd.DataFrame(
        {
            "season": seasons,
            "peak_km2": 16000.0 - 300 * (seasons - 2000) + (seasons % 3) * 50,
            "peak_doy": 281 + (seasons % 3) * 8,
        }
    )
    tests = inference(level3).set_index("test")
    assert tests.loc["Mann-Kendall (peak ~ year)", "statistic"] < -0.9
    assert tests.loc["OLS HC3 (peak ~ year)", "statistic"] == pytest.approx(
        -300, abs=10
    )
    assert tests.loc["Mann-Whitney (peak before vs from 2010)", "p"] < 0.001
