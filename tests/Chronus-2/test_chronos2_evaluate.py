"""Test bước 2 (chronos2_evaluate.py): các thước đo và kiểm định có đáp án biết trước."""
import numpy as np
import pandas as pd
import pytest

from chronos2_evaluate import (
    add_flood_phase,
    add_season_part,
    compare_with_anom_ar1,
    holm,
    load_predictions,
    pooled_rmse,
    season_mse,
)


# Dữ liệu giả có tỉ lệ MSE như nhau ở mọi mùa nên phương sai bằng 0: t-test/TOST chia cho 0 là bình thường.
DEGENERATE = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def fake_season_errors(ratio_of_mse):
    """MSE theo mùa của AnomAR1 và một mô hình có MSE = ratio_of_mse × AnomAR1."""
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) * 1e5
    rows = []
    for season, mse in zip(range(2010, 2016), base):
        rows.append({"model": "anom_ar1", "h": 1, "season": season, "mse": mse})
        rows.append({"model": "chronos2", "h": 1, "season": season, "mse": ratio_of_mse * mse})
    return pd.DataFrame(rows)


def test_holm_known_answer():
    np.testing.assert_allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


def test_pooled_rmse_known_answer():
    pred = pd.DataFrame({"model": ["m", "m"], "h": [1, 1], "y_target": [10.0, 10.0], "pred": [7.0, 14.0]})
    result = pooled_rmse(pred, by=["model", "h"])
    assert result["rmse"].iloc[0] == pytest.approx(np.sqrt(12.5))


def test_season_mse_known_answer():
    pred = pd.DataFrame({
        "model": ["m"] * 3, "h": [1] * 3, "season": [2010, 2010, 2011],
        "y_target": [0.0, 0.0, 0.0], "pred": [1.0, 3.0, 2.0],
    })
    result = season_mse(pred).set_index("season")["mse"]
    assert result[2010] == pytest.approx(5.0)
    assert result[2011] == pytest.approx(4.0)


@DEGENERATE
def test_compare_equal_models():
    row = compare_with_anom_ar1(fake_season_errors(1.0)).iloc[0]
    assert row["rmse_ratio"] == pytest.approx(1.0)
    assert row["mse_diff_pct"] == pytest.approx(0.0)
    assert row["wins"] == 0


@DEGENERATE
def test_compare_better_model():
    row = compare_with_anom_ar1(fake_season_errors(0.81)).iloc[0]
    assert row["rmse_ratio"] == pytest.approx(0.9)
    assert row["wins"] == row["n_seasons"] == 6


def test_phase_boundaries():
    pred = pd.DataFrame({"doy_target": [256, 257, 297, 298]})
    assert list(add_flood_phase(pred)["phase"]) == ["rising", "peak", "peak", "falling"]


def test_missing_predictions_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="chronos2_forecast.py"):
        load_predictions(tmp_path / "x.csv")


def test_season_part_split_at_origin_21():
    pred = pd.DataFrame({"t": [3, 20, 21, 26]})
    assert list(add_season_part(pred)["season_part"]) == ["t<=20", "t<=20", "t>=21", "t>=21"]
