"""Test bước 1 (run_forecast.py) trên dữ liệu thật, với pipeline giả thay cho Chronos-2."""
import numpy as np
import pytest

from chronos2_fakes import FakePipeline
from chronos_model import Chronos2ZeroShot
from flood_data import load_config, load_flood_matrix, step_to_doy
from protocol import climatology
from run_forecast import make_predictions


@pytest.fixture(scope="module")
def setup():
    config = load_config()
    Y = load_flood_matrix(config)
    zero_anomaly = Chronos2ZeroShot(FakePipeline(step_value=0.0))
    predictions = make_predictions(Y, config, zero_anomaly)
    return config, Y, predictions


def test_columns_match_team_table(setup):
    _, _, predictions = setup
    assert list(predictions.columns) == ["season", "t", "doy_target", "y_target", "model", "h", "pred"]


def test_row_count_matches_protocol(setup):
    _, _, predictions = setup
    counts = predictions["model"].value_counts()
    # 11 mùa test × (24 gốc cho h=1 + 23 cho h=2 + 21 cho h=4)
    assert counts["chronos2"] == 748
    assert counts["anom_ar1"] == 748


def test_target_is_h_steps_after_origin(setup):
    config, Y, predictions = setup
    first_year = config["first_year"]
    for row in predictions.itertuples():
        assert row.y_target == Y[row.season - first_year, row.t + row.h - 1]
        assert row.doy_target == step_to_doy(config, row.t + row.h)


def test_zero_anomaly_gives_climatology(setup):
    config, Y, predictions = setup
    chronos = predictions[predictions["model"] == "chronos2"]
    for row in chronos.itertuples():
        season_idx = row.season - config["first_year"]
        expected = climatology(Y, season_idx)[row.t + row.h - 1]
        assert row.pred == pytest.approx(expected)


def test_chronos_called_once_per_origin(setup):
    config, Y, _ = setup
    pipeline = FakePipeline()
    make_predictions(Y, config, Chronos2ZeroShot(pipeline))
    # 11 mùa × 24 gốc (t = 3..26); mỗi lần gọi trả luôn 4 bước tới
    assert len(pipeline.calls) == 11 * 24
    assert {call["prediction_length"] for call in pipeline.calls} == {4}
    context_lengths = [len(call["inputs"][0]) for call in pipeline.calls]
    assert context_lengths[:24] == list(range(3, 27))  # gốc t có ngữ cảnh đúng t bước
