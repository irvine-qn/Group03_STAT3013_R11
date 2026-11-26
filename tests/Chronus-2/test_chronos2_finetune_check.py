"""Test kiểm tra phụ fine-tune (chronos2_finetune_check.py) bằng pipeline giả: không học thật."""
import numpy as np
import pandas as pd

from chronos2_data import load_config, load_flood_matrix
from chronos2_fakes import FakeTrainablePipeline
from chronos2_finetune_check import FINETUNE_CONFIGS, finetuned_predictions, summary_table, training_series
from chronos2_protocol import anomaly


def test_training_series_uses_only_training_seasons():
    config = load_config()
    Y = load_flood_matrix(config)
    series = training_series(Y, 2010, config["first_year"])
    assert len(series) == 7  # mùa học 2003..2009
    np.testing.assert_allclose(series[0], anomaly(Y, 3))
    Y_future_changed = Y.copy()
    Y_future_changed[10:] = 1e9  # mùa test 2010 và các mùa sau không được lọt vào dữ liệu học
    for before, after in zip(series, training_series(Y_future_changed, 2010, config["first_year"])):
        np.testing.assert_allclose(before, after)


def test_finetuned_predictions_refits_for_every_test_season():
    config = load_config()
    Y = load_flood_matrix(config)
    base = FakeTrainablePipeline()
    settings = FINETUNE_CONFIGS["chronos2_ft_full"]
    pred = finetuned_predictions(Y, config, base, "chronos2_ft_full", settings)
    assert len(base.fit_calls) == 11  # fine-tune lại cho từng mùa test 2010..2020
    assert [len(call["inputs"]) for call in base.fit_calls] == list(range(7, 18))
    assert all(call["prediction_length"] == 4 for call in base.fit_calls)
    assert base.fit_calls[0]["settings"]["learning_rate"] == settings["learning_rate"]
    assert set(pred["model"]) == {"chronos2_ft_full"}
    assert len(pred) == 748


def test_summary_table_has_horizons_and_season_parts():
    pred = pd.DataFrame({
        "model": ["m"] * 4, "season": [2010] * 4, "t": [3, 21, 3, 21], "h": [1, 1, 4, 4],
        "y_target": [0.0] * 4, "pred": [1.0, 2.0, 3.0, 4.0],
    })
    table = summary_table(pred).loc["m"]
    assert table["8 days"] == np.sqrt((1 + 4) / 2)
    assert table["32 days"] == np.sqrt((9 + 16) / 2)
    assert table["32 days, t<=20"] == 3.0
    assert table["32 days, t>=21"] == 4.0
