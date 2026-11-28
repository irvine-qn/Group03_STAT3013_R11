"""Test Chronos-2 zero-shot (chronos2_model.py) bằng pipeline giả: không tải model thật."""
import numpy as np
import torch

from chronos2_fakes import FakePipeline
from chronos2_model import Chronos2ZeroShot, build_anomaly_context, load_pipeline
from chronos2_data import load_config, load_flood_matrix
from chronos2_protocol import climatology


def test_context_contains_only_steps_up_to_origin():
    Y = load_flood_matrix(load_config())
    season_idx, origin = 15, 10
    context = build_anomaly_context(Y, season_idx, origin)
    assert len(context) == origin
    np.testing.assert_allclose(context, Y[season_idx, :origin] - climatology(Y, season_idx)[:origin])
    Y_future_changed = Y.copy()
    Y_future_changed[season_idx, origin:] = 1e9  # tương lai của mùa test không được lọt vào
    np.testing.assert_allclose(build_anomaly_context(Y_future_changed, season_idx, origin), context)


def test_forecast_picks_requested_horizon():
    pipeline = FakePipeline()
    forecast = Chronos2ZeroShot(pipeline).forecast_anomaly(np.array([1.0, 2.0, 3.0]), n_ahead=4)
    np.testing.assert_allclose(forecast, [10, 20, 30, 40])
    assert pipeline.calls[0]["prediction_length"] == 4
    assert pipeline.calls[0]["quantile_levels"] == [0.5]


def test_context_passed_as_float32_tensor():
    pipeline = FakePipeline()
    Chronos2ZeroShot(pipeline).forecast_anomaly(np.array([1.0, 2.0, 3.0]), n_ahead=2)
    (series,) = pipeline.calls[0]["inputs"]
    assert isinstance(series, torch.Tensor)
    assert series.dtype == torch.float32
    assert series.ndim == 1


def test_load_pipeline_passes_device(monkeypatch):
    """load_pipeline phải chuyển đúng thiết bị (cpu/cuda) cho Chronos2Pipeline, mặc định là cpu."""
    import chronos

    received = []
    monkeypatch.setattr(chronos.Chronos2Pipeline, "from_pretrained",
                        lambda model_id, device_map: received.append((model_id, device_map)))
    load_pipeline()
    load_pipeline(device="cuda")
    assert received == [("amazon/chronos-2", "cpu"), ("amazon/chronos-2", "cuda")]
