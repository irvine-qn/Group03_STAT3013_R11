"""Chống rò rỉ ở mức DỰ BÁO cho anom_ar1, gam, lgbm_raw và chronos2 (fold 2010, dữ liệu giả).

Hai kịch bản, mỗi kịch bản thay giá trị "tương lai" bằng 1e9:
  1. mùa tương lai: mọi mùa sau mùa test 2010 (2011..2020);
  2. bước tương lai trong mùa test: mùa 2010, các bước > t với gốc t = 10.
Dự báo fold 2010 (kịch bản 1) hoặc dự báo tại các gốc ≤ 10 (kịch bản 2) phải không đổi.

Mỗi kịch bản có thêm một "đối chứng dương": đổi giá trị ở bước gốc t (thông tin hợp lệ) thì dự báo
PHẢI đổi, để chắc test có khả năng phát hiện.

Mỗi mô hình được chạy đúng như vòng lặp trong *_forecast.py / gam_run_model.py cho một fold:
rolling origin → build_rows (bên trong fold) → hàm mô hình. Chronos-2 dùng pipeline giả của
tests/Chronos-2/chronos2_fakes.py, bản con có đầu ra phụ thuộc ngữ cảnh.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src/AnomAR1", "src/GAM", "src/LightGBM", "src/Chronos-2", "tests/Chronos-2"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

import anomar1_data  # noqa: E402
import anomar1_protocol  # noqa: E402
import gam_data  # noqa: E402
import gam_model  # noqa: E402
import gam_protocol  # noqa: E402
import lightgbm_data  # noqa: E402
import lightgbm_model  # noqa: E402
import lightgbm_protocol  # noqa: E402

TEST_SEASON, ORIGIN, BIG = 2010, 10, 1e9
FIRST_YEAR = 2000
HORIZONS = (1, 2, 4)


def synthetic_Y(seed=0):
    """Ma trận giả 21 mùa × 27 bước (cùng cách tạo với test cũ của từng mô hình)."""
    return np.random.default_rng(seed).uniform(1000, 9000, size=(21, 27))


def future_seasons(Y):
    Y = Y.copy()
    Y[TEST_SEASON - FIRST_YEAR + 1:] = BIG
    return Y


def future_steps(Y):
    Y = Y.copy()
    Y[TEST_SEASON - FIRST_YEAR, ORIGIN:] = BIG  # cột ORIGIN (0-based) = bước ORIGIN + 1, tức các bước > t
    return Y


def bump_origin(Y):
    """Đối chứng dương: đổi các bước 1..t của mùa test (được phép biết) về 0, ngoài khoảng 1000–9000 của
    dữ liệu giả. Đổi nhỏ (ví dụ +500 ở bước t) có thể không làm đổi dự báo cây vì vẫn rơi vào cùng một lá."""
    Y = Y.copy()
    Y[TEST_SEASON - FIRST_YEAR, :ORIGIN] = 0.0
    return Y


# ---- Dự báo fold 2010 của từng mô hình: DataFrame (t, h, pred) ---------------------------------
def forecast_anom_ar1(Y):
    config = anomar1_data.load_config("mekong")
    seasons = anomar1_protocol.training_seasons(TEST_SEASON, config["first_year"])
    out = []
    for h in HORIZONS:
        train = anomar1_protocol.build_rows(Y, config, seasons, h)
        test = anomar1_protocol.build_rows(Y, config, [TEST_SEASON], h)
        out.append(test[["t"]].assign(h=h, pred=anomar1_protocol.anom_ar1(train, test, h)))
    return pd.concat(out, ignore_index=True)


def forecast_gam(Y):
    cfg = gam_data.load_region("mekong")
    fold = next(gam_protocol.rolling_origin(cfg))
    assert fold.test_season == TEST_SEASON
    idx = lambda seasons: [s - cfg.first_year for s in seasons]  # noqa: E731
    out = []
    for h in HORIZONS:
        train = gam_protocol.build_rows(Y, cfg, idx(fold.train_seasons), h)
        test = gam_protocol.build_rows(Y, cfg, idx([fold.test_season]), h)
        out.append(test[["t"]].assign(h=h, pred=gam_model.gam_forecast(train, test, h).pred))
    return pd.concat(out, ignore_index=True)


def forecast_lgbm_raw(Y):
    config = lightgbm_data.load_config("mekong")
    fold = next(lightgbm_protocol.rolling_origin(config))
    assert fold.test_season == TEST_SEASON
    out = []
    for h in HORIZONS:
        train = lightgbm_protocol.build_rows(Y, config, fold.train_seasons, h)
        test = lightgbm_protocol.build_rows(Y, config, [fold.test_season], h)
        out.append(test[["t"]].assign(h=h, pred=lightgbm_model.lgbm_raw(train, test, h)))
    return pd.concat(out, ignore_index=True)


def forecast_chronos2(Y):
    pytest.importorskip("torch")
    import chronos2_data
    from chronos2_fakes import FakePipeline
    from chronos2_forecast import chronos_season
    from chronos2_model import Chronos2ZeroShot

    class ContextFakePipeline(FakePipeline):
        """Như FakePipeline nhưng dự báo phụ thuộc ngữ cảnh: bước k = tổng ngữ cảnh × 0,001 + k × 10.

        Nếu 1e9 lọt vào ngữ cảnh thì dự báo đổi, nên test không thể qua một cách vô nghĩa.
        """

        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            quantiles, mean = super().predict_quantiles(inputs, prediction_length, quantile_levels)
            shift = float(inputs[0].double().sum()) * 1e-3
            return [q + shift for q in quantiles], [m + shift for m in mean]

    config = chronos2_data.load_config("mekong")
    rows = chronos_season(Y, config, Chronos2ZeroShot(ContextFakePipeline()), TEST_SEASON)
    return pd.DataFrame(rows)[["t", "h", "pred"]]


FORECASTS = {"anom_ar1": forecast_anom_ar1, "gam": forecast_gam, "lgbm_raw": forecast_lgbm_raw,
             "chronos2": forecast_chronos2}
_cache = {}


def base_forecast(model):
    if model not in _cache:
        _cache[model] = FORECASTS[model](synthetic_Y())
    return _cache[model]


def _at_or_before_origin(frame):
    return frame[frame["t"] <= ORIGIN].sort_values(["h", "t"]).reset_index(drop=True)


# ---- Kịch bản 1: mùa tương lai = 1e9 -------------------------------------------------------------
@pytest.mark.parametrize("model", list(FORECASTS))
def test_future_seasons_1e9_do_not_change_fold_2010_forecasts(model):
    base = base_forecast(model).sort_values(["h", "t"]).reset_index(drop=True)
    changed = FORECASTS[model](future_seasons(synthetic_Y())).sort_values(["h", "t"]).reset_index(drop=True)
    assert len(base) == 24 + 23 + 21
    np.testing.assert_allclose(changed["pred"], base["pred"])


# ---- Kịch bản 2: bước tương lai trong mùa test = 1e9 ---------------------------------------------
@pytest.mark.parametrize("model", list(FORECASTS))
def test_future_steps_1e9_do_not_change_forecasts_at_origin(model):
    base = _at_or_before_origin(base_forecast(model))
    changed = _at_or_before_origin(FORECASTS[model](future_steps(synthetic_Y())))
    assert set(base.loc[base["t"] == ORIGIN, "h"]) == set(HORIZONS)  # gốc t = 10 có đủ 3 tầm
    np.testing.assert_allclose(changed["pred"], base["pred"])


# ---- Đối chứng dương: thông tin hợp lệ ở bước gốc PHẢI làm đổi dự báo ---------------------------
@pytest.mark.parametrize("model", list(FORECASTS))
def test_value_at_origin_does_change_forecasts(model):
    base = base_forecast(model)
    bumped = FORECASTS[model](bump_origin(synthetic_Y()))
    at = lambda f: f[f["t"] == ORIGIN].sort_values("h")["pred"].to_numpy()  # noqa: E731
    assert np.all(np.abs(at(bumped) - at(base)) > 1e-6)
