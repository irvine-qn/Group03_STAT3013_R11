import numpy as np

from floodbench.cv import rolling_origin
from floodbench.data.regions import load_region
from floodbench.features import build_rows
from floodbench.models.baselines import anom_ar1
from floodbench.models.ml import lgbm_anomaly, lgbm_noclim, lgbm_raw
from floodbench.models.stat import gam

CFG = load_region("mekong")


def synthetic_Y(seed=0):
    """Mùa lũ hình chuông cộng nhiễu AR(1), đỉnh giảm dần theo năm."""
    rng = np.random.default_rng(seed)
    steps = np.arange(27)
    Y = np.zeros((21, 27))
    for i in range(21):
        noise = np.zeros(27)
        for k in range(1, 27):
            noise[k] = 0.8 * noise[k - 1] + rng.normal(0, 150)
        Y[i] = 2000 + (9000 - 150 * i) * np.exp(-((steps - 14) ** 2) / 40) + noise
    return Y


def test_every_model_beats_climatology_on_learnable_data_and_returns_right_shape():
    Y = synthetic_Y()
    fold = next(iter(rolling_origin(CFG, first_test=2012, last_test=2012)))
    idx = lambda ss: [s - CFG.first_year for s in ss]
    train = build_rows(Y, CFG, idx(fold.train_seasons), 2)
    test = build_rows(Y, CFG, idx([fold.test_season]), 2)
    clim_err = np.sqrt(np.mean((test["y_target"] - test["clim_all_target"]) ** 2))
    for fn in (anom_ar1, gam, lgbm_raw, lgbm_anomaly, lgbm_noclim):
        pred = fn(train, test, 2)
        assert pred.shape == (len(test),) and np.isfinite(pred).all()
        assert np.sqrt(np.mean((test["y_target"] - pred) ** 2)) < clim_err
