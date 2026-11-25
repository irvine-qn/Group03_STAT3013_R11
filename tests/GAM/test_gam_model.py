"""Test mô hình GAM viết lại.  Chạy từ gốc repo:  pytest tests/GAM

Nhóm 1 – đúng với mô hình chung của nhóm (floodbench.models.stat.gam).
Nhóm 2 – đáp án biết trước trên dữ liệu giả.
Nhóm 3 – không rò rỉ: mùa tương lai không được làm đổi dự báo của fold cũ.
"""
import numpy as np
import pandas as pd
import pytest

from floodbench.data.regions import load_region
from floodbench.features import build_rows
from common import TEAM_PREDICTIONS, load_data, season_index
from gam_model import GAM_GRID, contributions, coefficients, design_matrix, fit_gam, gam_forecast, predict
from run_model import run

CFG = load_region("mekong")


def synthetic_Y(seed=0):
    """Ma trận giả 21 mùa x 27 bước, giá trị ngẫu nhiên kiểu diện tích ngập."""
    return np.random.default_rng(seed).uniform(1000, 9000, size=(21, 27))


# ---- Nhóm 1: khớp mô hình chung --------------------------------------------------------------
def test_grid_is_the_team_grid():
    stat = pytest.importorskip("floodbench.models.stat")  # cần scikit-learn
    assert GAM_GRID == stat.GAM_GRID


def test_same_predictions_as_team_function_on_synthetic_data():
    stat = pytest.importorskip("floodbench.models.stat")
    Y = synthetic_Y()
    train, test = build_rows(Y, CFG, range(3, 12), 2), build_rows(Y, CFG, [12], 2)
    np.testing.assert_allclose(gam_forecast(train, test, 2).pred, stat.gam(train, test, 2), rtol=1e-10)


def test_reproduces_stored_team_forecasts_for_one_fold():
    """Fold 2010, tầm 16 ngày trên dữ liệu thật: phải trùng results/tables/probe_predictions.csv."""
    cfg, Y = load_data()
    train = build_rows(Y, cfg, season_index(cfg, range(2003, 2010)), 2)
    test = build_rows(Y, cfg, season_index(cfg, [2010]), 2)
    ours = test.assign(pred=gam_forecast(train, test, 2).pred).set_index("t")["pred"]
    team = pd.read_csv(TEAM_PREDICTIONS).query("model == 'gam' and h == 2 and season == 2010")
    np.testing.assert_allclose(ours.loc[team["t"]].to_numpy(), team["pred"].to_numpy(), atol=1e-6)


# ---- Nhóm 2: đáp án biết trước ---------------------------------------------------------------
def test_design_matrix_columns():
    rows = pd.DataFrame({"doy_target": [200], "y_t": [500.0], "c_t": [300.0],
                         "y_t1": [100.0], "c_t1": [400.0]})
    np.testing.assert_allclose(design_matrix(rows), [[200, 200, -300]])  # [ngày, a_t, a_{t-1}]


def make_rows(n=600, seed=1):
    """Dòng giả: anomaly đích = 100 + 0.7·a_t (không phụ thuộc a_{t-1}, không phụ thuộc ngày)."""
    rng = np.random.default_rng(seed)
    a_t, a_t1 = rng.normal(0, 1000, n), rng.normal(0, 1000, n)
    c_t, c_t1, c_target = (rng.uniform(3000, 9000, n) for _ in range(3))
    return pd.DataFrame({
        "season": np.repeat(np.arange(2003, 2003 + n // 24 + 1), 24)[:n],
        "doy_target": rng.choice(np.arange(177, 362, 8), n),
        "y_t": c_t + a_t, "c_t": c_t, "y_t1": c_t1 + a_t1, "c_t1": c_t1, "c_target": c_target,
        "y_target": c_target + 100 + 0.7 * a_t + rng.normal(0, 20, n),
    })


def test_recovers_known_coefficients():
    rows = make_rows()
    model = fit_gam(rows, (8, 10.0))
    co = coefficients(model)
    assert co["beta1_a_t"] == pytest.approx(0.7, abs=0.01)
    assert co["beta2_a_t1"] == pytest.approx(0.0, abs=0.01)
    # khi a_t = a_{t-1} = 0, anomaly dự báo ≈ 100 ở mọi ngày
    doys = np.arange(185, 355, 10)
    X0 = np.column_stack([doys, np.zeros_like(doys), np.zeros_like(doys)])
    np.testing.assert_allclose(model.predict(X0), 100, atol=15)


def test_contributions_add_up_to_prediction():
    rows = make_rows()
    model = fit_gam(rows, (6, 1.0))
    parts = contributions(model, rows.head(30))
    np.testing.assert_allclose(parts["pred"], predict(model, rows.head(30)), rtol=1e-9)


def test_tuning_never_sees_test_rows():
    """Đổi nhãn của dòng test không được làm đổi cấu hình được chọn."""
    Y = synthetic_Y()
    train, test = build_rows(Y, CFG, range(3, 12), 1), build_rows(Y, CFG, [12], 1)
    a = gam_forecast(train, test, 1)
    b = gam_forecast(train, test.assign(y_target=test["y_target"] * 5), 1)
    assert a.config == b.config and a.val_mse == b.val_mse
    np.testing.assert_allclose(a.pred, b.pred)


# ---- Nhóm 3: không rò rỉ ---------------------------------------------------------------------
def test_future_seasons_do_not_change_a_fold_prediction():
    Y = synthetic_Y()
    Y2 = Y.copy()
    Y2[11:] = synthetic_Y(seed=9)[11:]  # đổi toàn bộ mùa 2011 trở đi
    a = run(Y, CFG, windows=((None, ""),))["predictions"]
    b = run(Y2, CFG, windows=((None, ""),))["predictions"]
    key = (a["season"] == 2010) & (a["model"] == "gam")
    np.testing.assert_allclose(a.loc[key, "pred"], b.loc[key, "pred"])
