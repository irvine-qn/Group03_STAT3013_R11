"""Test mô hình GAM viết lại.  Chạy từ gốc repo:  pytest tests/GAM

Nhóm 1 – không trôi kết quả: khóa vài dự báo của benchmark thăm dò và bảng kết quả đã lưu.
Nhóm 2 – đáp án biết trước trên dữ liệu giả.
Nhóm 3 – không rò rỉ: mùa tương lai không được làm đổi dự báo của fold cũ.
"""
import numpy as np
import pandas as pd
import pytest

from gam_common import TABLES, load_data, season_index
from gam_data import load_region
from gam_model import GAM_GRID, contributions, coefficients, design_matrix, fit_gam, gam_forecast, predict
from gam_protocol import build_rows
from gam_run_model import run

CFG = load_region("mekong")


def synthetic_Y(seed=0):
    """Ma trận giả 21 mùa x 27 bước, giá trị ngẫu nhiên kiểu diện tích ngập."""
    return np.random.default_rng(seed).uniform(1000, 9000, size=(21, 27))


# ---- Nhóm 1: không trôi kết quả ---------------------------------------------------------------
def test_grid_is_the_logged_grid():
    """Lưới 6 cấu hình đã ghi trong nhật ký tinh chỉnh của nhóm (ngân sách bằng mọi mô hình)."""
    assert GAM_GRID == [(6, 0.1), (6, 1.0), (8, 0.6), (8, 5.0), (12, 1.0), (12, 10.0)]


# Dự báo GAM của mùa 2010 tại 3 gốc đầu tiên, lấy từ benchmark thăm dò của nhóm
# (results/tables/probe_predictions.csv, trước khi bỏ gói chung). (tầm h) -> [(gốc t, km²)]
PROBE_2010 = {1: [(3, 6671.6637), (4, 4615.695434), (5, 3769.79497)],
              2: [(3, 6047.387284), (4, 4436.486652), (5, 3785.364175)],
              4: [(3, 5479.095268), (4, 4400.177563), (5, 5327.206285)]}


@pytest.mark.parametrize("h", [1, 2, 4])
def test_matches_probe_benchmark_for_2010(h):
    cfg, Y = load_data()
    train = build_rows(Y, cfg, season_index(cfg, range(2003, 2010)), h)
    test = build_rows(Y, cfg, season_index(cfg, [2010]), h)
    ours = test.assign(pred=gam_forecast(train, test, h).pred).set_index("t")["pred"]
    t, expected = zip(*PROBE_2010[h])
    np.testing.assert_allclose(ours.loc[list(t)].to_numpy(), expected, atol=1e-4)


def test_reproduces_saved_predictions_for_one_fold():
    """Fold 2010, tầm 16 ngày: phải trùng results/GAM/tables/gam_predictions.csv đã lưu."""
    cfg, Y = load_data()
    train = build_rows(Y, cfg, season_index(cfg, range(2003, 2010)), 2)
    test = build_rows(Y, cfg, season_index(cfg, [2010]), 2)
    ours = test.assign(pred=gam_forecast(train, test, 2).pred).set_index("t")["pred"]
    saved = pd.read_csv(TABLES / "gam_predictions.csv").query("model == 'gam' and h == 2 and season == 2010")
    np.testing.assert_allclose(ours.loc[saved["t"]].to_numpy(), saved["pred"].to_numpy(), atol=1e-6)


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
