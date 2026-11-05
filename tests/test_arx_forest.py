import numpy as np
import pandas as pd

from floodbench.models.forest import RF_GRID, rf_anomaly, rf_noclim, rf_raw
from floodbench.models.stat import ARX_GRID, RIDGE_GRID, arx, ridge


def _rows(n, rng, phi=0.8, shift=0.0):
    """Dòng đặc trưng giả: anomaly AR(1) chính xác, không nhiễu, climatology phẳng."""
    a2 = rng.normal(0, 300, n)
    a1 = rng.normal(0, 300, n)
    a0 = phi * a1
    c = 5000.0 + shift
    rows = pd.DataFrame({
        "season": np.repeat(np.arange(2000, 2000 + n // 10), 10)[:n],
        "y_t": c + a0, "y_t1": c + a1, "y_t2": c + a2,
        "c_t": c, "c_t1": c, "c_t2": c, "c_target": c,
        "doy_target": np.tile(np.linspace(153, 361, 10), n // 10 + 1)[:n],
    })
    rows["y_target"] = c + phi * rows["y_t"] - phi * c  # anomaly đích = phi * anomaly hôm nay
    return rows


def test_arx_recovers_a_noiseless_ar1_exactly():
    rng = np.random.default_rng(0)
    train, test = _rows(120, rng), _rows(40, rng)
    pred = arx(train, test, 1)
    assert np.allclose(pred, test["y_target"], atol=1e-6)


def test_ridge_scaler_uses_training_rows_only():
    """Dịch cả bảng test khỏi miền huấn luyện không được làm đổi cách chuẩn hóa: dự báo = đường thẳng đã học."""
    rng = np.random.default_rng(1)
    train, test = _rows(120, rng), _rows(40, rng)
    base = ridge(train, test, 1)
    shifted = test.copy()
    shifted[["y_t", "y_t1", "y_t2"]] += 10_000  # anomaly = y - c lớn bất thường
    big = ridge(train, shifted, 1)
    # Nếu scaler bị fit lại trên test thì 2 dự báo sẽ gần nhau; học trên train thì chênh theo hệ số học được
    assert np.abs(big - base).mean() > 1_000


def test_grids_have_equal_budget():
    assert len(ARX_GRID) == len(RIDGE_GRID) == len(RF_GRID) == 6


def test_forest_variants_return_finite_right_shape():
    rng = np.random.default_rng(2)
    train, test = _rows(150, rng), _rows(30, rng)
    for fn in (rf_raw, rf_anomaly, rf_noclim):
        pred = fn(train, test, 1)
        assert pred.shape == (len(test),) and np.isfinite(pred).all()
